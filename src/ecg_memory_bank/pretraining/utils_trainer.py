"""Distributed contrastive training loop.

Purpose
-------
Train and validate the ECGCLIP model across multiple GPUs while recording
checkpoint provenance.

Method
------
Use distributed samplers, AMP, AdamW and learning-rate scheduling, aggregate
contrastive metrics, and periodically save the encoder and full model.

Inputs
------
Model, optimizer, HEEDB datasets, trainer YAML values, and evaluation
configuration.

Outputs
-------
External model checkpoints plus loss and retrieval-accuracy summaries; no data
is written on import.

Paper parameters
----------------
batch_size, max_epochs, checkpoint_interval, num_workers, and output path come
from config; each rank handles a sampler shard.

Shapes and meaning
------------------
Per-rank ECG batches are [B,12,5000]; projected paired features are [B,256];
similarity matrices for contrastive loss are [B,B].
"""

import os
from typing import Type

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as torch_dist
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml as yaml

from utils_loss import clip_loss

import wandb

class trainer_wBert:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.val_batch_size = args['val_batch_size']
        self.checkpoint = args['checkpoint']


    def train_w_TextEmb(self, train_dataset, val_dataset, args_zeroshot_eval):

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))

        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                num_workers=self.num_workers,
                                drop_last=True, shuffle=False,
                                sampler=DistributedSampler(val_dataset))


        model_checkpoints_folder = os.path.join(self.checkpoint)
        if self.device == 0:
            if not os.path.exists(model_checkpoints_folder):
                print('create directory "{}" for save checkpoint!'.format(
                    model_checkpoints_folder))
                print('---------------------------')
                os.makedirs(model_checkpoints_folder)
            else:
                print('directory "{}" existing for save checkpoint!'.format(
                    model_checkpoints_folder))

        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=1,
            eta_min=1e-8,
        )
        step = 0
        eval_step = 5000
        niter = 1
        next_save = 5000
        skip_scheduler = False
        scaler = GradScaler()

        f1_total = []
        acc_total = []
        auc_total = []

        zeroshot_csv = pd.DataFrame()
        best_auc = 0

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):

            epoch_loss = 0
            epoch_acc1 = []
            epoch_acc5 = []
            self.model.train()
            for data in tqdm(train_loader):
                self.model.train()
                report = data['txt']

                ecg = data['ecg'].to(torch.float32).to(
                    self.device).contiguous()

                self.optimizer.zero_grad()

                with autocast():
                    report_tokenize_output = self.model.module._tokenize(report)

                    input_ids = report_tokenize_output.input_ids.to(
                        self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(
                        self.device).contiguous()

                    labels = None


                    output_dict = self.model(ecg, input_ids, attention_mask, labels=labels)
                    ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']


                    world_size = torch_dist.get_world_size()
                    with torch.no_grad():
                        agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]

                        dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                        dist.all_gather(agg_proj_text_emb, proj_text_emb[0])

                        agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                        dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                        dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                        rank = torch_dist.get_rank()

                    agg_proj_img_emb[rank] = proj_ecg_emb[0]
                    agg_proj_text_emb[rank] = proj_text_emb[0]

                    agg_proj_ecg_emb1[rank] = ecg_emb[0]
                    agg_proj_ecg_emb2[rank] = ecg_emb[1]


                    agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                    agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                    agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                    agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)

                    cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                    gen_loss = output_dict.get('gen_loss', torch.tensor(0.0).to(self.device))


                    loss = cma_loss
                    if self.device == 0:
                        print(f'loss is {loss.item()}, acc1 is {acc1.item()}, acc5 is {acc5.item()}, cma_loss is {cma_loss.item()}, gen: {gen_loss.item():.4f}')

                        wandb.log({
                            'train_step_gen_loss': gen_loss.item(),
                            'train_step_cma_loss': cma_loss.item(),
                            'train_step_total_loss': loss.item(),
                            'train_step_acc1': acc1.item(),
                            'train_step_acc5': acc5.item()}
                            )

                    epoch_loss += loss.item()
                    epoch_acc1.append(acc1.item())
                    epoch_acc5.append(acc5.item())

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if not skip_scheduler:
                        scheduler.step()
                niter += 1
                step += 1
                if step % eval_step  == 0:
                    self.save_checkpoints(self.model.module.state_dict(), model_checkpoints_folder + self.model_name + f'_{step}_ckpt.pth')
                if niter % 5000 == 0 and self.device == 0:
                    encoder_path = os.path.join(model_checkpoints_folder, self.model_name + f'_{niter}_final_encoder.pth')
                    total_model_path = os.path.join(model_checkpoints_folder, self.model_name + f'_{niter}_final_total.pth')

                    torch.save(self.model.module.ecg_encoder.state_dict(), encoder_path)

                    torch.save(self.model.module.state_dict(), total_model_path)

                    print(f"在 batch {niter} 处保存检查点:")
                    print(f"   Encoder -> {encoder_path}")
                    print(f"   Total model -> {total_model_path}")
            val_log = self.val(val_loader)

            if self.device == 0:
                epoch_acc1 = np.array(epoch_acc1).mean()
                epoch_acc5 = np.array(epoch_acc5).mean()

                epoch_iter = (len(train_dataset)//self.train_batch_size)
                print(f'{epoch_counter} epoch loss is {epoch_loss/epoch_iter},\
                                    acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')

                wandb.log({
                            'train_epoch_loss': epoch_loss/epoch_iter,
                            'train_epoch_acc1': epoch_acc1,
                            'train_epoch_acc5': epoch_acc5,
                            'val_cma_loss': val_log['val_cma_loss'],
                            'val_epoch_loss': val_log['val_loss'],
                            'val_epoch_acc1': val_log['val_acc1'],
                            'val_epoch_acc5': val_log['val_acc5']}
                            )


        if self.checkpoint_interval != 1:
            torch.save(self.model.module.ecg_encoder.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_encoder.pth')
            torch.save(self.model.module.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_total.pth')

    def val(self, loader):
        print('start validation')
        self.model.eval()
        val_cma_loss = 0
        val_gen_loss = 0
        val_loss = 0
        val_epoch_acc1 = []
        val_epoch_acc5 = []

        for data in tqdm(loader):
            report = data['txt']
            ecg = data['ecg'].to(torch.float32).to(
                self.device).contiguous()

            report_tokenize_output = self.model.module._tokenize(report)

            input_ids = report_tokenize_output.input_ids.to(
                self.device).contiguous()
            attention_mask = report_tokenize_output.attention_mask.to(
                self.device).contiguous()
            labels = input_ids.clone()
            labels[labels == self.model.module.tokenizer.pad_token_id] = -100
            if torch.any(labels > 50256) or torch.any((labels < 0) & (labels != -100)):
                print("----------- [Rank:", self.device, "] ERROR: Invalid labels found! -----------")
                print("Labels Min:", torch.min(labels))
                print("Labels Max:", torch.max(labels))
                print("Vocab Size: 50257")
                problematic_labels = labels[(labels > 50256) | ((labels < 0) & (labels != -100))]
                print("Problematic Labels:", problematic_labels)
            with torch.no_grad():
                output_dict = self.model(ecg, input_ids, attention_mask, labels=labels)
                ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']


                world_size = torch_dist.get_world_size()
                with torch.no_grad():
                    agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                    agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]

                    dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                    dist.all_gather(agg_proj_text_emb, proj_text_emb[0])

                    agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                    agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                    dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                    dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                    rank = torch_dist.get_rank()

                agg_proj_img_emb[rank] = proj_ecg_emb[0]
                agg_proj_text_emb[rank] = proj_text_emb[0]

                agg_proj_ecg_emb1[rank] = ecg_emb[0]
                agg_proj_ecg_emb2[rank] = ecg_emb[1]

                agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)

                cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                gen_loss = output_dict.get('gen_loss', torch.tensor(0.0).to(self.device))
                beta = self.gen_loss_weight
                loss =  cma_loss
                val_gen_loss += gen_loss.item()
                val_cma_loss += cma_loss.item()
                val_loss += loss.item()
                val_epoch_acc1.append(acc1.item())
                val_epoch_acc5.append(acc5.item())

        if self.device == 0:
            val_cma_loss = val_cma_loss/len(val_epoch_acc1)
            val_gen_loss = val_gen_loss / len(loader)
            val_loss = val_loss/len(val_epoch_acc1)
            val_epoch_acc1 = np.array(val_epoch_acc1).mean()
            val_epoch_acc5 = np.array(val_epoch_acc5).mean()

            val_log = {'val_loss': val_loss,
                        'val_cma_loss': val_cma_loss,
                        'val_gen_loss': val_gen_loss,
                        'val_acc1': val_epoch_acc1,
                        'val_acc5': val_epoch_acc5}
            return val_log
        else:
            return None

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)


class test_wBert:
    def __init__(self, model, optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.test_batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.checkpoint = args['checkpoint']
        self.counter = 0
        self.total_samples = 0
        self.R1_ecg_to_text = 0
        self.R5_ecg_to_text = 0
        self.R10_ecg_to_text = 0
        self.R1_text_to_ecg = 0
        self.R5_text_to_ecg = 0
        self.R10_text_to_ecg = 0

        self.batch_size_for_recall = 1000

        self.csv_file =  self.csv_file = os.path.join(self.checkpoint,f"retrieval/test_results_{self.model_name}.csv")
        if dist.get_rank() == 0:
            with open(self.csv_file, 'w') as f:
                f.write("Samples,R@1 ECG to Text,R@5 ECG to Text,R@10 ECG to Text,"
                        "R@1 Text to ECG,R@5 Text to ECG,R@10 Text to ECG\n")

    def compute_recall(self, ecg_embeddings, text_embeddings):
        """Compute and export Recall@K metrics."""
        batch_size = ecg_embeddings.shape[0]
        print(f"\n[Computing Recall@K for {batch_size} samples...]")

        scores_ecg_to_text = torch.matmul(ecg_embeddings, text_embeddings.T).cpu().numpy()
        scores_text_to_ecg = torch.matmul(text_embeddings, ecg_embeddings.T).cpu().numpy()

        R1_ecg_to_text, R5_ecg_to_text, R10_ecg_to_text = 0, 0, 0
        R1_text_to_ecg, R5_text_to_ecg, R10_text_to_ecg = 0, 0, 0

        for i, score_row in enumerate(scores_ecg_to_text):
            sorted_indices = np.argsort(-score_row)
            if i in sorted_indices[:1]: R1_ecg_to_text += 1
            if i in sorted_indices[:5]: R5_ecg_to_text += 1
            if i in sorted_indices[:10]: R10_ecg_to_text += 1

        for i, score_row in enumerate(scores_text_to_ecg):
            sorted_indices = np.argsort(-score_row)
            if i in sorted_indices[:1]: R1_text_to_ecg += 1
            if i in sorted_indices[:5]: R5_text_to_ecg += 1
            if i in sorted_indices[:10]: R10_text_to_ecg += 1

        self.total_samples += batch_size

        self.R1_ecg_to_text += R1_ecg_to_text
        self.R5_ecg_to_text += R5_ecg_to_text
        self.R10_ecg_to_text += R10_ecg_to_text
        self.R1_text_to_ecg += R1_text_to_ecg
        self.R5_text_to_ecg += R5_text_to_ecg
        self.R10_text_to_ecg += R10_text_to_ecg

        batch_result = {
            'Samples': batch_size,
            'R@1 ECG to Text': R1_ecg_to_text / batch_size,
            'R@5 ECG to Text': R5_ecg_to_text / batch_size,
            'R@10 ECG to Text': R10_ecg_to_text / batch_size,
            'R@1 Text to ECG': R1_text_to_ecg / batch_size,
            'R@5 Text to ECG': R5_text_to_ecg / batch_size,
            'R@10 Text to ECG': R10_text_to_ecg / batch_size
        }

        print(f"Batch Testing Results: {batch_result}")

        if dist.get_rank() == 0:
            with open(self.csv_file, 'a') as f:
                f.write(f"{self.total_samples},{batch_result['R@1 ECG to Text']:.4f},"
                        f"{batch_result['R@5 ECG to Text']:.4f},{batch_result['R@10 ECG to Text']:.4f},"
                        f"{batch_result['R@1 Text to ECG']:.4f},{batch_result['R@5 Text to ECG']:.4f},"
                        f"{batch_result['R@10 Text to ECG']:.4f}\n")

        del ecg_embeddings, text_embeddings
        torch.cuda.empty_cache()

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size,
                                 num_workers=self.num_workers, drop_last=False, shuffle=False)

        self.model.eval()
        print('Start testing...')

        with torch.no_grad():
            batch_ecg_embeddings = []
            batch_text_embeddings = []

            for data in tqdm(test_loader):
                report = data['txt']
                ecg = data['ecg'].to(torch.float32).to(self.device).contiguous()

                report_tokenize_output = self.model.module._tokenize(report)
                input_ids = report_tokenize_output.input_ids.to(self.device).contiguous()
                attention_mask = report_tokenize_output.attention_mask.to(self.device).contiguous()

                ecg_embeddings = self.model.module.ext_ecg_emb(ecg)
                text_embeddings = self.model.module.get_text_emb(input_ids, attention_mask)

                proj_ecg_embeddings = F.normalize(ecg_embeddings, p=2, dim=-1)
                proj_text_embeddings = self.model.module.proj_t(text_embeddings)
                proj_text_embeddings = F.normalize(proj_text_embeddings, p=2, dim=-1)

                batch_ecg_embeddings.append(proj_ecg_embeddings)
                batch_text_embeddings.append(proj_text_embeddings)

                if len(batch_ecg_embeddings) * self.test_batch_size >= self.batch_size_for_recall:
                    ecg_emb_tensor = torch.cat(batch_ecg_embeddings, dim=0)
                    text_emb_tensor = torch.cat(batch_text_embeddings, dim=0)
                    self.compute_recall(ecg_emb_tensor, text_emb_tensor)

                    batch_ecg_embeddings = []
                    batch_text_embeddings = []

        if batch_ecg_embeddings:
            ecg_emb_tensor = torch.cat(batch_ecg_embeddings, dim=0)
            text_emb_tensor = torch.cat(batch_text_embeddings, dim=0)
            self.compute_recall(ecg_emb_tensor, text_emb_tensor)

        print("\nFinal Testing Results:")
        print(f"ECG to Text: R@1: {self.R1_ecg_to_text / self.total_samples:.4f}, R@5: {self.R5_ecg_to_text / self.total_samples:.4f}, R@10: {self.R10_ecg_to_text / self.total_samples:.4f}")
        print(f"Text to ECG: R@1: {self.R1_text_to_ecg / self.total_samples:.4f}, R@5: {self.R5_text_to_ecg / self.total_samples:.4f}, R@10: {self.R10_text_to_ecg / self.total_samples:.4f}")
