# ECGCLIP embedding 提取源码位置复核

本次先检查本地 `clip2026`，再用 SHA-256 核对服务器对应文件。以下都是**源码和配置路径**；病例数据与权重没有复制到代码归档。

| 数据集/用途 | 本地源码（相对 `C:\Users\knightlyflash\Desktop\博士\clip2026`） | 服务器对应源码 | 核对结果 |
| --- | --- | --- | --- |
| HEEDB 主提取 | `ECGCLIP/ecgfounder2/extract_heedb_embeddings_distributed_robust.py` | `/home/yanmingke/ecgfounder2/extract_heedb_embeddings_distributed_robust.py` | 提取逻辑相同；仅默认索引和输出路径不同：本地指向训练数据，服务器当前文件指向验证数据。 |
| MIMIC-IV-ECG 主提取 | `ECGCLIP/ecgfounder2/downstream/Preprocessing/feature_extract_mimic.py` | `/home/yanmingke/ecgfounder2/downstream/Preprocessing/feature_extract_mimic.py` | SHA-256 完全相同。 |
| CODE-15 主提取 | `ecgfounder2_extraction/extract_code15_ecgclip_single_gpu_robust.py` | `/home/yanmingke/ecgfounder2/ecgfounder2_extraction/extract_code15_ecgclip_single_gpu_robust.py` | SHA-256 完全相同。 |

三条主提取源码已在干净代码包中分别重命名为 `extraction/extract_heedb_embeddings.py`、`extract_mimic_embeddings.py`、`extract_code15_embeddings.py`；统一入口为 `scripts/run_extraction.py`。归档只包含源码，生成的 embedding 和中途检查点仍写入外部输出目录。

共同使用的模型配置：

- 本地：`ECGCLIP/ecgfounder2/retrieval/retrieval_config.yaml`
- 服务器：`/home/yanmingke/ecgfounder2/retrieval/retrieval_config.yaml`
- 两份配置的 SHA-256 均为 `bfd7b95601fd32b202f86ef004bb793bd3b42321a9666034fdcfd953f8034e4f`。配置指定 `model_name: ECGCLIP`、`ecg_model: ecgfounder`、`text_model: medcpt`。
- 配置指向的服务器权重：`/data2/2shared/yanmingke/ecg_model/retrieval_model/ecgfounder_medcpt_retrieval_F_10000_total.pth`；文件存在，大小约 777 MiB，SHA-256 为 `d8e387edd8e74d177492526a8990caa094e8862de2dd237a130ca351c5ecbec3`。

这些路径说明三条主提取链**默认指向**同一份 ECGCLIP 模型配置与权重；若某次实际运行覆盖了 `--config_path`，仍应以那次运行记录为准。HEEDB 本地/服务器脚本的两处默认路径差异不改变加载权重和波形提取函数。

**MIMIC 现存文件已只读核对：**原始索引 `mimic/mimic_final_with_labels.csv`、提取时保存的 `feature/mimic/metadata.csv`、论文使用的 `feature/mimic/metadata_cleaned_with_labels_correct.csv`，以及 `feature/mimic/ecg_embeddings.pt` 均为 **800,034 行/条**。三份 CSV 的 `study_id` 顺序逐行完全一致；原始索引的 `path` 与 `final_report` 缺失记录数为 **0**，因此提取数据集的 `dropna` 没有删行。这支持 MIMIC 元数据与 embedding 的位置对应；脚本对单条波形读取失败时会递归改用下一条记录，若要排除这种少数异常，还需查看当时运行日志或重查原始波形。

**CODE-15 现存文件已只读核对：**原始索引、保存的 metadata、embedding 均为 **342,755 条**，两份 CSV 的 `exam_id` 顺序逐行一致，metadata 中 ID 唯一。

**HEEDB `finalgallery` 已核对：**train/val/test 向量形状分别为 **(8,182,498, 256)**、**(1,041,322, 256)**、**(1,041,324, 256)**，首维均与 `finalgallery_summary.json` 的保留数量相同。`build_finalgallery_nonempty.py` 在写出时对元数据和原向量施加同一布尔掩码，并先检查两者原始行数一致。未对多 GB 的现存 CSV 逐行计算 ID 顺序哈希。

以上核对支持数据集内的行序一致性；原始数据集类遇到单条波形读取错误时，部分脚本会改用下一条样本。若要排除这种内容替代，应检查对应运行日志或逐条原始波形错误记录。

已找到的 HEEDB 验证/测试 `extraction_stats.txt` 记录了原模型配置；测试统计还明确记录了上面的 checkpoint 路径。CODE-15 的 `progress.json` 显示 `completed`，但摘要没有逐条读取失败或替代计数；MIMIC 输出目录中未找到对应的失败统计日志。
