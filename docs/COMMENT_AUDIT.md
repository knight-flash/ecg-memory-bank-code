# Source Comment and Language Audit

Scope: 95 Python files under `src/` and `scripts/`, including 86 substantive modules and 9 package initializers.

- All 86 substantive modules have a header with `Purpose`, `Method`, `Inputs`, `Outputs`, `Paper parameters`, and `Shapes and meaning`. These sections describe the module's role, algorithm, data contract, fixed paper settings, and array dimensions. Utilities without tensor inputs describe their file or table structure instead.
- Headers and inline comments use English throughout. Package initializers retain brief descriptions.
- Earlier cleanup removed commented-out code, obsolete experimental parameter alternatives, redundant step narration, and decorative separators. The inline-comment count fell from 639 to 140 at that stage. Nineteen Chinese docstrings were rewritten in English. Meaningful sampling, model, and statistical comments, plus shebangs and type markers, were retained.
- During the header expansion, executable ASTs were compared after removing module docstrings; those edits did not change computations. The subsequent language pass translated runtime messages, exceptions, progress labels, and CLI help while preserving algorithmic code and data keys.
- An unreferenced historical `reader_pair_ci.py` was removed. The formal reader-study analysis uses case-clustered `reader_study_ci.py`. The retained historical summary script identifies its fixed relative path in the header.
- All readable archive text, including the extraction scripts and release documentation, is now free of Chinese characters. A release test checks this property.

Verification command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The 86 substantive module headers are 27-34 lines long. The archive contains no experimental outputs.
