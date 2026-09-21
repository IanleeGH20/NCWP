# Bundled data

## `stsbenchmark/`

`sts_train.json`, `sts_valid.json`, `sts_test.json` are the official STS
Benchmark splits, redistributed here so that the exact fitting and evaluation
sentences are pinned (the paper fits on the 2,910 unique validation sentences and
evaluates Spearman on the 1,379 official test pairs). Each record keeps the
original fields, including `sentence1`, `sentence2` and the gold `score`.

- Source: STS Benchmark, <https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark>
- License: CC BY-SA 4.0 (as distributed with the STS Benchmark)
- Please cite:

```bibtex
@inproceedings{cer2017stsb,
  title     = {SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and
               Crosslingual Focused Evaluation},
  author    = {Cer, Daniel and Diab, Mona and Agirre, Eneko and
               Lopez-Gazpio, I{\~n}igo and Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic
               Evaluation (SemEval-2017)},
  year      = {2017}
}
```

All other datasets used in the paper (BEIR-Quora, CQADupStack, STS12–16, SICK-R)
are **not** bundled — they are downloaded through the `datasets` library on first
use and cached outside this repository.
