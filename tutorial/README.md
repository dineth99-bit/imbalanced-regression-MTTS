# Tutorial folder

This repository extends the official [YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression) codebase (ICML 2021, *Delving into Deep Imbalanced Regression*). All original benchmarks and the hands-on Colab tutorial live in that upstream project.

## What this fork adds

| Extension | Location | Description |
|-----------|----------|-------------|
| **SkyFinder-DIR** | [`skyfinder-dir/`](../skyfinder-dir/) | Temperature prediction from outdoor webcam images; camera-disjoint splits; LDS/FDS benchmarks |
| **MTTS & AgeDB experiments** | [`agedb-dir/`](../agedb-dir/) | Non-stationary label smoothing (MTTS), extended baselines, and result collection on AgeDB-DIR |

Everything else (`imdb-wiki-dir/`, `nyud2-dir/`, `sts-b-dir/`, `teaser/`, etc.) follows the upstream layout unless noted in the root [README.md](../README.md).

## Original hands-on tutorial

The Boston Housing DIR walkthrough (`tutorial.ipynb`) and Colab badge are maintained by the authors:

- Notebook: [tutorial/tutorial.ipynb](https://github.com/YyzHarry/imbalanced-regression/blob/main/tutorial/tutorial.ipynb) (upstream)
- Open in Colab: [link](https://colab.research.google.com/github/YyzHarry/imbalanced-regression/blob/master/tutorial/tutorial.ipynb)

A copy of `tutorial.ipynb` may remain in this folder for convenience; for the canonical version, use the upstream repo.

## Citation

Please cite Yang et al. (ICML 2021) for DIR, LDS, and FDS. If you use SkyFinder-DIR or MTTS experiments from this repository, cite the original DIR paper and acknowledge this extension.

```bibtex
@inproceedings{yang2021delving,
  title={Delving into Deep Imbalanced Regression},
  author={Yang, Yuzhe and Zha, Kaiwen and Chen, Ying-Cong and Wang, Hao and Katabi, Dina},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}
```
