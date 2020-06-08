# Scalable Uncertainty for Computer Vision with Functional Variational Inference

<div align="center">
  <img src="https://raw.githubusercontent.com/happenwah/FVI_CV/master/fvi_diagram.png?fbclid=IwAR0LoPUz-2Mw9rFS4hs8XsJBUZJGXTflsOxcxsaqmLtoedfTU9dURRWqw5k">
</div>

Code for [Scalable Uncertainty for Computer Visition with Functional Variational Inference](https://arxiv.org/abs/2003.03396), by [Eduardo D C Carvalho](https://twitter.com/happenwah), [Ronald Clark](http://www.ronnieclark.co.uk/), [Andrea Nicastro](https://andreanicastro.github.io/) and [Paul H J Kelly](https://www.doc.ic.ac.uk/~phjk/). To appear at [CVPR 2020](http://cvpr2020.thecvf.com/).

## BibTex:
`@InProceedings{Carvalho_2020_CVPR,
author = {Carvalho, Eduardo D. C. and Clark, Ronald and Nicastro, Andrea and Kelly, Paul H. J.},
title = {Scalable Uncertainty for Computer Vision With Functional Variational Inference},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}`

## Pre-requisites:
* **Download folder `/datasets`, containing [CamVid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) and pre-processed Make3D:** https://drive.google.com/open?id=1HaEnG8smGgCzUpUUMnVxXInrBGh-rjyl
* **Install cnn_gp package:** https://github.com/cambridge-mlg/cnn-gp

Put `/datasets` folder at the same directory level as `/FVI_CV`.

## Training all models:

Supply `BASE_DIR` as the path containing both `/FVI_CV` and `/datasets` folders.

If more convenient, set `--n_epochs` to a number smaller than 4000 and use `--load` in order to resume training. 

### Semantic segmentation on CamVid:

**Ours-Boltzmann:** `python3 run_fvi_seg.py --base_dir=BASE_DIR --training_mode`

**Deterministic-Boltzmann/MCDropout-Boltzmann:** `python3 run_mcd_seg.py --base_dir=BASE_DIR --training_mode`

### Depth regression on Make3D:

**Ours-Gaussian:** `python3 run_fvi_gaussian_depth.py --base_dir=BASE_DIR --training_mode`

**Ours-Laplace:** `python3 run_fvi_laplace_berhu_depth.py --base_dir=BASE_DIR --training_mode --likelihood="laplace"`

**Ours-berHu:** `python3 run_fvi_laplace_berhu_depth.py --base_dir=BASE_DIR --training_mode --likelihood="berhu"`

**Deterministic-Laplace:** `python3 run_deterministic_depth.py --base_dir=BASE_DIR --training_mode --loss="l1"`

**Deterministic-berHu:** `python3 run_deterministic_depth.py --base_dir=BASE_DIR --training_mode --loss="berhu"`

**MCDropout-Laplace:** `python3 run_mcd_laplace_depth.py --base_dir=BASE_DIR --training_mode`

## Computing test results:

Firstly, put all trained models (`.bin` files) and Ours-berHu likelihood threshold `c_test.txt` in folder `/models_test`. Change those filenames to the ones prescribed inside the `if __name__ == '__main__':` block in each `run_*.py` file.

Write same commands as in training, but replacing the `--training_mode` flag by `--test_mode`. For obtaining runtime comparison results, use `--test_runtime_mode` instead.

Finally, write `python3 compare_calibration.py` for computing the mean calibration scores.
