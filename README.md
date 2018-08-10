## Asynchronous Temporal Fields for Activity Recognition Codebase

Contributor: Gunnar Atli Sigurdsson

* This code implements a "Asynchronous Temporal Fields for Action Recognition" in Torch and **PyTorch**
* This code extends the framework from github.com/gsig/charades-algorithms

Details of the algorithm can be found in:
```
@inproceedings{sigurdsson2017asynchronous,
author = {Gunnar A. Sigurdsson and Santosh Divvala and Ali Farhadi and Abhinav Gupta},
title = {Asynchronous Temporal Fields for Action Recognition},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2017},
pdf = {http://arxiv.org/pdf/1612.06371.pdf},
code = {https://github.com/gsig/temporal-fields},
}
```

**We have updated the codebase with an improved and simplified PyTorch model. Detail can be found under [pytorch](pytorch/)**

Using the improved PyTorch code, a simple RGB model obtains **26.1% mAP** (evaluated with charades_v1_classify.m).
Using the original Torch code, combining the predictions (submission files) of those models using combine_rgb_flow.py
yields a final classification accuracy of 22.4% mAP (evaluated with charades_v1_classify.m).

Evaluation Scripts for localization and classification are available at [allenai.org/plato/charades/](http://allenai.org/plato/charades/)

Submission files for temporal fields and baselines for classification and localization that are compatible with the official evaluation codes on [allenai.org/plato/charades/](http://allenai.org/plato/charades/) are available here: [charades_submission_files.zip](https://www.dropbox.com/s/aw55dauebl87sth/charades_submission_files.zip?dl=1). This might be helpful for comparing and contrasting different algorithms.

Baseline Codes for Activity Localization / Classification are available at [github.com/gsig/charades-algorithms](https://github.com/gsig/charades-algorithms)


