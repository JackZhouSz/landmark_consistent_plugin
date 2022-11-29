<div align="center">

# 3D-aware Facial Landmark Detection via Multiview Consistent Training on Synthetic Data

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

</div>

![](./dad3d/readme_imgs/teaser_v3.png)

This is an official repository for the paper 
```
3D-aware Facial Landmark Detection via Multiview Consistent Training on Synthetic Data
Libing Zeng, Lele Chen, Wentao Bao, Zhong Li, Yi Xu, Junsong Yuan, Nima Kalantari
In Submission
```


## Installation
The proposed 3D-aware module can be plugged into any learning-based landmark detection algorithm. 
We applied our modual on DAD-3DNet and 3DDFA.
Please create the corresponding conda virtual environments refer to the following links respectively.
[DAD-3DNet](https://github.com/PinataFarms/DAD-3DHeads), 
[3DDFA](https://github.com/cleardusk/3DDFA)



## Demo
![](./dad3d/readme_imgs/all_compare.png)

The visual results of Dlib, FAN, 3DDFA, our refined 3DDFA+, 3DDFA-V2, DAD-3DNet, and our refined
DAD-3DNet+ on images randomly sampled from DAD-3DHeads testing set. We show the enlarged error region (white box) in the
middle row.

### Run demo:

#### DAD-3DNet
Please download checkpoints and dependent files following 
[checkpoints](https://drive.google.com/drive/folders/1JmdZwcweivDXx5_Q37TfsHQq9JogWmzo?usp=share_link), 
[dependence](https://drive.google.com/file/d/1uTuT8J7PN6YqW8XxXcCAp4Dy2pkO6Ngs/view?usp=share_link) respectively, and put them in the following paths:
```bash
# checkpoints path
landmark_consistent_plugin/dad3d/checkpoints

# dependent files path
landmark_consistent_plugin/dad3d/model_training/model/static
```

Then, run the demos using the following commands.

```bash
cd dad3d
# input images are in ./images folder
# output results are in ./outputs folder

# generate results using DAD-3DNet
./shell/run_test_dad3d.sh

# generate results using DAD-3DNet+
./shell/run_test_dad3d+.sh
```

#### 3DDFA
Please download checkpoints and dependent files following 
[checkpoints](https://drive.google.com/file/d/1l4Mh6F19ZSYHWuSNrMB-TXRqA2RzqlOu/view?usp=share_link), 
[dependence1](https://drive.google.com/file/d/1FI7qdbAkr5fauSazP5hNS4XUXvi7xVoD/view?usp=share_link) and 
[dependence2](https://drive.google.com/file/d/1l4Mh6F19ZSYHWuSNrMB-TXRqA2RzqlOu/view?usp=share_link) respectively, and put them in the following paths:

```bash
# checkpoints path
landmark_consistent_plugin/3ddfa/training/snapshot

# dependent files path
landmark_consistent_plugin/3ddfa/shape_predictor
landmark_consistent_plugin/3ddfa/train.configs
landmark_consistent_plugin/3ddfa/test.configs
landmark_consistent_plugin/3ddfa/models

```

Then, run the demos using the following commands.

```bash
cd 3ddfa
# input images are in ./images folder
# output results are in ./outputs folder

# generate results using 3DDFA
python demo_3ddfa.py

# generate results using 3DDFA+
python demo_3ddfa+.py
```

## Fine-Tuning




## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
