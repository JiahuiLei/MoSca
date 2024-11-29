# [MoSca](https://www.cis.upenn.edu/~leijh/projects/mosca/): A Modern 4D Reconstruction System for Monocular Videos

[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue)](https://www.cis.upenn.edu/~leijh/projects/mosca/)
[![Latest Paper](https://img.shields.io/badge/Latest%20Paper-Read-orange)](https://www.cis.upenn.edu/~leijh/projects/mosca/pub/mosca_v2.pdf)
[![Video](https://img.shields.io/badge/Video-Watch-red)](https://www.youtube.com/watch?v=7WrG5-xH1_k)
[![Contact Author](https://img.shields.io/badge/Contact%20Author-Email-green)](mailto:leijh@cis.upenn.edu)

## Updates:

- 2024.Nov.28th: code **pre**-release. This is a preview of MoSca. The code needs to be cleaned, and more functions should be added (see TODOs) in the future, but the first author does not have enough time to do so at this moment because he has to start finding the next position after PhD graduation in 2025.
- 2024.Nov.28th: **important note**, now the code supports TAP: [BootsTAPIR](https://github.com/google-deepmind/tapnet), [CoTracker](https://github.com/facebookresearch/co-tracker), and [SpaTracker](https://github.com/henry123-boy/SpaTracker); and Depth: [DepthCrafter](https://github.com/Tencent/DepthCrafter), [Metric3D-v2](https://github.com/YvanYin/Metric3D), and [UniDepth](https://github.com/lpiccinelli-eth/UniDepth). I authorize the MIT license only to the code I wrote (mostly in `lib_moca` and `lib_mosca`). The code from all those third-party foundational models and GS-Splatting (mostly in `lib_prior` and `lib_render`) is not included in the MIT license. Please refer to the original authors for the license of these third-party codes.

## Install

1. Simply run the following command. This script assumes you have an Ubuntu environment and Anaconda installed. The CUDA version used is 11.8. You may have to tweak the script to fit your own environment.
    ```bash
    bash install.sh
    ```

2. Download from [here](https://drive.google.com/file/d/15tveiv7ZkvBBAN3qkkB7Zfky9d7vSqLD/view?usp=sharing) some checkpoints for the 2D foundational models if they are not HG downloadables.

    WARNING: By downloading these checkpoints, you must agree and obey the original license from the original authors ([RAFT](https://github.com/princeton-vl/RAFT), [SpaTracker](https://github.com/henry123-boy/SpaTracker), and [TAPNet](https://github.com/google-deepmind/tapnet)). Unzip the weights into the following file structure:
    ```bash
    ProjRoot/weights
    ├── raft_models
    │   ├── raft-things.pth
    │   └── ...
    ├── spaT_final.pth
    └── tapnet
        └── bootstapir_checkpoint_v2.pt
    ```

## Get Started

A few demo scenes are provided in the `data` directory. A list of images is stored under `demo/SeqName/images`, and the main program is designed to process these lists of images into a 4D scene.

### Run the Full Pipeline: MoSca
```bash
# Infer off-the-shelf 2D models
python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck
# Fit the 4D scene
python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck
```

You should expect some output like this:

<video width="480" controls>
    <source src="./assets/duck_480.mp4" type="video/mp4">
</video>

**More examples are in `example.sh`**. `demo.ipynb` also provides some examples of the system.

### Run a Sub-Module: MoCa -- Moving Monocular Camera (if you only care camera pose)

We also ship a sub-module `MoCa`, which corresponds to "Moving Monocular Camera", that is a standalone module before MoSca for tracklet-based BA solving camera pose and depth alignment. To run this submodule, for example, simply:

```bash
# Infer off-the-shelf 2D models with a reduced mode
python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck --skip_dynamic_resample
# Fast solve a small BA
python lite_moca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck
```
You should expect some output like this:

<video width="480" controls>
    <source src="./assets/static_scaffold_init.mp4" type="video/mp4">
</video>


## Reproduce the Tables

- Now we provide instructions to reproduce our results for Tab.1 (Dycheck), Tab.2 (Nvidia), and Tab.3 (Tum and Sintel) in the new paper.
- (Option-A) Reproduce by running locally:
    - Download the data from [here](https://drive.google.com/file/d/1sSvVi5Bid_KQsuguVGuqUVgWPzVdb9jM/view?usp=sharing). By downloading the data, you must agree and obey the original license from the original authors ([Dycheck](https://github.com/KAIR-BAIR/dycheck), [Nvidia](https://github.com/gaochen315/DynamicNeRF?tab=readme-ov-file), [TUM](https://cvg.cit.tum.de/rgbd/dataset/), and [Sintel](http://sintel.is.tue.mpg.de/)). Unzip into the following file structure:
    
        ```bash
        ProjRoot/data/iphone
            ├── apple
            ├── ...
            └── wheel
        ```
    
    - Check the script `reproduce.sh`. For example, if you have 1 GPU, just run: 
        ```bash 
        bash reproduce.sh
        ```
        
        If you have multiple GPUs, you can run `bash reproduce.sh #GPU_ID #NUM_OF_TOTAL_DEVICES` in several terminals.
- (Option-B) Reproduce by downloading the checkpoints run by us from [here](https://drive.google.com/drive/folders/14awBsxTmY211ut9SnW5d1vuvVyThYhjl?usp=sharing). Unzip the downloaded subfolders in the same structure as above under `data`.
- Finally, you can collect all the results by checking `collect_metrics.ipynb` to form reports stored in `data/metrics_collected`.

## TO-DOs

- [ ] Sometimes, the system needs some tuning of the parameters. Add more detailed instructions for the parameters.
- [ ] Support manual labeling of the FG-BG masks.
- [ ] Support other focal initialization methods.
- [ ] Find a good visualizer for MoCa and MoSca.
- [ ] Replace the old render backends with the new GSplat.
- [ ] Only RAFT is used for optical flow now, check other checkpoints and methods.
- [ ] Docker environment.

## Acknowledgement

I authorize the MIT license only to the code I wrote (mostly in `lib_moca` and `lib_mosca`). The code from all those third-party foundational models and GS-Splatting (mostly in `lib_prior` and `lib_render`) is not included in the MIT license. Please refer to the original authors for the license of these third-party codes.

If you use either MoCa or MoSca, you should cite our technical paper:

```tex
@article{lei2024mosca,
  title={MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds},
  author={Lei, Jiahui and Weng, Yijia and Harley, Adam and Guibas, Leonidas and Daniilidis, Kostas},
  journal={arXiv preprint arXiv:2405.17421},
  year={2024}
}
```
