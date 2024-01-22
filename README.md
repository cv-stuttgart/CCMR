# CCMR
[WACV24] This is the inference code of our `CCMR` optical flow estimation method.

 **[CCMR: High Resolution Optical Flow Estimation via Coarse-to-Fine Context-Guided Motion Reasoning](https://openaccess.thecvf.com/content/WACV2024/papers/Jahedi_CCMR_High_Resolution_Optical_Flow_Estimation_via_Coarse-To-Fine_Context-Guided_Motion_WACV_2024_paper.pdf)**<br/>
> _WACV 2024_ <br/>
> Azin Jahedi, Maximilian Luz, Marc Rivinius and Andrés Bruhn

If you find our work useful please [cite via BibTeX](CITATIONS.bib).

## Requirements

The code has been tested with PyTorch 1.10.2+cu113.
Install the required dependencies via
```
pip install -r requirements.txt
```

Alternatively you can also manually install the following packages in your virtual environment:
- `torch`, `torchvision`, and `torchaudio` (e.g., with `--extra-index-url https://download.pytorch.org/whl/cu113` for CUDA 11.3)
- `matplotlib`
- `scipy`
- `tensorboard`
- `opencv-python`
- `tqdm`
- `parse`
- `timm`
- `flowpy`


## Pre-Trained Checkpoints

You can find our trained models in the release assets.


## Datasets

Datasets are expected to be located under `./data` in the following layout:
```
./data
  ├── kitti15                   # KITTI 2015
  │  └── dataset
  │     ├── testing/...
  │     └── training/...
  └── sintel                    # Sintel
     ├── test/...
     └── training/...
  
```

## Running CCMR(+)

For running `CCMR+` on MPI Sintel images you need about 4.5 GB of GPU VRAM. `CCMR` (the 3-scale version) needs about 3 GBs of VRAM, using the following Cuda module.
 
To compile the CUDA correlation module run the following once:
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```

And to reproduce our benchmark results after finetuning run:
```Shell
python evaluate.py --model_type "CCMR+" --model models/CCMR+_sintel.pth --dataset sintel_test
python evaluate.py --model_type "CCMR+" --model models/CCMR+_kitti.pth --dataset kitti_test

python evaluate.py --model_type "CCMR" --model models/CCMR_sintel.pth --dataset sintel_test
python evaluate.py --model_type "CCMR" --model models/CCMR_kitti.pth --dataset kitti_test
```

## License
- Our code is licensed under the BSD 3-Clause **No Military** License. See [LICENSE](LICENSE).
- The provided checkpoints are under the [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license.

## Acknowledgement
Parts of this repository are adapted from [RAFT](https://github.com/princeton-vl/RAFT) ([license](licenses/RAFT/LICENSE)), [MS-RAFT+](https://github.com/cv-stuttgart/MS_RAFT_plus) ([license](https://github.com/cv-stuttgart/MS_RAFT_plus/blob/main/LICENSE)), and [XCiT](https://github.com/facebookresearch/xcit/tree/main) ([license](https://github.com/facebookresearch/xcit/blob/main/LICENSE)).
We thank the authors.
