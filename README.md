# SAMO

This repo provides codes for paper ["SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofing"](https://arxiv.org/abs/2211.02718).

## Preparation
- Installing dependencies
```
pip install -r requirements.txt
```
- Running environment
  - 1 GPU: GeForce GTX 1080 Ti
  - ~11GB required for a batch size of 23 
- Dataset for train/val/eval:
  - Download ASVspoof 2019 logical access dataset [here](https://datashare.ed.ac.uk/handle/10283/3336)
  - Specify path to the 'LA' folder in argument `--path_to_database`

## Training
The `main.py` file contains train/val/eval steps for Softmax/OC-Softmax/SAMO.

For example, to train SAMO:
```angular2html
python3 train.py -o 'path_to_output_folder' -d 'path_to_database' -p 'path_to_protocol'
```

Please check argument setups in `main.py` to specify settings such as batch size and margins.

## Evaluation
To evaluate pretrained SAMO:
```angular2html
python3 main.py --test_only --test_model "./models/samo.pt" --scoring 'samo' --save_score "samo_score"
```
And the output will show `Test EER: 0.008751418248624953`
## Acknowledgement
This is built upon open-source repos:
- [OC-Softmax](https://github.com/yzyouzhang/AIR-ASVspoof)
- [AASIST](https://github.com/clovaai/aasist)

## References
```bibtex
@article{ding2022samo,
  title={SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofing},
  author={Ding, Siwen and Zhang, You and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2211.02718},
  year={2022}
}
```

```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

```bibtex
@ARTICLE{zhang21one,
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters}, 
  title={One-Class Learning Towards Synthetic Voice Spoofing Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={937-941},
  doi={10.1109/LSP.2021.3076358}}
```

```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```
