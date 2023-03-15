# OpenCastKit: an open-source solutions of global data-driven high-resolution weather forecasting

English | [简体中文](README.md)

This is an open-source solutions of global data-driven high-resolution weather forecasting, implemented and improved by [High-Flyer AI](https://www.high-flyer.cn/). It can compare with the ECMWF Integrated Forecasting System (IFS).

The model weights trained on the ERA5 data from 1979-01 to 2022-12 are released at [Hugging Face repository](https://huggingface.co/hf-ai/OpenCastKit). You can also have a look at [HF-Earth](https://www.high-flyer.cn/hf-earth/), a daily updated demo of weather prediction.

As shown in the following cases:

![Typhoon track comparison](./img/wind_small.gif)

![Water vapour comparison](./img/precipitation_small.gif)


## Requirements

- [hfai](https://doc.hfai.high-flyer.cn/index.html) >= 7.9.5
- torch >=1.8


## Training
The raw data is from the public dataset, [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) . We can use the script `data_factory/convert_ear5_hourly.py` to fetch featuers and convert them into the high-performance sample data of [FFRecord](https://www.high-flyer.cn/blog/ffrecord/) format.

### FourCastNet training

Run locally：
```shell
   python train_fourcastnet.py --pretrain-epochs 100 --fintune-epochs 40 --batch-size 4
```

We can conduct data-parallel training on the Yinghuo HPC:
```shell
   hfai python train_fourcastnet.py --pretrain-epochs 100 --fintune-epochs 40 --batch-size 4 -- -n 12 --name train_fourcastnet
```

### GraphCast training

Run locally：
```shell
   python train_graphcast.py --epochs 200 --batch-size 2
```

We can conduct pipeline-parallel training on the Yinghuo HPC:
```shell
   hfai python train_graphcast.py --epochs 200 --batch-size 2 -- -n 32 --name train_graphcast
```


## Reference

```bibtex
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and others},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```

```bibtex
@article{remi2022graphcast,
  title={GraphCast: Learning skillful medium-range global weather forecasting},
  author={Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Alexander Pritzel, Suman Ravuri, Timo Ewalds, Ferran Alet, Zach Eaton-Rosen, Weihua Hu, Alexander Merose, Stephan Hoyer, George Holland, Jacklynn Stott, Oriol Vinyals, Shakir Mohamed, Peter Battaglia},
  journal={arXiv preprint arXiv:2212.12794},
  year={2022}
}
```