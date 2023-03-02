# Global Data-driven High-resolution Weather Model

English | [简体中文](README.md)

This is a global data-driven high-resolution weather model implemented and improved by [High-Flyer AI](https://www.high-flyer.cn/). It is the first AI weather model, which can compare with the ECMWF Integrated
Forecasting System (IFS).

The model weights trained on the ERA5 data from 1979-01 to 2022-12 are released at [Hugging Face repository](https://huggingface.co/hf-ai/FourCastNet). You can also have a look at [HF-Earth](https://www.high-flyer.cn/hf-earth/), a daily updated demo of weather prediction.

Typhoon track comparison:

![](./img/wind_small.gif)

Water vapour comparison:

![](./img/precipitation_small.gif)


## Requirements

- [hfai](https://doc.hfai.high-flyer.cn/index.html)
- torch >=1.8


## Training
The raw data is from the public dataset, [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) , which is integrated into the dataset warehouse, `hfai.datasets`.

1. pretrain

   submit the task to Yinghuo HPC:
   ```shell
    hfai python train/pretrain.py -- -n 8 -p 30
   ```
   run locally:
   ```shell
    python train/pretrain.py
   ```

2. finetune

   submit the task to Yinghuo HPC:
   ```shell
    hfai python train/fine_tune.py -- -n 8 -p 30
   ```
   run locally:
   ```shell
    python train/fine_tune.py
   ```

3. precipitation train

   submit the task to Yinghuo HPC:
   ```shell
    hfai python train/precipitation.py -- -n 8 -p 30
   ```
   run locally:
   ```shell
    python train/precipitation.py
   ```


## Citation

```bibtex
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and others},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```
