# OpenCastKit: an open-source solutions of global data-driven high-resolution weather forecasting

简体中文 | [English](README_en.md)

本项目是由幻方AI团队复现优化，并开源的全球AI气象预报模型工具库。基于 [FourCastNet](https://arxiv.org/abs/2202.11214) 和 [GraphCast](https://arxiv.org/abs/2212.12794) 的论文，我们构建了一个新的全球AI气象预报项目——**OpenCastKit**，它能够与欧洲中期天气预报中心（ECMWF）的传统物理模型——高分辨率综合预测系统（IFS），进行直接比较。

我们将基于1979年1月到2022年12月的ERA5数据训练出来的模型参数开源到 [Hugging Face 仓库](https://huggingface.co/hf-ai/OpenCastKit)中，并上线了一个每日更新的 [HF-Earth](https://www.high-flyer.cn/hf-earth/)，展示模型的预测效果。

下面是一些预测案例：

![台风路径预测与真实路径比较](./img/wind_small.gif)

![汽水浓度预测与真实情况比较](./img/precipitation_small.gif)


## 依赖

- [hfai](https://doc.hfai.high-flyer.cn/index.html) >= 7.9.5
- torch >=1.8


## 训练
原始数据来自欧洲中期天气预报中心（ECMWF）提供的一个公开可用的综合数据集 [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) ，需要通过 `data_factory/convert_ear5_hourly.py` 脚本提出数据特征，转化为[高性能训练样本格式 FFRecord](https://www.high-flyer.cn/blog/ffrecord/) 下的样本数据。


### 训练 FourCastNet

本地运行：
```shell
   python train_fourcastnet.py --pretrain-epochs 100 --fintune-epochs 40 --batch-size 4
```

也可以提交任务至幻方萤火集群，使用96张A100进行数据并行训练
```shell
   hfai python train_fourcastnet.py --pretrain-epochs 100 --fintune-epochs 40 --batch-size 4 -- -n 12 --name train_fourcastnet
```

### 训练 GraphCast

本地运行：
```shell
   python train_graphcast.py --epochs 200 --batch-size 2
```

也可以提交任务至幻方萤火集群，使用256张A100进行流水线并行训练
```shell
   hfai python train_graphcast.py --epochs 200 --batch-size 2 -- -n 32 --name train_graphcast
```


## 引用

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