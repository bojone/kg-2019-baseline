# kg-2019-baseline
2019年百度的三元组抽取比赛（ http://lic2019.ccf.org.cn/kg ），一个baseline

注：正式版已经更新至 https://github.com/bojone/kg-2019

## 模型
用BiLSTM做联合标注，先预测subject，然后根据suject同时预测object和predicate，标注结构是“半指针-半标注”结构，以前也曾介绍过（ https://kexue.fm/archives/5409 ）

标注结构是自己设计的，我看了很多关系抽取的论文，没有发现类似的做法。所以，如果你基于此模型做出后的修改，最终获奖了或者发表paper什么的，烦请注明一下（其实也不是太奢望）

```
@misc{
  jianlin2019bdkg,
  title={Hybrid Structure of Pointer and Tagging for Relation Extraction: A Baseline},
  author={Jianlin Su},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/bojone/kg-2019-baseline}},
}
```

## 用法
`python trans.py`转换数据，`python kg.py`直接跑。

## 结果
5个epoch内dev集的F1应该就能到达0.71+了，最后基本上F1都能跑到0.72～0.73，自动保存F1最优的模型，有同学跑到过0.74甚至0.75的，我也表示很无辜，大家拼人品吧。反正都会比官方的baseline要高。

## 环境
Python 2.7 + Keras 2.2.4 + Tensorflow 1.8，其中关系最大的应该是Python 2.7了，如果你用Python 3，需要修改几行代码，至于修改哪几行，自己想办法，我不是你的debugger。

欢迎入坑Keras。人生苦短，我用Keras～

## 声明
欢迎测试、修改使用，但这是我比较早的模型，文件里边有些做法在我最新版已经被抛弃，所以以后如果发现有什么不合理的地方，不要怪我故意将大家引入歧途就行了。

欢迎跟我交流讨论，但请尽量交流一些有意义的问题，而不是debug。（如果Keras不熟悉，请先自学一个星期Keras。）

<strong>特别强调</strong>：baseline的初衷是供参赛选手测试使用，如果你已经错过了参赛日期，但想要训练数据，请自行想办法向主办方索取。我不负责提供数据下载服务。

## 链接
- https://kexue.fm
- https://keras.io
