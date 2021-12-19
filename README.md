# IW05

> @author:  汤远航
>
> @StuID: 181840211

这次作业分为几部分:

- 数据的读取: snacks 数据集的读取需要实现一个函数, 这只需要针对图片的大小和方框的比例进行乘法计算即可
- 模型的训练: turicreate一键完成, 在Ubuntu系统中CPU运行了两天左右, 得到的 iou只有 0.32 左右
- 模型的部署: 通过 `VNRecognizedObjectObservation` 可以获得所有的识别结果, 将其中 `confidence` 小于 0.8  的识别结果过滤, 通过 `VNImageRectForNormalizedRect` 和 `UIsreen.main.size` 将识别得到的`boundingbox`转化为合适的尺寸, 使用提供的`boundingboxview.show`函数绘制即可

主要的困难:

- 没有将`show`函数放进 `Dispatch.main.sync`里面, 导致方框未被展示
