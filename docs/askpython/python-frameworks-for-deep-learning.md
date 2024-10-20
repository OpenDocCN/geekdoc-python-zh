# 2021 年深度学习的 5 个 Python 框架

> 原文：<https://www.askpython.com/python/python-frameworks-for-deep-learning>

深度学习是目前最热门的行业技能之一。深度学习现在正被用于许多领域，以解决以前被认为无法解决的问题，例如:自动驾驶汽车、人脸识别/增强等。

从头开始编写深度学习模型是一项乏味且令人困惑的任务。这需要大量的专业知识和时间。所以我们使用某些框架，这些框架为我们提供了创建模型的基线。

## 深度学习的 Python 框架

站在 2021 年，它们是 Python 深度学习的很多框架。这些框架在抽象、使用和可移植性方面是不同的。从这个列表中选择您需要的框架可能有点困难。因此，这里列出了 2021 年你可以考虑学习的前 5 个框架/库。

### 1\. TensorFlow

[TensorFlow](https://www.askpython.com/python-modules/tensorflow-vs-pytorch-vs-jax) 是 2021 年最流行的深度学习框架。TensorFlow 是由谷歌大脑团队在 2015 年开源之前开发的。TensorFlow 的当前版本将 Keras 作为一个高级 API，它抽象出大量底层代码，使创建和训练我们的模型变得更容易、更快。

TensorFlow 可与各种计算设备配合使用——CPU、GPU(包括 NVIDIA 和 AMD)，甚至 TPU。对于低计算边缘设备，TensorFlow Lite 可以帮您节省时间。

TensorFlow 广泛支持其他语言，如 C++、JavaScript、Swift 等。如果您正在考虑生产，此功能使 TensorFlow 成为首选。一旦您训练并保存了一个模型，这个模型就可以在您选择的语言中使用，从而缓解多语言依赖性的问题。

### 2\. PyTorch

由脸书开发的 PyTorch 是流行度排名第二的框架。顾名思义就是 Python 版的 Torch(C++库)。PyTorch 与 Python 和 Numpy 无缝集成。PyTorch 在称为 Tensors 的多维数组上工作，它的 API 与 Numpy 非常相似。

PyTorch 提供了一个强大而灵活的 API 来处理 CPU 和 GPU。PyTorch 对 GPU 的出色支持使得分布式训练更加优化可行。PyTorch 非常具有可扩展性。由于这种可扩展性，许多其他框架和工具都建立在 PyTorch 之上，其中之一就是 HuggingFace TransFormers。

在 PyTorch 中，您必须定义自己的训练循环，手动更新权重。这有助于您获得对模型的更多控制。这是研究者倾向于 PyTorch 的主要原因。但是这种方法经常导致样板代码，这对软件部署是不利的。

### 3.法斯泰

FastAi 是杰瑞米·霍华德和雷切尔·托马斯创建的另一个深度学习库。它旨在为 DL 实践者提供高级组件，可以快速轻松地提供标准深度学习领域的最新成果，并为研究人员提供低级组件，可以混合和匹配以构建新方法。

它的目标是在不牺牲易用性、灵活性或性能的情况下做到这两点。FastAI 从两个世界——py torch 和 Keras 取长补短。FastAI 有定义良好的抽象层——高层、中层和低层。底层基于 PyTorch API。

FastAI 简化了生产过程，避免了样板代码和简单的语法，便于开发。

### 4.MxNet

Apache MxNet 可能是这个列表中最令人惊讶的标题之一。MxNet 由一个非常小的社区支持，不像这里列出的大多数其他框架那样受欢迎，但是它做了它承诺要做的事情。

MxNet 试图解决学习不同语言来进行机器学习的问题。MxNet 支持一系列语言，如 Scala、Python、R、Clojure、C++等。

MxNet API 与 PyTorch 非常相似。所以两者之间过渡不会很难。除了 PyTorch API 的好处之外，它还带来了部署方面的好处。它速度快，可伸缩，并且比其他框架占用更少的内存。

### 5\. PyTorch Lightning

PyTorch 照明是一个相对较新的框架。Lighting 只是原始 PyTorch 库的一个包装。它增加了一个抽象的薄层，减少了样板代码的数量，同时又不影响 PyTorch 的功能和美观。

Lightning 使剖析、度量记录和可视化以及分布式培训变得更加容易。此外，从 GPU 到 TPU 的过渡不需要额外的代码行。所以它让 PyTorch 更接近我们所说的可部署。

## 结论

这就把我们带到了本文的结尾。没有所谓的“最佳”框架。每个框架都有一些比其他框架更好的特性。因此，如果您正在寻找一个框架，首先，您可以选择其中的任何一个。随着你的深入，你会明白什么样的框架最适合你或你的工作，并相应地改变它。