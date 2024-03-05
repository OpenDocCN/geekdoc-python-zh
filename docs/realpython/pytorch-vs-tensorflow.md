# PyTorch vs TensorFlow，用于您的 Python 深度学习项目

> 原文：<https://realpython.com/pytorch-vs-tensorflow/>

PyTorch vs TensorFlow:有什么区别？两者都是开源 Python 库，使用图形对数据进行数值计算。两者都在学术研究和商业代码中广泛使用。两者都被各种 API、云计算平台和模型库所扩展。

如果它们如此相似，那么哪个最适合你的项目呢？

在本教程中，您将学习:

*   **PyTorch** 和 **TensorFlow** 有什么区别
*   每个人都有哪些**工具**和**资源**
*   如何为您的特定用例选择最佳选项

您将从仔细研究这两个平台开始，从稍旧的 TensorFlow 开始，然后探索一些可以帮助您确定哪个选择最适合您的项目的考虑因素。我们开始吧！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 什么是张量流？

TensorFlow 由谷歌开发，于 2015 年开源发布。它源于谷歌自主开发的机器学习软件，该软件经过重构和优化，可用于生产。

“TensorFlow”这个名称描述了如何组织和执行数据操作。TensorFlow 和 PyTorch 的基本数据结构是一个[张量](https://en.wikipedia.org/wiki/Tensor)。当你使用 TensorFlow 时，你通过构建一个[有状态数据流图](https://www.quora.com/What-are-the-differences-between-Data-flow-model-and-State-machine-model?)，对这些张量中的数据执行操作，有点像记忆过去事件的流程图。

[*Remove ads*](/account/join/)

### 谁用 TensorFlow？

TensorFlow 有着生产级深度学习库的美誉。它拥有大量活跃的用户，以及大量用于培训、部署和服务模型的官方和第三方工具和平台。

2016 年 PyTorch 发布后，TensorFlow 人气下滑。但在 2019 年末，谷歌发布了 [TensorFlow 2.0](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html) ，这是一次重大更新，简化了库，使其更加用户友好，引发了机器学习社区的新兴趣。

### 代码样式和功能

在 TensorFlow 2.0 之前，TensorFlow 需要你通过调用`tf.*` API 来手动拼接一个[抽象语法树](https://en.wikipedia.org/wiki/Abstract_syntax_tree)——图形。然后，它要求您通过向一个`session.run()`调用传递一组输出张量和输入张量来手动编译模型。

一个`Session`对象是一个用于运行 TensorFlow 操作的[类。它包含了评估`Tensor`对象和执行`Operation`对象的环境，它可以像`tf.Variable`对象一样拥有资源。使用`Session`最常见的方式是作为](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session)[上下文管理器](https://realpython.com/courses/python-context-managers-and-with-statement/)。

在 TensorFlow 2.0 中，您仍然可以用这种方式构建模型，但是使用[急切执行](https://www.tensorflow.org/guide/eager)更容易，这是 Python 通常的工作方式。急切执行会立即评估操作，因此您可以使用 Python 控制流而不是图形控制流来编写代码。

为了看出区别，让我们看看如何用每种方法将两个张量相乘。下面是一个使用旧 TensorFlow 1.0 方法的示例:

>>>

```py
>>> import tensorflow as tf

>>> tf.compat.v1.disable_eager_execution()

>>> x = tf.compat.v1.placeholder(tf.float32, name = "x")
>>> y = tf.compat.v1.placeholder(tf.float32, name = "y")

>>> multiply = tf.multiply(x, y)

>>> with tf.compat.v1.Session() as session:
...     m = session.run(
...         multiply, feed_dict={x: [[2., 4., 6.]], y: [[1.], [3.], [5.]]}
...     )
...     print(m)
[[ 2\.  4\.  6.]
 [ 6\. 12\. 18.]
 [10\. 20\. 30.]]
```

这段代码使用 TensorFlow 2.x 的`tf.compat` API 来访问 TensorFlow 1.x 方法，并禁用急切执行。

首先使用 [`tf.compat.v1.placeholder`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder) 张量对象声明输入张量`x`和`y`。然后定义要对它们执行的操作。接下来，使用`tf.Session`对象作为上下文管理器，创建一个容器来封装运行时环境，并通过用`feed_dict`将实值输入占位符来执行乘法。最后，还是在会话里面，你 [`print()`](https://realpython.com/python-print/) 的结果。

借助 TensorFlow 2.0 中的热切执行，您只需 [`tf.multiply()`](https://www.tensorflow.org/api_docs/python/tf/math/multiply) 即可实现相同的结果:

>>>

```py
>>> import tensorflow as tf

>>> x = [[2., 4., 6.]]
>>> y = [[1.], [3.], [5.]]
>>> m = tf.multiply(x, y)

>>> m
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 2.,  4.,  6.],
 [ 6., 12., 18.],
 [10., 20., 30.]], dtype=float32)>
```

在这段代码中，您使用 Python 列表表示法声明您的张量，当您调用时，`tf.multiply()`立即执行**元素级乘法**。

如果你不想或者不需要构建底层组件，那么推荐使用 TensorFlow 的方式是 [Keras](https://keras.io/) 。它具有更简单的 API，将常见用例转化为预制组件，并提供比 base TensorFlow 更好的错误消息。

### 特殊功能

TensorFlow 拥有庞大而成熟的用户群，以及大量帮助生产机器学习的工具。对于移动开发，它有用于 [JavaScript](https://realpython.com/python-vs-javascript/) 和 Swift 的 API，而 [TensorFlow Lite](https://www.tensorflow.org/lite) 可以让你压缩和优化物联网设备的模型。

您可以快速开始使用 TensorFlow，因为谷歌和第三方都提供了丰富的数据、预训练模型和[谷歌 Colab 笔记本](https://keras.io/examples/)。

TensorFlow 内置了很多流行的机器学习算法和数据集，随时可以使用。除了内置的数据集，你可以访问[谷歌研究数据集](https://research.google/tools/datasets/)或使用谷歌的[数据集搜索](https://datasetsearch.research.google.com/)来找到更多。

Keras 使模型的建立和运行变得更加容易，因此您可以在更短的时间内尝试新的技术。事实上，Keras 是 [Kaggle](https://www.kaggle.com/) 上五个获胜团队中使用最多的深度学习框架。

一个缺点是，从 TensorFlow 1.x 到 TensorFlow 2.0 的更新[改变了太多的功能](https://www.tensorflow.org/guide/effective_tf2)，你可能会发现自己很困惑。[升级代码](https://www.tensorflow.org/guide/migrate)繁琐且容易出错。许多资源，如教程，可能包含过时的建议。

PyTorch 没有同样大的向后兼容性问题，这可能是选择它而不是 TensorFlow 的一个原因。

[*Remove ads*](/account/join/)

### Tensorflow Ecosystem

TensorFlow 扩展生态系统的 API、扩展和有用工具的一些亮点包括:

*   [TensorFlow Hub](https://tfhub.dev/) ，一个可重用机器学习模块的库
*   [模型花园](https://github.com/tensorflow/models/tree/master/official)，使用 TensorFlow 高级 API 的官方模型集合
*   [*用 Scikit-Learn、Keras 和 TensorFlow*](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) 进行动手机器学习，全面介绍使用 TensorFlow 进行机器学习

## PyTorch 是什么？

PyTorch 由脸书开发，于 2016 年首次[公开发布](https://github.com/pytorch/pytorch/releases/tag/v0.1.1)。创建它是为了提供类似 TensorFlow 的生产优化，同时使模型更容易编写。

因为 Python 程序员发现它使用起来如此自然，PyTorch 迅速获得了用户，激励 TensorFlow 团队在 TensorFlow 2.0 中采用了 PyTorch 的许多最受欢迎的功能。

### 谁用 PyTorch？

PyTorch 以在研究中比在生产中应用更广泛而闻名。然而，自从在 TensorFlow 发布一年后，PyTorch 被专业开发人员大量使用。

[2020 Stack Overflow 开发者调查](https://insights.stackoverflow.com/survey/2020#technology-other-frameworks-libraries-and-tools-professional-developers3)最受欢迎的“其他框架、库和工具”列表显示，10.4%的专业开发者选择 TensorFlow，4.1%选择 PyTorch。在 [2018](https://insights.stackoverflow.com/survey/2018#technology-_-frameworks-libraries-and-tools) 中，TensorFlow 的比例为 7.6%，PyTorch 仅为 1.6%。

至于研究，PyTorch 是一个受欢迎的选择，像斯坦福大学的计算机科学项目现在用它来教授深度学习。

### 代码样式和功能

PyTorch 基于 [Torch](http://torch.ch/) ，这是一个用 c 语言编写的快速计算框架。Torch 有一个用于构建模型的 [Lua](https://www.lua.org/about.html) 包装器。

PyTorch [将相同的 C 后端](https://discuss.pytorch.org/t/roadmap-for-torch-and-pytorch/38/2)包装在 Python 接口中。但它不仅仅是一个包装纸。开发人员从头开始构建它是为了让 Python 程序员更容易编写模型。底层的低级 C 和 C++代码针对运行 Python 代码进行了优化。由于这种紧密集成，您可以:

*   更好的内存和优化
*   更合理的错误消息
*   模型结构的细粒度控制
*   更透明的模型行为
*   与 NumPy 的兼容性更好

这意味着你可以直接用 Python 编写高度定制的神经网络组件，而不必使用大量的底层函数。

PyTorch 的 [eager execution](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) ，可以立即动态地计算张量运算，启发了 TensorFlow 2.0，所以两者的 API 看起来非常相似。

将 NumPy 对象转换成张量是 PyTorch 的核心数据结构。这意味着您可以轻松地在`torch.Tensor`对象和`numpy.array`对象之间来回切换。

例如，您可以使用 PyTorch 将 NumPy 数组转换为张量的本机支持来创建两个`numpy.array`对象，使用`torch.from_numpy()`将每个对象转换为`torch.Tensor`对象，然后获取它们的[元素级乘积](https://pytorch.org/docs/master/generated/torch.mul.html):

>>>

```py
>>> import torch
>>> import numpy as np

>>> x = np.array([[2., 4., 6.]])
>>> y = np.array([[1.], [3.], [5.]])

>>> m = torch.mul(torch.from_numpy(x), torch.from_numpy(y))

>>> m.numpy()
array([[ 2.,  4.,  6.],
 [ 6., 12., 18.],
 [10., 20., 30.]])
```

使用`torch.Tensor.numpy()`可以将矩阵乘法的结果——它是一个`torch.Tensor`对象——作为一个`numpy.array`对象打印出来。

一个`torch.Tensor`对象和一个`numpy.array`对象最重要的区别就是`torch.Tensor` [类](https://github.com/pytorch/pytorch/blob/master/torch/tensor.py)有不同的方法和属性，比如 [`backward()`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.backward) ，它计算渐变， [CUDA](https://en.wikipedia.org/wiki/CUDA) 兼容性。

[*Remove ads*](/account/join/)

### 特殊功能

PyTorch 为 Torch 后端添加了一个用于自动分化的 C++模块。自动微分在[反向传播](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#backprop)期间自动计算`torch.nn`中定义的函数的梯度。

默认情况下，PyTorch 使用**急切模式**计算。您可以在构建神经网络时一行一行地运行它，这样更容易调试。这也使得构造具有条件执行的神经网络成为可能。对于大多数 Python 程序员来说，这种动态执行更加直观。

### PyTorch 生态系统

PyTorch 扩展生态系统的 API、扩展和有用工具的一些亮点包括:

*   fast . ai API，这使得快速构建模型变得非常容易
*   TorchServe ，AWS 和脸书合作开发的开源模型服务器
*   [TorchElastic](http://pytorch.org/elastic/0.2.0rc0/kubernetes.html) 使用 [Kubernetes](https://kubernetes.io/docs/tutorials/kubernetes-basics/) 大规模训练深度神经网络
*   [PyTorch Hub](https://pytorch.org/hub/research-models) ，一个分享和推广前沿模型的活跃社区

## PyTorch vs TensorFlow 决策指南

使用哪个库取决于您自己的风格和偏好、您的数据和模型以及您的项目目标。当您开始您的项目时，稍微研究一下哪个库最好地支持这三个因素，您将为自己的成功做好准备！

### 风格

如果你是一个 Python 程序员，那么 PyTorch 会感觉很容易上手。开箱即可按照您的预期方式运行。

另一方面，TensorFlow 比 PyTorch 支持更多的编码语言，py torch 有一个 C++ API。在 JavaScript 和 Swift 中都可以使用 TensorFlow。如果你不想写太多的底层代码，那么 Keras 抽象出了很多常见用例的细节，这样你就可以构建 TensorFlow 模型，而不用担心细节。

### 数据和模型

你用的是什么型号？如果你想使用一个特定的预训练模型，像[伯特](https://github.com/google-research/bert)或[深梦](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb)，那么你应该研究它与什么兼容。一些预训练的模型只在一个库中可用，而一些在两个库中都可用。模型花园、PyTorch 和 TensorFlow 中心也是很好的资源。

需要什么数据？如果您想要使用预处理数据，那么它可能已经被构建到一个或另一个库中。查看文档，这将使你的开发更快！

### 项目目标

你的模特会住在哪里？如果您想在移动设备上部署模型，那么 TensorFlow 是一个不错的选择，因为 TensorFlow Lite 及其 Swift API。对于服务模型，TensorFlow 与 Google Cloud 紧密集成，但 PyTorch 集成到 AWS 上的 TorchServe 中。如果你想参加 Kaggle 比赛，那么 Keras 会让你快速迭代实验。

在项目开始时考虑这些问题和例子。确定两三个最重要的组件，TensorFlow 或 PyTorch 将成为正确的选择。

## 结论

在本教程中，您已经了解了 PyTorch 和 TensorFlow，了解了谁使用它们以及它们支持哪些 API，并了解了如何为您的项目选择 PyTorch 和 TensorFlow。您已经看到了每种语言支持的不同编程语言、工具、数据集和模型，并了解了如何选择最适合您独特风格和项目的语言。

**在本教程中，您学习了:**

*   **PyTorch** 和 **TensorFlow** 有什么区别
*   如何使用**张量**在每个中进行计算
*   对于不同类型的项目，哪个平台最适合
*   各自支持哪些**工具**和**数据**

既然已经决定了使用哪个库，就可以开始用它们构建神经网络了。查看进一步阅读中的链接以获取想法。

[*Remove ads*](/account/join/)

## 延伸阅读

以下教程是实践 PyTorch 和 TensorFlow 的好方法:

*   [用 Python 和 Keras 进行实用的文本分类](https://realpython.com/python-keras-text-classification/)教你用 PyTorch 构建一个自然语言处理应用。

*   [在 Windows 上设置 Python 进行机器学习](https://realpython.com/python-windows-machine-learning-setup/)有关于在 Windows 上安装 PyTorch 和 Keras 的信息。

*   [纯 Python vs NumPy vs TensorFlow 性能对比](https://realpython.com/numpy-tensorflow-performance/)教你如何使用 TensorFlow 和 NumPy 做梯度下降，以及如何对你的代码进行基准测试。

*   [Python 上下文管理器和“with”语句](https://realpython.com/courses/python-context-managers-and-with-statement/)将帮助您理解为什么需要在 TensorFlow 1.0 中使用`with tf.compat.v1.Session() as session`。

*   [生成对抗网络:构建您的第一个模型](https://realpython.com/generative-adversarial-networks/)将带您使用 PyTorch 构建一个生成对抗网络来生成手写数字！

*   Python 系列中的[机器学习是更多项目想法的伟大来源，比如构建语音识别引擎或执行人脸识别。](https://realpython.com/learning-paths/machine-learning-python/)****