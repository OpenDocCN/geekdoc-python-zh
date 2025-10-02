# 科学应用

## 背景

Python 被经常使用在高性能科学应用中。它之所以在学术和科研项目中得到 如此广泛的应用是因为它容易编写而且执行效果很好。

由于科学计算对高性能的要求，Python 中相关操作经常借用外部库，通常是 以更快的语言（如 C，或者 FORTRAN 来进行矩阵操作）写的。其主要的库有 [Numpy](http://numpy.scipy.org/) [http://numpy.scipy.org/], [Scipy](http://scipy.org/) [http://scipy.org/] 以及 [Matplotlib](http://matplotlib.sourceforge.net/) [http://matplotlib.sourceforge.net/]。关于这些库的细节超出了本指南的范围。不过， 关于 Python 的科学计算生态的综合介绍可以在这里找到 [Python Scientific Lecture Notes](http://scipy-lectures.github.com/) [http://scipy-lectures.github.com/]。

## 工具

### IPython

[IPython](http://ipython.org/) [http://ipython.org/] 是一个加强版 Python 解释器，它提供了科学工作者 感兴趣的特性。其中，inline mode 允许将图像绘制到终端中（基于 Qt）。 进一步的，notebook 模式支持文学化编程（literate programming， 译者注：作者这里可能是指其富文本性不是那个编程范式）与可重现性（reproducible， 译者注：作者可能是指每段程序可以单独重新计算的特性），它产生了一个基于 web 的 python 笔记本。这个笔记本允许你保存一些代码块，伴随着它们的计算结果以及增强的 注释（HTML,LaTex,Markdown）。这个笔记本可以被共享并以各种文件格式导出。

## 库

### NumPy

[NumPy](http://numpy.scipy.org/) [http://numpy.scipy.org/] 是一个用 C 和 FORTRAN 写的底层库，它提供一些高层 数学函数。NumPy 通过多维数组和操作这些数组的函数巧妙地解决了 Python 运行算法较慢的问题。 任何算法只要被写成数组中的函数，就可以运行得很快。

NumPy 是 SciPy 项目中的一部分，它被发布为一个独立的库，这样对于只需基本功能的人来说， 就不用安装 SciPy 的其余部分。

NumPy 兼容 Python 2.4-2.7.2 以及 3.1+。

### Numba

[Numba](http://numba.pydata.org) [http://numba.pydata.org] 是一个针对 NumPy 的 Python 编译器（即时编译器,JIT） 它通过特殊的装饰器，将标注过的 Python（以及 NumPy）代码编译到 LLVM（Low Level Virtual Machine，底层虚拟机）中。简单地说，Python 使用一种机制，用 LLVM 将 Python 代码编译为 能够在运行时执行的本地代码。

### SciPy

[SciPy](http://scipy.org/) [http://scipy.org/] 是基于 NumPy 并提供了更多的数学函数的库。 SciPy 使用 NumPy 数组作为基本数据结构，并提供完成各种常见科学编程任务的模块， 包括线性代数，积分（微积分），常微分方程求解以及信号过程。

### Matplotlib

[Matplotlib](http://matplotlib.sourceforge.net/) [http://matplotlib.sourceforge.net/] 是一个可以灵活绘图的库，它 能够创建 2D、3D 交互式图形，并能保存成具有稿件质量（manuscript-quality）的图表。 其 API 很像 [MATLAB](http://www.mathworks.com/products/matlab/) [http://www.mathworks.com/products/matlab/]，这使得 MATLAB 用户 很容易转移到 Python。在 [matplotlib gallery](http://matplotlib.sourceforge.net/gallery.html) [http://matplotlib.sourceforge.net/gallery.html] 中可以找到很多例子以及实现它们的源代码（可以在此基础上再创造）。

### Pandas

[Pandas](http://pandas.pydata.org/) [http://pandas.pydata.org/] 是一个基于 NumPy 的数据处理库，它提供了 许多有用的函数能轻松地对数据进行访问、索引、合并以及归类。其主要数据结构（DataFrame） 与 R 统计学包十分相近；也就是，使用名称索引的异构数据（heterogeneous data）表、时间序列操作以及对数据的自动对准（auto-alignment）。

### Rpy2

[Rpy2](http://rpy.sourceforge.net/rpy2.html) [http://rpy.sourceforge.net/rpy2.html] 是一个对 R 统计学包的 Python 绑定， 它能够让 Python 执行 R 函数，并在两个环境中交换数据。Rpy2 是 对 [Rpy](http://rpy.sourceforge.net/rpy.html) [http://rpy.sourceforge.net/rpy.html] 绑定的面向对象实现。

### PsychoPy

[PsychoPy](http://www.psychopy.org/) [http://www.psychopy.org/] 是面向认知科学家的库，它允许创建 认知心理学和神经科学实验（译者注：指的是那种你坐在电脑前，给你一个刺激测 你反应的实验，基本上就是个 UI）。这个库能够处理刺激表示、实验设计脚本以及 数据收集。

## 资源

安装这些科学计算 Python 包可能会有些麻烦，因为它们中很多是用 Python 的 C 扩展实现的， 这就意味着需要编译。这一节列举了各种科学计算 Python 发行版，它们提供了预编译编译 且易于安装的科学计算 Python 包。

### Python 扩展包的非官方 Windows 二进制文件（库）

很多人在 Windows 平台上做科学计算，然而众所周知的是，其中很多科学计算包在该平台上 难以构建和安装。不过， [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/pythonlibs/) [http://www.lfd.uci.edu/~gohlke/pythonlibs/] 将一系列有用的 Python 包编译成了 Windows 的二进制文件，其数量还在不断增长。如果你在 Windows 上工作，你也许想要看看。

### Anaconda

[Continuum Analytics](http://continuum.io/) [http://continuum.io/] 提供了 [Anaconda Python Distribution](https://store.continuum.io/cshop/anaconda) [https://store.continuum.io/cshop/anaconda]，它 拥有所有常见的 Python 科学包，也包括与数据分析和大数据相关的包。Anaconda 是免费的 而 Continuum 销售一些专有的额外组件。学术研究者可以获取这些组件的免费许可。

### Canopy

[Canopy](https://www.enthought.com/products/canopy/) [https://www.enthought.com/products/canopy/] 是另一个 Python 科学发布版，由 [Enthought](https://www.enthought.com/) [https://www.enthought.com/] 提供。其受限制的 ‘Canopy Express’ 版本 是免费提供的，但是 Enthought 负责完整版。学术研究者可以获取到免费许可。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.