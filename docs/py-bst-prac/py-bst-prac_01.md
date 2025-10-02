# 选择一个解释器

 ## Python 的现状 (2 vs 3)

当选择 Python 解释器的时候，一个首先要面对的问题是：“我应该选择 Python 2 还是 Python 3？” 答案并不像人们想象的那么明显。

现状的基本要点如下：

1.  Python 2.7 作为标准有 *很长* 一段时间了。
2.  Python 3 将重大改变引入到语言中，其中有不少开发者并不满意。
3.  Python 2.7 直到 2020 前都会得到必要的安全更新 [[1]](#pep373-eol)。
4.  Python 3 正在不断发展，就像 Python 2 在过去几年一样。

所以，你现在可以看到为什么这不是一个简单的决定了。 

## 建议

那我直言不讳：

**使用 Python 3，如果：**

*   你不在乎。
*   你喜欢 Python 3。
*   你并不关心 Python2 和 3 的比较。
*   你不知道该用哪个。
*   你拥抱变化。

**使用 Python 2，如果：**

*   你热爱 Python 2，并对未来的 Python 3 感到悲伤。
*   一门语言及它的运行时永不改变，有助改善你的软件对稳定性的要求。
*   你依赖的软件需要它。

## 所以.... 3？

如果你想选择一种 Python 的解释器，你又不是固执己见的人，我推荐你用最新的 Python 3.x， 因为每个版本都带来了新的改进了的标准库模块、安全性以及 bug 修复。进步就是进步。

鉴于此，如果你有一个强有力的理由只用 Python 2，比如 Python 3 没有足够能替代的 Python 2 的特有库，或者你（像我）非常喜而且受 Python 2 启发。

查看 [Can I Use Python 3?](https://caniusepython3.com/) [https://caniusepython3.com/] 来看看是否有你 依赖的软件阻止你用 Python 3。

[延伸阅读](http://wiki.python.org/moin/Python2orPython3) [http://wiki.python.org/moin/Python2orPython3]

写 [能够同时兼容 Python 2.6，2.7，和 Python 3 的代码](https://docs.python.org/3/howto/pyporting.html) [https://docs.python.org/3/howto/pyporting.html] 是可能的。这包括从简单到困难的各种难度，取决于你所写软件的类型；如果你是初学者，其实有更重要的东西要操心。

## 实现

当人们谈论起 *Python*，他们不仅是在说语言本身，还包括其 CPython 实现。 *Python* 实际上是一个可以用许多不同的方式来实现的语言规范。

### CPython

[CPython](http://www.python.org) [http://www.python.org] 是 Python 的参考实现，用 C 编写。它把 Python 代码编译成 中间态的字节码，然后由虚拟机解释。CPython 为 Python 包和 C 扩展模块提供了最大限度的兼容。

如果你正在写开源的 Python 代码，并希望有尽可能广泛的用户，用 CPython 是最好的。使用依赖 C 扩展的包，CPython 是你唯一的选择。

所有版本的 Python 语言都用 C 实现，因为 CPython 是参考实现。

### PyPy

[PyPy](http://pypy.org/) [http://pypy.org/] 是用 RPython 实现的解释器。RPython 是 Python 的子集， 具有静态类型。这个解释器的特点是即时编译，支持多重后端（C, CLI, JVM）。

PyPy 旨在提高性能，同时保持最大兼容性（参考 CPython 的实现）。

如果你正在寻找提高你的 Python 代码性能的方法，值得试一试 PyPy。在一套的基准测试下， 它目前比 CPython 的速度快超过 5 倍 。

PyPy 支持 Python 2.7。PyPy3 [[2]](#pypy-ver)，发布的 Beta 版，支持 Python 3。

### Jython

[Jython](http://www.jython.org/) [http://www.jython.org/] 是一个将 Python 代码编译成 Java 字节码的实现， 运行在 JVM (Java Virtual Machine) 上。另外，它可以像是用 Python 模块一样，导入 并使用任何 Java 类。

如果你需要与现有的 Java 代码库对接或者基于其他原因需要为 JVM 编写 Python 代码，那么 Jython 是最好的选择。

Jython 现在支持到 Python 2.7 [[3]](#jython-ver)。

### IronPython

[IronPython](http://ironpython.net/) [http://ironpython.net/] 是一个针对 .NET 框架的 Python 实现。它 可以用 Python 和.NET framework 的库，也能将 Python 代码暴露给给.NET 框架中的其他语言。

[Python Tools for Visual Studio](http://ironpython.net/tools/) [http://ironpython.net/tools/] 直接集成了 IronPython 到 Visual Studio 开发环境中，使之成为 Windows 开发者的理想选择。

IronPython 支持 Python 2.7 [[4]](#iron-ver)。

### PythonNet

[Python for .NET](http://pythonnet.github.io/) [http://pythonnet.github.io/] 是一个近乎无缝集成的， 提供给本机已安装的 Python .NET 公共语言运行时（CLR）包。它采取与 IronPython （见上文）相反的方法，与其说是竞争，不如说是互补。

PythonNet 与 Mono 相结合使用，通过.NET 框架，能使 Python 在非 windows 系统上（如 OS X 和 Linux）完成操作。它可以在除外 IronPython 的环境中无冲突运行。

PythonNet 支持 Python 2.3 到 2.7 [[5]](#pythonnet-ver)。

| [[1]](#id2) | [`www.python.org/dev/peps/pep-0373/#id2`](https://www.python.org/dev/peps/pep-0373/#id2) |

| [[2]](#id9) | [`pypy.org/compat.html`](http://pypy.org/compat.html) |

| [[3]](#id11) | [`hg.python.org/jython/file/412a8f9445f7/NEWS`](https://hg.python.org/jython/file/412a8f9445f7/NEWS) |

| [[4]](#id13) | [`ironpython.codeplex.com/releases/view/81726`](http://ironpython.codeplex.com/releases/view/81726) |

| [[5]](#id14) | [`pythonnet.github.io/readme.html`](http://pythonnet.github.io/readme.html) |

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.