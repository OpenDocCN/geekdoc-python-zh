# 什么是 Python 轮子，为什么要关心？

> 原文：<https://realpython.com/python-wheels/>

Python `.whl`文件，或者说[轮子](https://packaging.python.org/glossary/#term-wheel)，是 Python 中很少被讨论的部分，但是它们对于 [Python 包](https://realpython.com/python-modules-packages/)的安装过程是一个福音。如果你已经使用 [`pip`](https://realpython.com/what-is-pip/) 安装了一个 Python 包，那么很有可能是一个轮子使得安装更快更有效。

Wheels 是 Python 生态系统的一个组件，它有助于让包安装*正常工作*。它们允许更快的安装和更稳定的软件包分发过程。在本教程中，您将深入了解什么是轮子，它们有什么好处，以及它们是如何获得牵引力并使 Python 变得更加有趣的。

在本教程中，您将学习:

*   什么是车轮，它们与**源分布**相比如何
*   如何使用轮子来控制**包装安装**过程
*   如何**为你自己的 Python 包创建和分发**轮子

您将从用户和开发人员的角度看到使用流行的开源 Python 包的例子。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 设置

接下来，激活一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，并确保你已经安装了最新版本的`pip`、`wheel`和`setuptools`:

```py
$ python -m venv env && source ./env/bin/activate
$ python -m pip install -U pip wheel setuptools
Successfully installed pip 20.1 setuptools-46.1.3 wheel-0.34.2
```

这就是你安装和制造轮子所需要的全部实验！

[*Remove ads*](/account/join/)

## Python 打包变得更好:Python Wheels 简介

在您学习如何将一个项目打包到一个轮子之前，了解从用户的角度看使用一个轮子是什么样子会有所帮助。这听起来可能有点落后，但是学习轮子如何工作的一个好方法是从安装一个不是轮子的东西开始。

您可以像平常一样，通过在您的环境中安装一个 Python 包来开始这个实验。在这种情况下，安装 [uWSGI](https://github.com/unbit/uwsgi) 版本 2.0.x:

```py
 1$ python -m pip install 'uwsgi==2.0.*'
 2Collecting uwsgi==2.0.*
 3 Downloading uwsgi-2.0.18.tar.gz (801 kB) 4 |████████████████████████████████| 801 kB 1.1 MB/s
 5Building wheels for collected packages: uwsgi
 6 Building wheel for uwsgi (setup.py) ... done
 7 Created wheel for uwsgi ... uWSGI-2.0.18-cp38-cp38-macosx_10_15_x86_64.whl
 8 Stored in directory: /private/var/folders/jc/8_hqsz0x1tdbp05 ...
 9Successfully built uwsgi
10Installing collected packages: uwsgi
11Successfully installed uwsgi-2.0.18
```

要完全安装 uWSGI，`pip`需要经过几个不同的步骤:

1.  在第 3 行的**上，它下载了一个名为`uwsgi-2.0.18.tar.gz`的 TAR 文件(tarball)，这个文件已经用 [gzip](https://www.gnu.org/software/gzip/manual/gzip.html) 压缩过了。**
2.  在**的第 6 行**，它获取 tarball 并通过调用`setup.py`构建一个`.whl`文件。
3.  在**线 7** 上，它标记车轮`uWSGI-2.0.18-cp38-cp38-macosx_10_15_x86_64.whl`。
4.  在**第 10 行**上，它在构建完车轮后安装实际的包。

`pip`取回的`tar.gz` tarball 是一个**源分布**，或`sdist`，而不是一个轮子。在某些方面，`sdist`是轮子的反义词。

**注意**:如果您看到 uWSGI 安装出错，您可能需要[安装 Python 开发头文件](https://uwsgi-docs.readthedocs.io/en/latest/Install.html#installing-from-source)。

一个[源代码分发](https://packaging.python.org/glossary/#term-source-distribution-or-sdist)包含源代码。这不仅包括 Python 代码，还包括与软件包捆绑在一起的任何扩展模块的源代码(通常用 [C](https://realpython.com/build-python-c-extension-module/) 或 [C++](https://realpython.com/python-vs-cpp/) )。对于源代码发行版，扩展模块是在用户端编译的，而不是在开发人员端。

源代码发行版还包含一个名为 [`<package-name>.egg-info`](https://setuptools.readthedocs.io/en/latest/formats.html) 的目录中的元数据包。这些元数据有助于构建和安装软件包，但是用户实际上不需要做任何事情。

从开发人员的角度来看，源代码发行版是在运行以下命令时创建的:

```py
$ python setup.py sdist
```

现在尝试安装一个不同的包， [chardet](https://github.com/chardet/chardet/blob/master/docs/index.rst) :

```py
 1$ python -m pip install 'chardet==3.*'
 2Collecting chardet
 3 Downloading chardet-3.0.4-py2.py3-none-any.whl (133 kB) 4 |████████████████████████████████| 133 kB 1.5 MB/s
 5Installing collected packages: chardet
 6Successfully installed chardet-3.0.4
```

您可以看到与 uWSGI 安装明显不同的输出。

安装 chardet 直接从 PyPI 下载一个`.whl`文件。轮子名称`chardet-3.0.4-py2.py3-none-any.whl`遵循一个特定的命名约定，您将在后面看到。从用户的角度来看，更重要的是，当`pip`在 PyPI 上找到一个兼容的轮子时，没有构建阶段。

从开发人员的角度来看，轮子是运行以下命令的结果:

```py
$ python setup.py bdist_wheel
```

为什么 uWSGI 给你一个源码分发而 chardet 提供一个轮子？通过查看 PyPI 上每个项目的页面，并导航到*下载文件*区域，你就能明白其中的原因。本节将向您展示`pip`在 PyPI 索引服务器上实际看到的内容:

*   **uw SGI**T3】由于与项目复杂性相关的原因，只提供了一个源分布 ( `uwsgi-2.0.18.tar.gz`)。
*   **chardet** [提供了一个轮子和一个源分布](https://pypi.org/project/chardet/3.0.4/#files)，但是如果与你的系统兼容的话`pip`会更喜欢轮子*。稍后您将看到如何确定这种兼容性。*

用于轮子安装的兼容性检查的另一个例子是 [`psycopg2`](https://pypi.org/project/psycopg2/2.8.5/#files) ，它为 Windows 提供了一系列轮子，但不为 Linux 或 macOS 客户端提供任何轮子。这意味着`pip install psycopg2`可以根据您的具体设置获取一个轮或一个源分布。

为了避免这些类型的兼容性问题，一些包提供了多个轮子，每个轮子都适合特定的 Python 实现和底层操作系统。

到目前为止，你已经看到了轮子和`sdist`之间的一些明显区别，但是更重要的是这些区别对安装过程的影响。

[*Remove ads*](/account/join/)

### 轮子让东西跑得更快

在上面，您看到了获取预建轮子的安装和下载`sdist`的安装的比较。Wheels 使得 Python 包的端到端安装更快，原因有二:

1.  在其他条件相同的情况下，轮子通常比源分布的尺寸小，这意味着它们可以在网络中更快地移动。
2.  从 wheels 直接安装避免了**在源代码发行版之外构建**包的中间步骤。

几乎可以保证，chardet 的安装只需要 uWSGI 所需时间的一小部分。然而，这可能是一个不公平的比较，因为 chardet 是一个非常小和不复杂的包。使用不同的命令，您可以创建一个更直接的比较，来演示轮子产生了多大的差异。

您可以通过传递`--no-binary`选项使`pip`忽略其向车轮的倾斜:

```py
$ time python -m pip install \
      --no-cache-dir \
      --force-reinstall \
 --no-binary=:all: \      cryptography
```

该命令对 [`cryptography`](https://pypi.org/project/cryptography/) 包的安装进行计时，告诉`pip`即使有合适的轮子也要使用源分布。包含`:all:`使得规则适用于`cryptography`及其所有[依赖项](https://realpython.com/courses/managing-python-dependencies/)。

在我的机器上，从开始到结束大约需要*32 秒*。不仅安装要花很长时间，而且构建`cryptography`还需要有 OpenSSL 开发头文件，并且 Python 可以使用。

**注意**:在`--no-binary`中，你很可能会看到一个关于`cryptography`安装所需的头文件丢失的错误，这是使用源代码发行版令人沮丧的原因之一。如果是这样的话，`cryptography`文档的[安装部分](https://cryptography.io/en/latest/installation/#building-cryptography-on-linux)会建议你为一个特定的操作系统需要哪些库和头文件。

现在你可以重新安装`cryptography`，但是这次要确保`pip`使用 PyPI 的轮子。因为`pip`更喜欢一个轮子，这类似于没有任何参数地调用`pip install`。但是在这种情况下，您可以通过要求一个带有`--only-binary`的轮子来明确意图:

```py
$ time python -m pip install \
      --no-cache-dir \
      --force-reinstall \
 --only-binary=cryptography \      cryptography
```

这个选项只需要 4 秒多一点，或者说*只需要对`cryptography`及其依赖项使用源代码发行版的八分之一*的时间。

### 什么是巨蟒轮？

Python `.whl`文件本质上是一个 ZIP ( `.zip`)档案，带有一个特制的文件名，告诉安装人员 wheel 将支持哪些 Python 版本和平台。

一个轮子是一种[型的**建成分布**型的](https://packaging.python.org/glossary/#term-built-distribution)。在这种情况下，*build*意味着这个轮子以一种现成的格式出现，允许你跳过源代码发行版所需要的构建阶段。

**注意**:值得一提的是，尽管使用了术语*构建*，但是一个轮子并不包含`.pyc`文件，或者编译的 Python 字节码。

车轮文件名被分成由连字符分隔的多个部分:

```py
{dist}-{version}(-{build})?-{python}-{abi}-{platform}.whl
```

`{brackets}`中的每一部分都是一个**标签**，或者是轮子名称的一个组成部分，承载着轮子包含的内容以及轮子在哪里可以工作或者不可以工作的一些含义。

下面是一个使用 [`cryptography`](https://github.com/pyca/cryptography) 滚轮的示例:

```py
cryptography-2.9.2-cp35-abi3-macosx_10_9_x86_64.whl
```

`cryptography`分配多个车轮。每个轮子都是一个**平台轮子**，这意味着它只支持 Python 版本、Python ABIs、操作系统和机器架构的特定组合。您可以将命名约定分成几个部分:

*   **`cryptography`** 是包名。

*   **`2.9.2`** 是`cryptography`的打包版本。版本是符合 [PEP 440](https://www.python.org/dev/peps/pep-0440/) 的字符串，例如`2.9.2`、`3.4`或`3.9.0.a3`。

*   **`cp35`** 是 [Python 标签](https://www.python.org/dev/peps/pep-0425/#python-tag)，表示轮子需要的 Python 实现和版本。`cp`代表 [CPython](https://realpython.com/cpython-source-code-guide/) ，Python 的参考实现，`35`代表 Python [3.5](https://docs.python.org/3/whatsnew/3.5.html) 。例如，这个轮子与 [Jython](https://www.jython.org/) 不兼容。

*   **`abi3`** 是 ABI 的标记。ABI 代表[应用二进制接口](https://docs.python.org/3/c-api/stable.html)。你真的不需要担心它需要什么，但是`abi3`是 Python C API 二进制兼容性的一个独立版本。

*   **`macosx_10_9_x86_64`** 是站台标签，恰好相当拗口。在这种情况下，它可以进一步分解为子部分:

    *   **`macosx`** 就是 [macOS](https://en.wikipedia.org/wiki/MacOS) 操作系统。
    *   **`10_9`** 是 macOS developer tools SDK 版本，用于编译 Python，进而构建这个轮子。
    *   **`x86_64`** 是对 x86-64 指令集架构的引用。

最后一个组件在技术上不是标签，而是标准的`.whl`文件扩展名。综合起来看，上述部件表示该`cryptography`轮设计用于的目标机器。

现在让我们来看一个不同的例子。以下是您在上述 chardet 案例中看到的内容:

```py
chardet-3.0.4-py2.py3-none-any.whl
```

你可以把它分解成标签:

*   **`chardet`** 是包名。
*   **`3.0.4`** 是 chardet 的包版本。
*   **`py2.py3`** 是 Python 标签，意思是轮子支持任何 Python 实现的 Python 2 和 3。
*   **`none`** 是 ABI 的标记，意思是 ABI 不是一个因素。
*   **`any`** 是站台。这个轮子几乎可以在任何平台上工作。

车轮名称的`py2.py3-none-any.whl`段很常见。这是一个**万向轮**，它将与 Python 2 或 3 一起安装在任何带有 [ABI](https://stackoverflow.com/a/2456882/7954504) 的平台上。如果轮子以`none-any.whl`结束，那么它很可能是一个不关心特定 Python ABI 或 CPU 架构的纯 Python 包。

另一个例子是`jinja2`模板引擎。如果你导航到 Jinja 3.x alpha 版本的[下载页面](https://pypi.org/project/Jinja2/3.0.0a1/#files)，那么你会看到下面的轮子:

```py
Jinja2-3.0.0a1-py3-none-any.whl
```

注意这里缺少了`py2`。这是一个纯 Python 项目，可以在任何 Python 3.x 版本上工作，但它不是万向轮，因为它不支持 Python 2。相反，它被称为**纯蟒蛇轮**。

**注**:2020 年，多个项目也在[放弃对 Python 2](https://www.python.org/doc/sunset-python-2/) 的支持，Python 2 于 2020 年 1 月 1 日达到寿命终止(EOL)。Jinja 版[于 2020 年 2 月放弃 Python 2 支持](https://github.com/pallets/jinja/pull/1136)。

以下是为一些流行的开源包分发的`.whl`名称的几个例子:

| 车轮 | 事实真相 |
| --- | --- |
| `PyYAML-5.3.1-cp38-cp38-win_amd64.whl` | [PyYAML](https://pypi.org/project/PyYAML/5.3.1/#files) 用于采用 AMD64 (x86-64)架构的 Windows 上的 CPython 3.8 |
| `numpy-1.18.4-cp38-cp38-win32.whl` | 用于 Windows 32 位上的 CPython 3.8 的 NumPy |
| `scipy-1.4.1-cp36-cp36m-macosx_10_6_intel.whl` | [SciPy](https://pypi.org/project/scipy/1.4.1/#files) 用于 macOS 10.6 SDK 上的 CPython 3.6，带有胖二进制(多指令集) |

既然你对什么是轮子有了透彻的了解，是时候谈谈它们有什么好处了。

[*Remove ads*](/account/join/)

### Python 车轮的优势

这里有一个来自 [Python 打包权威](https://www.pypa.io/en/latest/) (PyPA)的轮子的证明:

> 并不是所有的开发人员都有合适的工具或经验来构建用这些编译语言编写的组件，所以 Python 创建了 wheel，这是一种旨在将库与编译后的工件一起发布的包格式。事实上，Python 的包安装程序`pip`总是更喜欢轮子，因为安装总是更快，所以即使是纯 Python 的包也能更好地使用轮子。([来源](https://packaging.python.org/overview/#python-binary-distributions))

更全面的描述是，wheels [在几个方面对 Python 包的用户和维护者](https://pythonwheels.com/#advantages)都有好处:

*   对于纯 Python 包和[扩展模块](https://realpython.com/build-python-c-extension-module/)，轮子的安装速度都比源代码发行版快。

*   **轮子比源分布更小**。比如 [`six`](https://pypi.org/project/six/#files) 轮大约是[对应源分布](https://pypi.org/project/six/#files)的三分之一大小。当您考虑到单个包的`pip install`实际上可能会引发一系列依赖项的下载时，这种差异就变得更加重要了。

*   **车轮切`setup.py`执行出方程式。**从源代码安装运行*无论*包含在那个项目的`setup.py`中。正如 [PEP 427](https://www.python.org/dev/peps/pep-0427/#rationale) 所指出的，这相当于任意代码执行。轮子完全避免了这一点。

*   不需要编译器来安装包含已编译扩展模块的轮子。扩展模块包含在针对特定平台和 Python 版本的 wheel 中。

*   **`pip`自动生成`.pyc`文件**在轮子中匹配正确的 Python 解释器。

*   **轮子提供了一致性**,它将安装包所涉及的许多变量排除在方程式之外。

您可以使用 PyPI 上项目的*下载文件*选项卡来查看不同的可用发行版。例如，[熊猫](https://pypi.org/project/pandas/#files)分发各种各样的轮子。

### 告诉`pip`下载什么

可以对`pip`进行细粒度控制，并告诉它喜欢或避免哪种格式。您可以使用`--only-binary`和`--no-binary`选项来完成此操作。您已经在安装`cryptography`包的前一节中看到了它们，但是有必要仔细看看它们做了什么:

```py
$ pushd "$(mktemp -d)"
$ python -m pip download --only-binary :all: --dest . --no-cache six
Collecting six
 Downloading six-1.14.0-py2.py3-none-any.whl (10 kB)
 Saved ./six-1.14.0-py2.py3-none-any.whl
Successfully downloaded six
```

在本例中，您使用`pushd "$(mktemp -d)"`切换到一个临时目录来存储下载内容。你使用`pip download`而不是`pip install`，这样你就可以检查最终的轮子，但是你可以用`install`代替`download`，同时保持相同的选项集。

你下载的 [`six`](https://github.com/benjaminp/six) 模块有几个标志:

*   **`--only-binary :all:`** 告诉`pip`约束自己使用轮子并忽略源分布。如果没有这个选项，`pip`将只会*偏好*轮子，但在某些情况下会退回到源分布。
*   **`--dest .`** 告诉`pip`将`six`下载到当前目录。
*   **`--no-cache`** 告诉`pip`不要在本地下载缓存中查找。您使用这个选项只是为了演示从 PyPI 的实时下载，因为您很可能在某个地方有一个`six`缓存。

我前面提到过，wheel 文件本质上是一个`.zip`档案。你可以从字面上理解这句话，也可以这样看待轮子。例如，如果您想查看一个轮子的内容，您可以使用`unzip`:

```py
$ unzip -l six*.whl
Archive:  six-1.14.0-py2.py3-none-any.whl
 Length      Date    Time    Name
---------  ---------- -----   ----
 34074  01-15-2020 18:10   six.py
 1066  01-15-2020 18:10   six-1.14.0.dist-info/LICENSE
 1795  01-15-2020 18:10   six-1.14.0.dist-info/METADATA
 110  01-15-2020 18:10   six-1.14.0.dist-info/WHEEL
 4  01-15-2020 18:10   six-1.14.0.dist-info/top_level.txt
 435  01-15-2020 18:10   six-1.14.0.dist-info/RECORD
---------                     -------
 37484                     6 files
```

`six`是一个特例:它实际上是一个单独的 Python 模块，而不是一个完整的包。Wheel 文件也可能非常复杂，稍后您会看到这一点。

与`--only-binary`相反，您可以使用`--no-binary`来做相反的事情:

```py
$ python -m pip download --no-binary :all: --dest . --no-cache six
Collecting six
 Downloading six-1.14.0.tar.gz (33 kB)
 Saved ./six-1.14.0.tar.gz
Successfully downloaded six
$ popd
```

本例中唯一的变化是切换到`--no-binary :all:`。这告诉`pip`忽略轮子，即使它们可用，而是下载一个源发行版。

`--no-binary`什么时候可能有用？这里有几个案例:

*   **对应的轮子坏了。**这是对轮子的讽刺。它们的设计是为了减少东西损坏的频率，但在某些情况下，轮子可能会配置错误。在这种情况下，为自己下载并构建源代码发行版可能是一个可行的替代方案。

*   **你想将一个小的改变或[补丁文件](https://en.wikipedia.org/wiki/Patch_%28Unix%29)** 应用到项目中，然后安装它。这是从其[版本控制系统](https://realpython.com/python-git-github-intro/#version-control) URL 克隆项目的一种替代方法。

您也可以使用上述带有`pip install`的标志。此外，`:all:`不仅会将`--only-binary`规则应用到您正在安装的包，还会应用到它的所有依赖项，您可以向`--only-binary`和`--no-binary`传递应用该规则的特定包的列表。

下面举几个安装网址库 [`yarl`](https://github.com/aio-libs/yarl/) 的例子。包含 Cython 代码，依赖 [`multidict`](https://github.com/aio-libs/multidict) ，包含纯 C 代码。对于`yarl`及其依赖项，有几个严格使用或严格忽略轮子的选项:

```py
$ # Install `yarl` and use only wheels for yarl and all dependencies
$ python -m pip install --only-binary :all: yarl

$ # Install `yarl` and use wheels only for the `multidict` dependency
$ python -m pip install --only-binary multidict yarl

$ # Install `yarl` and don't use wheels for yarl or any dependencies
$ python -m pip install --no-binary :all: yarl

$ # Install `yarl` and don't use wheels for the `multidict` dependency
$ python -m pip install --no-binary multidict yarl
```

在本节中，您了解了如何微调`pip install`将使用的发布类型。虽然常规的`pip install`应该没有选项，但了解这些选项对于特殊情况是有帮助的。

[*Remove ads*](/account/join/)

### `manylinux`车轮标签

Linux 有许多变体和风格，比如 Debian、CentOS、Fedora 和 Pacman。其中的每一个都可能在共享库(如`libncurses`)和核心 C 库(如`glibc`)中略有不同。

如果你正在写一个 C/C++扩展，那么这可能会产生一个问题。用 C 编写并在 Ubuntu Linux 上编译的源文件不能保证在 CentOS 机器或 Arch Linux 发行版上是可执行的。你需要为每一个 Linux 变种建立一个单独的轮子吗？

幸运的是，答案是否定的，这要归功于一组特别设计的标签，称为 **`manylinux`** 平台标签家族。目前有三种变体:

1.  **`manylinux1`** 是[人教版 513](https://www.python.org/dev/peps/pep-0513/) 中规定的原始格式。

2.  **`manylinux2010`** 是 [PEP 571](https://www.python.org/dev/peps/pep-0571/) 中指定的更新，升级到 CentOS 6 作为 Docker 镜像所基于的底层操作系统。理由是 CentOS 5.11，即`manylinux1`中允许的库列表的来源，于 2017 年 3 月达到 EOL，并停止接收安全补丁和错误修复。

3.  **`manylinux2014`** 是 [PEP 599](https://www.python.org/dev/peps/pep-0599/) 中指定的升级到 CentOS 7 的更新，因为 CentOS 6 计划于 2020 年 11 月达到 EOL。

你可以在[熊猫](https://realpython.com/pandas-python-explore-dataset/)项目中找到一个`manylinux`分布的例子。这里是从 PyPI 下载的可用[熊猫列表中的两个(从许多中选出来的):](https://pypi.org/project/pandas/1.0.3/#files)

```py
pandas-1.0.3-cp37-cp37m-manylinux1_x86_64.whl
pandas-1.0.3-cp37-cp37m-manylinux1_i686.whl
```

在这种情况下，pandas 为 CPython 3.7 构建了`manylinux1`轮子，支持 x86-64 和 [i686](https://en.wikipedia.org/wiki/P6_(microarchitecture)) 架构。

在它的核心，`manylinux`是一个 [Docker 镜像](https://www.python.org/dev/peps/pep-0513/#docker-image)，构建于 CentOS 操作系统的某个版本之上。它捆绑了一个编译器套件、多个版本的 Python 和`pip`，以及一组允许的共享库。

**注意**:术语**允许**表示一个低级的库，默认情况下会出现在几乎所有的 Linux 系统上[。这个想法是，依赖关系应该存在于基本操作系统上，而不需要额外安装。](https://www.python.org/dev/peps/pep-0513/#rationale)

截至 2020 年中期，`manylinux1`仍然是主要的`manylinux`标签。其中一个原因可能只是习惯。另一个原因可能是客户端(用户)对`manylinux2010`及以上版本的支持仅限于[的更新版本](https://pip.pypa.io/en/stable/news/)或`pip`:

| 标签 | 要求 |
| --- | --- |
| `manylinux1` | 8.1.0 或更高版本 |
| `manylinux2010` | `pip` 19.0 或更高版本 |
| `manylinux2014` | `pip` 19.3 或更高版本 |

换句话说，如果你是一个构建`manylinux2010`轮子的包开发者，那么使用你的包的人将需要`pip`19.0(2019 年 1 月发布)或更高版本来让`pip`从 PyPI 找到并安装`manylinux2010`轮子。

幸运的是，虚拟环境已经变得越来越普遍，这意味着开发人员可以在不接触系统`pip`的情况下更新虚拟环境的`pip`。然而，情况并非总是如此，一些 Linux 发行版仍然附带了过时版本的`pip`。

也就是说，如果你正在 Linux 主机上安装 Python 包，那么如果包的维护者不怕麻烦地创建了`manylinux`轮子，你应该感到幸运。这将几乎保证软件包的安装没有任何麻烦，不管您的具体 Linux 变体或版本如何。

**注意**:注意 [PyPI 轮在 Alpine Linux](https://pythonspeed.com/articles/alpine-docker-python/) (或者 [BusyBox](https://hub.docker.com/_/busybox/) )上不工作。这是因为 Alpine 使用 [`musl`](https://wiki.musl-libc.org/) 代替了标准 [`glibc`](https://www.gnu.org/software/libc/libc.html) 。`musl libc`图书馆标榜自己是“一个新的`libc`，努力做到快速、简单、轻量级、免费和正确。”不幸的是，说到轮子，`glibc`就不是了。

### 平台车轮的安全注意事项

从用户安全的角度来看，wheels 的一个值得考虑的特性是，wheels[可能会受到版本腐烂](https://github.com/asottile/no-manylinux#what-why)的影响，因为它们捆绑了一个二进制依赖项，而不允许系统包管理器更新该依赖项。

例如，如果一个轮子包含了 [`libfortran`](https://gcc.gnu.org/fortran/) 共享库，那么该轮子的发行版将使用它们所捆绑的`libfortran`版本，即使你用一个包管理器如`apt`、`yum`或`brew`来升级你自己机器的`libfortran`版本。

如果您在一个具有高度安全防范措施的环境中进行开发，某些平台轮子的这个特性是需要注意的。

[*Remove ads*](/account/join/)

## 召集所有开发者:打造你的车轮

本教程的标题是“你为什么要关心？”作为一名开发人员，如果您计划向社区分发 Python 包，那么您应该非常关心为您的项目分发 wheels，因为它们使最终用户的安装过程更干净，更简单。

你可以用兼容的轮子支持的目标平台越多，你就会越少看到标题为“在 XYZ 平台上安装失败”的问题为 Python 包分发轮子客观上降低了包的用户在安装过程中遇到问题的可能性。

在本地构建一个轮子你需要做的第一件事就是安装`wheel`。确保`setuptools`也是最新的也无妨:

```py
$ python -m pip install -U wheel setuptools
```

接下来的几节将带您在各种不同的场景中构建轮子。

### 不同类型的车轮

正如本教程中所提到的，轮子有几种不同的[变体，轮子的类型反映在其文件名中:](https://packaging.python.org/guides/distributing-packages-using-setuptools/#wheels)

*   一个**万向轮**包含`py2.py3-none-any.whl`。它在任何操作系统和平台上都支持 Python 2 和 Python 3。在[巨蟒轮](https://pythonwheels.com/)网站上列出的大多数轮子都是万向轮。

*   一个**纯蟒轮**包含`py3-none-any.whl`或`py2.none-any.whl`。它支持 Python 3 或 Python 2，但不支持两者。它在其他方面与万向轮相同，但它将贴上`py2`或`py3`的标签，而不是`py2.py3`的标签。

*   一个**平台轮**支持特定的 Python 版本和平台。它包含指示特定 Python 版本、ABI、操作系统或架构的段。

轮子类型之间的差异取决于它们支持的 Python 版本以及它们是否针对特定的平台。以下是车轮变化之间差异的简明摘要:

| 车轮类型 | 支持 Python 2 和 3 | 支持所有 ABI、操作系统和平台 |
| --- | --- | --- |
| 普遍的 | -好的 | -好的 |
| 纯 Python 语言 |  | -好的 |
| 平台 |  |  |

正如您接下来将看到的，您可以通过相对较少的设置来构建万向轮和纯 Python 轮，但是平台轮可能需要一些额外的步骤。

### 打造纯 Python 车轮

您可以使用`setuptools` 为任何[项目构建一个纯 Python 轮子或通用轮子，只需一个命令:](https://realpython.com/pypi-publish-python-package/)

```py
$ python setup.py sdist bdist_wheel
```

这将创建一个源分布(`sdist`)和一个轮(`bdist_wheel`)。默认情况下，两者都将放在当前目录下的`dist/`中。为了自己看，你可以为 [HTTPie](https://github.com/jakubroztocil/httpie) 构建一个轮子，这是一个用 Python 编写的命令行 HTTP 客户端，旁边还有一个`sdist`。

下面是为 HTTPie 包构建两种类型的发行版的结果:

```py
$ git clone -q git@github.com:jakubroztocil/httpie.git
$ cd httpie
$ python setup.py -q sdist bdist_wheel $ ls -1 dist/
httpie-2.2.0.dev0-py3-none-any.whl
httpie-2.2.0.dev0.tar.gz
```

这就够了。您克隆项目，移动到它的根目录，然后调用`python setup.py sdist bdist_wheel`。你可以看到`dist/`包含了一个轮子和一个源分布。

默认情况下，得到的分布放在`dist/`中，但是您可以用`-d` / `--dist-dir`选项来改变它。您可以将它们放在临时目录中，而不是用于构建隔离:

```py
$ tempdir="$(mktemp -d)"  # Create a temporary directory
$ file "$tempdir"
/var/folders/jc/8_kd8uusys7ak09_lpmn30rw0000gk/T/tmp.GIXy7XKV: directory

$ python setup.py sdist -d "$tempdir"
$ python setup.py bdist_wheel --dist-dir "$tempdir"
$ ls -1 "$tempdir"
httpie-2.2.0.dev0-py3-none-any.whl
httpie-2.2.0.dev0.tar.gz
```

您可以将`sdist`和`bdist_wheel`步骤合并成一个，因为`setup.py`可以接受多个子命令:

```py
$ python setup.py sdist -d "$tempdir" bdist_wheel -d "$tempdir"
```

如此处所示，您需要将选项如`-d`传递给每个子命令。

[*Remove ads*](/account/join/)

### 指定万向轮

万向轮是支持 Python 2 和 3 的纯 Python 项目的轮子。有多种方法可以告诉`setuptools`和`distutils`一个轮子应该是通用的。

选项 1 是在您项目的 [`setup.cfg`](https://docs.python.org/3/distutils/configfile.html) 文件中指定选项:

```py
[bdist_wheel] universal  =  1
```

选项 2 是在命令行传递恰当命名的`--universal`标志:

```py
$ python setup.py bdist_wheel --universal
```

选项 3 是使用它的`options`参数告诉`setup()`它自己关于标志的信息:

```py
# setup.py
from setuptools import setup

setup(
    # ....
    options={"bdist_wheel": {"universal": True}}
    # ....
)
```

虽然这三个选项中的任何一个都可以，但前两个是最常用的。你可以在 [chardet 设置配置](https://github.com/chardet/chardet/blob/master/setup.cfg)中看到这样的例子。之后，您可以使用前面所示的`bdist_wheel`命令:

```py
$ python setup.py sdist bdist_wheel
```

无论您选择哪一个选项，最终的控制盘都是相同的。这种选择很大程度上取决于开发人员的偏好以及哪种工作流最适合您。

### 构建平台轮(macOS 和 Windows)

[**二进制发行版**](https://packaging.python.org/glossary/#term-binary-distribution) 是包含编译扩展的**构建发行版**的子集。**扩展**是你的 Python 包的非 Python 依赖或组件。

通常，这意味着你的包包含一个扩展模块或者依赖于一个用静态类型语言编写的库，比如 C，C++，Fortran，甚至是 [Rust](https://github.com/ijl/orjson) 或者 Go。**平台轮**的存在主要是针对单个平台，因为它们包含或依赖于扩展模块。

说了这么多，是时候造一个平台轮了！

根据您现有的开发环境，您可能需要完成一两个额外的先决步骤来构建平台轮子。下面的步骤将帮助您建立 C 和 C++扩展模块，这是目前最常见的类型。

在 macOS 上，您需要通过 [`xcode`](https://www.unix.com/man-page/OSX/1/xcode-select/) 获得命令行开发工具:

```py
$ xcode-select --install
```

在 Windows 上，你需要安装[微软 Visual C++](https://docs.microsoft.com/en-us/cpp/?view=vs-2019) :

1.  在浏览器中打开 [Visual Studio 下载页面](https://visualstudio.microsoft.com/downloads/)。
2.  选择*Visual Studio 工具→Visual Studio 构建工具→下载*。
3.  运行产生的`.exe`安装程序。
4.  在安装程序中，选择 *C++构建工具→安装*。
5.  重启你的机器。

在 Linux 上，你需要一个 [`gcc`](https://linux.die.net/man/1/gcc) 或者`g++` / `c++`这样的编译器。

做好准备后，您就可以为 UltraJSON ( `ujson`)构建一个平台轮了，这是一个用纯 C 编写的 [JSON](https://realpython.com/python-json/) 编码器和解码器，使用 Python 3 [绑定](https://realpython.com/python-bindings-overview/)。使用`ujson`是一个很好的玩具示例，因为它涵盖了几个基础:

1.  它包含一个扩展模块， [`ujson`](https://github.com/ultrajson/ultrajson/blob/master/python/ujson.c) 。
2.  它依赖于要编译的 Python 开发头文件(`#include <Python.h>`)，但并不复杂。`ujson`就是为了做一件事，并且做好这件事，就是读写 JSON！

您可以从 GitHub 克隆这个项目，导航到它的目录，然后编译它:

```py
$ git clone -q --branch 2.0.3 git@github.com:ultrajson/ultrajson.git
$ cd ultrajson
$ python setup.py bdist_wheel
```

您应该会看到大量的输出。这里有一个在 macOS 上的精简版本，其中使用了 [Clang](https://clang.llvm.org/) 编译器驱动程序:

```py
clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g ...
...
creating 'dist/ujson-2.0.3-cp38-cp38-macosx_10_15_x86_64.whl'
adding 'ujson.cpython-38-darwin.so'
```

以`clang`开头的代码行显示了对编译器的实际调用，包括一组编译标志。根据操作系统的不同，你可能还会看到像`MSVC` (Windows)或`gcc` (Linux)这样的工具。

如果在执行完上面的代码后遇到了一个`fatal error`，不用担心。你可以展开下面的方框，学习如何处理这个问题。



对`ujson`的这个`setup.py bdist_wheel`调用需要 [Python 开发头文件](https://github.com/python/cpython/tree/master/Include)，因为`ujson.c`拉入了`<Python.h>`。如果您没有将它们放在可搜索的位置，那么您可能会看到如下错误:

```py
fatal error: 'Python.h' file not found
#include <Python.h>
```

要编译扩展模块，您需要将开发头文件保存在编译器可以找到它们的地方。

如果您使用的是 Python 3 的最新版本和虚拟环境工具，比如`venv`，那么 Python 开发头很可能会默认包含在编译和链接中。

否则，您可能会看到一个错误，指示找不到头文件:

```py
fatal error: 'Python.h' file not found
#include <Python.h>
```

在这种情况下，您可以通过设置`CFLAGS`来告诉`setup.py`在其他地方寻找头文件。要找到头文件本身，可以使用`python3-config`:

```py
$ python3-config --include
-I/Users/<username>/.pyenv/versions/3.8.2/include/python3.8
```

这告诉您 Python 开发头文件位于所示的目录中，您现在可以将它与`python setup.py bdist_wheel`一起使用:

```py
$ CFLAGS="$(python3-config --include)" python setup.py bdist_wheel
```

更一般地说，您可以传递您需要的任何路径:

```py
$ CFLAGS='-I/path/to/include' python setup.py bdist_wheel
```

在 Linux 上，您可能还需要单独安装头文件:

```py
$ apt-get install -y python3-dev  # Debian, Ubuntu
$ yum install -y python3-devel  # CentOS, Fedora, RHEL
```

如果你检查 UltraJSON 的 [`setup.py`](https://github.com/ultrajson/ultrajson/blob/master/setup.py) ，那么你会看到它定制了一些编译器标志比如`-D_GNU_SOURCE`。通过`setup.py`控制编译过程的复杂性超出了本教程的范围，但是您应该知道，对编译和链接如何发生进行[细粒度的控制是可能的](https://pythonextensionpatterns.readthedocs.io/en/latest/compiler_flags.html#setting-flags-automatically-in-setup-py)。

如果你查看`dist`，那么你应该会看到创建的轮子:

```py
$ ls dist/
ujson-2.0.3-cp38-cp38-macosx_10_15_x86_64.whl
```

请注意，该名称可能因平台而异。例如，您会在 64 位 Windows 上看到`win_amd64.whl`。

您可以查看 wheel 文件，发现它包含编译后的扩展名:

```py
$ unzip -l dist/ujson-*.whl
...
 Length      Date    Time    Name
---------  ---------- -----   ----
 105812  05-10-2020 19:47   ujson.cpython-38-darwin.so
 ...
```

这个例子显示了 macOS 的输出，`ujson.cpython-38-darwin.so`，这是一个共享对象(`.so`)文件，也称为动态库。

[*Remove ads*](/account/join/)

### Linux:构建`manylinux`轮子

作为一名软件包开发人员，您很少想为一个单一的 Linux 变种构建轮子。Linux wheels 需要一套专门的约定和工具，以便它们可以跨不同的 Linux 环境工作。

与 macOS 和 Windows 的 wheels 不同，在一个 Linux 版本上构建的 wheels 不能保证在另一个 Linux 版本上工作，即使是具有相同机器架构的版本。事实上，如果您在现成的 Linux 容器上构建一个轮子，那么如果您试图上传它，PyPI 甚至不会接受这个轮子！

如果您希望您的包可以在一系列 Linux 客户机上使用，那么您需要一个`manylinux`轮子。`manylinux`轮是一种特殊类型的平台轮，被大多数 Linux 变体所接受。它必须在一个特定的环境中构建，并且需要一个名为`auditwheel`的工具来重命名车轮文件，以表明它是一个`manylinux`车轮。

**注意**:即使你是从开发者的角度而不是从用户的角度来阅读本教程，在继续本节之前，请确保你已经阅读了关于[和`manylinux`滚轮标签](#the-manylinux-wheel-tag)的章节。

建立一个`manylinux`轮子可以让你瞄准更广泛的用户平台。 [PEP 513](https://www.python.org/dev/peps/pep-0513/) 指定了 CentOS 的一个特定(和古老)版本，并提供了一系列 Python 版本。在 CentOS 和 Ubuntu 或任何其他发行版之间的选择没有任何特殊的区别。要点是构建环境由一个普通的 Linux 操作系统和一组有限的外部共享库组成，这些库对于不同的 Linux 变体是通用的。

谢天谢地，你不必亲自去做。PyPA [提供了一组 Docker 图像](https://github.com/pypa/manylinux),只需点击几下鼠标就能提供这个环境:

*   **选项 1** 是从您的开发机器上运行`docker`，并使用 Docker 卷挂载您的项目，以便它可以在容器文件系统中被访问。
*   **选项 2** 是使用一个 [CI/CD](https://en.wikipedia.org/wiki/CI/CD) 解决方案，比如 CircleCI、GitHub Actions、Azure DevOps 或 Travis-CI，它们将提取你的项目并在一个动作(比如 push 或 tag)上运行构建。

为不同的`manylinux`口味提供了 Docker 图像:

| `manylinux`标签 | 体系结构 | Docker 图像 |
| --- | --- | --- |
| `manylinux1` | x86-64 | [quay.io/pypa/manylinux1_x86_64](https://quay.io/pypa/manylinux1_x86_64) |
| `manylinux1` | i686 | [quay.io/pypa/manylinux1_i686](https://quay.io/pypa/manylinux1_i686) |
| `manylinux2010` | x86-64 | [quay.io/pypa/manylinux2010_x86_64](https://quay.io/pypa/manylinux2010_x86_64) |
| `manylinux2010` | i686 | [quay.io/pypa/manylinux2010_i686](https://quay.io/pypa/manylinux2010_i686) |
| `manylinux2014` | x86-64 | [quay.io/pypa/manylinux2014_x86_64](https://quay.io/pypa/manylinux2014_x86_64) |
| `manylinux2014` | i686 | [quay.io/pypa/manylinux2014_i686](https://quay.io/pypa/manylinux2014_i686) |
| `manylinux2014` | aarh64 足球俱乐部 | [quay . io/pypa/manylinox 2014 _ aach 64](https://quay.io/pypa/manylinux2014_aarch64) |
| `manylinux2014` | ppc64le | [quay . io/pypa/manylinox 2014 _ ppc64 le](https://quay.io/pypa/manylinux2014_ppc64le) |
| `manylinux2014` | s390x | [quay.io/pypa/manylinux2014_s390x](https://quay.io/pypa/manylinux2014_s390x) |

为了开始，PyPA 还提供了一个示例库， [python-manylinux-demo](https://github.com/pypa/python-manylinux-demo) ，这是一个结合 [Travis-CI](https://travis-ci.org/) 构建`manylinux`轮子的演示项目。

虽然构建轮子作为远程托管 CI 解决方案的一部分很常见，但是您也可以在本地构建`manylinux`轮子。为此，你需要安装 [Docker](https://www.docker.com/get-started) 。Docker 桌面可用于 macOS、Windows 和 Linux。

首先，克隆演示项目:

```py
$ git clone -q git@github.com:pypa/python-manylinux-demo.git
$ cd python-manylinux-demo
```

接下来，分别为`manylinux1` Docker 映像和平台定义几个 shell 变量:

```py
$ DOCKER_IMAGE='quay.io/pypa/manylinux1_x86_64'
$ PLAT='manylinux1_x86_64'
```

`DOCKER_IMAGE`变量是 PyPA 为建造`manylinux`车轮维护的图像，托管在 [Quay.io](https://quay.io/) 。平台(`PLAT`)是提供给`auditwheel`的必要信息，让它知道应用什么平台标签。

现在，您可以提取 Docker 图像并在容器中运行 wheel-builder 脚本:

```py
$ docker pull "$DOCKER_IMAGE"
$ docker container run -t --rm \
      -e PLAT=$PLAT \
      -v "$(pwd)":/io \
      "$DOCKER_IMAGE" /io/travis/build-wheels.sh
```

这告诉 Docker 在`manylinux1_x86_64` Docker 容器中运行`build-wheels.sh` shell 脚本，将`PLAT`作为容器中可用的环境变量传递。由于您使用了`-v`(或`--volume`)来[绑定挂载一个卷](https://docs.docker.com/engine/reference/commandline/service_create/#add-bind-mounts-volumes-or-memory-filesystems)，容器中生成的轮子现在可以在您的主机上的`wheelhouse`目录中访问:

```py
$ ls -1 wheelhouse
python_manylinux_demo-1.0-cp27-cp27m-manylinux1_x86_64.whl
python_manylinux_demo-1.0-cp27-cp27mu-manylinux1_x86_64.whl
python_manylinux_demo-1.0-cp35-cp35m-manylinux1_x86_64.whl
python_manylinux_demo-1.0-cp36-cp36m-manylinux1_x86_64.whl
python_manylinux_demo-1.0-cp37-cp37m-manylinux1_x86_64.whl
python_manylinux_demo-1.0-cp38-cp38-manylinux1_x86_64.whl
```

在几个简短的命令中，您就有了一组用于 CPython 2.7 到 3.8 的`manylinux1`轮子。一种常见的做法也是迭代不同的架构。例如，您可以对`quay.io/pypa/manylinux1_i686` Docker 图像重复这个过程。这将建立针对 32 位(i686)架构的`manylinux1`轮子。

如果你想更深入地研究制造轮子，那么下一步最好是向最好的人学习。从 [Python Wheels](https://pythonwheels.com/) 页面开始，选择一个项目，导航到它的源代码(在 GitHub、GitLab 或 Bitbucket 之类的地方)，亲自看看它是如何构建轮子的。

Python Wheels 页面上的许多项目都是纯 Python 项目，并分发通用轮子。如果您正在寻找更复杂的情况，那么请留意使用扩展模块的包。这里有两个例子可以吊起你的胃口:

1.  [**`lxml`**](https://github.com/lxml/lxml/blob/master/tools/manylinux/build-wheels.sh) 使用从`manylinux1` Docker 容器中调用的独立构建脚本。
2.  [**`ultrajson`**](https://github.com/ultrajson/ultrajson/blob/master/.github/workflows/deploy-wheels.yml) 做同样的事情，并使用 GitHub 动作调用构建脚本。

如果你对建造`manylinux`车轮感兴趣，这两个项目都是著名的项目，提供了很好的学习范例。

[*Remove ads*](/account/join/)

### 捆绑共享库

另一个挑战是为依赖外部共享库的包构建轮子。`manylinux`图像包含一组预先筛选的库，如`libpthread.so.0`和`libc.so.6`。但是如果你依赖于列表之外的东西，比如 [ATLAS](http://math-atlas.sourceforge.net/) 或者 [GFortran](https://gcc.gnu.org/fortran/) 呢？

在这种情况下，有几种解决方案可供选择:

*   [**`auditwheel`**](https://github.com/pypa/auditwheel) 将外部库捆绑成一个已经构建好的轮子。
*   [**`delocate`**](https://github.com/matthew-brett/delocate) 在 macOS 上也是如此。

便利地，`auditwheel`出现在`manylinux` Docker 图像上。使用`auditwheel`和`delocate`只需要一个命令。只需告诉他们有关车轮文件的信息，剩下的工作由他们来完成:

```py
$ auditwheel repair <path-to-wheel.whl>  # For manylinux
$ delocate-wheel <path-to-wheel.whl>  # For macOS
```

这将通过项目的`setup.py`检测所需的外部库，并将它们捆绑到轮子中，就像它们是项目的一部分一样。

利用`auditwheel`和`delocate`的项目的一个例子是 [`pycld3`](https://github.com/bsolomon1124/pycld3) ，它为紧凑语言检测器 v3 (CLD3)提供 Python 绑定。

`pycld3`包依赖于 [`libprotobuf`](https://github.com/protocolbuffers/protobuf) ，不是一般安装的库。如果你偷看一个 [`pycld3` macOS 轮](https://pypi.org/project/pycld3/#files)的内部，那么你会看到`libprotobuf.22.dylib`包含在那里。这是一个**动态链接的共享库**，它被捆绑到轮子中:

```py
$ unzip -l pycld3-0.20-cp38-cp38-macosx_10_15_x86_64.whl
...
 51  04-10-2020 11:46   cld3/__init__.py
 939984  04-10-2020 07:50   cld3/_cld3.cpython-38-darwin.so
 2375836  04-10-2020 07:50   cld3/.dylibs/libprotobuf.22.dylib ---------                     -------
 3339279                     8 files
```

车轮预装了`libprotobuf`。一个`.dylib`类似于一个 Unix `.so`文件或者 Windows `.dll`文件，但是我承认我不知道除此之外的本质区别。

`auditwheel`和`delocate`知道包括`libprotobuf`是因为 [`setup.py`通过`libraries`的论证告诉他们](https://docs.python.org/3/distutils/setupscript.html#describing-extension-modules):

```py
setup(
    # ...
    libraries=["protobuf"],
    # ...
)
```

这意味着`auditwheel`和`delocate`为用户省去了安装`protobuf`的麻烦，只要他们从一个平台和 Python 组合中安装，这个平台和 Python 组合有一个匹配的轮子。

如果你正在发布一个像这样有外部依赖的包，那么你可以帮你的用户一个忙，使用`auditwheel`或者`delocate`来省去他们自己安装依赖的额外步骤。

### 在持续集成中构建车轮

在本地机器上构建轮子的另一种方法是在项目的 [CI 管道](https://realpython.com/python-continuous-integration/)中自动构建轮子。

有无数的 CI 解决方案与主要的代码托管服务相集成。其中有 [Appveyor](https://www.appveyor.com/) 、 [Azure DevOps](https://azure.microsoft.com/en-us/services/devops/) 、 [BitBucket Pipelines](https://bitbucket.org/product/features/pipelines) 、 [Circle CI](https://circleci.com/) 、 [GitLab](https://about.gitlab.com/stages-devops-lifecycle/continuous-integration/) 、 [GitHub Actions](https://github.com/features/actions) 、 [Jenkins](https://www.jenkins.io/) 和 [Travis CI](https://travis-ci.org/) 等等。

本教程的目的不是判断哪种 CI 服务最适合构建车轮，并且考虑到 CI 支持的发展速度，任何列出哪些 CI 服务支持哪些容器的列表都会很快过时。然而，这一节可以帮助你开始。

如果你正在开发一个纯 Python 包，那么`bdist_wheel`步骤是一个幸福的单行程序:你在哪个容器操作系统和平台上构建轮子基本上无关紧要。事实上，所有主要的 CI 服务都应该通过在项目中的一个特殊的 [YAML 文件](https://realpython.com/python-yaml/)中定义步骤，使您能够以一种简洁的方式做到这一点。

例如，下面是您可以在 [GitHub 操作](https://help.github.com/en/actions/language-and-framework-guides/using-python-with-github-actions)中使用的语法:

```py
 1name:  Python wheels 2on: 3  release: 4  types: 5  -  created 6jobs: 7  wheels: 8  runs-on:  ubuntu-latest 9  steps: 10  -  uses:  actions/checkout@v2 11  -  name:  Set up Python 3.x 12  uses:  actions/setup-python@v2 13  with: 14  python-version:  '3.x' 15  -  name:  Install dependencies 16  run:  python -m pip install --upgrade setuptools wheel 17  -  name:  Build wheels 18  run:  python setup.py bdist_wheel 19  -  uses:  actions/upload-artifact@v2 20  with: 21  name:  dist 22  path:  dist
```

在此配置文件中，您使用以下步骤构建一个轮子:

1.  在**第 8 行**中，您指定作业应该在 Ubuntu 机器上运行。
2.  在**第 10 行**中，您使用 [`checkout`](https://github.com/actions/checkout) 动作来设置您的项目存储库。
3.  在**第 14 行**中，您告诉 CI 运行程序使用 Python 3 的最新稳定版本。
4.  在**第 21 行**中，您请求生成的轮子作为工件可用，一旦作业完成，您可以从 UI 下载。

然而，如果您有一个复杂的项目(可能是一个带有 C 扩展或 Cython 代码的项目),并且您正在努力创建一个 CI/CD 管道来自动构建轮子，那么可能会涉及到额外的步骤。这里有几个项目，你可以通过例子来学习:

*   [T2`yarl`](https://github.com/aio-libs/yarl)
*   [T2`msgpack`](https://github.com/msgpack/msgpack-python)
*   [T2`markupsafe`](https://github.com/pallets/markupsafe)
*   [T2`cryptography`](https://github.com/pyca/cryptography)

许多项目推出自己的配置项配置。然而，已经出现了一些解决方案来减少在配置文件中指定的构建轮子的代码量。您可以直接在 CI 服务器上使用 [cibuildwheel](https://github.com/joerick/cibuildwheel) 工具来减少构建多个平台轮子所需的代码和配置行。还有 [multibuild](https://github.com/matthew-brett/multibuild) ，它提供了一组 shell 脚本来帮助在 Travis CI 和 AppVeyor 上构建轮子。

[*Remove ads*](/account/join/)

### 确保你的车轮旋转正确

构建结构正确的车轮可能是一项精细的操作。例如，如果你的 Python 包使用了一个 [`src`布局](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure)，而你忘记了[在`setup.py`](https://github.com/jwodder/check-wheel-contents#w005--wheel-contains-common-toplevel-name-in-library) 中正确地指定它，那么产生的轮子可能在错误的位置包含了一个目录。

在`bdist_wheel`之后可以使用的一个检查是 [`check-wheel-contents`](https://github.com/jwodder/check-wheel-contents) 工具。它会查找常见问题，例如包目录结构异常或存在重复文件:

```py
$ check-wheel-contents dist/*.whl
dist/ujson-2.0.3-cp38-cp38-macosx_10_15_x86_64.whl: OK
```

在这种情况下，`check-wheel-contents`表示使用`ujson`滚轮的一切都检查完毕。如果没有，`stdout`将会显示一个可能问题的概要，就像`flake8`中的过磅信息一样。

另一种确认你建造的轮子是否有合适的材料的方法是使用 [TestPyPI](https://packaging.python.org/guides/using-testpypi/) 。首先，您可以在那里上传软件包:

```py
$ python -m twine upload \
      --repository-url https://test.pypi.org/legacy/ \
      dist/*
```

然后，您可以下载相同的包进行测试，就像它是真实的一样:

```py
$ python -m pip install \
      --index-url https://test.pypi.org/simple/ \
      <pkg-name>
```

这允许你通过上传然后下载你自己的项目来测试你的轮子。

### 上传 Python 轮子到 PyPI

现在是[上传你的 Python 包](https://realpython.com/pypi-publish-python-package/)的时候了。由于默认情况下`sdist`和轮子都放在`dist/`目录中，所以您可以使用 [`twine`](https://pypi.org/project/twine/) 工具来上传它们，这是一个用于将包发布到 PyPI 的实用程序:

```py
$ python -m pip install -U twine
$ python -m twine upload dist/*
```

由于默认情况下`sdist`和`bdist_wheel`都输出到`dist/`，您可以放心地告诉`twine`使用 shell 通配符(`dist/*`)上传`dist/`下的所有内容。

## 结论

理解轮子在 Python 生态系统中扮演的关键角色可以让 Python 包的用户和开发者的生活更加轻松。此外，提高您在轮子方面的 Python 素养将有助于您更好地理解当您安装一个包时会发生什么，以及在越来越少的情况下，操作何时出错。

**在本教程中，您学习了:**

*   什么是车轮，它们与**源分布**相比如何
*   如何使用轮子来控制**包装安装**过程
*   **万向**、**纯蟒**、**平台**车轮有什么区别
*   如何**为你自己的 Python 包创建和分发**轮子

现在，您已经从用户和开发人员的角度对轮子有了很好的理解。您已经准备好打造自己的车轮，让项目的安装过程变得快速、方便和稳定。

请参阅下一节的附加阅读材料，深入了解快速扩张的车轮生态系统。

[*Remove ads*](/account/join/)

## 资源

Python Wheels 页面致力于跟踪 PyPI 上下载最多的 360 个包中对 Wheels 的支持。在本教程中，采用率是相当可观的，360 个中有 331 个，大约 91%。

有许多 Python 增强提案(pep)帮助规范和发展了 wheel 格式:

*   [PEP 425 -已构建发行版的兼容性标签](https://www.python.org/dev/peps/pep-0425/)
*   [PEP 427 -车轮二进制包格式 1.0](https://www.python.org/dev/peps/pep-0427/)
*   [PEP 491 -车轮二进制包格式 1.9](https://www.python.org/dev/peps/pep-0491/)
*   [PEP 513 -一个用于可移植 Linux 构建发行版的平台标签](https://www.python.org/dev/peps/pep-0513/)
*   [PEP 571-`manylinux2010`平台标签](https://www.python.org/dev/peps/pep-0571/)
*   [PEP 599-`manylinux2014`平台标签](https://www.python.org/dev/peps/pep-0599/)

以下是本教程中提到的各种车轮包装工具的列表:

*   [T2`pypa/wheel`](https://github.com/pypa/wheel)
*   [T2`pypa/auditwheel`](https://github.com/pypa/auditwheel)
*   [T2`pypa/manylinux`](https://github.com/pypa/manylinux)
*   [`pypa/python` -manylinux-demo](https://github.com/pypa/python-manylinux-demo)
*   [T2`jwodder/check-wheel-contents`](https://github.com/jwodder/check-wheel-contents)
*   [T2`matthew-brett/delocate`](https://github.com/matthew-brett/delocate)
*   [T2`matthew-brett/multibuild`](https://github.com/matthew-brett/multibuild)
*   [T2`joerick/cibuildwheel`](https://github.com/joerick/cibuildwheel)

Python 文档中有几篇文章介绍了 wheels 和源代码发行版:

*   [生成发行档案](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives)
*   [创建源分布](https://docs.python.org/3/distutils/sourcedist.html)

最后，这里有一些来自 PyPA 的更有用的链接:

*   [打包您的项目](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project)
*   [Python 打包概述](https://packaging.python.org/overview/)**********