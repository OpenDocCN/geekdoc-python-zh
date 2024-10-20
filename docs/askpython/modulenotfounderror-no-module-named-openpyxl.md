# ModuleNotFoundError:没有名为 openpyxl 的模块

> 原文：<https://www.askpython.com/python/modulenotfounderror-no-module-named-openpyxl>

您是否陷入了“ModuleNotFoundError:没有名为 openpyxl 的模块”错误？初学 Python 的人在开始使用 [openpyxl 库](https://www.askpython.com/python-modules/openpyxl-in-python)时经常会遇到困难。这是一个全行业的问题，会出现在许多从 Python 开始的程序员的屏幕上。

这通常是由于 Python 和 Openpyxl 之间的版本错误或不正确的安装造成的。找不到正确的解决方案可能是一个令人头疼的问题，这就是为什么在本文中，我们将讨论编码人员在使用 openpyxl 时遇到的最常见的错误，并针对每一个错误提出解决方案。

## 什么是 openpyxl？

缺少读取 Open XML 格式的 Python 库产生了对 openpyxl 的需求，openpyxl 是当前读取和写入 excel 格式文件的行业标准 Python 库。

支持的格式有:xlsx/xlsm/xltx/xltm。

大多数情况下，openpyxl 遇到的错误是由于安装不正确造成的。这些错误可以通过简单的重新安装来解决。我们来看看安装 openpyxl 的正确方法。

## 通过包管理器安装 openpyxl

**通过使用画中画**

安装任何 Python 库最简单的方法就是使用 [pip](https://www.askpython.com/python-modules/python-pip) 。Pip 是一个用 Python 编写的系统软件，它通过一个开源库来安装和管理你的 Python 库。通过 pip 安装需要了解您的 Python 版本。

**如果您使用 Python2** ，安装任何库的语法是:

```py
pip install 'package_name'

```

只使用“pip”就可以安装 Python2 版本的任何 Python 包。

要为 Python3 安装它，请编写:

```py
pip3 install 'x'

```

注意:在 pip 的较新版本中，主要是 3.0 以上的版本，当只键入“pip”时，pip 默认安装 Python3 的包。在最新版本的 Python 中可以发现这种异常，如果您遇到这种情况，建议您降级您的 Python 版本。

如果您的系统中有**多个 Python 版本**，并且您不确定 vanilla pip3 是否会获取正确的版本，您可以使用以下语法:

```py
python3 -m pip install --user xlsxwriter

```

**通过使用 Conda**

许多编码人员更喜欢使用 [Conda 而不是 pip](https://www.askpython.com/python/conda-vs-pip) ，因为它的虚拟环境特性。如果要使用 Conda 安装 openpyxl，请键入以下语法:

```py
conda install -c anaconda openpyxl

```

**或**

```py
conda install openpyxl

```

第二种语法主要针对最新的 Conda 版本(Conda 4.7.6 或更高版本)

## 根据您的操作系统安装

让我们看看基于您使用的操作系统的不同安装方式！

### 1.Windows 操作系统

如果您使用的是 windows，则需要以下软件包来读写 excel 文件。要安装它们，请键入以下语法:

```py
pip install openpyxl
pip install --user xlsxwriter
pip install xlrd==1.2.0

```

### 2.人的本质

Linux 发行版更适合 Python，但是由于系统错误或安装不当，您可能仍然会遇到一些错误。在 Ubuntu 中安装 openpyxl 的安全而正确的方法是执行以下操作。

在您的终端中键入以下语法:

**对于 Python2:**

```py
sudo apt-get install python-openpyxl

```

**对于 Python3:**

```py
sudo apt-get install python3-openpyxl

```

上面的案例类似于上面的例子。很多时候，不正确的安装是由于将 Python2 的 openpyxl 安装到 Python3 的系统中造成的。

了解您的 python 版本可以解决大多数问题，如果问题仍然存在，那么降级您的 Python 版本并重新安装软件包是另一个选择。

## 脚本路径错误

很多时候，即使安装正确，openpyxl 也可能抛出“modulenotfounderror”。使用什么安装管理器并不重要，因为软件包可以正确安装，但是软件包管理器会将它安装在其他目录中。这主要有几个原因:

*   您的 Python 脚本保存在不同的目录中，您的包管理器将它们安装在不同的目录中。
*   最近的更新可能已经更改了目录的名称或路径
*   手动安装可能会创建多个脚本文件夹
*   您的系统无法识别正确的脚本目录。

要解决此问题，请在终端中使用以下语法:

```py
import sys
sys.append(full path to the site-package directory)

```

注意:上面的代码重定向 Python，从给定的路径目录中搜索导入包。这不是一个永久的解决办法，而是一个转变的方法。

彻底根除这个问题是一个漫长的过程，因为它需要识别所有的脚本目录，并将它们添加到“path”(安装软件包的脚本目录的路径)。为了避免这样的错误，我们识别并记住我们的脚本安装在哪里是很重要的。知道哪里发生了位移可以解决一半的问题，另一半可以通过安装与您的 python 版本兼容的包来解决。

## 结论

在本文中，我们了解了人们在使用 openpyxl 时遇到的错误，以及解决这些错误的不同方法。这些方法不仅适用于 openpyxl，还适用于许多显示“ [filenotfounderror](https://www.askpython.com/python/examples/python-filenotfounderror) ”的 python 库包。大多数情况下，错误是由于不兼容的版本、错误的安装或安装在错误的目录中而发生的。