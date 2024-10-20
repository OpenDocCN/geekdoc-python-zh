# python 101:pip——easy _ install 的替代品

> 原文：<https://www.blog.pythonlibrary.org/2012/07/16/python-101-pip-a-replacement-for-easy_install/>

[Pip](http://www.pip-installer.org/en/latest/) 安装 Python 或者说 Pip 是一个安装和管理 Python 包的工具，很多都在 [Python 包索引](http://pypi.python.org/pypi) (PyPI)上。它是 easy_install 的替代品。在本文中，我们将花一点时间试用 pip，看看它是如何工作的，以及它如何帮助我们开发 Python。

### 装置

你需要出去找[分发](http://pypi.python.org/pypi/distribute)或[设置工具](http://pypi.python.org/pypi/setuptools)来让 pip 工作。如果您使用的是 Python 3，那么 distribute 是您唯一的选择，因为在撰写本文时 setuptools 还不支持它。在 pip 的网站上有一个安装程序，你可以使用，名为 [get-pip.py](https://raw.github.com/pypa/pip/master/contrib/get-pip.py) ，或者你可以直接去 [PyPI](http://pypi.python.org/pypi/pip/) 下载它作为源代码。

希望你已经知道了这一点，但是要从源代码安装大多数模块，你必须压缩它，然后打开一个终端或命令行窗口。然后将目录(cd)切换到解压后的文件夹，运行“ *python setup.py install* ”(去掉引号)。请注意，您可能需要提升权限来安装它(即 root 或管理员)。pip 网站建议在 virtualenv 中使用 pip，因为它是自动安装的，并且“不需要 root 访问权限或修改您的系统 Python 安装”。我会让你自己决定。

### pip 使用

pip 最常见的用法是安装、升级或卸载软件包。这在 pip 网站上都有涉及，但我们还是在这里看一下。因为我们一直在提到 virtualenv，所以让我们试着用 pip 安装它:

```py

pip install virtualenv

```

如果您在终端中运行上面的命令，您应该得到类似如下的输出:

```py

Downloading/unpacking virtualenv
  Downloading virtualenv-1.7.2.tar.gz (2.2Mb): 2.2Mb downloaded
  Running setup.py egg_info for package virtualenv
    warning: no previously-included files matching '*' found under directory 'do
cs\_templates'
    warning: no previously-included files matching '*' found under directory 'do
cs\_build'
Installing collected packages: virtualenv
  Running setup.py install for virtualenv
    warning: no previously-included files matching '*' found under directory 'do
cs\_templates'
    warning: no previously-included files matching '*' found under directory 'do
cs\_build'
    Installing virtualenv-script.py script to C:\Python26\Scripts
    Installing virtualenv.exe script to C:\Python26\Scripts
    Installing virtualenv.exe.manifest script to C:\Python26\Scripts
    Installing virtualenv-2.6-script.py script to C:\Python26\Scripts
    Installing virtualenv-2.6.exe script to C:\Python26\Scripts
    Installing virtualenv-2.6.exe.manifest script to C:\Python26\Scripts
Successfully installed virtualenv
Cleaning up...

```

这似乎很有效。请注意，pip 在开始安装之前下载软件包，这是 easy_install 所不做的(关于其他区别，请参见这个[比较](http://www.pip-installer.org/en/latest/other-tools.html#pip-compared-to-easy-install))。比如说这篇文章写完之后 virtualenv 出了一个新版本，你想升级？皮普掩护你！

```py

pip install --upgrade virtualenv

```

那不是很容易吗？另一方面，如果你喜欢总是在沙盒安全之外的危险边缘工作，那该怎么办？卸载沙盒和安装沙盒一样简单:

```py

pip uninstall virtualenv

```

没错。就这么简单。真的！

Pip 还可以从文件路径、URL 和版本控制系统(如 Subversion、Git、Mercurial 和 Bazaar)安装。更多信息参见[文档](www.pip-installer.org/en/latest/usage.html)。

### 其他画中画功能

Pip 还使您能够创建配置文件，这些文件可以保存您通常以类似 INI 文件的格式使用的所有命令行选项。你可以在这里阅读[。遗憾的是，看起来 pip 只在一个特定的位置寻找配置，所以您实际上不能将不同的配置传递给 pip。](http://www.pip-installer.org/en/latest/configuration.html#examples)

我想强调的另一个特性是它的**需求文件**概念。这些文件只是要安装的软件包列表。它们提供了安装软件包和所有依赖项的方法，包括依赖项的特定版本。您甚至可以添加可选库和支持工具的列表。如果你需要知道你当前的安装程序已经安装了什么，你可以像这样把它们“冻结”到一个需求文件中:

```py

pip freeze > myrequirements.txt

```

这在 virtualenv 中非常有用，因为您可能会在主 Python 套件中安装许多与当前项目没有任何关系的包。这也是为什么与 virtualenv 合作是个好主意的另一个原因。

### 包扎

现在，您已经了解了足够的知识，可以开始使用 pip 了。这是一个非常方便的工具，可以添加到您的工具包中，并使安装和管理软件包变得轻而易举。玩得开心！