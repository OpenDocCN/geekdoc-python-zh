# 如何使用 Pip(安装、更新、卸载软件包的简单指南)

> 原文：<https://www.pythoncentral.io/how-to-use-pip-simple-guide-to-install-update-uninstall-packages/>

Python 包包含大量代码，你可以在不同的程序中重复使用。它们消除了编写已经编写好的代码的需要，因此对于程序员和其他广泛使用 Python 的职业来说非常方便，比如机器学习工程师 T2 和数据分析师。 可以将 matplotlib、numpy 等包导入到项目中。每个包都附带了几个可以在代码中使用的函数。然而，在使用一个包之前，您需要使用 pip 来安装它，pip 是 Python 的默认包管理器。 你还需要了解如何更新软件包，卸载不需要的软件包。这本简明指南涵盖了你需要知道的一切。

## **Pip 操作说明**

### **安装 pip**

如果你使用 python.org 上的安装程序在电脑上安装 Python，pip 会和 Python 一起安装。 如果你在虚拟环境中工作或者使用未被再发行商修改的 Python 版本，Pip 也会被自动安装。再发行商通常会从 Python 安装中删除 ensurepip 模块。 如果使用的是 Python 的修改版，可以使用 get-pip.py 或 ensurepip 安装 pip。这在 Windows 和 macOS 机器上都可以工作。然而，用 python.org 安装程序建立一个新的 Python 环境通常更容易。

### **使用不同版本的画中画**

如果你已经在电脑上安装了 Python2 和 Python3，那么除了使用 pip 命令之外，你应该还能使用 pip2 和 pip3。 Pip 可以设置为在一台机器上运行 Python2 或 Python3。两者都不行。例如，如果您将 pip 设置为与 Python3 一起工作，那么您用它安装的包将不能在 Python2 上工作。 pip2 可用于 Python2 中的包管理，pip3 可用于 Python3 中的包管理。

### **检查已安装软件包的详细信息**

如果您不确定是否安装了 pip，您可以使用 pip show 命令来查找任何已安装软件包的详细信息。pip 显示的语法是:

```py
pip show <package-name>
```

如果机器上安装了 pip，运行代码来查找 pip 的许可证和依赖项应该会得到如下类似的结果:

```py
pip show pip
Name: pip
Version: 18.1
Summary: The PyPA recommended tool for installing Python packages.
Home-page: 

|   1   
 |  |

		(https://pip.pypa.io/)
Author: The pip developers
Author-email: pypa-dev@groups.google.com
License: MIT
Location: /usr/local/lib/python2.7/site-packages
Requires:
Required-by:

请记住，上面的代码显示了 pip 命令在…/python2.7/site-中安装包。但是，根据环境的不同，pip 也可以在 Python3 上使用。

### **列出所有已安装的软件包**

Pip 还能让你生成一个在你的计算机上为 Python 安装的包的列表。 在解释器上输入“pip list”应该会得到类似的输出:

```
pip list
Package    Version
---------- -------
future     0.16.0
pip        18.1
setuptools 39.2.0
six        1.11.0
wheel      0.31.1
```py

您可以改变命令的输出格式，只输出最新的包、过期的包或没有依赖关系的包。你也可以使用 pip 冻结命令。该命令不会输出 pip 和包管理包，如车轮和设置工具。 

```
pip freeze
future==0.16.0
six==1.11.0
```py

**安装包** 用 pip 安装包很容易。如果您想要安装的软件包已经在 PyPI 中注册，您需要做的就是指定软件包的名称，就像这样:

```
pip install <package-name>
```py

运行这个命令将会安装最新版本的软件包。您也可以使用 pip install 一次安装多个软件包。

```
pip install <package_name_1> <package_name_2> <package_name_3> ...
```py

 Pip 还允许您使用语法安装特定版本的软件包:

```
pip install <package-name>==<version>
```py

### **从 GitHub/本地驱动器安装**

有时，PyPI 上一些包的新版本没有及时更新。在这种情况下，开发人员从本地目录或 GitHub 库安装它。 使用 pip install 从本地目录安装软件包，指定包含 setup.py 文件的路径，如下:

```
pip install path/to/dir
```py

您可以从以下网站安装软件包。whl 和。zip 文件，只要它们包含 setup.py 文件。

```
pip install path/to/zipfile.zip
```py

要从 Git 存储库中安装一个包，可以使用下面的命令:

```
pip install git+<repository-url>
```py

要指定标签或分支并安装软件包的特定版本，请在存储库的 URL 末尾加上一个“@”，然后输入软件包的标签。 记住，要用这种方法从 GitHub 安装软件包，你需要在你的系统上安装 git。要安装 git，您也可以从 GitHub 下载要安装的包的 zip 文件，并以这种方式安装。

### **更新包**

您可以使用- upgrade 选项和 pip install 将软件包更新到最新版本:

```
pip install --upgrade <package-name>
#or
pip install -U <package-name>
```py

### **更新画中画**

您可以毫不费力地使用 pip 更新 pip。当 pip 的更新可用时，如果您运行 pip 命令，您将看到一条消息，说明“您使用的是 pip 版本 xy.a，但是版本 xy.b 可用。” 您可以运行“pip 安装-升级 pip”来安装和使用新版 pip。 要使用该命令更新 pip2 或 pip3，只需用 pip 版本替换第一个 pip。

```
pip3 install --upgrade pip
#or
pip2 install --upgrade pip
```py

### **卸载软件包**

用 pip 卸载软件包就像用它安装软件包一样简单。您可以使用以下命令卸载软件包:

```
pip uninstall <package-name>
```py

除了允许您同时安装多个软件包，pip 还允许您一次卸载多个软件包。您必须使用命令:

```
pip uninstall <package-name1> <package-name2> <package-name3> ...
```py

当您使用 pip 删除包时，会要求您确认是否要删除文件。如果您不希望提示您要卸载的软件包，您可以在“pip 卸载”后使用- yes 或-y 选项

### **检查依赖性**

pip check 命令允许您检查和验证已安装的软件包是否具有兼容的依赖关系。 当您在一台所有包都具有所需依赖关系的机器上运行该命令时，屏幕上会出现一个“没有发现不符合要求”的提示。 另一方面，如果一个依赖包没有安装或者有版本不匹配，pip 检查会输出这个包和依赖。如果您看到这样的消息，请使用 pip install -U. 更新相应的软件包

了解如何使用 pytest 测试 python 应用程序，点击这里查看文章。** 
```