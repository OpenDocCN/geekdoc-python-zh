# 如何安装 matplotlib python

> 原文：<https://pythonguides.com/how-to-install-matplotlib-python/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本 [Python 教程](https://pythonguides.com)中，我们将讨论**如何安装 matplotlib python** ，以及在 Python 中使用数据可视化所需的所有依赖项，我们还将涵盖以下主题:

*   如何安装 matplotlib python
*   如何安装 matplotlib python ubuntu
*   如何安装 matplotlib python 窗口
*   如何安装 matplotlib python mac
*   如何安装 matplotlib python conda
*   如何安装 matplotlib python pip
*   如何安装 matplotlib python venv
*   如何安装 matplotlib python3
*   如何安装 matplotlib python2

目录

[](#)

*   [如何安装 matplotlib python](#How_to_install_matplotlib_python "How to install matplotlib python")
*   [如何安装 matplotlib python ubuntu](#How_to_install_matplotlib_python_ubuntu "How to install matplotlib python ubuntu")
*   [如何安装 matplotlib python windows](#How_to_install_matplotlib_python_windows "How to install matplotlib python windows")
*   [如何安装 matplotlib python mac](#How_to_install_matplotlib_python_mac "How to install matplotlib python mac")
*   [如何安装 matplotlib python conda](#How_to_install_matplotlib_python_conda "How to install matplotlib python conda")
*   [如何安装 matplotlib python3](#How_to_install_matplotlib_python3 "How to install matplotlib python3")
*   [如何安装 matplotlib python2](#How_to_install_matplotlib_python2 "How to install matplotlib python2")

## 如何安装 matplotlib python

可以安装 [matplotlib 库](https://pythonguides.com/what-is-matplotlib/)在所有常用的三大操作系统中用 python 使用:

*   Linux (Ubuntu，redhat 等。,)
*   Windows 操作系统
*   马科斯

您可以在这些操作系统中安装 matplotlib，方法是使用 `pip` 命令(使用 python 包管理器)来安装已发布的**轮包**，或者从 python 和 matplotlib 的其他安装中为 matplotlib 创建一个单独的**虚拟环境**，或者使用另一个环境，如 `anaconda` ，它提供 `conda` 作为包管理器来安装包。

**注—**

*   `Wheel Package` 是 python 的内置包格式，具有。whl 文件扩展名。它包含与 python 安装包及其元数据相关的所有文件。
*   **虚拟环境**在 python 中只是一个命名的开发目录，包含所有必需的依赖项和安装在该目录中的包。
*   Anaconda 是 python 的一个发行版，它提供了一个基于科学研究开发 python 项目的环境。

Read: [modulenotfounderror:没有名为“matplotlib”的模块](https://pythonguides.com/no-module-named-matplotlib/)

## 如何安装 matplotlib python ubuntu

#### 如何在 Linux (Ubuntu)中使用 pip 安装 matplotlib python

你可以在任何一个 `Linux` 发行版中安装 matplotlib for python，包括 `Ubuntu` ，通过使用 python 包管理器，它提供了 `pip` 命令来安装任何为 python 发布的轮子包。首先，确保您已经在系统中安装了 python 和 pip。如果你没有安装 pip，首先你必须安装它，然后使用 pip 安装 matplotlib。在终端中执行以下命令:

```py
python -m pip install -U pip
python -m pip install -U matplotlib [--prefer-binary]
```

在上述命令中，

*   第一个命令更新 pip python 包管理器。
*   在第二个命令中，**–prefere-binary**是可选的，如果不包括–prefere-binary 选项的命令无法安装或更新 matplotlib 包。然后添加这个选项，它会根据你安装的操作系统和 python 的预编译轮选择最新的版本。

您可以通过在终端中执行以下命令来检查 matplotlib 是否已成功安装在您的系统上:

```py
import matplotlib
matplotlib.__version__
```

#### 如何安装 matplotlib python Linux 包管理器

在 Linux 中，python 预装在 OS 发行版中，如果您使用的是该版本，那么您可以通过使用 Linux 包管理器来安装 matplotlib，不同的发行版有不同的包管理器:

*   对于 Debian / Ubuntu，您可以使用以下命令:

```py
sudo apt-get install python3-matplotlib
```

*   对于 Red Hat，您可以使用以下命令:

```py
sudo yum install python3-matplotlib
```

*   对于 Fedora，您可以使用以下命令:

```py
sudo dnf install python3-matplotlib
```

*   对于 Arch，您可以使用以下命令:

```py
sudo pacman -S python-matplotlib
```

#### 如何在 Linux 上安装 matplotlib python venv

可以在 Linux 的虚拟开发环境中安装 matplotlib，通过使用 Python 的虚拟环境 `venv` 创建一个虚拟环境。下面给出了执行此操作的步骤:

*   创建虚拟环境:

```py
python -m venv <directory_path>
```

上面的命令在位置 *<目录 _ 路径>* 创建了一个虚拟环境(专用目录)。

*   激活创建的环境:

```py
source <directory_path>/bin/activate
```

上面的命令激活了开发环境。无论何时开始使用 matplotlib，都必须首先在 shell 中激活开发环境。

*   从位于[https://github.com/matplotlib/matplotlib.git](https://github.com/matplotlib/matplotlib.git)的 git 获取 matplotlib 的最新版本。以下命令将 matplotlib 的最新源代码检索到当前工作目录中:

```py
git clone https://github.com/matplotlib/matplotlib.git
```

*   现在，在可编辑(开发)模式下安装 matplotlib，因为开发模式允许 python 从您的开发环境源目录导入 matplotlib，即从 git 源导入，这允许您在源发生任何更改后导入最新版本的 matplotlib，而无需重新安装。以下命令允许您这样做:

```py
python -m pip install -ve
```

现在，您可以导入 matplotlib 包并在您的开发环境中使用它。

阅读:[在 Python 中切片字符串](https://pythonguides.com/slicing-string-in-python/)

## 如何安装 matplotlib python windows

#### 如何在 Windows 中安装 matplotlib python pip

您可以在 `Windows` OS 中安装 matplotlib for python，方法是使用 python 包管理器，它提供了 `pip` 命令来安装为 python 发布的任何 wheel 包。

首先，确保您已经在系统中安装了 python 和 pip。如果你没有安装 pip，首先你必须安装它，然后使用 pip 安装 matplotlib。在 cmd 中执行以下命令:

```py
python -m pip install -U pip        # Update the pip package manager
python -m pip install -U matplotlib [--prefer-binary]
```

上面的命令与我们在上面的主题中在 Linux 发行版中所做的一样。

您可以通过在 cmd 中执行以下命令来检查 matplotlib 是否已成功安装在您的系统上:

```py
import matplotlib
matplotlib.__version__
```

#### 如何在 Windows 中安装 matplotlib python venv

您可以在 python 中创建一个**虚拟环境**,并按照给定的步骤对其进行配置，以便在 Windows 中开发 matplotlib:

*   创建虚拟环境:

```py
python -m venv <directory_path>
```

上面的命令在位置 *<目录 _ 路径>* 创建了一个虚拟环境(专用目录)。

*   激活创建的环境:

```py
source <directory_path>/bin/activate.bat

# Note that, this command was different for Linux distribution
```

上面的命令激活了开发环境。无论何时开始使用 matplotlib，都必须首先在 shell 中激活开发环境。

*   所有步骤都与 Linux 发行版相同:

```py
# Retrieve the latest version of matplotlib from the git source

git clone https://github.com/matplotlib/matplotlib.git

# Install matplotlib in the editable mode

python -m pip install -ve
```

前面的主题已经讨论了上述命令。

阅读: [Python NumPy Random](https://pythonguides.com/python-numpy-random/)

## 如何安装 matplotlib python mac

#### 如何在 macOS 中安装 matplotlib python pip

你可以在一个 `macOS` 中安装 python 的 matplotlib，通过使用 python 包管理器，它提供了 `pip` 命令来安装任何为 python 发布的轮子包。首先，确保您已经在系统中安装了 python 和 pip。如果你没有安装 pip，首先你必须安装它，然后使用 pip 安装 matplotlib。在 cmd 中执行以下命令:

```py
python -m pip install -U pip        # Update the pip package manager
python -m pip install -U matplotlib [--prefer-binary]
```

上面的命令也与我们为 Linux 发行版所做的和讨论的一样。

您可以通过在终端中执行以下命令来检查 matplotlib 是否已成功安装在您的系统上:

```py
import matplotlib
matplotlib.__version__
```

#### 如何在 macOS 中安装 matplotlib python venv

在 macOS 中为 matplotlib python 创建专用开发环境的步骤与我们为 Linux 发行版所做和讨论的步骤相同。

```py
# You can see that all the steps are same, as done for the Linux

# Creating a development environment
python -m venv <directory_path>

# Activate the created environment
source <directory_path>/bin/activate

# Retrieve the latest version of matplotlib from the git source
git clone https://github.com/matplotlib/matplotlib.git

# Install matplotlib in the editable mode
python -m pip install -ve
```

阅读: [Python Tkinter 选项菜单](https://pythonguides.com/python-tkinter-optionmenu/)

## 如何安装 matplotlib python conda

Matplotlib 也是一些主要 Python 发行版的一部分，比如 anaconda。因此，您可以在 python 的这个发行版中安装 matplotlib，它为 matplotlib 提供了环境。Anaconda 适用于所有三大操作系统，Linux、Windows、macOS。您可以使用 anaconda 提供的包管理器 conda 来安装 matplotlib。您必须在系统中安装了 anaconda，然后才能在 cmd /conda 提示符/终端中执行以下命令:

```py
conda install matplotlib
```

上面的命令将从 **anaconda 主通道**在 anaconda 开发环境中安装 matplotlib。

您也可以通过执行下面的命令从 **anaconda 社区通道**安装 matplotlib。

```py
conda install -c conda-forge matplotlib
```

## 如何安装 matplotlib python3

如果你使用 python3，那么使用 `pip3` 代替 pip 来安装 matplotlib。所有安装过程与上述主题中给出的相同，只是使用 `pip3` 代替。

## 如何安装 matplotlib python2

如果你使用 python2，那么使用 `pip` 来安装 matplotlib。所有安装过程与上述主题中给出的相同。

你可能也喜欢读下面的文章。

*   [如何安装 Django](https://pythonguides.com/how-to-install-django/)
*   [什么是 Python Django](https://pythonguides.com/what-is-python-django/)
*   [Python 绘制多条线](https://pythonguides.com/python-plot-multiple-lines/)
*   [Matplotlib 绘制一条线](https://pythonguides.com/matplotlib-plot-a-line/)
*   [什么是 matplotlib 内联](https://pythonguides.com/what-is-matplotlib-inline/)
*   [Matplotlib 支线剧情教程](https://pythonguides.com/matplotlib-subplot-tutorial/)

在本 [Python 教程](https://pythonguides.com)中，我们讨论了**如何安装 matplotlib python** ，以及在 Python 中使用数据可视化所需的所有依赖项，我们还讨论了以下主题:

*   如何安装 matplotlib python
*   如何安装 matplotlib python ubuntu
*   如何安装 matplotlib python 窗口
*   如何安装 matplotlib python mac
*   如何安装 matplotlib python conda
*   如何安装 matplotlib python pip
*   如何安装 matplotlib python venv
*   如何安装 matplotlib python3
*   如何安装 matplotlib python2

![Bijay Kumar MVP](img/9cb1c9117bcc4bbbaba71db8d37d76ef.png "Bijay Kumar MVP")[Bijay Kumar](https://pythonguides.com/author/fewlines4biju/)

Python 是美国最流行的语言之一。我从事 Python 工作已经有很长时间了，我在与 Tkinter、Pandas、NumPy、Turtle、Django、Matplotlib、Tensorflow、Scipy、Scikit-Learn 等各种库合作方面拥有专业知识。我有与美国、加拿大、英国、澳大利亚、新西兰等国家的各种客户合作的经验。查看我的个人资料。

[enjoysharepoint.com/](https://enjoysharepoint.com/)[](https://www.facebook.com/fewlines4biju "Facebook")[](https://www.linkedin.com/in/fewlines4biju/ "Linkedin")[](https://twitter.com/fewlines4biju "Twitter")