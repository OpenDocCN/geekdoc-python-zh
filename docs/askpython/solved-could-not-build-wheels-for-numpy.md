# 无法为 Numpy 构建轮子(已解决)

> 原文：<https://www.askpython.com/python-modules/numpy/solved-could-not-build-wheels-for-numpy>

对于许多人来说，Python 包的安装可能是一个令人困惑和沮丧的过程，尤其是在为 Numpy 构建轮子的时候。对于许多 Python 包来说，轮子是安装过程中的一个重要部分，如果不能构建轮子，包就可能无法安装。

在本文中，我们将讨论在尝试为 Numpy 构建轮子时可能出现的常见问题，以及如何解决这些问题。

## 什么是巨蟒轮？

Python Wheels 是 Python 环境中不可或缺的一部分，旨在简化 Python 包的安装。

它们通常与一个. whl 文件相关联，该文件是软件包的较小版本，因此安装起来更高效、更快速。这是因为它包含 Python 源代码和所有必需的元数据。

“wheel”这个名称是对 Python 存储库的原始名称“cheeseShop”的半开玩笑引用，cheeseShop 是一个奶酪轮子。

## 车轮和 Numpy 版本不兼容:

大多数时候，当你面对“无法为 NumPy 构建轮子”的错误时，通常是因为你的 NumPy 版本和你的 python 版本之间的版本不匹配。您要么需要更新您的 python 版本，要么显式设置您的 NumPy 版本以匹配您系统的 python 版本。

### 错误:无法为 NumPy 构建轮子

该错误主要出现在安装 numpy 包或与当前 python 版本不兼容的特定版本的 Numpy 包时，使用:

```py
pip install numpy

```

或者

```py
pip install numpy=1.24.0 #the latest version

```

给出输出:

```py
ERROR: Failed building wheel for numpy
Failed to build numpy
ERROR: Could not build wheels for numpy

```

## 如何修复“无法为 NumPy 构建轮子”

有三种非常简单的方法可以解决这个错误，当你面对这个错误时，你不需要担心，因为它很容易解决。

### 1.从命令提示符或 python 终端升级 pip

通过在命令提示符下运行以下代码，在[升级您的 pip](https://www.askpython.com/python-modules/python-pip) 之前检查 python 版本:

```py
python --version

```

接下来，建议使用命令提示符而不是 python IDE 终端来升级 pip。在命令提示符下运行以下命令:

```py
python.exe -m pip install --upgrade pip

```

或者，如果使用 python IDE 终端，请使用:

```py
pip install --upgrade pip

```

升级 pip 后，再次检查 python 版本。应该更新版本以匹配最新版本。

现在，通过运行以下命令尝试安装 numpy:

```py
pip install numpy

```

现在，您应该会收到以下消息:

```py
Successfully installed numpy- "latest version name"

```

### 2.根据 python 安装特定于版本的 NumPy

这是避免版本不匹配的轮子错误的另一个解决方案。在命令提示符下运行以下命令，首先检查 python 版本:

```py
python --version

```

输出将给出您计算机上安装的 python 的具体版本。我的是最新版本，因此输出显示 python 3.11，如下所示:

```py
Python 3.11.0

```

现在检查 numpy 的哪个版本与您计算机中的 python 版本兼容。[点击此处](https://pypi.org/project/numpy/#history)访问官方网站，从所有版本中选择兼容版本。接下来，在命令提示符下运行以下命令:

```py
pip install numpy == "required version number"

```

Numpy 应该安装成功。现在，您应该能够在所有项目中导入 numpy 了。

### 3.卸载 numpy 并重新安装 numpy

运行以下命令:

```py
pip uninstall numpy

```

直到您收到一条消息:

```py
no files available with numpy to uninstall

```

现在再次新鲜安装 numpy，你可以指定版本或不根据您的使用。

```py
pip install numpy

```

这个问题现在应该已经解决了。

### 验证 numpy 版本

您可以在安装后通过运行以下命令来检查 numpy 版本:

```py
pip show numpy

```

它将给出您已安装的 numpy 包的完整详细信息，如下所示:

```py
Name: numpy
Version: 1.23.5
Summary: NumPy is the fundamental package for array computing with Python.
Home-page: https://www.numpy.org
Author: Travis E. Oliphant et al.
Author-email:
License: BSD
Location: C:\Users\AUTHOR\AppData\Local\Programs\Python\Python311\Lib\site-packages
Requires:
Required-by: contourpy, matplotlib, opencv-python

```

## 结论

“无法为 NumPy 构建轮子”这一错误可以通过几个简单的步骤来解决。升级 pip，按照 python 安装特定版本的 numpy，卸载并重新安装 NumPy 都是可能的解决方案。如果您仍然遇到问题，请确保检查您安装的 python 和 numpy 的版本，并确保它们是兼容的。