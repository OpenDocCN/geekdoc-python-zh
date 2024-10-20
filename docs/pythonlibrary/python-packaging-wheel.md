# Python:使用轮子打包

> 原文：<https://www.blog.pythonlibrary.org/2014/01/10/python-packaging-wheel/>

Python 的第一个主流包是。鸡蛋文件。现在城里有一种新的形式叫做轮子(*。whl)。一个[轮](https://wheel.readthedocs.org/en/latest/)“被设计成以非常接近于磁盘格式的方式包含 PEP 376 兼容安装的所有文件”。在这篇文章中，我们将学习如何创建一个轮子，然后在一个虚拟环境中安装我们的轮子。

### 入门指南

你需要画中画来创建轮子。要学习如何安装 pip，我强烈推荐阅读 pip 的[安装页面](http://www.pip-installer.org/en/latest/installing.html)。如果您已经有 pip，那么您可能需要将其升级到最新版本。方法如下:在控制台窗口中，键入以下内容:

```py

pip install --upgrade pip

```

一旦你完成了，我们就可以开始学习如何制作轮子了！

### 创建轮子

首先，您需要安装转轮套件:

```py

pip install wheel

```

那很容易！接下来，我们将使用 unidecode 包来创建我们的第一个轮子，因为在撰写本文时它还没有制作出来，我自己在几个项目中使用过这个包。

```py

pip wheel --wheel-dir=my_wheels Unidecode

```

现在你应该在一个名为 **my_wheels** 的文件夹中有一个名为**Unidecode-0 . 04 . 14-py26-none-any . whl**的轮子。让我们学习如何安装我们的新车轮！

### 安装一个巨蟒轮

让我们创建一个 [virtualenv](https://pypi.python.org/pypi/virtualenv) 来进行测试。你可以在这里阅读更多关于虚拟 T2 的信息。安装后，运行以下命令:

```py

virtualenv test

```

这将为我们创建一个虚拟沙盒，其中包括 pip。在继续之前，确保从**脚本**文件夹中运行**激活**来启用虚拟菜单。现在 virtualenv 不包括 wheel，所以您必须重新安装 wheel:

```py

pip install wheel

```

安装完成后，我们可以使用以下命令安装我们的轮子:

```py

pip install --use-wheel --no-index --find-links=path/to/my_wheels Unidecode

```

要测试这是否可行，请从 virtualenv 中的 Scripts 文件夹运行 Python，并尝试导入 unidecode。如果它导入，那么你成功地安装了你的车轮！

*注:我原来装了一个旧版本的 virtualenv，相当麻烦。一定要升级你的，否则你将不得不做很多徒劳的事情来让它工作。*

*。whl 文件类似于*。它基本上是一个*。伪装的 zip 文件。如果您将扩展名从*重命名为。whl 到*。你可以在闲暇时打开你选择的 zip 应用程序，检查里面的文件和文件夹。

### 包扎

现在你应该准备好创建你自己的轮子了。它们看起来是为您的项目创建一个快速安装的本地依赖库的好方法。您可以创建几个不同的 wheel 存储库，以方便地在不同的版本集之间进行切换，用于测试目的。当与 virtualenv 结合使用时，您可以非常容易地看到新版本的依赖项如何影响您的项目，而无需多次下载它们。

### 附加说明

*   车轮[文档](https://wheel.readthedocs.org/en/latest/)
*   Python 打包[用户指南](https://python-packaging-user-guide.readthedocs.org/en/latest/current.html) 
*   virtualenv [文档](http://www.virtualenv.org/en/latest/)