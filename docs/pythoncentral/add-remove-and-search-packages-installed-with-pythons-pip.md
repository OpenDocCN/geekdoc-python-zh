# 使用 pip 在 Python 中添加、删除和搜索包

> 原文：<https://www.pythoncentral.io/add-remove-and-search-packages-installed-with-pythons-pip/>

## 使用 pip 管理 Python 包

像许多有用的编程生态系统一样，Python 提供了一个强大且易于使用的包管理系统，称为 pip。它是用来取代一个叫做 easy_install 的旧工具的。从高层次的角度来看，pip 比 easy_install 具有以下优势:

*   所有的软件包在安装前都被下载，以防止部分安装。
*   输出信息是经过预处理的，因此比古怪的消息更有用。
*   它记录了执行操作的原因。例如，需要一个包的原因被记录下来以备将来参考。
*   包可以作为平面模块安装，这使得库代码调试比 egg 档案容易得多。
*   对其他版本控制系统的本机支持。例如，如果设置正确，您可以直接从 [GitHub](https://github.com/ "GitHub") 库安装 Python 包。
*   软件包可以卸载。这是 easy_install 的一个巨大优势，easy _ install 需要程序员手动卸载软件包。
*   定义需求集很简单，因此跨不同环境复制一组包也很容易。

## 设置 virtualenv 和 pip 以执行简单操作

virtualenv 是一个用于沙盒 Python 环境的工具。virtualenv 允许程序员设置类似于独立沙箱的独立 Python 上下文，而不是修改全局 Python 环境。将您的 Python 环境沙箱化的一个优点是，您可以在不同的 Python 版本和包依赖项下毫不费力地测试您的代码，而无需在虚拟机之间切换。

要安装和设置 pip 和 virtualenv，请运行以下命令:

```py

# easy_install is the default package manager in CPython

% easy_install pip

# Install virtualenv using pip

% pip install virtualenv

% virtualenv python-workspace # create a virtual environment under the folder 'python-workspace'

```

如果您当前的 Python 环境不包含任何包管理器，您可以从这里的[下载一个 Python 文件并运行它:](https://raw.github.com/pypa/virtualenv/master/virtualenv.py "Virtualenv download")

```py

# Create a virtual environment under the folder 'python-workspace'

% python virtualenv.py python-workspace

```

一旦在“python-workspace”下设置了新的 virtualenv，您需要激活它，以便当前 shell 中的 python 环境将过渡到 virtualenv:

```py

% cd python-workspace

% source ./bin/activate # activate the virtualenv 'python-workspace'

% python # enter an interpreter shell by executing the current virtual python program under 'python-workspace/bin/python'

```

如果你看一下文件夹“python-workspace/bin”的内容，你会看到一个可执行程序的列表，如“python”、“pip”和“easy_install”等。不是在系统默认的 Python 环境下执行程序，而是执行 virtualenv 的 Python 程序。

## 使用 pip 添加/安装软件包

要安装软件包，请使用“install”命令。

```py

% pip install sqlalchemy # install the package 'sqlalchemy' and its dependencies

```

有时，您可能希望在安装软件包之前检查它的源代码。例如，您可能希望在安装软件包之前检查其新版本的源代码，以确保它能与您当前的代码一起工作。

```py

% pip install --download sqlalchemy_download sqlalchemy # download the package 'sqlalchemy' archives into 'sqlalchemy_download' instead of installing it

% pip install --no-install sqlalchemy # unpack the downloaded package archives into 'python-workspace/build' for inspection

% pip install --no-download sqlalchemy # install the unpacked package archives

```

如果您想要安装软件包的最新版本，您可以直接从它的 Git 或 Subversion 存储库中安装它:

```py

% pip install git+https://github.com/simplejson/simplejson.git

% pip install svn+svn://svn.zope.org/repos/main/zope.interface/trunk

```

## 使用 pip 升级软件包

对于已安装的软件包，您可以通过以下方式升级它们:

```py

% pip install --upgrade sqlalchemy # upgrade sqlalchemy if there’s a newer version available. Notice that --upgrade will recursively upgrade sqlalchemy and all of its dependencies.

```

如果不希望升级软件包及其依赖项(有时您可能希望测试软件包的向后兼容性)，您可以通过以下方式执行非递归升级:

```py

% pip install --upgrade --no-deps sqlalchemy # only upgrade sqlalchemy but leave its dependencies alone

```

## 保存您的 pip 包列表

到目前为止，您应该已经使用 pip 安装了一堆包。为了在不同的环境下测试已安装的软件包，您可以将已安装的软件包列表保存或“冻结”到一个需求文件中，并在另一个环境中使用该需求文件重新安装所有的软件包:

```py

% pip freeze > my-awesome-env-req.txt # create a requirement file that contains a list of all installed packages in the current virtualenv 'python-workspace'

% virtualenv ../another-python-workspace # create a new virtualenv 'another-python-workspace'

% cd ../another-python-workspace

% source ./bin/activate # activate the new empty virtualenv 'another-python-workspace'

% pip install -r ../python-workspace/my-awesome-env-req.txt # install all packages specified in 'my-awesome-env-req.txt'

```

## 移除/卸载 pip 软件包

如果不知何故，你决定某些包对你的项目不再有用，你想删除它们来清理 virtualenv。您只需输入以下命令即可删除软件包:

```py

% pip uninstall sqlalchemy # uninstall the package 'sqlalchemy'

```

或者通过以下方式删除软件包列表:

```py

% pip uninstall -r my-awesome-env-req.txt # uninstall all packages specified in 'my-awesome-env-req.txt'

```

请注意，pip 不知道如何卸载两种类型的软件包:

*   用纯发行版安装的软件包:'【T0]'
*   使用脚本包装安装的包:'【T0]'

因为这两种类型的已安装软件包不包含任何元数据，所以 pip 不知道应该删除哪些文件来卸载它们。

## 搜索 pip 包

如果要搜索解决特定类型问题的软件包，可以通过以下方式执行搜索:

```py

% pip search database # search package titles and descriptions that contain the word 'database'

```

搜索包对于检索问题域中所有包的概述非常有用，这样您就可以比较和选择最适合您需要做的事情的包。

## 使用 pip 的提示

1.为了防止意外运行 pip 将不需要的软件包安装到全局环境中，您可以通过设置 shell 环境变量来告诉 pip 仅在 virtualenv 当前处于活动状态时运行:

```py

% export PIP_REQUIRE_VIRTUALENV=true

```

2.通常，软件包将安装在“站点软件包”目录下。但是，如果您想要对包进行修改和调试，那么直接从包的源代码树中运行包是有意义的。您可以通过告诉 pip 使用“-e”选项/参数来安装软件包，使其进入“编辑模式”,如下所示:

```py

# create a .pth file for sqlalchemy in 'site-packages' instead of installing it into 'site-packages'

# so that you can make changes to the package and debug the changes immediately

% pip install -e path/to/sqlalchemy

```

## 软件包索引，以及关于 pip 的更多信息

Python 有一个网站，包含许多您可能想要查看的有用包，称为“ [Python 包索引](https://pypi.python.org/pypi?%3Aaction=browse "Python Package Index")”，或“ [PyPI](https://pypi.python.org/pypi?%3Aaction=browse "PyPI") ”。这是许多常用 Python 库/模块的一个很好的存储库，它们都是预打包的，很容易安装。

关于 Python 的 pip - checkout [的更多信息，请访问本网站](http://www.pip-installer.org/en/latest/ "pip installer")。