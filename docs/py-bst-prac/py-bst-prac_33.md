# 虚拟环境

虚拟环境是一个将不同项目所需求的依赖分别放在独立的地方的一个工具，它给这些工程创建虚拟的 Python 环境。它解决了“项目 X 依赖于版本 1.x，而项目 Y 需要项目 4.x”的两难问题，而且使你的全局 site-packages 目录保持干净和可管理。

比如，你可以工作在一个需求 Django 1.3 的工程，同时维护一个需求 Django 1.0 的工程。

## virtualenv

[virtualenv](http://pypi.python.org/pypi/virtualenv) [http://pypi.python.org/pypi/virtualenv] 是一个创建隔绝的 Python 环境的工具。virtualenv 创建一个包含所有必要的可执行文件的文件夹，用来使用 Python 工程所需的包。

通过 pip 安装 virtualenv：

```py
$ pip install virtualenv 
```

### 基本使用

1.  为一个工程创建一个虚拟环境：

```py
$ cd my_project_folder
$ virtualenv venv 
```

`virtualenv venv` 将会在当前的目录中创建一个文件夹，包含了 Python 可执行文件，以及 `pip` 库的一份拷贝，这样就能安装其他包了。虚拟环境的名字（此例中是 `venv` ）可以是任意的；若省略名字将会把文件均放在当前目录。

在任何你运行命令的目录中，这会创建 Python 的拷贝，并将之放在叫做 `venv` 的文件中。

你可以选择使用一个 Python 解释器：

```py
$ virtualenv -p /usr/bin/python2.7 venv 
```

这将会使用 `/usr/bin/python2.7` 中的 Python 解释器。

2.  要开始使用虚拟环境，其需要被激活：

```py
$ source venv/bin/activate 
```

当前虚拟环境的名字会显示在提示符左侧（比如说 `(venv)你的电脑:你的工程 用户名$）以让你知道它是激活的。从现在起，任何你使用 pip 安装的包将会放在 ``venv` 文件夹中，与全局安装的 Python 隔绝开。

像平常一样安装包，比如：

```py
$ pip install requests 
```

3.  如果你在虚拟环境中暂时完成了工作，则可以停用它：

```py
$ deactivate 
```

这将会回到系统默认的 Python 解释器，包括已安装的库也会回到默认的。

要删除一个虚拟环境，只需删除它的文件夹。（要这么做请执行 `rm -rf venv` ）

然后一段时间后，你可能会有很多个虚拟环境散落在系统各处，你将有可能忘记它们的名字或者位置。

### 其他注意

运行带 `--no-site-packages` 选项的 `virtualenv` 将不会包括全局安装的包。这可用于保持包列表干净，以防以后需要访问它。（这在 `virtualenv` 1.7 及之后是默认行为）

为了保持你的环境的一致性，“冷冻住（freeze）”环境包当前的状态是个好主意。要这么做，请运行：

```py
$ pip freeze > requirements.txt 
```

这将会创建一个 `requirements.txt` 文件，其中包含了当前环境中所有包及各自的版本的简单列表。你可以使用 “pip list”在不产生 requirements 文件的情况下，查看已安装包的列表。这将会使另一个不同的开发者（或者是你，如果你需要重新创建这样的环境）在以后安装相同版本的相同包变得容易。

```py
$ pip install -r requirements.txt 
```

这能帮助确保安装、部署和开发者之间的一致性。

最后，记住在源码版本控制中排除掉虚拟环境文件夹，可在 ignore 的列表中加上它。

 ## virtualenvwrapper

[virtualenvwrapper](http://virtualenvwrapper.readthedocs.org/en/latest/index.html) [http://virtualenvwrapper.readthedocs.org/en/latest/index.html] 提供了一系列命令使得和虚拟环境工作变得愉快许多。它把你所有的虚拟环境都放在一个地方。

安装（确保 **virtualenv** 已经安装了）：

```py
$ pip install virtualenvwrapper
$ export WORKON_HOME=~/Envs
$ source /usr/local/bin/virtualenvwrapper.sh 
```

([virtualenvwrapper 的完整安装指引](http://virtualenvwrapper.readthedocs.org/en/latest/install.html) [http://virtualenvwrapper.readthedocs.org/en/latest/install.html].)

对于 Windows，你可以使用 [virtualenvwrapper-win](https://github.com/davidmarble/virtualenvwrapper-win/) [https://github.com/davidmarble/virtualenvwrapper-win/] 。

To install (make sure **virtualenv** is already installed): 安装（确保 **virtualenv** 已经安装了）：

```py
$ pip install virtualenvwrapper-win 
```

在 Windows 中，WORKON_HOME 默认的路径是 %USERPROFILE%Envs 。

### 基本使用

1.  创建一个虚拟环境：

```py
$ mkvirtualenv venv 
```

这会在 `~/Envs` 中创建 `venv` 文件夹。

2.  在虚拟环境上工作：

```py
$ workon venv 
```

或者，你可以创建一个项目，它会创建虚拟环境，并在 `$PROJECT_HOME` 中创建一个项目目录。当你使用 `workon myproject` 时，会 `cd` -ed 到项目目录中。

```py
$ mkproject myproject 
```

**virtualenvwrapper** 提供环境名字的 tab 补全功能。当你有很多环境，并且很难记住它们的名字时，这就显得很有用。

`workon` 也能停止你当前所在的环境，所以你可以在环境之间快速的切换。

3.  停止是一样的：

```py
$ deactivate 
```

4.  删除：

```py
$ rmvirtualenv venv 
```

### 其他有用的命令

`lsvirtualenv`

列举所有的环境。

`cdvirtualenv`

导航到当前激活的虚拟环境的目录中，比如说这样你就能够浏览它的 `site-packages` 。

`cdsitepackages`

和上面的类似，但是是直接进入到 `site-packages` 目录中。

`lssitepackages`

显示 `site-packages` 目录中的内容。

[virtualenvwrapper 命令的完全列表](http://virtualenvwrapper.readthedocs.org/en/latest/command_ref.html) [http://virtualenvwrapper.readthedocs.org/en/latest/command_ref.html] 。 

## virtualenv-burrito

有了 [virtualenv-burrito](https://github.com/brainsik/virtualenv-burrito) [https://github.com/brainsik/virtualenv-burrito] ，你就能使用单行命令拥有 virtualenv + virtualenvwrapper 的环境。

## autoenv

当你 `cd` 进入一个包含 `.env` 的目录中，就会 [autoenv](https://github.com/kennethreitz/autoenv) [https://github.com/kennethreitz/autoenv] 自动激活那个环境。

使用 `brew` 在 Mac OS X 上安装它：

```py
$ brew install autoenv 
```

在 Linux 上:

```py
$ git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
$ echo 'source ~/.autoenv/activate.sh' >> ~/.bashrc 
```

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.