## 现在就开始吧！

跟着我一起安装合适的 Python 吧。

在正式安装之前，应先安装 GCC。GCC 的获取方式包括：下载安装 [XCode](http://developer.apple.com/xcode/) [http://developer.apple.com/xcode/], 或安装小巧一些的 [Command Line Tools](https://developer.apple.com/downloads/) [https://developer.apple.com/downloads/] (需要一个 Apple 账号) 或者更轻巧的 [OSX-GCC-Installer](https://github.com/kennethreitz/osx-gcc-installer#readme) [https://github.com/kennethreitz/osx-gcc-installer#readme] 。

尽管 OS X 系统附带了大量 UNIX 工具，熟悉 Linux 系统的人员使用时会发现缺少一个重要的组件——合适的包管理工具， [Homebrew](http://brew.sh) [http://brew.sh] 正好填补了这个空缺。

[安装 Homebrew](http://brew.sh/#install) [http://brew.sh/#install] 只需打开 `终端` 或个人常用的终端模拟器并运行：

```py
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" 
```

运行这段脚本将列出它会引起的改变，并在安装开始前提示你。 安装完成 Homebrew 后，需将其所在路径插入到 `PATH` 环境变量的最前面，即在你所登录用户的 `~/.profile` 文件末尾加上这一行：

```py
export PATH=/usr/local/bin:/usr/local/sbin:$PATH 
```

接下来可以开始安装 Python 2.7：

```py
$ brew install python 
```

耗时大概几分钟。

## Setuptools & Pip

Homebrew 会自动安装好 Setuptools 和 `pip` 。 Setuptools 提供 `easy_install` 命令，实现通过网络（通常 Internet）下载和安装第三方 Python 包。 还可以轻松地将这种网络安装的方式加入到自己开发的 Python 应用中。

`pip` 是一款方便安装和管理 Python 包的工具， 在 [一些方面](https://python-packaging-user-guide.readthedocs.org/en/latest/pip_easy_install/#pip-vs-easy-install) [https://python-packaging-user-guide.readthedocs.org/en/latest/pip_easy_install/#pip-vs-easy-install] ， 它更优于 `easy_install` ，故更推荐它。

## 虚拟环境(Virtual Environment)

虚拟环境工具(virturalenv)通过为不同项目创建专属的 Python 虚拟环境，以实现其依赖的库独立保存在不同的路径。 这解决了“项目 X 依赖包版本 1.x，但项目 Y 依赖包版本为 4.x”的难题，并且维持全局的 site-packages 目录干净、易管理。

举个例子，通过这个工具可以实现依赖 Django 1.3 的项目与依赖 Django 1.0 的项目共存。

进一步了解与使用请参考文档 [Virtual Environments](http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst) [http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst] 。

* * *

该页是 [另一份指南](http://www.stuartellis.eu/articles/python-development-windows/) [http://www.stuartellis.eu/articles/python-development-windows/] 的混合版本，可通过同一份许可获取。 © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.