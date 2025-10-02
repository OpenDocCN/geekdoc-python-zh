## Setuptools & Pip

[setuptools](https://pypi.python.org/pypi/setuptools) [https://pypi.python.org/pypi/setuptools] 和 [pip](https://pip.pypa.io/en/stable/) [https://pip.pypa.io/en/stable/] 是最重要的两个 Python 第三方软件包。一旦安装了它们，就可以通过一条指令下载、安装和卸载可获取到的 Python 应用包，还可以轻松地将这种网络安装的方式加入到自己开发的 Python 应用中。

Python 2.7.9 以及之后版本(Python2 系列)，和 Python 3.4 以及之后版本均默认包含 pip。

运行以下命令行代码检查 pip 是否已经安装：

```py
$ command -v pip 
```

[参考官方 pip 安装指南](https://pip.pypa.io/en/latest/installing/) [https://pip.pypa.io/en/latest/installing/] 获取 pip 工具，并自动安装最新版本的 setuptools。

## Virtual Environments

虚拟环境工具(virturalenv)通过为不同项目创建专属的 Python 虚拟环境，以实现其依赖的库独立保存在不同的路径。 这解决了“项目 X 依赖包版本 1.x，但项目 Y 依赖包版本为 4.x”的难题, 并且维持全局的 site-packages 目录干净、易管理。

举个例子，通过这个工具可以实现依赖 Django 1.3 的项目与依赖 Django 1.0 的项目共存。

进一步了解与使用请参考文档 [Virtual Environments](http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst) [http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst] 。

也可使用 virtualenvwrapper 更轻松地管理你的虚拟环境。

* * *

该页是 [另一份指南](http://www.stuartellis.eu/articles/python-development-windows/) [http://www.stuartellis.eu/articles/python-development-windows/] 的混合版本，可通过同一份许可获取。 © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.