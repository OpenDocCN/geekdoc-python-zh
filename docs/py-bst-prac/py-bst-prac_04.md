## Setuptools + Pip

Setuptools 是一款非常重要的 Python 第三方工具，它是标准包自带的 distutils 工具的增强版。一旦安装 Setuptools 后， 就可以通过一行指令下载和安装任何可获取到的 Python 应用包，还可以轻松地将这种网络安装的方式加入到自己开发 的 Python 应用中。

通过运行 Python 脚本 [ez_setup.py](https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py) [https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py] 获取最新 Windows 版本的 Setuptools。

安装完后就可以使用 **easy_install** 命令，但由于该命令已经被大多数人弃用，我们将安装替代它的 **pip** 命令。 Pip 支持包的卸载，而且与 easy_install 不同，它一直处于维护下。

通过运行 Python 脚本 [get-pip.py](https://raw.github.com/pypa/pip/master/contrib/get-pip.py) [https://raw.github.com/pypa/pip/master/contrib/get-pip.py] 可安装 pip

## 虚拟环境(Virtual Environment)

虚拟环境工具(virturalenv)通过为不同项目创建专属的 Python 虚拟环境，以实现其依赖的库独立保存在不同的路径。 这解决了“项目 X 依赖包版本 1.x，但项目 Y 依赖包版本为 4.x”的难题, 并且维持全局的 site-packages 目录干净、易管理。

举个例子，通过这个工具可以实现依赖 Django 1.3 的项目与依赖 Django 1.0 的项目共存。

进一步了解与使用请参考文档 [Virtual Environments](http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst) [http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst] 。

* * *

该页是 [另一份指南](http://www.stuartellis.eu/articles/python-development-windows/) [http://www.stuartellis.eu/articles/python-development-windows/] 的混合版本，可通过同一份许可获取。 © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.