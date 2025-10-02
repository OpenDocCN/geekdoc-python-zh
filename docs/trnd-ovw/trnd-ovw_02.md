# 下载和安装

## 下载和安装

**自动安装：** Tornado 已经列入 [PyPI](http://pypi.python.org/pypi/tornado) ，因此可以通过 `pip` 或者 `easy_install` 来安装。如果你没有安装 libcurl 的话，你需要将其单独安装到系统中。请参见下面的安装依赖一节。注意一点，使用 `pip` 或 `easy_install` 安装的 Tornado 并没有包含源代码中的 demo 程序。

**手动安装：** 下载 [tornado-2.0.tar.gz](http://github.com/downloads/facebook/tornado/tornado-2.0.tar.gz)

```py
tar xvzf tornado-2.0.tar.gz
cd tornado-2.0
python setup.py build
sudo python setup.py install 
```

Tornado 的代码托管在 [GitHub](http://sebug.net/paper/books/tornado/) 上面。对于 Python 2.6 以上的版本，因为标准库中已经包括了对 `epoll` 的支持，所以你可以不用 `setup.py` 编译安装，只要简单地将 tornado 的目录添加到 `PYTHONPATH` 就可以使用了。