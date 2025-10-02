# 安装需求

### 安装需求

Tornado 在 Python 2.5, 2.6, 2.7 中都经过了测试。要使用 Tornado 的所有功能，你需要安装 [PycURL](http://pycurl.sourceforge.net/) (7.18.2 或更高版本) 以及 [simplejson](http://pypi.python.org/pypi/simplejson/) (仅适用于 Python 2.5，2.6 以后的版本标准库当中已经包含了对 JSON 的支持)。为方便起见，下面将列出 Mac OS X 和 Ubuntu 中的完整安装方式：

Mac OS X 10.6 (Python 2.6+)

```py
sudo easy_install setuptools pycurl 
```

Ubuntu Linux (Python 2.6+)

```py
sudo apt-get install python-pycurl 
```

Ubuntu Linux (Python 2.5)

```py
sudo apt-get install python-dev python-pycurl python-simplejson 
```