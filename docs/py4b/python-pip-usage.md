# 如何在 Python 中使用 Pip

> 原文：<https://www.pythonforbeginners.com/basics/python-pip-usage>

Pip 是一个软件包管理系统，用于安装和管理软件包，例如那些在 [Python 包索引](https://pypi.python.org/pypi "PYPI")中找到的软件包。

### 皮普是什么？

```py
 Pip is a replacement for easy_install. 

Packages installs the packages default under site-packages. 
```

### 安装 Pip

```py
 To install Pip on your system, you can use either the source tarball or
by using easy_install.
>> $ easy_install pip

After that, the pip application is installed. 
```

### Pip 使用

```py
 How to use Pip 
```

##### 安装软件包

```py
$ pip install simplejson
[... progress report ...]
Successfully installed simplejson

```

##### 升级软件包

```py
$ pip install --upgrade simplejson
[... progress report ...]
Successfully installed simplejson

```

##### 移除包

```py
$ pip uninstall simplejson
Uninstalling simplejson:
  /home/me/env/lib/python2.7/site-packages/simplejson
  /home/me/env/lib/python2.7/site-packages/simplejson-2.2.1-py2.7.egg-info
Proceed (y/n)? y
  Successfully uninstalled simplejson

```

##### 搜索包

```py
#Search PyPI for packages
$ pip search "query"

```

##### 检查包的状态

```py
# To get info about an installed package, including its location and files:
pip show ProjectName

```

### 为什么使用 Pip 而不是 easy_install？

```py
 (The answer is taken from this [post](https://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install "pip-over-easy_install") on stackoverflow)

All packages are downloaded before installation. 

Partially-completed installation doesn’t occur as a result.

Care is taken to present useful output on the console.

The reasons for actions are kept track of. 

For instance, if a package is being installed, pip keeps track of why that 
package was required.

Error messages should be useful.

The code is relatively concise and cohesive, making it easier to use 
programmatically.

Packages don’t have to be installed as egg archives, they can be installed flat.

Native support for other version control systems (Git, Mercurial and Bazaar)

Uninstallation of packages.

Simple to define fixed sets of requirements and reliably reproduce a set of 
packages. 
```