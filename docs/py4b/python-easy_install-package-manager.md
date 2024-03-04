# Python : Easy_Install 包管理器

> 原文：<https://www.pythonforbeginners.com/basics/python-easy_install-package-manager>

## 简易安装软件包管理器

```py
 This post will be the first in a series of "Python Packing Manager" posts.

To know which one you are going to use, can be hard, as there are a few
different ones to choose from.

Let's start out with Easy_Install. 
```

## 简单安装

```py
 Distributes Python programs and libraries (based on the Python Eggs wrapper)

It's a python module (easy_install) that is bundled with setuptools.

It lets you automatically download, build, install, and manage Python packages.

Easy_Install looks in the Python Package Index (PyPI) for the desired packages
and uses the metadata there to download and install the package and its
dependencies. 

It is also hosted itself on the PyPI. 
```

## 蟒蛇蛋

```py
 A Python egg is a way of distributing Python packages. 

"Eggs are to Pythons as Jars are to Java..."

For a great introduction about this, I suggest you head over to [this](http://mrtopf.de/blog/en/a-small-introduction-to-python-eggs/ "python_eggs") post 
```

## setuptools

```py
 setuptools is a collection of enhancements to the Python distutils that allow
you to more easily build and distribute Python packages, especially ones that
have dependencies on other packages. 
```

## 简易安装用法

##### 下载和安装软件包

```py
 For basic use of easy_install, you need only supply the filename or URL of a 
source distribution or .egg file (Python Egg). 

By default, packages are installed to the running Python installation's
site-packages directory. 
```

## 简易安装示例

```py
 Install a package by name, searching PyPI for the latest version, 
and automatically downloading, building, and installing it:
**>>easy_install SQLObject**

Install or upgrade a package by name and version by finding links on
a given "download page":
**>>easy_install -f http://pythonpaste.org/package_index.html SQLObject**

Download a source distribution from a specified URL, automatically
building and installing it:
**>>easy_install http://example.com/path/to/MyPackage-1.2.3.tgz**

Install an already-downloaded .egg file:
**>>easy_install /my_downloads/OtherPackage-3.2.1-py2.3.egg**

Upgrade an already-installed package to the latest version listed on
PyPI:
**>>easy_install --upgrade PyProtocols**

Install a source distribution that's already downloaded and extracted
in the current directory (New in 0.5a9):
**>>easy_install .** 
```

## 升级软件包

```py
 You don't need to do anything special to upgrade a package

Just install the new version, either by requesting a specific version, e.g.:
**>>easy_install "SomePackage==2.0"** 
```

```py
 If you're installing to a directory on PYTHONPATH, or a configured "site"
directory (and not using -m), installing a package automatically replaces
any previous version in the easy-install.pth file, so that Python will
import the most-recently installed version by default. 

So, again, installing the newer version is the only upgrade step needed. 
```

```py
 A version greater than the one you have now:
**>>easy_install "SomePackage>2.0"**

Using the upgrade flag, to find the latest available version on PyPI:
**>>easy_install --upgrade SomePackage**

Or by using a download page, direct download URL, or package filename:
**>>easy_install -f http://example.com/downloads ExamplePackage
>>easy_install 	  http://example.com/downloads/ExamplePackage-2.0-py2.4.egg
>>easy_install my_downloads/ExamplePackage-2.0.tgz** 
```

## 更改活动版本

```py
 If you've upgraded a package, but need to revert to a previously-installed
version, you can do so like this: 
```

```py
 **>>easy_install PackageName==1.2.3** 
(where 1.2.3 is replaced by the exact version number you wish to switch to. 

If a package matching the requested name and version is not already installed in
a directory on sys.path, it will be located via PyPI and installed.)

To switch to the latest installed version of PackageName:
**>>easy_install PackageName**

This will activate the latest installed version! 
```

## 卸载软件包

```py
 If you have replaced a package with another version, then you can just delete
the package(s) you don't need by deleting the PackageName-versioninfo.egg file
or directory (found in the installation directory). 
```

```py
 If you want to delete the currently installed version of a package 
(or all versions of a package), you should first run:
**>>easy_install -mxN PackageName**

This will ensure that Python doesn't continue to search for a package you're 
planning to remove. 

After you've done this, you can safely delete the .egg files or directories,
along with any scripts you wish to remove. 
```

```py
 For more information on how to use easy_install:
[http://peak.telecommunity.com/DevCenter/EasyInstall](http://peak.telecommunity.com/DevCenter/EasyInstall "EasyInstall") 
```