# Python 3 -从 github 导入

> 原文：<https://www.blog.pythonlibrary.org/2016/02/02/python-3-import-from-github/>

前几天，我偶然发现了这个有趣的实验包，叫做 [import_from_github_com](https://github.com/nvbn/import_from_github_com) 。这个包使用了 [PEP 302](https://www.python.org/dev/peps/pep-0302/) 中提供的新的导入钩子，基本上允许你从 github 导入一个包。这个包实际上看起来要做的是安装这个包并把它添加到本地。反正你需要 Python 3.2 或者更高版本，git 和 pip 才能使用这个包。

安装完成后，您可以执行以下操作:

```py

>>> from github_com.zzzeek import sqlalchemy
Collecting git+https://github.com/zzzeek/sqlalchemy
  Cloning https://github.com/zzzeek/sqlalchemy to /tmp/pip-acfv7t06-build
Installing collected packages: SQLAlchemy
  Running setup.py install for SQLAlchemy ... done
Successfully installed SQLAlchemy-1.1.0b1.dev0
>>> locals()
{'__builtins__': , '__spec__': None,
 '__package__': None, '__doc__': None, '__name__': '__main__', 
'sqlalchemy': <module from="">,
 '__loader__': <class>}
```

软件包的 github 页面上没有提到的一个重要注意事项是，您需要以管理员身份运行 Python，否则它将无法安装其软件包。至少在 Xubuntu 上我是这样。总之，我认为这是一个整洁的小软件包，演示了一些可以添加到 Python 3 中的整洁的小导入挂钩。