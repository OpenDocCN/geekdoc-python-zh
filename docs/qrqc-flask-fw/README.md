# 浅入浅出 Flask 框架

## 在 ubuntu 下安装 Flask

2014-06-01

Flask 是一个轻量级的基于 python 的 web 框架。

一般情况下，只要通过 pip 安装 Flask 即可：

```py
pip install Flask 
```

可以根据需要安装其他的工具与 Flask 配合，请参考[The Flask Mega-Tutorial, Part I: Hello, World!](http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)。

笔者安装的 flask 信息如下：

```py
>>> import flask

>>> print flask.__doc__

    flask
    ~~~~~

    A microframework based on Werkzeug.  It's extensively documented
    and follows best practice patterns.

    :copyright: (c) 2011 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.

>>> print flask.__version__
0.10.1 
```