# 烧瓶扩展——什么是扩展，如何安装？

> 原文：<https://www.askpython.com/python-modules/flask/flask-extensions>

在本文中，我们将讨论一些基本的 Flask 扩展。这些扩展方便易用。所以让我们来研究一下吧！！

## 为什么我们需要长颈瓶？

正如我们所知，Flask 是一个微型 web 框架，因为它的核心功能只包括基于 Werkzueg 的 WSGI、路由和基于 Jinja2 的模板引擎。

它还可以支持 cookies、会话和前端特性，如 JSON、静态文件等。

但这不足以构建全面安全的 web 应用程序。这就是长颈瓶延伸进入画面的地方。有了 Flask-Extensions，我们可以使用 Flask 框架执行更多的任务。

有许多长颈瓶扩展可用。我们现在来看看一些最常用的烧瓶延长管

## **一些重要的长颈瓶扩展**

一些最常用的长颈瓶延伸部分有:

| 延长 | 效用 |
| --- | --- |
| [烧瓶-SQLAlchemy](https://www.askpython.com/python-modules/flask/flask-postgresql) | 它提供了一个模型类型的接口来轻松地与数据库表进行交互。 |
| 烧瓶-WTF | 它提供了在 Flask web 应用程序中设计表单的另一种方法。使用 WT 表单，我们可以验证和保护用户发送的表单数据。
 |
| 烧瓶邮件 | 它为 Flask 应用程序提供了一个 SMTP 接口，用于向客户端/用户发送电子邮件。 |
| [烧瓶-登录](https://www.askpython.com/python-modules/flask/flask-user-authentication) | 它为 Flask Web 应用程序提供用户认证功能 |
| [瓶调试工具](https://www.askpython.com/python-modules/flask/flask-debug-mode) | 它提供了一个强大的调试工具栏，用于调试 Flask 应用程序 |
| 烧瓶-Sijax | 它有助于添加 Sijax，这是一个 Python/ [jQuery](https://jquery.com/) 库，使 AJAX 易于在 web 应用程序中使用，并支持 Flask 应用程序。 |

这些扩展是 **Python 模块**，它扩展了 Flask 应用程序的功能。因此，我们可以使用 [pip](https://www.askpython.com/python-modules/python-pip) 实用程序像安装 Python 库一样安装它们。

安装 Flask-Extension "**Flask-foo**的语法是:

```py
pip install flask-foo

#pip install flask-Sqlalchemy
#pip install flask-wtf
#pip install flask-mail
#pip install flask-login
#pip install flask-debugtoolbar
#pip install flask-sijax

```

导入也类似于我们导入 python 库的方式:

```py
from flask_foo import <Class>, <function>...

```

对于高于 0.7 的 Flask 版本，您也可以通过 **flask.ext.** 导入扩展

语法是:

```py
from flask.ext import foo #sqlalchemy, login .....

```

如果您的**兼容模块**未激活**，您会得到一个错误。**要激活它，请使用代码:

```py
import flaskext_compat
flaskext_compat.activate()

from flask.ext import foo

```

一旦我们激活它，我们可以像以前一样使用 **flask.ext** 。

## **参考文献:**

*   **烧瓶 SQLAlchemy:**【https://flask-sqlalchemy.palletsprojects.com/en/2.x/ 
*   **Flask WT Forms:**[https://Flask . pallets projects . com/en/1.1 . x/patterns/WT Forms/](https://flask.palletsprojects.com/en/1.1.x/patterns/wtforms/)
*   **烧瓶邮件:**【https://pythonhosted.org/Flask-Mail/】T2
*   **烧瓶登录:**【https://flask-login.readthedocs.io/en/latest/ 
*   **烧瓶调试工具栏:**【https://flask-debugtoolbar.readthedocs.io/en/latest/ 
*   **烧瓶 Sijax:**【https://pythonhosted.org/Flask-Sijax/ 

## **结论**

就这样，伙计们！这是 Flask 教程系列的最后一篇文章。一定要看看我们的其他 [Flask 教程](https://www.askpython.com/python-modules/flask)来了解更多关于 Flask 的知识。

编码快乐！！