# 数据库

## DB-API

Python 数据库 API（DB-API）定义了一个 Python 数据库访问模块的标准接口。它的文档在 [**PEP 249**](https://www.python.org/dev/peps/pep-0249) [https://www.python.org/dev/peps/pep-0249] 可以查看。 几乎所有 Python 数据库模块，诸如 sqlite3， psycopg 以及 mysql-python 都遵循这个接口。

关于如何与遵守这一接口的模块交互的教程可以在这里找到： [这里](http://halfcooked.com/presentations/osdc2006/python_databases.html) [http://halfcooked.com/presentations/osdc2006/python_databases.html] 以及 [这里](http://web.archive.org/web/20120815130844/http://www.amk.ca/python/writing/DB-API.html) [http://web.archive.org/web/20120815130844/http://www.amk.ca/python/writing/DB-API.html] 。

## SQLAlchemy

[SQLAlchemy](http://www.sqlalchemy.org/) [http://www.sqlalchemy.org/] 是一个流行的数据库工具。不像很多 数据库库，它不仅提供一个 ORM 层，而且还有一个通用 API 来编写避免 SQL 的数据库无关代码。

```py
$ pip install sqlalchemy 
```

## Django ORM

Django ORM 是 [Django](http://www.djangoproject.com) [http://www.djangoproject.com] 用来进行数据库访问的接口。

它的思想建立在 [models](https://docs.djangoproject.com/en/dev/#the-model-layer) [https://docs.djangoproject.com/en/dev/#the-model-layer] ， 之上。这是一个致力于简化 Python 中数据操作的抽象层。

基础：

*   每个 model 是 django.db.models.Model 的子类。
*   model 的每个属性表示数据库的域（field）。
*   Django 给你一个自动生成的数据库访问 API，参见 [Making queries](https://docs.djangoproject.com/en/dev/topics/db/queries/) [https://docs.djangoproject.com/en/dev/topics/db/queries/]。

## peewee

[peewee](http://docs.peewee-orm.com/en/latest/) [http://docs.peewee-orm.com/en/latest/] 是另一个 ORM，它致力于轻量级和支持 Python2.6+与 3.2+默认支持的 SQLite，MySQL 以及 Postgres。 [model layer](https://peewee.readthedocs.org/en/latest/peewee/quickstart.html#model-definition) [https://peewee.readthedocs.org/en/latest/peewee/quickstart.html#model-definition] 与 Django ORM 类似并且它拥有 [SQL-like methods](https://peewee.readthedocs.org/en/latest/peewee/quickstart.html#retrieving-data) [https://peewee.readthedocs.org/en/latest/peewee/quickstart.html#retrieving-data] 来查询数据。除了将 SQLite，MySQL 以及 Postgres 变为开箱即用，还有进一步的扩展功能可以在这里找到： [collection of add-ons](https://peewee.readthedocs.org/en/latest/peewee/playhouse.html#playhouse) [https://peewee.readthedocs.org/en/latest/peewee/playhouse.html#playhouse]。

## PonyORM

[PonyORM](http://ponyorm.com/) [http://ponyorm.com/] 是一个 ORM，它使用与众不同的方法查询数据库，有别于 使用类似 SQL 的语言或者布尔表达式，它使用 Python 的生成器达到目的。而且还有一个图形化 schema 编辑器生成 PonyORM 实体。它支持 Python2.6+与 3.3+并且可以连接 SQLite，MySQL，Postgres 与 Oracle。

## SQLObject

[SQLObject](http://www.sqlobject.org/) [http://www.sqlobject.org/] 是另一个 ORM。它支持广泛的数据库，常见的 MySQL，Postgres 以及 SQLite 与更多的特别系统如 SAP DB，SyBase 与 MSSQL。它只支持 Python 2

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.