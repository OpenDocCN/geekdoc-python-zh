# SQLAlchemy–一些常见问题

> 原文：<https://www.pythoncentral.io/sqlalchemy-faqs/>

## 常见问题

在我们深入 SQLAlchemy 之前，让我们回答一系列关于 ORM 的问题:

*   能否**阻止 SQLAlchemy 自动创建模式**？相反，可以将 SQLAlchemy 模型绑定到现有的模式吗？
*   与编写原始 SQL 相比，使用 SQLAlchemy 时有没有**性能开销？如果有，多少？**
*   如果您**没有足够的权限在数据库中创建表**，SQLAlchemy 会抛出异常吗？
*   **模式是如何修改的**？它是自动完成的还是你写代码来完成的？
*   有没有对**触发器**的支持？

在这篇文章中，我们将回答所有的问题。一些问题将被详细讨论，而另一些问题将在另一篇文章中被总结和讨论。

## SQLAlchemy 架构反射/自省

我们也可以指示一个`Table`对象从数据库中已经存在的相应数据库模式对象中加载关于自身的信息，而不是像前面的文章中所示的那样，使用`Base.metadata.create_all(engine)`从 SQLAlchemy 中自动创建一个模式。

让我们创建一个示例 sqlite3 数据库，其中一个表`person`存储一条记录:

```py

import sqlite3
conn = sqlite3 . connect(" example . db ")
c = conn . cursor()
c . execute(' '
创建表人
(姓名文本，邮件文本)
' ' ')
c . execute(" INSERT INTO person VALUES(' John '，' John @ example . com '))"
c . close()

```

现在我们可以使用`Table`构造函数中的参数`autoload`和`autoload_with`来反映表`person`的结构。

```py

>>> from sqlalchemy import create_engine, MetaData, Table

>>>

>>> engine = create_engine('sqlite:///example.db')

>>> meta = MetaData(bind=engine)

>>> person = Table("person", meta, autoload=True, autoload_with=engine)

>>> person

Table('person', MetaData(bind=Engine(sqlite:///example.db)), Column(u'name', TEXT(), table=), Column(u'email', TEXT(), table=), schema=None)

>>> [c.name for c in person.columns]

[u'name', u'email']

```

我们还可以使用`MetaData.reflect`方法反映数据库中的所有表。

```py

>>> meta = MetaData()

>>> meta.reflect(bind=engine)

>>> person = meta.tables['person']

>>> person

Table(u'person', MetaData(bind=None), Column(u'name', TEXT(), table=), Column(u'email', TEXT(), table=), schema=None)

```

尽管反射非常强大，但它也有其局限性。记住反射构造`Table`元数据只使用关系数据库中可用的信息是很重要的。当然，这样的过程不能恢复实际上没有存储在数据库中的模式的某些方面。不可用的方面包括但不限于:

1.  使用`Column`构造函数的`default`关键字定义的客户端默认值、Python 函数或 SQL 表达式。
2.  列信息，在`Column.info`字典中定义。
3.  `Column`或`Table`的`.quote`设定值。
4.  特定`Sequence`与给定`Column`的关联。

SQLAlchemy 最近的改进允许反映视图、索引和外键选项等结构。像检查约束、表注释和触发器这样的结构没有被反映出来。

## SQLAlchemy 的性能开销

由于 SQLAlchemy 在将变更(即`session.commit()`)同步到数据库时使用工作单元模式，因此它不仅仅是像在原始 SQL 语句中那样“插入”数据。它跟踪对会话对象所做的更改，并维护所有对象的标识映射。它还执行相当数量的簿记，并维护任何 CRUD 操作的完整性。总的来说，工作单元自动化了将复杂对象图持久化到关系数据库中的任务，而无需编写显式的过程性持久化代码。当然，如此先进的自动化是有代价的。

因为 SQLAlchemy 的 ORM 不是为处理批量插入而设计的，所以我们可以写一个例子来测试它对原始 SQL 的效率。除了批量插入测试用例的 ORM 和原始 SQL 实现，我们还实现了一个使用 SQLAlchemy 核心系统的版本。由于 SQLAlchemy 的核心是原始 SQL 之上的一个抽象薄层，我们希望它能达到与原始 SQL 相当的性能水平。

```py

import time

import sqlite3
from sqlalchemy . ext . declarative import declarative _ base
from sqlalchemy import Column，Integer，String，create _ engine
from sqlalchemy . ORM import scoped _ session，sessionmaker
base = declarative _ base()
session = scoped _ session(session maker())
类 User(Base):
_ _ tablename _ _ = " User "
id = Column(Integer，primary _ key = True)
name = Column(String(255))
def init _ db(dbname = ' SQLite:///example . db '):
engine = create _ engine(dbname，echo = False)
session . remove()
session . configure(bind = engine，autoflush=False，expire _ on _ commit = False)
base . metadata . drop _ all(engine)
base . metadata . create _ all(engine)
返回引擎
def test _ SQLAlchemy _ ORM(number _ of _ records = 100000):
init _ db()
start = time . time()
for I in range(number _ of _ records):
User = User()
User . NAME = ' NAME '+str(I)
session . add(User)
session . commit()
end = time . time()
print " SQLAlchemy ORM:在{1}秒内插入{0}条记录"。格式(
 str(记录数)，str(结束-开始)
)
def test _ sqlalchemy _ core(number _ of _ records = 100000):
engine = init _ db()
start = time . time()
engine . execute(
User。__ 表 _ _。insert()，
[{ " NAME ":" NAME "+str(I)} for I in range(number _ of _ records)]
)
end = time . time()
打印“SQLAlchemy Core:在{1}秒内插入{0}条记录”。格式(
 str(记录数)，str(结束-开始)
)
def init _ sqlite3(dbname = " sqlite3 . db "):
conn = sqlite3 . connect(dbname)
cursor = conn . cursor()
cursor . execute(" DROP TABLE IF EXISTS user ")
cursor . execute(" CREATE TABLE user(id INTEGER NOT NULL，name VARCHAR(255)，PRIMARY KEY(id))"
conn . commit()
return conn
def test _ sqlite3(number _ of _ records = 100000):
conn = init _ sqlite3()
cursor = conn . cursor()
start = time . time()
for I in range(number _ of _ records):
cursor . execute(" INSERT INTO user(name)VALUES(？)"，(" NAME " + str(i)，))
conn . commit()
end = time . time()
打印" sqlite3:Insert { 0 } records in { 1 } seconds "。格式(
 str(记录数)，str(结束-开始)
)
if _ _ name _ _ = = " _ _ main _ _ ":
test _ sqlite3()
test _ sqlalchemy _ core()
test _ sqlalchemy _ ORM()

```

在前面的代码中，我们比较了使用原始 SQL、SQLAlchemy 的 Core 和 SQLAlchemy 的 ORM 向 sqlite3 数据库中批量插入 100000 条用户记录的性能。如果您运行该代码，您将得到类似如下的输出:

```py

$ python orm_performance_overhead.py

sqlite3: Insert 100000 records in 0.226176977158 seconds

SQLAlchemy Core: Insert 100000 records in 0.371157169342 seconds

SQLAlchemy ORM: Insert 100000 records in 10.1760079861 seconds

```

注意，核心和原始 SQL 获得了相当的插入速度，而 ORM 比其他两个慢得多。尽管看起来 ORM 会导致很大的性能开销，但是请记住，只有当有大量数据需要插入时，开销才会变得很大。由于大多数 web 应用程序在一个请求-响应周期中运行小的 CRUD 操作，由于额外的便利和更好的可维护性，最好使用 ORM 而不是核心。

## SQLAlchemy 和数据库权限

到目前为止，我们的示例在 sqlite3 数据库中运行良好，该数据库没有细粒度的访问控制，如用户和权限管理。如果我们想在 MySQL 或 PostgreSQL 中使用 SQLAlchemy 怎么办？当连接到数据库的用户没有足够的权限创建表、索引等时会发生什么？？SQLAlchemy 会抛出数据库访问异常吗？

让我们用一个例子来测试当用户没有足够的权限时 SQLAlchemy 的 ORM 的行为。首先，我们创建一个测试数据库“test_sqlalchemy”和一个测试用户“sqlalchemy”。

```py

$ psql

postgres=# create database test_sqlalchemy;

CREATE DATABASE

postgres=# create user sqlalchemy with password 'sqlalchemy';

CREATE ROLE

postgres=# grant all on database test_sqlalchemy to sqlalchemy;

GRANT

```

目前，测试用户“sqlalchemy”拥有对测试数据库“test_sqlalchemy”的所有访问权限。因此，我们希望数据库初始化调用成功，并将一条记录插入数据库“test_sqlalchemy”。

```py

import time

import sqlite3
from sqlalchemy . ext . declarative import declarative _ base
from sqlalchemy import Column，Integer，String，create _ engine
from sqlalchemy . ORM import scoped _ session，sessionmaker
base = declarative _ base()
session = scoped _ session(session maker())
类 User(Base):
_ _ tablename _ _ = " User "
id = Column(Integer，primary _ key = True)
name = Column(String(255))
def init _ db(dbname):
engine = create _ engine(dbname，echo = False)
session . configure(bind = engine，autoflush=False，expire _ on _ commit = False)
base . metadata . create _ all(engine)
return engine
if _ _ name _ _ = = " _ _ main _ _ ":
init _ db(" PostgreSQL://sqlalchemy:sqlalchemy @ localhost/test _ sqlalchemy ")
u = User(name = " other _ User ")
session . add(u)
session . commit()
session . close()
执行完脚本后，您可以检查“user”表中是否有新的`User`记录。

```

$ psql test_sqlalchemy

psql (9.3.3)

Type "help" for help.
test _ sqlalchemy = # select * from " user "；
 id |姓名
 - + - 
 1 |其他 _ 用户

```py

现在假设我们取消测试用户“sqlalchemy”的插入权限。那么我们应该预料到运行相同的代码将会失败并出现异常。

```

# inside a psql shell

test_sqlalchemy=# revoke INSERT on "user" from sqlalchemy;

REVOKE
# inside a bash shell
$ python permission_example.py
trace back(最近一次调用 last): 
文件“permission _ example . py”，第 32 行，在
 session.commit() 
文件“/home/vagger/python central/local/lib/python 2.7/site-packages/sqlalchemy/ORM/scoping . py”，第 149 行，在 do 
返回 getattr(self.registry()，name)(*args，**kwargs) 
文件“/home......
File "/home/vagger/python central/local/lib/python 2.7/site-packages/sqlalchemy/engine/default . py "，第 425 行，在 do _ execute
cursor . execute(statement，parameters)
sqlalchemy . exc . programming error:(programming error)对关系用户的权限被拒绝
' INSERT INTO " user "(name)VALUES(%)(name)s)返回" user "。id' {'name': 'other_user'} 

```py

如您所见，抛出了一个异常，表明我们没有权限将记录插入到关系用户中。
SQLAlchemy 的模式迁移
至少有两个库可用于执行 SQLAlchemy 迁移:`migrate` [文档链接](https://sqlalchemy-migrate.readthedocs.org/en/latest/ "documentation link")和`alembic` [文档链接](http://alembic.readthedocs.org/en/latest/ "documentation link")。
由于`alembic`是 SQLAlchemy 的作者写的，并且是积极开发的，所以我们推荐你用它来代替`migrate`。`alembic`不仅允许您手动编写迁移脚本，它还提供了自动生成脚本的方法。我们将在另一篇文章中进一步探讨如何使用`alembic`。
SQLAlchemy 对触发器的支持
可以使用定制的 DDL 结构创建 SQL 触发器，并与 SQLAlchemy 的事件挂钩。虽然它不是对触发器的直接支持，但它很容易实现并插入到任何系统中。我们将在另一篇文章中研究自定义 DDL 和事件。
提示和总结
在本文中，我们从 SQL 数据库管理员的角度回答了一些关于 SQLAlchemy 的常见问题。虽然 SQLAlchemy 默认为您创建一个数据库模式，但是它也允许您反映现有的模式并为您生成`Table`对象。使用 SQLAlchemy 的 ORM 时会有性能开销，但在执行批量插入时这一点最明显，而大多数 web 应用程序执行相对较小的 CRUD 操作。如果您的数据库用户没有足够的权限在表上执行某些操作，SQLAlchemy 将抛出一个异常，显示您无法执行这些操作的确切原因。SQLAlchemy 有两个迁移库，强烈推荐使用`alembic`。尽管不直接支持触发器，但是您可以很容易地在原始 SQL 中编写它们，并使用自定义 DDL 和 SQLAlchemy 事件将它们连接起来。

```