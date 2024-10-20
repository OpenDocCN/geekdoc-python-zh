# Python 的 SQLAlchemy vs 其他 ORM

> 原文：<https://www.pythoncentral.io/sqlalchemy-vs-orms/>

Update: A review of PonyORM has been added.

## Python ORMs 概述

作为一门优秀的语言，除了 [SQLAlchemy](http://www.sqlalchemy.org/) 之外，Python 还有很多 ORM 库。在本文中，我们将看看几个流行的替代 [ORM](https://en.wikipedia.org/wiki/Object-relational_mapping) 库，以更好地理解 Python ORM 的全貌。通过编写一个脚本来读写一个包含两个表`person`和`address`的简单数据库，我们将更好地理解每个 ORM 库的优缺点。

## SQLObject

[SQLObject](http://sqlobject.org/ "SQLObject") 是一个 Python ORM，在 SQL 数据库和 Python 之间映射对象。由于与 Ruby on Rails 的 ActiveRecord 模式相似，它在编程社区中越来越受欢迎。SQLObject 的第一个版本发布于 2002 年 10 月。它在 LGPL 的许可下。

在 [SQLObject](http://sqlobject.org/ "SQLObject") 中，数据库概念以一种非常类似于 SQLAlchemy 的方式映射到 Python 中，其中表被映射为类，行被映射为实例，列被映射为属性。它还提供了一种基于 Python 对象的查询语言，使得 [SQL](https://en.wikipedia.org/wiki/SQL) 更加抽象，从而为应用程序提供了数据库不可知论。

```py

$ pip install sqlobject

Downloading/unpacking sqlobject

Downloading SQLObject-1.5.1.tar.gz (276kB): 276kB downloaded

Running setup.py egg_info for package sqlobject
警告:未找到匹配“*”的文件。html' 
警告:未找到匹配' * '的文件。css' 
警告:没有找到与' docs/*匹配的文件。html' 
警告:没有找到匹配' * '的文件。“tests”目录下的“py”要求已经满足(使用-升级来升级):form encode>= 1 . 1 . 1 in/Users/xiaonuogantan/python 2-workspace/lib/python 2.7/site-packages(来自 sqlobject) 
安装收集的包:sqlobject 
运行 setup.py install for sqlobject 
将 build/scripts-2.7/SQL object-admin 的模式从 644 更改为 755 
更改 build/scripts-2.7 的模式
警告:未找到匹配“*”的文件。html' 
警告:未找到匹配' * '的文件。css' 
警告:没有找到与' docs/*匹配的文件。html' 
警告:没有找到匹配' * '的文件。“tests”目录下的“py”
将/Users/xiaonuogantan/python 2-workspace/bin/SQL object-admin 的模式更改为 755 
将/Users/xiaonuogantan/python 2-workspace/bin/SQL object-convertOldURI 的模式更改为 755 
成功安装 sqlobject 
清理...

```

```py

>>> from sqlobject import StringCol, SQLObject, ForeignKey, sqlhub, connectionForURI

>>> sqlhub.processConnection = connectionForURI('sqlite:/:memory:')

>>>

>>> class Person(SQLObject):

...     name = StringCol()

...

>>> class Address(SQLObject):

...     address = StringCol()

...     person = ForeignKey('Person')

...

>>> Person.createTable()

[]

>>> Address.createTable()

[]

```

上面的代码创建了两个简单的表:`person`和`address`。要在这两个表中创建或插入记录，我们只需像普通 Python 对象一样实例化一个人和一个地址:

```py

>>> p = Person(name='person')

>>> a = Address(address='address', person=p)

>>> p
>>>答

[/python] 

为了从数据库中获取或检索新记录，我们使用附加到`Person`和`Address`类的神奇的`q`对象:

```

>>> persons = Person.select(Person.q.name == 'person')

>>> persons
> > > list(persons)
[]
>>>P1 = persons[0]
>>>P1 = = p
True
>>>addresses = address . select(address . q . person = = P1)
>>>addresses
>>>列表(地址)
 [
]

>>> a1 = addresses[0]

>>> a1 == a

True

[/python] 

## 暴风雨

 [Storm](https://storm.canonical.com/ "Storm") 是一个 Python ORM，在一个或多个数据库和 Python 之间映射对象。它允许开发人员跨多个数据库表构建复杂的查询，以支持对象信息的动态存储和检索。它是由 Ubuntu 背后的公司 Canonical Ltd .用 Python 开发的，用于 Launchpad 和景观应用程序，随后在 2007 年作为自由软件发布。该项目是在 LGPL 许可证下发布的，贡献者需要向 Canonical 分配版权。

像 SQLAlchemy 和 SQLObject 一样， [Storm](https://storm.canonical.com/ "Storm") 也将表映射到类，将行映射到实例，将列映射到属性。与其他两个相比，Storm 的表类不必是特定于框架的特殊超类的子类。在 SQLAlchemy 中，每个表类都是`sqlalchemy.ext.declarative.declarative_bas`的子类。在 SQLObject 中，每个表类都是`sqlobject.SQLObject`的子类。

与 SQLAlchemy 类似，Storm 的`Store`对象充当后端数据库的代理，所有操作都缓存在内存中，一旦在存储上调用方法 commit，就将`committed`缓存到数据库中。每个存储拥有自己的映射 Python 数据库对象集，就像一个 SQLAlchemy 会话拥有不同的 Python 对象集一样。

Storm 的具体版本可以从[下载页面](https://launchpad.net/storm/+download "Storm Downloads")下载。在本文中，示例代码是用 Storm 版编写的。

```py

>>> from storm.locals import Int, Reference, Unicode, create_database, Store

>>>

>>>

>>> db = create_database('sqlite:')

>>> store = Store(db)

>>>

>>>

>>> class Person(object):

...     __storm_table__ = 'person'

...     id = Int(primary=True)

...     name = Unicode()

...

>>>

>>> class Address(object):

...     __storm_table__ = 'address'

...     id = Int(primary=True)

...     address = Unicode()

...     person_id = Int()

...     person = Reference(person_id, Person.id)

...

```

上面的代码创建了一个内存 sqlite 数据库和一个引用该数据库对象的*存储库*。Storm store 类似于 SQLAlchemy DBSession 对象，两者都管理附加到它们的实例对象的生命周期。例如，下面的代码创建一个人和一个地址，并通过刷新存储将这两个记录插入到数据库中。

```py

>>> store.execute("CREATE TABLE person "

... "(id INTEGER PRIMARY KEY, name VARCHAR)")
> > > store.execute("创建表地址"
...”(id 整数主键，address VARCHAR，person_id 整数，“
..."外键(person_id)引用人员(id))")
> > > Person = Person()
>>>Person . name = u ' Person '
>>>打印人
> > > print "%r，%r" % (person.id，person.name) 
 None，u'person' #请注意，person.id 为 None，因为 person 实例尚未附加到有效的数据库存储。
>>>store . add(person)
>>>
>>>print " % r，%r" % (person.id，person.name) 
 None，u'person' #由于 store 还没有将 Person 实例刷新到 sqlite 数据库中，person.id 仍然是 None。
>>>store . flush()
>>>print " % r，%r" % (person.id，person.name) 
 1，u'person' #现在 store 已经刷新了 person 实例，我们得到了 person 的 id 值。
>>>Address = Address()
>>>address.person = person
>>>Address . Address = ' Address '
>>>print " % r，%r" % (address.id，Address . person，address.address) 
 None，，' Address '
>>>Address . person = = person
True

```

为了获取或检索插入的 Person 和 Address 对象，我们调用`store.find()`来找到它们:

```py

>>> person = store.find(Person, Person.name == u'person').one()

>>> print "%r, %r" % (person.id, person.name)

1, u'person'

>>> store.find(Address, Address.person == person).one()
> > > address = store.find(Address，Address.person == person)。one() 
 > > > print "%r，%r" % (address.id，address.address) 
 1，u'address' 

```

## 姜戈氏 ORM

Django 是一个免费的开源 web 应用框架，它的 ORM 被紧密地嵌入到系统中。在最初发布之后，Django 因其简单的设计和易于使用的网络特性而变得越来越受欢迎。它于 2005 年 7 月在 BSD 许可下发布。由于 Django 的 ORM 紧密地构建在 web 框架中，所以不推荐在独立的非 Django Python 应用程序中使用它的 ORM，尽管这是可能的。

Django 是最流行的 Python web 框架之一，它有自己专用的 ORM。与 SQLAlchemy 相比，Django 的 ORM 更适合直接的 SQL 对象操作，它公开了数据库表和 Python 类之间简单直接的映射。

```py

$ django-admin.py startproject demo

$ cd demo

$ python manage.py syncdb

Creating tables ...

Creating table django_admin_log

Creating table auth_permission

Creating table auth_group_permissions

Creating table auth_group

Creating table auth_user_groups

Creating table auth_user_user_permissions

Creating table auth_user

Creating table django_content_type

Creating table django_session
您刚刚安装了 Django 的 auth 系统，这意味着您没有定义任何超级用户。您想现在创建一个吗？(是/否):否
安装自定义 SQL...
安装索引...
从 0 个固定设备
 $ python manage.py shell 
安装了 0 个对象
```

因为我们必须先创建一个项目才能执行 Django 的代码，所以我们在之前的 shell 中创建了一个 Django 项目“demo ”,并进入 Django shell 来测试我们的 ORM 示例。

```py

# demo/models.py

>>> from django.db import models

>>>

>>>

>>> class Person(models.Model):

...     name = models.TextField()

...

...     class Meta:

...         app_label = 'demo'

...

>>>

>>> class Address(models.Model):

...     address = models.TextField()

...     person = models.ForeignKey(Person)

...

...     class Meta:

...         app_label = 'demo'

```

上面的代码声明了两个 Python 类，`Person`和`Address`，每个类都映射到一个数据库表。在执行任何数据库操作代码之前，我们需要在本地 sqlite 数据库中创建表。

```py

python manage.py syncdb

Creating tables ...

Creating table demo_person

Creating table demo_address

Installing custom SQL ...

Installing indexes ...

Installed 0 object(s) from 0 fixture(s)

```

为了将一个人和一个地址插入数据库，我们实例化相应的对象并调用这些对象的`save()`方法。

```py

>>> from demo.models import Person, Address

>>> p = Person(name='person')

>>> p.save()

>>> print "%r, %r" % (p.id, p.name)

1, 'person'

>>> a = Address(person=p, address='address')

>>> a.save()

>>> print "%r, %r" % (a.id, a.address)

1, 'address'

```

为了获取或检索 person 和 address 对象，我们使用模型类的神奇的`objects`属性从数据库中获取对象。

```py

>>> persons = Person.objects.filter(name='person')

>>> persons

[]

>>> p = persons[0]

>>> print "%r, %r" % (p.id, p.name)

1, u'person'

>>> addresses = Address.objects.filter(person=p)

>>> addresses

[
]

>>> a = addresses[0]

>>> print "%r, %r" % (a.id, a.address)

1, u'address'

[/python] 

## 叫声类似“皮威”的鸟

peewee 是一个小型的、富有表现力的 ORM。与其他 ORM 相比，`peewee`专注于极简主义的原则，API 简单，库易于使用和理解。

```

pip install peewee

Downloading/unpacking peewee

Downloading peewee-2.1.7.tar.gz (1.1MB): 1.1MB downloaded

Running setup.py egg_info for package peewee
安装收集的包:peewee 
运行 setup.py 安装 peewee 
将 build/scripts-2.7/pwiz.py 的模式从 644 更改为 755
将/Users/xiaonuogantan/python 2-workspace/bin/pwiz . py 的模式更改为 755 
成功安装 peewee 
清理...

```py

为了创建数据库模型映射，我们实现了映射到相应数据库表的一个`Person`类和一个`Address`类。

```

>>> from peewee import SqliteDatabase, CharField, ForeignKeyField, Model

>>>

>>> db = SqliteDatabase(':memory:')

>>>

>>> class Person(Model):

...     name = CharField()

...

...     class Meta:

...         database = db

...

>>>

>>> class Address(Model):

...     address = CharField()

...     person = ForeignKeyField(Person)

...

...     class Meta:

...         database = db

...

>>> Person.create_table()

>>> Address.create_table()

```py

为了将对象插入数据库，我们实例化对象并调用它们的`save()`方法。从对象创建的角度来看，`peewee`和 Django 很像。

```

>>> p = Person(name='person')

>>> p.save()

>>> a = Address(address='address', person=p)

>>> a.save()

```py

为了从数据库中获取或检索对象，我们从它们各自的类中`select`对象。

```

>>> person = Person.select().where(Person.name == 'person').get()

>>> person

>>>

>>> print '%r, %r' % (person.id, person.name)

1, u'person'

>>> address = Address.select().where(Address.person == person).get()

>>> print '%r, %r' % (address.id, address.address)

1, u'address'

```py

## 波尼奥姆

 [PonyORM](https://ponyorm.org/) 允许您使用 Python 生成器查询数据库。这些生成器被翻译成 SQL，结果被自动映射成 Python 对象。将查询编写为 Python 生成器使得程序员可以很容易地快速构造某些查询。

例如，让我们使用 PonyORM 查询 SQLite 数据库中以前的`Person`和`Address`模型。

```

>>> from pony.orm import Database, Required, Set

>>>

>>> db = Database('sqlite', ':memory:')

>>>

>>>

>>> class Person(db.Entity):

...     name = Required(unicode)

...     addresses = Set("Address")

...

>>>

>>> class Address(db.Entity):

...     address = Required(unicode)

...     person = Required(Person)

...

>>> db.generate_mapping(create_tables=True)

```py

现在我们在内存中有了一个 SQLite 数据库，两个表映射到了`db`对象，我们可以将两个对象插入到数据库中。

```

>>> p = Person(name="person")

>>> a = Address(address="address", person=p)

>>> db.commit()

```py

调用`db.commit()`实际上将新对象`p`和`a`提交到数据库中。现在我们可以使用生成器语法查询数据库。

```

>>> from pony.orm import select

>>> select(p for p in Person if p.name == "person")[:]

[Person[1]]

>>> select(p for p in Person if p.name == "person")[:][0].name

u'person'

>>> select(a for a in Address if a.person == p)[:]

[Address[1]]

>>> select(a for a in Address if a.person == p)[:][0].address

u'address'

```py

## sqllcemy(SQL 语法)

 [SQLAlchemy](http://www.sqlalchemy.org/ "SQLAlchemy") 是在 MIT 许可下发布的 Python 编程语言的开源 SQL 工具包和 ORM。它最初于 2006 年 2 月发布，作者是迈克尔·拜尔。它提供了“一整套众所周知的企业级持久化模式，设计用于高效和高性能的数据库访问，适应于简单和 Pythonic 化的领域语言”。它采用了数据映射模式(像 Java 中的 Hibernate)而不是活动记录模式(像 Ruby on Rails 中的模式)。

SQLAlchemy 的工作单元原则使得有必要将所有数据库操作代码限制在特定的数据库会话中，该会话控制该会话中每个对象的生命周期。与其他 ORM 类似，我们从定义`declarative_base()`的子类开始，以便将表映射到 Python 类。

```

>>> from sqlalchemy import Column, String, Integer, ForeignKey

>>> from sqlalchemy.orm import relationship

>>> from sqlalchemy.ext.declarative import declarative_base

>>>

>>> Base = declarative_base()

>>>

>>>

>>> class Person(Base):

...     __tablename__ = 'person'

...     id = Column(Integer, primary_key=True)

...     name = Column(String)

...

>>>

>>> class Address(Base):

...     __tablename__ = 'address'

...     id = Column(Integer, primary_key=True)

...     address = Column(String)

...     person_id = Column(Integer, ForeignKey(Person.id))

...     person = relationship(Person)

...

```py

在我们编写任何数据库代码之前，我们需要为我们的数据库会话创建一个数据库引擎。

```

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///')

```py

一旦我们创建了一个数据库引擎，我们就可以继续创建一个数据库会话，并为之前定义为`Person`和`Address`的所有数据库类创建表。

```

>>> from sqlalchemy.orm import sessionmaker

>>> session = sessionmaker()

>>> session.configure(bind=engine)

>>> Base.metadata.create_all(engine)

```py

现在，`session`对象变成了我们的工作单元构造器，所有后续的数据库操作代码和对象都将被附加到一个通过调用它的`__init__()`方法构建的 db 会话上。

```

>>> s = session()

>>> p = Person(name='person')

>>> s.add(p)

>>> a = Address(address='address', person=p)

>>> s.add(a)

```py

为了获取或检索数据库对象，我们从 db session 对象中调用`query()`和`filter()`方法。

```

>>> p = s.query(Person).filter(Person.name == 'person').one()

>>> p
> > > print "%r，%r" % (p.id，p.name) 
 1、' person '
>>>a = s . query(地址)。filter(Address.person == p)。一()
 > > >打印“%r，% r”%(a . id，a.address) 
 1、‘地址’

```py

请注意，到目前为止，我们还没有提交对数据库的任何更改，因此新的 person 和 address 对象实际上还没有存储在数据库中。调用`s.commit()`将实际提交更改，即向数据库中插入一个新的人和一个新的地址。

```

>>> s.commit()

>>> s.close()

```py

## Python ORMs 之间的比较

对于本文中介绍的每个 Python ORM，我们将在这里列出它们的优缺点:

### SQLObject

**优点:**

1.  采用了易于理解的 ActiveRecord 模式

2.  相对较小的代码库

**缺点:**

1.  方法和类的命名遵循 Java 的 camelCase 风格

2.  不支持数据库会话来隔离工作单元

### 暴风雨

**优点:**

1.  一个简洁、轻量级的 API，可以缩短学习曲线，实现长期可维护性

2.  不需要特殊的类构造函数，也不需要命令式基类

**缺点:**

1.  迫使程序员编写手动创建表的 DDL 语句，而不是从模型类中自动派生出来

2.  Storm 的贡献者必须将他们贡献的版权给 Canonical 有限公司。

### 姜戈氏 ORM

**优点:**

1.  易于使用，学习曲线短

2.  与 Django 紧密集成，使其成为处理 Django 数据库时的 de-factor 标准

**缺点:**

1.  不能很好地处理复杂的查询；迫使开发人员回到原始 SQL

2.  与 Django 紧密结合；这使得它很难在 Django 上下文之外使用

### 叫声类似“皮威”的鸟

**优点:**

1.  一个 Django-ish API；使其易于使用

2.  轻量级实现；使其易于与任何 web 框架集成

**缺点:**

1.  不支持自动模式迁移

2.  编写多对多查询并不直观

### sqllcemy(SQL 语法)

**优点:**

1.  企业级 APIs 使代码健壮且适应性强

2.  灵活的设计；使得编写复杂的查询变得轻松

**缺点:**

1.  工作单元的概念并不常见

2.  一个重量级的 API 导致漫长的学习曲线

### 波尼奥姆

**优点:**

1.  用于编写查询的非常方便的语法

2.  自动查询优化

3.  简化的设置和使用

**缺点:**

1.  不是为同时处理数十万或数百万条记录而设计的

## 总结和提示

与其他 ORM 相比，SQLAlchemy 突出了它对工作单元概念的关注，这在您编写 SQLAlchemy 代码时非常普遍。最初，DBSession 的概念可能很难理解和正确使用，但稍后您会体会到额外的复杂性，它将与数据库提交时间相关的意外错误减少到几乎为零。在 SQLAlchemy 中处理多个数据库可能很棘手，因为每个数据库会话都被限制在一个数据库连接中。然而，这种限制实际上是一件好事，因为它迫使您认真考虑多个数据库之间的交互，并使调试数据库交互代码变得更容易。

在以后的文章中，我们将全面探索 SQLAlchemy 更高级的用例，以真正掌握其强大的 API。

```

```py

```