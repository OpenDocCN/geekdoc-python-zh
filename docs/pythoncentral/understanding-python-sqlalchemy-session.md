# 了解 Python SQLAlchemy 的会话

> 原文：<https://www.pythoncentral.io/understanding-python-sqlalchemy-session/>

## 什么是 SQLAlchemy 会话？会话是做什么的？

SQLAlchemy 的核心概念之一是`Session`。一个`Session`建立并维护你的程序和数据库之间的所有对话。它代表所有已加载到其中的 Python 模型对象的中间区域。它是启动对数据库查询的入口点之一，查询结果被填充并映射到`Session`中的唯一对象。唯一对象是`Session`中唯一具有特定主键的对象。

一只`Session`的典型寿命如下:

*   构建了一个`Session`，此时它不与任何模型对象相关联。
*   `Session`接收查询请求，其结果被持久化/与`Session`相关联。
*   构建任意数量的模型对象，然后添加到`Session`，之后`Session`开始维护和管理这些对象。
*   一旦对`Session`中的对象进行了所有的修改，我们可以决定将`Session`中的修改`commit`到数据库中，或者将`Session`中的修改`rollback`。`Session.commit()`表示到目前为止对`Session`中的对象所做的更改将被保存到数据库中，而`Session.rollback()`表示这些更改将被丢弃。
*   `Session.close()`将关闭`Session`及其对应的连接，这意味着我们已经完成了对`Session`的操作，并希望释放与之关联的连接对象。

## 通过示例了解 SQLAlchemy 会话

让我们用一个简单的例子来说明如何使用`Session`将对象插入数据库。

```py

from sqlalchemy import Column, String, Integer, ForeignKey

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
类 User(Base):
_ _ tablename _ _ = ' User '
id = Column(Integer，primary _ key = True)
name = Column(String)
从 sqlalchemy 导入 create _ engine
engine = create _ engine(' SQLite:///')
从 sqlalchemy.orm 导入会话标记
#构造一个 sessionmaker 对象
 session = sessionmaker()
#将 sessionmaker 绑定到引擎
session . configure(Bind = engine)
#在数据库中创建所有由 base 的子类定义的
 #表，例如用户
Base . metadata . Create _ all(engine)

```

### 创建和保持会话对象

一旦我们有了一个`session`，我们就可以创建对象并将它们添加到`session`中。

```py

# Make a new Session object

s = session()

john = User(name='John')
#将用户 john 添加到会话对象
 s.add(john)
#将新用户 John 提交到数据库
 s.commit() 

```

让我们插入另一个用户 Mary，并在插入过程的每一步检查新对象的`id`。

```py

>>> mary = User(name='Mary')

>>> print(mary.id, mary.name)

(None, 'Mary')

>>> s.add(mary)

>>> print(mary.id, mary.name)

(None, 'Mary')

>>> s.commit()

>>> print(mary.id, mary.name)

(1, u'Mary')

```

注意在调用`s.commit()`之前`mary.id`是`None`。为什么？因为对象`mary`在构造并添加到`s`时还没有提交到数据库，所以它没有底层 SQLite 数据库分配的主键。一旦新对象`mary`被`s`提交，它就会被底层 SQLite 数据库赋予一个`id`值。

### 查询对象

一旦数据库中有了 John 和 Mary，我们就可以使用`Session`查询他们。

```py

>>> mary = s.query(User).filter(User.name == 'Mary').one()

>>> john = s.query(User).filter(User.name == 'John').one()

>>> mary.id

2

>>> john.id

1

```

如您所见，被查询的对象具有来自数据库的有效的`id`值。

### 更新对象

我们可以像更改普通 Python 对象的属性一样更改`Mary`的名称，只要我们记得在最后调用`session.commit()`即可。

```py

>>> mary.name = 'Mariana'

>>> s.commit()

>>> mary.name

u'Mariana'

>>> s.query(User).filter(User.name == 'Mariana').one()

>>>

>>> mary.name = 'Mary'

>>> s.commit()

>>> s.query(User).filter(User.name == 'Mariana').one()

Traceback (most recent call last):

......

sqlalchemy.orm.exc.NoResultFound: No row was found for one()

>>> s.query(User).filter(User.name == 'Mary').one()
删除对象
现在我们有两个`User`对象保存在数据库中，`Mary`和`John`。我们将通过调用会话对象的`delete()`来删除它们。

```

>>> s.delete(mary)

>>> mary.id

2

>>> s.commit()

>>> mary
> > > Mary . id
2
>T5>玛丽。_ sa _ instance _ state . persistent
False # Mary 不再持久存在于数据库中，因为她已经被会话删除

```py

由于`Mary`已被会话标记为删除，并且该删除已被会话提交到数据库中，我们将无法再在数据库中找到`Mary`。

```

>>> mary = s.query(User).filter(User.name == 'Mary').one()

Traceback (most recent call last):

......

    raise orm_exc.NoResultFound("No row was found for one()")

sqlalchemy.orm.exc.NoResultFound: No row was found for one()

```py

会话对象状态
因为我们已经看到了一个`Session`对象的运行，所以了解会话对象的四种不同状态也很重要:

*   *Transient* :不包含在会话中的实例，还没有被持久化到数据库中。
*   *Pending* :一个已经添加到会话中但还没有持久化到数据库中的实例。它将在下一个`session.commit()`保存到数据库中。
*   *Persistent* :持久化到数据库中的实例，也包含在会话中。您可以通过将模型对象提交到数据库或从数据库中查询它来使其持久化。
*   *Detached* :一个实例已经被持久化到数据库中，但是没有包含在任何会话中。

让我们用`sqlalchemy.inspect`来看看一个新的`User`对象`david`的状态。

```

>>> from sqlalchemy import inspect

>>> david = User(name='David')

>>> ins = inspect(david)

>>> print('Transient: {0}; Pending: {1}; Persistent: {2}; Detached: {3}'.format(ins.transient, ins.pending, ins.persistent, ins.detached))

Transient: True; Pending: False; Persistent: False; Detached: False

>>> s.add(david)

>>> print('Transient: {0}; Pending: {1}; Persistent: {2}; Detached: {3}'.format(ins.transient, ins.pending, ins.persistent, ins.detached))

Transient: False; Pending: True; Persistent: False; Detached: False

>>> s.commit()

>>> print('Transient: {0}; Pending: {1}; Persistent: {2}; Detached: {3}'.format(ins.transient, ins.pending, ins.persistent, ins.detached))

Transient: False; Pending: False; Persistent: True; Detached: False

>>> s.close()

>>> print('Transient: {0}; Pending: {1}; Persistent: {2}; Detached: {3}'.format(ins.transient, ins.pending, ins.persistent, ins.detached))

Transient: False; Pending: False; Persistent: False; Detached: True

```py

注意在插入过程的每个步骤中，`david`的状态从*瞬时*到*分离*的变化。熟悉对象的这些状态是很重要的，因为轻微的误解可能会导致程序中难以发现的错误。
作用域会话与普通会话
到目前为止，我们从`sessionmaker()`调用构建的用于与数据库通信的会话对象是一个普通的会话。如果您第二次调用`sessionmaker()`，您将获得一个新的会话对象，其状态独立于前一个会话。例如，假设我们有两个按以下方式构造的会话对象:

```

from sqlalchemy import Column, String, Integer, ForeignKey

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
类 User(Base):
_ _ tablename _ _ = ' User '
id = Column(Integer，primary _ key = True)
name = Column(String)
从 sqlalchemy 导入 create _ engine
engine = create _ engine(' SQLite:///')
从 sqlalchemy.orm 导入 session maker
session = session maker()
session . configure(bind = engine)
base . metadata . create _ all(engine)
#构造第一个会话对象
 s1 = session() 
 #构造第二个会话对象
 s2 = session() 

```py

然后，我们将无法同时向`s1`和`s2`添加同一个`User`对象。换句话说，一个对象最多只能附加一个唯一的`session`对象。

```

>>> jessica = User(name='Jessica')

>>> s1.add(jessica)

>>> s2.add(jessica)

Traceback (most recent call last):

......

sqlalchemy.exc.InvalidRequestError: Object '' is already attached to session '2' (this is '3')

```py

然而，如果会话对象是从一个`scoped_session`对象中检索的，那么我们就没有这样的问题，因为`scoped_session`对象为同一个会话对象维护了一个注册表。

```

>>> session_factory = sessionmaker(bind=engine)

>>> session = scoped_session(session_factory)

>>> s1 = session()

>>> s2 = session()

>>> jessica = User(name='Jessica')

>>> s1.add(jessica)

>>> s2.add(jessica)

>>> s1 is s2

True

>>> s1.commit()

>>> s2.query(User).filter(User.name == 'Jessica').one()
请注意，`s1`和`s2`是同一个会话对象，因为它们都是从一个维护同一个会话对象的引用的`scoped_session`对象中检索的。
总结和提示
在本文中，我们回顾了如何使用`SQLAlchemy`的`Session`以及模型对象的四种不同状态。由于*工作单元*是 SQLAlchemy 中的一个核心概念，所以完全理解并熟悉如何使用`Session`和模型对象的四种不同状态是至关重要的。在下一篇文章中，我们将向您展示如何利用`Session`来管理复杂的模型对象并避免常见的错误。

```py

```