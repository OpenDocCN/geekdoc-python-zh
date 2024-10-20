# SQLAlchemy 表达式语言，更高级的用法

> 原文：<https://www.pythoncentral.io/sqlalchemy-expression-language-advanced-usage/>

## 概观

在上一篇文章 [SQLAlchemy 表达式语言，高级用法](https://www.pythoncentral.io/sqlalchemy-expression-language-advanced/ "SQLAlchemy Expression Language, Advanced Usage")中，我们通过包含`User`、`ShoppingCart`和`Product`的三表数据库了解了 SQLAlchemy 表达式语言的强大功能。在本文中，我们将回顾 SQLAlchemy 中物化路径的概念，并使用它来实现产品包含关系，其中某些产品可能包含其他产品。例如，DSLR 相机包是一种产品，其可以包含主体、三脚架、镜头和一组清洁工具，而主体、三脚架、镜头和该组清洁工具中的每一个也是一种产品。在这种情况下，DSLR 相机包产品*包含*其他产品。

## 物化路径

`Materialized Path`是一种在关系数据库中存储分层数据结构(通常是树)的方法。它可以用来处理数据库中任何类型的实体之间的层次关系。`sqlamp`是一个第三方 SQLAlchemy 库，我们将使用它来演示如何建立一个包含基于关系的分层数据结构的产品。要安装`sqlamp`，在您的 shell 中运行以下命令:

```py

$ pip install sqlamp

Downloading/unpacking sqlamp

...

Successfully installed sqlamp

Cleaning up...

```

首先，让我们回顾一下我们在上一篇文章中所做的事情。

```py

from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, Float

from sqlalchemy.orm import relationship, backref

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
类 User(Base):
_ _ tablename _ _ = ' User '
id = Column(Integer，primary _ key = True)
name = Column(String)
class shopping cart(Base):
_ _ tablename _ _ = ' shopping _ cart '
id = Column(Integer，primary _ key = True)
owner _ id = Column(Integer，foreign key(User . id))
owner = relationship(
User，backref = backref(' shopping _ carts '，use list = True)
)
products = relationship(
' Product '，
secondary = ' shopping _ cart _ Product _ link '
)
defr})”。格式(购物车、自助)
class Product(Base):
_ _ tablename _ _ = ' Product '
id = Column(Integer，primary _ key = True)
name = Column(String)
#使用浮点数不是对货币值建模的正确方式。
 #我们将在另一篇文章中探讨这个话题。
price = Column(Float)
shopping _ carts = relationship(
' shopping cart '，
secondary = ' shopping _ cart _ product _ link '
)
def _ _ repr _ _(self):
return '({ 0 }:{ 1 . name！r}:{1.price！r})”。格式(产品、自身)
类 ShoppingCartProductLink(Base):
_ _ tablename _ _ = ' shopping _ cart _ product _ link '
shopping _ cart _ id = Column(Integer，ForeignKey('shopping_cart.id ')，primary _ key = True)
product _ id = Column(Integer，ForeignKey('product.id ')，primary_key=True) 

```

我们定义了四个模型，`User`表示一组用户，`Product`表示一组产品，`ShoppingCart`表示一组购物车，每个购物车都由一个`User`拥有并包含多个`Product`，还有`ShoppingCartProductLink`，它是一个连接`Product`和`ShoppingCart`的链接表。

然后，让我们将`sqlamp`引入模型类，看看我们如何使用它来为`Product` s 创建一个物化的路径。

```py

import sqlamp
from sqlalchemy 导入列，DateTime，String，Integer，ForeignKey，Float 
 from sqlalchemy.orm 导入关系，back ref
from sqlalchemy . ext . declarative import declarative _ base
Base = declarative_base(元类=sqlamp。DeclarativeMeta)
类 User(Base):
_ _ tablename _ _ = ' User '
id = Column(Integer，primary _ key = True)
name = Column(String)
class shopping cart(Base):
_ _ tablename _ _ = ' shopping _ cart '
id = Column(Integer，primary _ key = True)
owner _ id = Column(Integer，foreign key(User . id))
owner = relationship(
User，backref = backref(' shopping _ carts '，use list = True)
)
products = relationship(
' Product '，
secondary = ' shopping _ cart _ Product _ link '
)
defr})”。格式(购物车、自助)
class Product(Base):
_ _ tablename _ _ = ' Product '
# _ _ MP _ manager _ _ 指定产品的哪个字段是物化路径管理器，
 #用于管理产品的子代和祖先的查询。
_ _ MP _ manager _ _ = ' MP '
id = Column(Integer，primary _ key = True)
name = Column(String)
#使用浮点数不是对货币值建模的正确方式。我们将在另一篇文章中探讨这个话题。
price = Column(Float)
shopping _ carts = relationship(
' shopping cart '，
secondary = ' shopping _ cart _ product _ link '
)
#使用自引用外键引用包含此产品的父产品
 #。
 parent_id = Column(Integer，foreign key(' Product . id ')
parent = relationship(' Product '，remote _ side =[id])
def _ _ repr _ _(self):
return '({ 0 }:{ 1 . name！r}:{1.price！r})”。格式(产品、自身)
类 ShoppingCartProductLink(Base):
_ _ tablename _ _ = ' shopping _ cart _ product _ link '
shopping _ cart _ id = Column(Integer，ForeignKey('shopping_cart.id ')，primary _ key = True)
product _ id = Column(Integer，ForeignKey('product.id ')，primary_key=True) 

```

注意，我们在`Product`模型中插入了一个新的外键`parent_id`和一个新的关系`parent`，并引入了一个新的类成员字段`__mp_manager__`。现在我们可以使用`Product.mp`来查询任何`product`的子代和祖先。

```py

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///')

>>>

>>>

>>> from sqlalchemy.orm import sessionmaker

>>> DBSession = sessionmaker()

>>> DBSession.configure(bind=engine)

>>> Base.metadata.create_all(engine)

>>>

>>>

>>> camera_package = Product(name='DSLR Camera Package', price=1600.00)

>>> tripod = Product(name='Camera Tripod', price=200.00, parent=camera_package)

>>> body = Product(name='Camera Body', price=400.00, parent=camera_package)

>>> lens = Product(name='Camera Lens', price=1000.00, parent=camera_package)

>>> session = DBSession()

>>> session.add_all([camera_package, tripod, body, lens])

>>> session.commit()

```

```py

>>> camera_package.mp.query_children().all()

[( :u'Camera Tripod':200.0 ), ( :u'Camera Body':400.0 ), ( :u'Camera Lens':1000.0 )]

>>> tripod.mp.query_ancestors().all()

[( :u'DSLR Camera Package':1600.0 )]

>>> lens.mp.query_ancestors().all()

[( :u'DSLR Camera Package':1600.0 )]

```

## 递归处理产品树

为了递归地遍历一棵`Product`树，我们可以调用`sqlamp.tree_recursive_iterator`并使用递归函数遍历树的所有后代。

```py

>>> def recursive_tree_processor(nodes):

...     for node, children in nodes:

...         print('{0}'.format(node.name))

...         if children:

...             recursive_tree_processor(children)

...

>>> query = camera_package.mp.query_descendants(and_self=True)

>>> recursive_tree_processor(

...     sqlamp.tree_recursive_iterator(query, Product.mp)

... )

DSLR Camera Package

Camera Tripod

Camera Body

Camera Lens

```

## 摘要

在本文中，我们使用前一篇文章的`Product`来说明如何使用`sqlamp`在 SQLAlchemy 中实现物化路径。通过简单地在`Product`中插入一个自引用外键和一个 *__mp_manager__* 字段，我们能够为`Product`实现一个分层数据结构。由于`sqlamp`是在 SQLAlchemy 之上编写的，它应该可以与 SQLAlchemy 支持的任何数据库后端一起工作。