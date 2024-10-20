# SQLAlchemy 表达式语言，高级用法

> 原文：<https://www.pythoncentral.io/sqlalchemy-expression-language-advanced/>

## 表达语言

SQLAlchemy 的核心组件之一是表达式语言。它允许程序员在 Python 结构中指定 SQL 语句，并在更复杂的查询中直接使用这些结构。由于表达式语言是后端中立的，并且全面涵盖了原始 SQL 的各个方面，所以它比 SQLAlchemy 中的任何其他组件都更接近原始 SQL。在本文中，我们将使用一个三表数据库来说明表达式语言的强大功能。

## 数据库模型

假设我们想要对多个购物车进行建模，每个购物车都由一个用户创建，并存储多种产品。从规范中，我们可以推断出一个用户拥有多个购物车，一个购物车包含多个产品，一个产品可以包含在多个购物车中。因此，我们希望在`ShoppingCart`和`Product`之间建立多对多的关系，在`User`和`ShoppingCart`之间建立一对多的关系。让我们创建数据库模型:

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
class ShoppingCartProductLink(Base):
_ _ tablename _ _ = ' shopping _ cart _ product _ link '
shopping _ cart _ id = Column(Integer，ForeignKey('shopping_cart.id ')，primary_key=True)
product _ id = Column(Integer，ForeignKey('product.id ')，primary _ key = True)
从 sqlalchemy 导入 create _ engine
engine = create _ engine(' SQLite:///')
从 sqlalchemy.orm 导入 session maker
DBSession = session maker()
DBSession . configure(bind = engine)
base . metadata . create _ all(engine)

```

## 创建用户、产品和购物车

现在让我们创建一个用户和几个产品。

```py

>>> session = DBSession()

>>> cpu = Product(name='CPU', price=300.00)

>>> motherboard = Product(name='Motherboard', price=150.00)

>>> coffee_machine = Product(name='Coffee Machine', price=30.00)

>>> john = User(name='John')

>>> session.add(cpu)

>>> session.add(motherboard)

>>> session.add(coffee_machine)

>>> session.add(john)

>>> session.commit()

>>> session.close()

```

在继续之前，让我们验证一下现在数据库中有一个用户和三个产品。

```py

>>> session = DBSession()

>>> cpu = session.query(Product).filter(Product.name == 'CPU').one()

>>> motherboard = session.query(Product).filter(Product.name == 'Motherboard').one()

>>> coffee_machine = session.query(Product).filter(Product.name == 'Coffee Machine').one()

>>> john = session.query(User).filter(User.name == 'John').one()

>>> session.close()

```

现在我们可以为用户`John`创建两个购物车。

```py

>>> session = DBSession()

>>> cpu = session.query(Product).filter(Product.name == 'CPU').one()

>>> motherboard = session.query(Product).filter(Product.name == 'Motherboard').one()

>>> coffee_machine = session.query(Product).filter(Product.name == 'Coffee Machine').one()

>>> john = session.query(User).filter(User.name == 'John').one()

>>> john_shopping_cart_computer = ShoppingCart(owner=john)

>>> john_shopping_cart_kitchen = ShoppingCart(owner=john)

>>> john_shopping_cart_computer.products.append(cpu)

>>> john_shopping_cart_computer.products.append(motherboard)

>>> john_shopping_cart_kitchen.products.append(coffee_machine)

>>> session.add(john_shopping_cart_computer)

>>> session.add(john_shopping_cart_kitchen)

>>> session.commit()

>>> session.close()

```

## 使用表达式语言查询数据库

现在我们在数据库中有了一个用户、三个产品和两个购物车，我们可以开始使用表达式语言了。首先，让我们编写一个查询来回答这个问题:哪些产品的价格高于$100.00？

```py

>>> from sqlalchemy import select

>>> product_higher_than_one_hundred = select([Product.id]).where(Product.price > 100.00)

>>>

>>> session = DBSession()

>>> session.query(Product).filter(Product.id.in_(product_higher_than_one_hundred)).all()

[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )]

>>> session.close()

```

然后，让我们编写一个查询来回答一个更复杂的问题:哪些购物车包含至少一个价格高于$100.00 的产品？

```py

>>> shopping_carts_with_products_higher_than_one_hundred = select([ShoppingCart.id]).where(

...     ShoppingCart.products.any(Product.id.in_(product_higher_than_one_hundred))

... )

>>> session = DBSession()

>>> session.query(ShoppingCart).filter(ShoppingCart.id.in_(shopping_carts_with_products_higher_than_one_hundred)).one()

( :John:[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )] )

>>> session.close()

```

然后，让我们编写一个查询来回答一个稍微不同的问题:哪些购物车中没有价格低于$100.00 的产品？

```py

>>> products_lower_than_one_hundred = select([Product.id]).where(Product.price < 100.00)
>>> from sqlalchemy import not_

>>> shopping_carts_with_no_products_lower_than_one_hundred = select([ShoppingCart.id]).where(

...     not_(ShoppingCart.products.any(Product.id.in_(products_lower_than_one_hundred)))

... )

>>> session = DBSession()

>>> session.query(ShoppingCart).filter(ShoppingCart.id.in_(

...     shopping_carts_with_no_products_lower_than_one_hundred)

... ).all()

[( :John:[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )] )]

>>> session.close()

```

或者，前一个问题可以以不同的方式形成:我们如何找到所有产品价格都高于$100.00 的购物车？

```py

>>> from sqlalchemy import and_

>>> shopping_carts_with_all_products_higher_than_one_hundred = select([ShoppingCart.id]).where(

...     and_(

...         ShoppingCartProductLink.product_id.in_(product_higher_than_one_hundred),

...         ShoppingCartProductLink.shopping_cart_id == ShoppingCart.id

...     )

... )

>>> session = DBSession()

>>> session.query(ShoppingCart).filter(ShoppingCart.id.in_(

...     shopping_carts_with_all_products_higher_than_one_hundred)

... ).all()

[( :John:[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )] )]

>>> session.close()

```

现在，我们可以就`Product.price`上的聚合提出一种不同的问题。例如，我们可以问:哪些购物车中产品的总价格高于 200 美元？

```py

>>> from sqlalchemy import func

>>> total_price_of_shopping_carts = select([

...     ShoppingCart.id.label('shopping_cart_id'),

...     func.sum(Product.price).label('product_price_sum')

... ]).where(

...     and_(

...         ShoppingCartProductLink.product_id == Product.id,

...         ShoppingCartProductLink.shopping_cart_id == ShoppingCart.id,

...     )

... ).group_by(ShoppingCart.id)

>>> session = DBSession()

>>> session.query(total_price_of_shopping_carts).all()

[(1, 450.0), (2, 30.0)]

>>> session.query(ShoppingCart).filter(

...     ShoppingCart.id == total_price_of_shopping_carts.c.shopping_cart_id,

...     total_price_of_shopping_carts.c.product_price_sum > 200.00

... ).all()

[( :John:[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )] )]

>>> session.query(ShoppingCart).filter(

...     ShoppingCart.id == total_price_of_shopping_carts.c.shopping_cart_id,

...     total_price_of_shopping_carts.c.product_price_sum < 200.00
... ).all()
[( :John:[( :u'Coffee Machine':30.0 )] )]
>>> session.close()

```

在前面的例子中，我们从构建一个 SQLAlchemy `selectable` `total_price_of_shopping_carts`开始，它的“列”是每个购物车的`ShoppingCart.id`和每个相应购物车中所有产品价格的总和。一旦我们有了这样一个`selectable`，就很容易编写查询来查找所有产品价格总和高于$200.00 的购物车。

## 潜在陷阱

到目前为止，我们的示例程序似乎运行得很好。但是，如果我们以非预期的方式编写和使用这些结构，从而意外地破坏了程序呢？SQLAlchemy 会通知我们程序有什么问题吗，以便我们调试它？

例如，列`Product.price`被定义为一个`Float`。如果我们创建一个带有字符串的`price`的`Product`对象会怎么样？SQLAlchemy 会因为价格输入的数据类型与定义不同而中断吗？让我们试一试。

```py

>>> session = DBSession()

>>> cpu = Product(name='CPU', price='0.15')

>>> session.add(cpu)

>>> session.commit()

>>> cpu = session.query(Product).filter(Product.name == 'CPU').one()

>>> cpu.price

0.15

```

因此，带有字符串 price 的产品 CPU 被成功地插入到数据库中。用一个根本不是数字的字符串来表示价格怎么样？

```py
> > > cpu_two = Product(name='CPU Two '，price = ' asdf ')
>>>session . add(CPU _ Two)
>>>session . commit()
...
sqlalchemy . exc . statement error:无法将字符串转换为浮点数:asdf(原始原因:ValueError:无法将字符串转换为浮点数:asdf) u'INSERT INTO product (name，price) VALUES(？, ?)'[{'price': 'asdf '，' name': 'CPU Two'}]
哎呀。现在 SQLAlchemy 引发了一个`StatementError`，因为“asdf”不能转换成一个`Float`。这是一个很好的特性，因为它消除了由粗心引起的潜在编程错误。
您可能还注意到，我们示例中的`filter()`方法使用了像`Product.name == 'CPU'`和`Product.price > 100.0`这样的表达式。这些表达式不是先被求值，然后将结果布尔值传递给`filter()`函数以获得实际的过滤结果吗？让我们用几个例子来验证`filter()`的行为。

```

>>> session.query(Product).filter(True).all()

[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 ), ( :u'Coffee Machine':30.0 )]

>>> session.query(Product).filter(Product.name='CPU').all()

  File "", line 1

SyntaxError: keyword can't be an expression

>>> session.query(Product).filter(Product.price > '100.0').all()

[( :u'CPU':300.0 ), ( :u'Motherboard':150.0 )]

```py

从上面的例子中，我们看到`filter()`确实接受像`True`这样简单的布尔值，它返回数据库中的所有产品。但是，它不接受像`Product.name = 'CPU'`这样在过滤器上下文中含义不明确的表达式。像`Product`构造函数一样，它也将一个字符串值`'100.0'`转换成一个浮点数，并根据最终标准过滤产品表。
现在让我们检查几个 SQLAlchemy API 看起来不太直观的例子。首先，`select()`语句似乎只接受一列作为第一个参数，就像`select([Product.id])`一样。如果我们能写出类似`select(Product.id)`的东西不是很好吗？

```

>>> products_lower_than_one_hundred = select(Product.id).where(Product.price < 100.00)
...
NotImplementedError: Operator 'getitem' is not supported on this expression
[/python]

Oops. SQLAlchemy does not like a single element as the first argument of select(). Remember, always pass in a list.
第二，一些 where 子句中的语法看起来不像 Pythonic 化的:`ShoppingCart.products.any(Product.id.in_(product_higher_than_one_hundred))`。如果我们能写出类似`ShoppingCart.products.any(Product.id in product_higher_than_one_hundred))`的东西不是很好吗？

```py

>>> shopping_carts_with_products_higher_than_one_hundred = select([ShoppingCart.id]).where(

...     ShoppingCart.products.any(Product.id in product_higher_than_one_hundred)

... )

...

TypeError: argument of type 'Select' is not iterable

```

因为 SQLAlchemy 的`'Select'`对象是不可迭代的，所以在`in`上下文中使用它不起作用。这看起来可能是一个缺点，但是它是有意义的，因为一个`'Select'`对象在 SQLAlchemy 中非常灵活。如示例所示，一个`'Select'`对象可以传递给任何一个`filter()`或`where()`，成为另一个`'Select'`对象的更复杂查询或事件的一部分。在这样的对象上支持`iterable`需要对底层实现进行大量的修改。
第三，`query()`的结果似乎是返回格式良好的对象，例如`( :u'CPU':300.0 )`作为一个`Product`对象的显示。它看起来不同于典型的物体，比如:

```py

>>> class C:

...   pass

...

>>> c = C()

>>> c
为什么？这是因为我们覆盖了`Product`的`__repr__()`方法，并且来自 Python 解释器的`print()`命令正在调用`Product`和`ShoppingCart`对象的结果数组上的`repr()`，这些对象调用每个相应类的实现的`__repr__()`。
最后，SQLAlchemy 为什么要实现自己的`Float`列类型？为什么他们不能重用 Python 内部的`float`类型？
简单来说，SQLAlchemy 是一个 ORM，ORM 使用定义的类型系统将 Python 结构映射到 SQL 结构，类型系统必须是数据库不可知的，这意味着它必须处理具有相同列类型的不同数据库后端。最长的答案是，模型中定义的每一列都必须由 SQLAlchemy 定义，定义/列类型实现自定义方法，这些方法由 SQLAlchemy 的低级 API 调用，以将 Python 构造转换为相应的 SQL 构造。
提示和总结
在本文中，我们使用了一个三表数据库来说明如何使用 SQLAlchemy 的表达式语言。需要记住的一点是，我们使用数学集合来指导我们编写 SQL 查询。每当我们遇到一个不小的问题，尤其是涉及多个表格的问题，我们应该分而治之，先回答问题的一部分。例如，问题“我们如何找到所有产品的价格总和高于$200.00 的购物车”可以分为以下几个部分:1 .我们如何计算产品价格的总和？(`func.sum()` ) 2。如何在一个`selectable`中列出所有的元组(`ShoppingCart.id`、`func.sum(Product.price)`)？3.我们如何使用`selectable`来编写实际的查询呢？

```

```py

```