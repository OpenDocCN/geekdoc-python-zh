# 用 Python 在 Django 中建模多态性

> 原文：<https://realpython.com/modeling-polymorphism-django-python/>

在关系数据库中建模多态性是一项具有挑战性的任务。在本文中，我们介绍了几种使用 Django 对象-关系映射( [ORM](https://en.wikipedia.org/wiki/Object-relational_mapping) )在关系数据库中表示多态对象的建模技术。

本中级教程是为已经熟悉 [Django](https://realpython.com/tutorials/django/) 基本设计的读者设计的。

**免费奖励:** ，提高您的 Django + Python web 开发技能。

## 什么是多态性？

多态是一个对象采取多种形式的能力。多态对象的常见例子包括电子商务网站中的事件流、不同类型的用户和产品。当单个实体需要不同的功能或信息时，使用多态模型。

在上面的例子中，所有事件都被记录下来以备将来使用，但是它们可以包含不同的数据。所有用户都需要能够登录，但是他们可能有不同的配置文件结构。在每个电子商务网站中，用户都希望将不同的产品放入购物车。

[*Remove ads*](/account/join/)

## 为什么对多态性建模具有挑战性？

有许多方法可以对多态性进行建模。有些方法使用 Django ORM 的标准特性，有些使用 Django ORM 的特殊特性。在对多态对象建模时，您将会遇到以下主要挑战:

*   **如何表示单个多态对象:**多态对象有不同的属性。Django ORM 将属性映射到数据库中的列。在这种情况下，Django ORM 应该如何将属性映射到表中的列呢？不同的对象应该驻留在同一个表中吗？你应该有多个表吗？

*   **如何引用多态模型的实例:**要利用数据库和 Django ORM 特性，您需要使用外键来引用对象。如何决定表示单个多态对象对您引用它的能力至关重要。

为了真正理解建模多态性的挑战，你要把一个小书店从它的第一个在线网站变成一个销售各种产品的大网店。在这个过程中，您将体验和分析使用 Django ORM 对多态性建模的不同方法。

**注意:**要学习本教程，建议您使用 PostgreSQL 后端、Django 2.x 和 Python 3。

也可以使用其他数据库后端。在使用 PostgreSQL 独有特性的地方，将为其他数据库提供一个替代方案。

## 天真的实现

你在镇上的一个好地方有一家书店，就在咖啡店旁边，你想开始在网上卖书。

你只卖一种产品:书。在您的在线商店中，您希望显示图书的详细信息，如名称和价格。你希望你的用户浏览网站并收集许多书籍，所以你还需要一个购物车。你最终需要把书运送给用户，所以你需要知道每本书的重量来计算运费。

让我们为您的新书店创建一个简单的模型:

```py
from django.contrib.auth import get_user_model
from django.db import models

class Book(models.Model):
    name = models.CharField(
        max_length=100,
    )
    price = models.PositiveIntegerField(
        help_text='in cents',
    )
    weight = models.PositiveIntegerField(
        help_text='in grams',
    )

    def __str__(self) -> str:
        return self.name

class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
    books = models.ManyToManyField(Book)
```

要创建一本新书，您需要提供名称、价格和重量:

>>>

```py
>>> from naive.models import Book
>>> book = Book.objects.create(name='Python Tricks', price=1000, weight=200)
>>> book
<Product: Python Tricks>
```

要创建购物车，首先需要将其与用户相关联:

>>>

```py
>>> from django.contrib.auth import get_user_model
>>> haki = get_user_model().create_user('haki')

>>> from naive.models import Cart
>>> cart = Cart.objects.create(user=haki)
```

然后，用户可以开始向其中添加项目:

>>>

```py
>>> cart.products.add(book)
>>> cart.products.all()
<QuerySet [<Book: Python Tricks>]>
```

**Pro**

*   **易于理解和维护:**对于单一类型的产品就足够了。

**Con**

*   **限于同质产品:**只支持属性集相同的产品。多态性根本不被捕获或允许。

[*Remove ads*](/account/join/)

## 稀疏模型

随着你的网上书店的成功，用户开始问你是否也卖电子书。电子书对你的网上商店来说是一个很好的产品，你想马上开始销售它们。

实体书不同于电子书:

*   一本电子书没有重量。这是一个虚拟产品。

*   一本电子书不需要发货。用户从网站上下载。

为了使您现有的模型支持销售电子书的附加信息，您向现有的`Book`模型添加了一些字段:

```py
from django.contrib.auth import get_user_model
from django.db import models

class Book(models.Model):
    TYPE_PHYSICAL = 'physical'
    TYPE_VIRTUAL = 'virtual'
    TYPE_CHOICES = (
        (TYPE_PHYSICAL, 'Physical'),
        (TYPE_VIRTUAL, 'Virtual'),
    )
 type = models.CharField( max_length=20, choices=TYPE_CHOICES, ) 
    # Common attributes
    name = models.CharField(
        max_length=100,
    )
    price = models.PositiveIntegerField(
        help_text='in cents',
    )

    # Specific attributes
    weight = models.PositiveIntegerField(
        help_text='in grams',
    )
 download_link = models.URLField( null=True, blank=True, ) 
    def __str__(self) -> str:
        return f'[{self.get_type_display()}] {self.name}'

class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
    books = models.ManyToManyField(
        Book,
    )
```

首先，您添加了一个 type 字段来指示这是哪种类型的书。然后，您添加了一个 URL 字段来存储电子书的下载链接。

要将实体书添加到书店，请执行以下操作:

>>>

```py
>>> from sparse.models import Book
>>> physical_book = Book.objects.create(
...     type=Book.TYPE_PHYSICAL,
...     name='Python Tricks',
...     price=1000,
...     weight=200,
...     download_link=None,
... )
>>> physical_book
<Book: [Physical] Python Tricks>
```

要添加新的电子书，请执行以下操作:

>>>

```py
>>> virtual_book = Book.objects.create(
...     type=Book.TYPE_VIRTUAL,
...     name='The Old Man and the Sea',
...     price=1500,
...     weight=0,
...     download_link='https://books.com/12345',
... )
>>> virtual_book
<Book: [Virtual] The Old Man and the Sea>
```

您的用户现在可以将图书和电子书添加到购物车中:

>>>

```py
>>> from sparse.models import Cart
>>> cart = Cart.objects.create(user=user)
>>> cart.books.add(physical_book, virtual_book)
>>> cart.books.all()
<QuerySet [<Book: [Physical] Python Tricks>, <Book: [Virtual] The Old Man and the Sea>]>
```

虚拟图书大受欢迎，你决定雇佣员工。新员工显然不太懂技术，您开始在数据库中看到奇怪的东西:

>>>

```py
>>> Book.objects.create(
...     type=Book.TYPE_PHYSICAL,
...     name='Python Tricks',
...     price=1000,
...     weight=0,
...     download_link='http://books.com/54321',
... )
```

那本书显然有`0`磅重，并且有下载链接。

这款电子书明显重 100g，没有下载链接:

>>>

```py
>>> Book.objects.create(
...     type=Book.TYPE_VIRTUAL,
...     name='Python Tricks',
...     price=1000,
...     weight=100,
...     download_link=None,
... )
```

这没有任何意义。你有数据完整性问题。

为了克服完整性问题，您需要向模型中添加验证:

```py
from django.core.exceptions import ValidationError

class Book(models.Model):

    # ...

    def clean(self) -> None:
        if self.type == Book.TYPE_VIRTUAL:
            if self.weight != 0:
                raise ValidationError(
                    'A virtual product weight cannot exceed zero.'
                )

            if self.download_link is None:
                raise ValidationError(
                    'A virtual product must have a download link.'
                )

        elif self.type == Book.TYPE_PHYSICAL:
            if self.weight == 0:
                raise ValidationError(
                    'A physical product weight must exceed zero.'
                )

            if self.download_link is not None:
                raise ValidationError(
                    'A physical product cannot have a download link.'
                )

        else:
            assert False, f'Unknown product type "{self.type}"'
```

您使用了 [Django 的内置验证机制](https://docs.djangoproject.com/en/2.1/ref/models/instances/#django.db.models.Model.clean)来实施数据完整性规则。`clean()`仅由 Django 表单自动调用。对于不是由 Django 表单创建的对象，您需要确保显式验证该对象。

为了保持`Book`模型的完整性，您需要对创建图书的方式做一点小小的改变:

>>>

```py
>>> book = Book(
...    type=Book.TYPE_PHYSICAL,
...    name='Python Tricks',
...    price=1000,
...    weight=0,
...    download_link='http://books.com/54321',
... )
>>> book.full_clean() ValidationError: {'__all__': ['A physical product weight must exceed zero.']} 
>>> book = Book(
...    type=Book.TYPE_VIRTUAL,
...    name='Python Tricks',
...    price=1000,
...    weight=100,
...    download_link=None,
... )
>>> book.full_clean() ValidationError: {'__all__': ['A virtual product weight cannot exceed zero.']}
```

当使用默认管理器(`Book.objects.create(...)`)创建对象时，Django 将创建一个对象并立即将它保存到数据库中。

在您的情况下，您希望在将对象保存到数据库之前对其进行验证。首先创建对象(`Book(...)`)，验证它(`book.full_clean()`)，然后保存它(`book.save()`)。

**反规格化:**

稀疏模型是[反规格化](https://en.wikipedia.org/wiki/Denormalization)的产物。在反规范化过程中，您将来自多个规范化模型的属性内联到一个表中，以获得更好的性能。非规范化的表通常会有许多可空的列。

非规范化通常用于决策支持系统，如读取性能非常重要的数据仓库。与 [OLTP 系统](https://en.wikipedia.org/wiki/Online_transaction_processing)不同，数据仓库通常不需要执行数据完整性规则，这使得反规范化成为理想。

**Pro**

*   **易于理解和维护:**当某些类型的对象需要更多信息时，稀疏模型通常是我们采取的第一步。非常直观，容易理解。

**缺点**

*   **无法利用非空数据库约束:**空值用于没有为所有类型的对象定义的属性。

*   **复杂验证逻辑:**需要复杂验证逻辑来实施数据完整性规则。复杂的逻辑也需要更多的测试。

*   许多空字段会造成混乱:在一个模型中表示多种类型的产品会增加理解和维护的难度。

*   **新类型需要模式更改:**新类型的产品需要额外的字段和验证。

**用例**

当您表示共享大部分属性的异构对象，并且不经常添加新项目时，稀疏模型是理想的。

[*Remove ads*](/account/join/)

## 半结构化模型

你的书店现在非常成功，你卖出了越来越多的书。你有不同流派和出版商的书，不同格式的电子书，形状和大小都很奇怪的书，等等。

在稀疏模型方法中，您为每种新产品添加了字段。该模型现在有许多可空字段，新开发人员和员工很难跟上。

为了解决混乱的问题，您决定只保留模型中的公共字段(`name`和`price`)。您将剩余的字段存储在一个单独的`JSONField`中:

```py
from django.contrib.auth import get_user_model
from django.contrib.postgres.fields import JSONField
from django.db import models

class Book(models.Model):
    TYPE_PHYSICAL = 'physical'
    TYPE_VIRTUAL = 'virtual'
    TYPE_CHOICES = (
        (TYPE_PHYSICAL, 'Physical'),
        (TYPE_VIRTUAL, 'Virtual'),
    )
    type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
    )

    # Common attributes
    name = models.CharField(
        max_length=100,
    )
    price = models.PositiveIntegerField(
        help_text='in cents',
    )

 extra = JSONField() 
    def __str__(self) -> str:
        return f'[{self.get_type_display()}] {self.name}'

class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
    books = models.ManyToManyField(
        Book,
        related_name='+',
    )
```

**JSONField:**

在本例中，您使用 PostgreSQL 作为数据库后端。Django 在`django.contrib.postgres.fields`中为 PostgreSQL 提供了内置的 JSON 字段。

对于其他数据库，如 SQLite 和 MySQL，有 T2 包提供类似的功能。

你的`Book`模型现在整洁了。公共属性被建模为字段。不是所有类型产品共有的属性存储在`extra` JSON 字段中:

>>>

```py
>>> from semi_structured.models import Book
>>> physical_book = Book(
...     type=Book.TYPE_PHYSICAL,
...     name='Python Tricks',
...     price=1000,
...     extra={'weight': 200}, ... )
>>> physical_book.full_clean()
>>> physical_book.save()
<Book: [Physical] Python Tricks>

>>> virtual_book = Book(
...     type=Book.TYPE_VIRTUAL,
...     name='The Old Man and the Sea',
...     price=1500,
...     extra={'download_link': 'http://books.com/12345'}, ... )
>>> virtual_book.full_clean()
>>> virtual_book.save()
<Book: [Virtual] The Old Man and the Sea>

>>> from semi_structured.models import Cart
>>> cart = Cart.objects.create(user=user)
>>> cart.books.add(physical_book, virtual_book)
>>> cart.books.all()
<QuerySet [<Book: [Physical] Python Tricks>, <Book: [Virtual] The Old Man and the Sea>]>
```

清理杂物很重要，但这是有代价的。验证逻辑要复杂得多:

```py
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator

class Book(models.Model):

    # ...

    def clean(self) -> None:

        if self.type == Book.TYPE_VIRTUAL:

            try:
                weight = int(self.extra['weight'])
            except ValueError:
                raise ValidationError(
                    'Weight must be a number'
                )
            except KeyError:
                pass
            else:
                if weight != 0:
                    raise ValidationError(
                        'A virtual product weight cannot exceed zero.'
                    )

            try:
                download_link = self.extra['download_link']
            except KeyError:
                pass
            else:
                # Will raise a validation error
                URLValidator()(download_link)

        elif self.type == Book.TYPE_PHYSICAL:

            try:
                weight = int(self.extra['weight'])
            except ValueError:
                raise ValidationError(
                    'Weight must be a number'
                 )
            except KeyError:
                pass
            else:
                if weight == 0:
                    raise ValidationError(
                        'A physical product weight must exceed zero.'
                     )

            try:
                download_link = self.extra['download_link']
            except KeyError:
                pass
            else:
                if download_link is not None:
                    raise ValidationError(
                        'A physical product cannot have a download link.'
                    )

        else:
            raise ValidationError(f'Unknown product type "{self.type}"')
```

使用适当字段的好处是它可以验证类型。Django 和 Django ORM 都可以执行检查，以确保字段使用了正确的类型。当使用`JSONField`时，您需要验证类型和值:

>>>

```py
>>> book = Book.objects.create(
...     type=Book.TYPE_VIRTUAL,
...     name='Python Tricks',
...     price=1000,
...     extra={'weight': 100},
... )
>>> book.full_clean()
ValidationError: {'__all__': ['A virtual product weight cannot exceed zero.']}
```

使用 JSON 的另一个问题是，并非所有数据库都支持查询和索引 JSON 字段中的值。

以 PostgreSQL 为例，可以查询所有重量超过`100`的书籍:

>>>

```py
>>> Book.objects.filter(extra__weight__gt=100)
<QuerySet [<Book: [Physical] Python Tricks>]>
```

然而，并不是所有的数据库供应商都支持这一点。

使用 JSON 的另一个限制是不能使用数据库约束，比如 not null、unique 和 foreign keys。您必须在应用程序中实现这些约束。

这种半结构化的方法类似于 NoSQL 的架构，有很多优点和缺点。JSON 字段是一种绕过关系数据库的严格模式的方法。这种混合方法为我们提供了将许多对象类型压缩到单个表中的灵活性，同时还保留了关系型、严格型和强类型数据库的一些优点。对于许多常见的 NoSQL 用例，这种方法实际上可能更合适。

**优点**

*   **减少杂乱:**公共字段存储在模型上。其他字段存储在单个 JSON 字段中。

*   **更容易添加新类型:**新类型的产品不需要改变模式。

**缺点**

*   **复杂和特殊的验证逻辑**:验证 JSON 字段需要验证类型和值。这个挑战可以通过使用其他解决方案来验证 JSON 数据来解决，比如 [JSON 模式](https://json-schema.org/)。

*   **无法利用数据库约束**:不能使用数据库约束，如 null null、unique 和 foreign key 约束，它们在数据库级别强制类型和数据完整性。

*   **受限于数据库对 JSON 的支持**:并不是所有的数据库厂商都支持查询和索引 JSON 字段。

*   **数据库系统**不强制执行模式:模式更改可能需要向后兼容或临时迁移。数据可能会“腐烂”

*   **没有与数据库元数据系统**深度集成:关于字段的元数据没有存储在数据库中。模式仅在应用程序级别实施。

**用例**

当您需要表示没有很多公共属性的异构对象，以及经常添加新项目时，半结构化模型是理想的。

半结构化方法的一个经典用例是存储事件(如日志、分析和事件存储)。大多数事件都有时间戳、类型和元数据，如设备、用户代理、用户等等。每种类型的数据都存储在 JSON 字段中。对于分析和日志事件，能够以最小的努力添加新类型的事件非常重要，因此这种方法是理想的。

[*Remove ads*](/account/join/)

## 抽象基础模型

到目前为止，您已经解决了将产品视为异类的问题。您假设产品之间的差异很小，因此在相同的模型中维护它们是有意义的。这个假设只能带你到这里。

你的小店发展很快，你想开始销售完全不同类型的产品，如电子阅读器、笔和笔记本。

书和电子书都是产品。产品是使用名称和价格等公共属性来定义的。在面向对象的环境中，你可以把一个`Product`看作一个基类或者一个[接口](https://realpython.com/python-interface/)。您添加的每一个新类型的产品都必须实现`Product`类，并用它自己的属性扩展它。

Django 提供了创建[抽象基类](https://docs.djangoproject.com/en/2.1/topics/db/models/#abstract-base-classes)的能力。让我们定义一个`Product`抽象基类，并为`Book`和`EBook`添加两个模型:

```py
from django.contrib.auth import get_user_model
from django.db import models

class Product(models.Model):
 class Meta: abstract = True 
    name = models.CharField(
        max_length=100,
    )
    price = models.PositiveIntegerField(
        help_text='in cents',
    )

    def __str__(self) -> str:
        return self.name

class Book(Product):
    weight = models.PositiveIntegerField(
        help_text='in grams',
    )

class EBook(Product):
    download_link = models.URLField()
```

注意，`Book`和`EBook`都继承自`Product`。基类`Product`中定义的字段是继承的，所以派生的模型`Book`和`Ebook`不需要重复。

要添加新产品，可以使用派生类:

>>>

```py
>>> from abstract_base_model.models import Book
>>> book = Book.objects.create(name='Python Tricks', price=1000, weight=200)
>>> book
<Book: Python Tricks>

>>> ebook = EBook.objects.create(
...     name='The Old Man and the Sea',
...     price=1500,
...     download_link='http://books.com/12345',
... )
>>> ebook
<Book: The Old Man and the Sea>
```

您可能已经注意到`Cart`模型不见了。您可以尝试创建一个带有`ManyToMany`字段的`Cart`模型来`Product`:

```py
class Cart(models.Model):
    user = models.OneToOneField(
       get_user_model(),
       primary_key=True,
       on_delete=models.CASCADE,
    )
 items = models.ManyToManyField(Product)
```

如果您试图将一个`ManyToMany`字段引用到一个抽象模型，您将得到以下错误:

```py
abstract_base_model.Cart.items: (fields.E300) Field defines a relation with model 'Product', which is either not installed, or is abstract.
```

外键约束只能指向具体的表。抽象基础模型`Product`只存在于代码中，所以数据库中没有 products 表。Django ORM 只会为派生的模型`Book`和`EBook`创建表格。

鉴于无法引用抽象基类`Product`，需要直接引用书籍和电子书:

```py
class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
 books = models.ManyToManyField(Book) ebooks = models.ManyToManyField(EBook)
```

现在，您可以将书籍和电子书添加到购物车中:

>>>

```py
>>> user = get_user_model().objects.first()
>>> cart = Cart.objects.create(user=user)
>>> cart.books.add(book)
>>> cart.ebooks.add(ebook)
```

这个模型现在有点复杂了。让我们查询购物车中商品的总价:

>>>

```py
>>> from django.db.models import Sum
>>> from django.db.models.functions import Coalesce
>>> (
...     Cart.objects
...     .filter(pk=cart.pk)
...     .aggregate(total_price=Sum( ...         Coalesce('books__price', 'ebooks__price') ...     )) ... )
{'total_price': 1000}
```

因为您有多种类型的书，所以您使用 [`Coalesce`](https://docs.djangoproject.com/en/2.1/ref/models/database-functions/#coalesce) 来获取每行的书的价格或电子书的价格。

**Pro**

*   **更容易实现特定的逻辑**:每个产品的独立模型使得实现、测试和维护特定的逻辑更加容易。

**缺点**

*   **需要多个外键**:为了引用所有类型的产品，每种类型都需要一个外键。

*   **更难实现和维护**:对所有类型产品的操作都需要检查所有外键。这增加了代码的复杂性，使得维护和测试更加困难。

*   **非常难以扩展**:新型产品需要额外的型号。管理许多模型可能会很繁琐，并且很难扩展。

**用例**

当只有很少类型的对象需要非常独特的逻辑时，抽象基础模型是一个很好的选择。

一个直观的例子是为你的网上商店建模一个支付过程。您希望接受信用卡、PayPal 和商店信用支付。每种支付方式都经历一个非常不同的过程，需要非常独特的逻辑。添加一种新的支付方式并不常见，而且您近期也不打算添加新的支付方式。

您可以使用信用卡付款流程、PayPal 付款流程和商店信用付款流程的派生类来创建付款流程基类。对于每个派生类，您以一种非常不同的方式实现支付过程，这种方式不容易共享。在这种情况下，具体处理每个支付过程可能是有意义的。

[*Remove ads*](/account/join/)

## 混凝土基础模型

Django 提供了另一种在模型中实现[继承](https://realpython.com/inheritance-composition-python/)的方法。您可以将基类具体化，而不是使用只存在于代码中的抽象基类。“具体”是指基类以表的形式存在于数据库中，不像抽象基类解决方案中，基类只存在于代码中。

使用抽象基础模型，您无法引用多种类型的产品。您被迫为每种类型的产品创建多对多关系。这使得在公共字段上执行任务变得更加困难，比如获取购物车中所有商品的总价。

使用一个具体的基类，Django 将在数据库中为`Product`模型创建一个表。`Product`模型将拥有您在基础模型中定义的所有公共字段。衍生模型如`Book`和`EBook`将使用一对一字段引用`Product`表。要引用一个产品，您需要为基本模型创建一个外键:

```py
from django.contrib.auth import get_user_model
from django.db import models

class Product(models.Model):
    name = models.CharField(
        max_length=100,
    )
    price = models.PositiveIntegerField(
        help_text='in cents',
    )

    def __str__(self) -> str:
        return self.name

class Book(Product):
    weight = models.PositiveIntegerField()

class EBook(Product):
    download_link = models.URLField()
```

这个例子和上一个例子的唯一区别是`Product`模型没有用`abstract=True`定义。

要创建新产品，您可以直接使用派生的`Book`和`EBook`模型:

>>>

```py
>>> from concrete_base_model.models import Book, EBook
>>> book = Book.objects.create(
...     name='Python Tricks',
...     price=1000,
...     weight=200,
... )
>>> book
<Book: Python Tricks>

>>> ebook = EBook.objects.create(
...     name='The Old Man and the Sea',
...     price=1500,
...     download_link='http://books.com/12345',
... )
>>> ebook
<Book: The Old Man and the Sea>
```

在具体基类的情况下，看看底层数据库中发生了什么是很有趣的。让我们看看 Django 在数据库中创建的表:

```py
> \d concrete_base_model_product

Column |          Type          |                         Default
--------+-----------------------+---------------------------------------------------------
id     | integer                | nextval('concrete_base_model_product_id_seq'::regclass)
name   | character varying(100) |
price  | integer                |

Indexes:
 "concrete_base_model_product_pkey" PRIMARY KEY, btree (id)

Referenced by:
 TABLE "concrete_base_model_cart_items" CONSTRAINT "..." FOREIGN KEY (product_id) 
 REFERENCES concrete_base_model_product(id) DEFERRABLE INITIALLY DEFERRED

 TABLE "concrete_base_model_book" CONSTRAINT "..." FOREIGN KEY (product_ptr_id) 
 REFERENCES concrete_base_model_product(id) DEFERRABLE INITIALLY DEFERRED

 TABLE "concrete_base_model_ebook" CONSTRAINT "..." FOREIGN KEY (product_ptr_id) 
 REFERENCES concrete_base_model_product(id) DEFERRABLE INITIALLY DEFERRED
```

product 表有两个熟悉的字段:名称和价格。这些是您在`Product`模型中定义的公共字段。Django 还为您创建了一个 ID 主键。

在“约束”部分，您会看到多个引用 product 表的表。两个突出的表是`concrete_base_model_book`和`concrete_base_model_ebook`:

```py
> \d concrete_base_model_book

 Column     |  Type
---------------+---------
product_ptr_id | integer weight         | integer

Indexes:
 "concrete_base_model_book_pkey" PRIMARY KEY, btree (product_ptr_id)

Foreign-key constraints:
 "..." FOREIGN KEY (product_ptr_id) REFERENCES concrete_base_model_product(id) DEFERRABLE INITIALLY DEFERRED
```

`Book`模型只有两个字段:

*   **`weight`** 是您在派生的`Book`模型中添加的字段。
*   **`product_ptr_id`** 既是表的主键，也是基本产品模型的外键。

在幕后，Django 为 product 创建了一个基表。然后，对于每个派生的模型，Django 创建了另一个表，其中包含附加字段，以及一个既充当 product 表的主键又充当外键的字段。

让我们来看看 Django 生成的获取一本书的查询。下面是`print(Book.objects.filter(pk=1).query)`的结果:

```py
SELECT "concrete_base_model_product"."id", "concrete_base_model_product"."name", "concrete_base_model_product"."price", "concrete_base_model_book"."product_ptr_id", "concrete_base_model_book"."weight" FROM "concrete_base_model_book" INNER  JOIN  "concrete_base_model_product"  ON "concrete_base_model_book"."product_ptr_id"  =  "concrete_base_model_product"."id"  WHERE  "concrete_base_model_book"."product_ptr_id"  =  1
```

为了拿到一本书，姜戈加入了`product_ptr_id`球场的`concrete_base_model_product`和`concrete_base_model_book`。名称和价格在产品表中，重量在图书表中。

由于所有产品都在 Product 表中进行管理，所以现在可以在来自`Cart`模型的外键中引用它:

```py
class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
 items = models.ManyToManyField(Product)
```

向购物车添加商品与之前相同:

>>>

```py
>>> from concrete_base_model.models import Cart
>>> cart = Cart.objects.create(user=user)
>>> cart.items.add(book, ebook)
>>> cart.items.all()
<QuerySet [<Book: Python Tricks>, <Book: The Old Man and the Sea>]>
```

使用公共字段也很简单:

>>>

```py
>>> from django.db.models import Sum
>>> cart.items.aggregate(total_price=Sum('price'))
{'total_price': 2500}
```

**迁移 Django 中的基类:**

当一个派生模型被创建时，Django 向[迁移](https://realpython.com/django-migrations-a-primer/)添加一个`bases`属性:

```py
 migrations.CreateModel(
      name='Book',
      fields=[...],
 bases=('concrete_base_model.product',),  ),
```

如果将来您删除或更改了基类，Django 可能无法自动执行迁移。您可能会得到以下错误:

```py
TypeError: metaclass conflict: the metaclass of a derived class must 
be a (non-strict) subclass of the metaclasses of all its bases
```

这是姜戈( [#23818](https://code.djangoproject.com/ticket/23818) ， [#23521](https://code.djangoproject.com/ticket/23521) ， [#26488](https://code.djangoproject.com/ticket/26488) )的一个已知问题。要解决此问题，您必须手动编辑原始迁移并调整“基础”属性。

**优点**

*   **主键在所有类型中保持一致**:产品由基表中的单个序列发出。通过使用 UUID 而不是序列，可以很容易地解决这种限制。

*   **单表查询常用属性**:总价、产品名称列表、价格等常用查询可以直接从基表中取出。

**缺点**

*   **新产品类型需要模式变更**:新类型需要新型号。

*   **会产生低效的查询**:单个项目的数据在两个数据库表中。提取产品需要与基表连接。

*   **无法从基类实例**访问扩展数据:需要类型字段来向下转换项目。这增加了代码的复杂性。 [`django-polymorphic`](https://django-polymorphic.readthedocs.io/en/stable/) 是一个流行的模块，可能会消除这些挑战。

**用例**

当基类中的公共字段足以满足大多数公共查询时，具体的基模型方法是有用的。

例如，如果您经常需要查询购物车的总价，显示购物车中的商品列表，或者对购物车模型运行特定的分析查询，那么在一个数据库表中包含所有的通用属性会让您受益匪浅。

[*Remove ads*](/account/join/)

## 通用外键

遗产继承有时会是一件令人讨厌的事情。它迫使你创建([可能是不成熟的](https://www.sandimetz.com/blog/2016/1/20/the-wrong-abstraction))抽象，并且它并不总是很好地适应 ORM。

您遇到的主要问题是从购物车模型中引用不同的产品。您首先试图将所有的产品类型压缩到一个模型中(稀疏模型、半结构化模型)，并且您得到了混乱。然后，您尝试将产品分成不同的模型，并使用具体的基础模型提供统一的界面。你得到了一个复杂的模式和许多连接。

Django 提供了一种引用项目中任何模型的特殊方式，称为 [`GenericForeignKey`](https://docs.djangoproject.com/en/2.1/ref/contrib/contenttypes/#django.contrib.contenttypes.fields.GenericForeignKey) 。通用外键是 Django 内置的[内容类型框架](https://docs.djangoproject.com/en/2.1/ref/contrib/contenttypes/)的一部分。Django 自己使用内容类型框架来跟踪模型。这对于一些核心功能(如迁移和权限)是必要的。

为了更好地理解什么是内容类型以及它们如何促进通用外键，让我们来看一下与`Book`模型相关的内容类型:

>>>

```py
>>> from django.contrib.contenttypes.models import ContentType
>>> ct = ContentType.objects.get_for_model(Book) >>> vars(ct)
{'_state': <django.db.models.base.ModelState at 0x7f1c9ea64400>,
'id': 22,
'app_label': 'concrete_base_model',
'model': 'book'}
```

每个型号都有唯一的标识符。如果你想引用一本 PK 54 的书，你可以说:“在内容类型 22 表示的模型中获取 PK 54 的对象。”

`GenericForeignKey`就是这样实现的。要创建通用外键，需要定义两个字段:

*   对内容类型(模型)的引用
*   被引用对象的主键(模型实例的`pk`属性)

要使用`GenericForeignKey`实现多对多关系，您需要手动创建一个模型来连接购物车和商品。

`Cart`模型与您目前看到的大致相似:

```py
from django.db import models
from django.contrib.auth import get_user_model

class Cart(models.Model):
    user = models.OneToOneField(
        get_user_model(),
        primary_key=True,
        on_delete=models.CASCADE,
    )
```

与以前的`Cart`型号不同，这个`Cart`不再包括一个`ManyToMany`字段。你需要自己去做。

要表示购物车中的单个商品，您需要引用购物车和任何产品:

```py
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType

class CartItem(models.Model):
    cart = models.ForeignKey(
        Cart,
        on_delete=models.CASCADE,
        related_name='items',
    )
    product_object_id = models.IntegerField()
    product_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.PROTECT,
    )
 product = GenericForeignKey( 'product_content_type', 'product_object_id',    )
```

要在购物车中添加新商品，您需要提供内容类型和主键:

>>>

```py
>>> book = Book.objects.first()

>>> CartItem.objects.create(
...     product_content_type=ContentType.objects.get_for_model(book), ...     product_object_id=book.pk, ... )
>>> ebook = EBook.objects.first()

>>> CartItem.objects.create(
...    product_content_type=ContentType.objects.get_for_model(ebook), ...    product_object_id=ebook.pk, ... )
```

将商品添加到购物车是一项常见任务。您可以在购物车中添加一种方法，将任何产品添加到购物车中:

```py
class Cart(models.Model):

    # ...

    def add_item(self, product) -> 'CartItem':
        product_content_type = ContentType.objects.get_for_model(product)

        return CartItem.objects.create(
            cart=self,
            product_content_type=product_content_type,
            product_object_id=product.pk,
        )
```

现在，向购物车添加新商品的时间大大缩短了:

>>>

```py
>>> cart.add_item(book)
>>> cart.add_item(ebook)
```

获取购物车中商品的信息也是可能的:

>>>

```py
>>> cart.items.all()
<QuerySet [<CartItem: CartItem object (1)>, <CartItem: CartItem object (2)>]

>>> item = cart.items.first()
>>> item.product
<Book: Python Tricks>

>>> item.product.price
1000
```

到目前为止一切顺利。关键在哪里？

让我们尝试计算购物车中产品的总价:

>>>

```py
>>> from django.db.models import Sum
>>> cart.items.aggregate(total=Sum('product__price')) 
FieldError: Field 'product' does not generate an automatic reverse 
relation and therefore cannot be used for reverse querying. 
If it is a GenericForeignKey, consider adding a GenericRelation.
```

Django 告诉我们，不可能从通用模型到引用模型遍历通用关系。原因是 Django 不知道要连接到哪个表。记住，`Item`模型可以指向任何一个`ContentType`。

错误信息确实提到了一个 [`GenericRelation`](https://docs.djangoproject.com/en/2.1/ref/contrib/contenttypes/#django.contrib.contenttypes.fields.GenericRelation) 。使用一个`GenericRelation`，你可以定义一个从参考模型到`Item`模型的反向关系。例如，您可以定义从`Book`模型到图书项目的反向关系:

```py
from django.contrib.contenttypes.fields import GenericRelation

class Book(model.Model):
    # ...
    cart_items = GenericRelation(
        'CartItem',
        'product_object_id',
        'product_content_type_id',
 related_query_name='books',    )
```

使用反向关系，您可以回答这样的问题，比如有多少购物车包含了某本书:

>>>

```py
>>> book.cart_items.count()
4

>>> CartItem.objects.filter(books__id=book.id).count()
4
```

这两种说法完全相同。

您仍然需要知道整个购物车的价格。您已经看到，使用 ORM 不可能从每个产品表中获取价格。为此，您必须迭代这些项目，分别获取每个项目，然后聚合:

>>>

```py
>>> sum(item.product.price for item in cart.items.all())
2500
```

这是泛型外键的主要缺点之一。这种灵活性伴随着巨大的性能成本。仅仅使用 Django ORM 很难优化性能。

**结构子类型**

在抽象和具体的基类方法中，您使用了基于类层次结构的**名义子类型**。Mypy 能够检测两个类之间的这种形式的关系，并从中推断出类型。

在一般关系方法中，您使用了结构化子类型。当一个类实现另一个类的所有方法和属性时，结构子类型存在。当您希望避免模块之间的直接依赖时，这种形式的子类型非常有用。

Mypy 提供了一种使用[协议](https://mypy.readthedocs.io/en/latest/protocols.html)利用结构化子类型的方法。

您已经确定了具有通用方法和属性的产品实体。您可以定义一个`Protocol`:

```py
from typing_extensions import Protocol

class Product(Protocol):
    pk: int
    name: str
    price: int

    def __str__(self) -> str:
        ...
```

**注意:**在方法定义中使用类属性和省略号(`...`)是 Python 3.7 中的新特性。在 Python 的早期版本中，不可能使用这种语法定义协议。方法体中应该有`pass`而不是省略号。像`pk`和`name`这样的类属性可以使用`@attribute`装饰器来定义，但是它不能用于 Django 模型。

您现在可以使用`Product`协议来添加类型信息。例如，在`add_item()`中，您接受一个产品实例并将其添加到购物车中:

```py
def add_item(
    self,
 product: Product, ) -> 'CartItem':
    product_content_type = ContentType.objects.get_for_model(product)

    return CartItem.objects.create(
        cart=self,
        product_content_type=product_content_type,
        product_object_id=product.pk,
    )
```

在此功能上运行`mypy`不会产生任何警告。假设您将`product.pk`更改为`product.id`，这在`Product`协议中没有定义:

```py
def add_item(
    self,
 product: Product, ) -> 'CartItem':
    product_content_type = ContentType.objects.get_for_model(product)

    return CartItem.objects.create(
        cart=self,
        product_content_type=product_content_type,
 product_object_id=product.id,    )
```

您将从 Mypy 收到以下警告:

```py
$ mypy
models.py:62: error: "Product" has no attribute "id"
```

**注:** `Protocol`还不是 Mypy 的一部分。它是补充包的一部分，叫做 [`mypy_extentions`](https://pypi.org/project/mypy_extensions/) 。这个包是由 Mypy 团队开发的，包含了他们认为还没有准备好用于主 Mypy 包的特性。

**优点**

*   **添加产品类型不需要迁移:**通用外键可以引用任何型号。添加新类型的产品不需要迁移。

*   **任何模型都可以作为条目:**使用通用外键，任何模型都可以被`Item`模型引用。

*   **内置管理支持:** Django 在管理中内置了[对通用外键的支持。例如，它可以在详细页面中内嵌关于引用模型的信息。](https://docs.djangoproject.com/en/2.1/ref/contrib/contenttypes/#generic-relations-in-admin)

*   **独立模块:**产品模块和购物车模块之间没有直接的依赖关系。这使得这种方法非常适合现有的项目和可插拔模块。

**缺点**

*   **会产生低效的查询:**ORM 无法预先确定通用外键引用的是什么模型。这使得 it 部门很难优化获取多种产品的查询。

*   更难理解和维护:通用外键消除了一些需要访问特定产品模型的 Django ORM 特性。从产品模型中访问信息需要编写更多的代码。

*   **类型化需要`Protocol` :** Mypy 无法提供通用模型的类型检查。需要一个`Protocol`。

**用例**

通用外键是可插拔模块或现有项目的最佳选择。`GenericForeignKey`和结构化子类型的使用抽象了模块之间的任何直接依赖。

在书店示例中，图书和电子书模型可以存在于一个单独的应用程序中，并且可以在不更改购物车模块的情况下添加新产品。对于现有的项目，可以添加一个`Cart`模块，只需对现有代码做最小的改动。

本文中介绍的模式配合得很好。使用混合模式，您可以消除一些缺点，并为您的用例优化模式。

例如，在通用外键方法中，您无法快速获得整个购物车的价格。您必须分别获取每个项目并进行汇总。您可以通过在`Item`模型中内嵌产品价格(稀疏模型方法)来解决这个具体问题。这将允许您只查询`Item`型号，以便非常快速地获得总价。

[*Remove ads*](/account/join/)

## 结论

在这篇文章中，你从一个小镇书店开始，发展成为一个大型电子商务网站。您解决了不同类型的问题，并调整了您的模型以适应这些变化。您了解了诸如复杂代码和难以向团队中添加新程序员之类的问题通常是更大问题的征兆。你学会了如何识别这些问题并解决它们。

现在您知道了如何使用 Django ORM 计划和实现多态模型。您熟悉多种方法，并且了解它们的优缺点。您能够分析您的用例并决定最佳的行动方案。*******