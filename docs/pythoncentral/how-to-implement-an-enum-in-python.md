# 如何在 Python 中实现“枚举”

> 原文：<https://www.pythoncentral.io/how-to-implement-an-enum-in-python/>

## 什么是 enum，我们为什么需要它？

枚举类型(也称为枚举)是一种数据类型，由一组命名值组成，这些命名值称为该类型的元素、成员或枚举器。这些枚举的命名值在计算语言中充当常数。例如，`COLOR`枚举可以包括命名值，如`RED`、`GREEN`和`BLUE`。请注意，所有命名的值都以大写形式书写，以区别它们是常量，这与变量有很大的不同。

那么，为什么我们需要使用枚举呢？想象一下这样一个场景，您可能希望将网站中用户的性别类型限制为`MALE`、`FEMALE`和`N/A`。当然，字符串列表也能很好地完成这项工作，比如`user.gender = 'MALE'`或`user.gender = 'FEMALE'`。然而，使用字符串作为性别属性的值对于程序员错误和恶意攻击来说并不健壮。程序员可以很容易地将“男性”错打成“麦芽酒”，或者将“女性”错打成“电子邮件”，代码仍然会运行，并产生有趣但可怕的结果。攻击者可能会向系统提供精心构建的垃圾字符串值，试图导致系统崩溃或获得根用户访问权限。如果我们使用 enum 将`user.gender`属性限制在有限的值列表中，上述问题都不会发生。

### **Python enum 简单易行**

在 Python 中，内置函数`type`接受一个或三个参数。根据 [Python 文档](http://docs.python.org/3/library/functions.html#type "Python type built-in function")，当你传递一个参数给`type`时，它返回一个对象的类型。当你向`type`传递三个参数时，比如`type(name, bases, dict):`

> 它返回一个新的类型对象。这本质上是一种动态形式的`class`语句。第一个参数`name`字符串是类名。第二个参数`bases` tuple 指定了新类型的基类。第三个参数`dict`是包含类体定义的名称空间。
> 
> [http://docs.python.org/3/library/functions.html#type](http://docs.python.org/3/library/functions.html#type "Python build-in type function")

使用`type`，我们可以通过以下方式构建一个`enum`:

```py

>>> def enum(**named_values):

... return type('Enum', (), named_values)

...

>>> Color = enum(RED='red', GREEN='green', BLUE='blue')

>>> Color.RED

'red'

>>> Color.GREEN

'green'

>>> Gender = enum(MALE='male', FEMALE='female', N_A='n/a')

>>> Gender.N_A

'n/a'

>>> Gender.MALE

'male'

```

现在`Color`和`Gender`是行为类似于`enum`的类，你可以以如下方式使用它们:

```py

>>> class User(object):

... def __init__(self, gender):

... if gender not in (Gender.MALE, Gender.FEMALE, Gender.N_A):

... raise ValueError('gender not valid')

... self.gender = gender

...

>>> u = User('malicious string')

Traceback (most recent call last):

File "", line 1, in

File "", line 4, in __init__

ValueError: gender not valid

```

注意传入的无效字符串被`User`的构造函数拒绝。

### **Python enum 的奇特方式**

尽管前面实现`enum`的方法很简单，但它确实需要您为每个`named value`显式指定一个值。例如，在`enum(MALE='male', FEMALE='female', N_A='n/a')`中，`MALE`是一个名称，`'male'`是该名称的值。由于大部分时间我们只通过名称使用`enum`，我们可以通过以下方式实现自动赋值的`enum`:

```py

>>> def enum(*args):

... enums = dict(zip(args, range(len(args))))

... return type('Enum', (), enums)

...

>>> Gender = enum('MALE', 'FEMALE', 'N_A')

>>> Gender.N_A

2

>>> Gender.MALE

0

```

使用`zip`和`range`，代码自动为`args`中的每个`name`分配一个整数值。

### **Python 枚举的技巧和建议**

*   `enum`有助于限制属性值的公开选择，这在现实编程中非常有用。当处理数据库时，在 Python 和数据库管理系统中相同的`enum`可以防止许多难以发现的错误。
*   使用一个`enum`来保护你的系统免受恶意输入。在系统接受输入值之前，一定要检查它们。