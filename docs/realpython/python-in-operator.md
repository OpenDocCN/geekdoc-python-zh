# Python 的“in”和“not in”操作符:检查成员资格

> 原文：<https://realpython.com/python-in-operator/>

Python 的 **`in`** 和 **`not in`** 运算符允许您快速确定一个给定值是否是值集合的一部分。这种类型的检查在编程中很常见，在 Python 中通常被称为**成员测试**。因此，这些算子被称为**隶属算子**。

**在本教程中，您将学习如何:**

*   使用 **`in`** 和 **`not in`** 操作符执行**成员测试**
*   使用不同**数据类型**的`in`和`not in`
*   与`operator.contains()`、**一起工作，相当于`in`操作员的功能**
*   在你的**自己的班级**中为`in`和`not in`提供支持

为了充分利用本教程，您将需要 Python 的基础知识，包括内置数据类型，如[列表、元组](https://realpython.com/python-lists-tuples/)、[范围](https://realpython.com/python-range/)、[字符串](https://realpython.com/python-strings/)、[集合](https://realpython.com/python-sets/)和[字典](https://realpython.com/python-dicts/)。您还需要了解 Python [生成器](https://realpython.com/introduction-to-python-generators/)、[综合](https://realpython.com/list-comprehension-python/)和[类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-in-operator-code/)，你将使用它们用`in`和`not in`在 Python 中执行成员测试。

## Python 成员测试入门

有时，您需要找出一个值是否存在于一个值集合中。换句话说，您需要检查给定的值是否是值集合的成员。这种检查通常被称为[会员资格测试](https://docs.python.org/3/reference/expressions.html#membership-test-operations)。

可以说，执行这种检查的自然方式是迭代这些值，并将它们与目标值进行比较。你可以借助一个 [`for`循环](https://realpython.com/python-for-loop/)和一个[条件语句](https://realpython.com/python-conditional-statements/)来完成这个任务。

考虑下面的`is_member()`函数:

>>>

```py
>>> def is_member(value, iterable):
...     for item in iterable:
...         if value is item or value == item:
...             return True
...     return False
...
```

这个函数有两个参数，目标值`value`和一组值，通常称为`iterable`。循环在`iterable`上迭代，同时条件语句检查目标`value`是否等于当前值。注意，该条件使用`is`检查[对象标识](https://realpython.com/python-is-identity-vs-equality/#comparing-identity-with-the-python-is-and-is-not-operators)，或者使用相等运算符(`==`)检查[值相等](https://realpython.com/python-is-identity-vs-equality/#comparing-equality-with-the-python-and-operators)。这些测试略有不同，但互为补充。

如果条件为真，那么函数[返回](https://realpython.com/python-return-statement/) `True`，退出循环。这种提前返回[短路](https://realpython.com/python-return-statement/#short-circuiting-loops)的循环操作。如果循环结束而没有任何匹配，那么函数返回`False`:

>>>

```py
>>> is_member(5, [2, 3, 5, 9, 7])
True

>>> is_member(8, [2, 3, 5, 9, 7])
False
```

对`is_member()`的第一次调用返回`True`，因为目标值`5`是当前列表`[2, 3, 5, 9, 7]`的成员。对该函数的第二次调用返回`False`，因为`8`不在输入值列表中。

像上面这样的成员资格测试在编程中是如此普遍和有用，以至于 Python 有专门的操作符来执行这些类型的检查。您可以通过下表了解**隶属运算符**:

| 操作员 | 描述 | 句法 |
| --- | --- | --- |
| [T2`in`](https://docs.python.org/3/reference/expressions.html#in) | 如果目标值*是值集合中的*，则返回`True`。否则返回`False`。 | `value in collection` |
| [T2`not in`](https://docs.python.org/3/reference/expressions.html#not-in) | 如果目标值是给定值集合中的*而不是*，则返回`True`。否则返回`False`。 | `value not in collection` |

与[布尔运算符](https://realpython.com/python-boolean/)一样，Python 通过使用普通的英语单词而不是潜在的混淆符号作为运算符来提高可读性。

**注意:**当`in` [关键字](https://realpython.com/python-keywords/)在`for`循环语法中作为成员操作符时，不要将它与`in`关键字混淆。它们有完全不同的含义。`in`操作符检查一个值是否在一个值集合中，而`for`循环中的`in`关键字表示您想要从中提取的 iterable。

和其他很多运算符一样，`in`和`not in`都是二元运算符。这意味着你可以通过连接两个操作数来创建表达式。在这种情况下，它们是:

1.  **左操作数:**要在值集合中查找的目标值
2.  **右操作数:**可以找到目标值的值的集合

成员资格测试的语法如下所示:

```py
value in collection

value not in collection
```

在这些表达式中，`value`可以是任何 Python 对象。同时，`collection`可以是能够保存值集合的任何数据类型，包括[列表、元组](https://realpython.com/python-lists-tuples/)、[字符串](https://realpython.com/python-strings/)、[集合](https://realpython.com/python-sets/)和[字典](https://realpython.com/python-dicts/)。它也可以是实现`.__contains__()`方法的类，或者是明确支持成员测试或迭代的用户定义的类。

如果您正确使用了`in`和`not in`操作符，那么您用它们构建的表达式将总是计算出一个[布尔](https://realpython.com/python-boolean/)值。换句话说，这些表达式将总是返回`True`或`False`。另一方面，如果你试图在不支持成员测试的东西中找到一个值，那么你将得到一个 [`TypeError`](https://realpython.com/python-traceback/#typeerror) 。[稍后](#using-in-and-not-in-with-different-python-types)，您将了解更多关于支持成员测试的 Python 数据类型。

因为成员运算符总是计算为布尔值，Python 将它们视为布尔运算符，就像 [`and`](https://realpython.com/python-and-operator/) 、 [`or`](https://realpython.com/python-or-operator/) 和 [`not`](https://realpython.com/python-not-operator/) 运算符一样。

现在您已经知道了什么是成员资格操作符，是时候学习它们如何工作的基础知识了。

[*Remove ads*](/account/join/)

### Python 的`in`操作符

为了更好地理解`in`操作符，您将从编写一些小的演示示例开始，这些示例确定给定值*是否在*列表中:

>>>

```py
>>> 5 in [2, 3, 5, 9, 7]
True

>>> 8 in [2, 3, 5, 9, 7]
False
```

第一个表达式返回`True`，因为`5`出现在数字列表中。第二个表达式返回`False`，因为`8`不在列表中。

根据`in`操作符[文档](https://docs.python.org/3/reference/expressions.html#in)，类似`value in collection`的表达式相当于下面的代码:

```py
any(value is item or value == item for item in collection)
```

包装在对 [`any()`](https://realpython.com/any-python/) 的调用中的[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)构建了一个布尔值列表，该列表是通过检查目标`value`是否具有相同的身份或者是否等于`collection`中的当前`item`而得到的。对`any()`的调用检查是否有任何一个结果布尔值为`True`，在这种情况下，函数返回`True`。如果所有的值都是`False`，那么`any()`返回`False`。

### Python 的`not in`操作符

`not in`成员操作符做的正好相反。使用这个操作符，您可以检查给定值*是否不在*值集合中:

>>>

```py
>>> 5 not in [2, 3, 5, 9, 7]
False

>>> 8 not in [2, 3, 5, 9, 7]
True
```

在第一个例子中，您得到了`False`，因为`5`在`[2, 3, 5, 9, 7]`中。在第二个例子中，您得到了`True`,因为`8`不在值列表中。这种消极的逻辑看起来像绕口令。为了避免混淆，请记住您正在尝试确定值是否是给定值集合的*而不是*部分。

**注意:**`not value in collection`构造与`value not in collection`构造的工作原理相同。然而，前一种结构更难阅读。因此，你应该使用`not in`作为单个运算符，而不是使用`not`来否定`in`的结果。

通过对成员操作符如何工作的快速概述，您已经准备好进入下一个层次，学习`in`和`not in`如何处理不同的内置数据类型。

## 使用不同 Python 类型的`in`和`not in`

所有内置的[序列](https://docs.python.org/3/glossary.html#term-sequence)——比如列表、元组、 [`range`](https://realpython.com/python-range/) 对象和字符串——都支持使用`in`和`not in`操作符进行成员测试。像集合和字典这样的集合也支持这些测试。默认情况下，字典上的成员操作检查字典是否有给定的键。但是，字典也有显式的方法，允许您对键、值和键值对使用成员操作符。

在接下来的几节中，您将了解对不同的内置数据类型使用`in`和`not in`的一些特殊之处。您将从列表、元组和`range`对象开始。

### 列表、元组和范围

到目前为止，您已经编写了一些使用`in`和`not in`操作符来确定一个给定值是否存在于一个现有的值列表中的例子。对于这些例子，你已经明确地使用了`list`对象。因此，您已经熟悉了成员资格测试如何处理列表。

对于元组，成员运算符的工作方式与列表相同:

>>>

```py
>>> 5 in (2, 3, 5, 9, 7)
True

>>> 5 not in (2, 3, 5, 9, 7)
False
```

这里没有惊喜。这两个例子的工作方式都与以列表为中心的例子相同。在第一个例子中，`in`操作符返回`True`，因为目标值`5`在元组中。在第二个示例中，`not in`返回相反的结果。

对于列表和元组，成员操作符使用一个**搜索算法**，该算法迭代底层集合中的项目。因此，随着 iterable 变长，搜索时间也成正比增加。使用[大 O 符号](https://en.wikipedia.org/wiki/Big_O_notation)，你会说对这些数据类型的成员操作具有[的时间复杂度](https://wiki.python.org/moin/TimeComplexity)为 *O(n)* 。

如果您对`range`对象使用`in`和`not in`操作符，那么您会得到类似的结果:

>>>

```py
>>> 5 in range(10)
True

>>> 5 not in range(10)
False

>>> 5 in range(0, 10, 2)
False

>>> 5 not in range(0, 10, 2)
True
```

当涉及到`range`对象时，使用成员测试乍一看似乎是不必要的。大多数情况下，您会事先知道结果范围内的值。但是，如果您使用的`range()`带有在运行时确定的偏移量呢？

**注意:**创建`range`对象时，最多可以传递三个参数给`range()`。这些论点是`start`、`stop`和`step`。它们定义了*开始*范围的次数，范围必须*停止*生成值的次数，以及生成值之间的*步长*。这三个参数通常被称为**偏移**。

考虑以下示例，这些示例使用[随机](https://realpython.com/python-random/)数来确定运行时的偏移量:

>>>

```py
>>> from random import randint

>>> 50 in range(0, 100, randint(1, 10))
False

>>> 50 in range(0, 100, randint(1, 10))
False

>>> 50 in range(0, 100, randint(1, 10))
True

>>> 50 in range(0, 100, randint(1, 10))
True
```

在您的机器上，您可能会得到不同的结果，因为您正在使用随机范围偏移。在这些具体示例中，`step`是唯一变化的偏移量。在实际代码中，`start`和`stop`偏移量也可以有不同的值。

对于`range`对象，成员测试背后的算法使用表达式`(value - start) % step) == 0`计算给定值的存在，这取决于用来创建当前范围的偏移量。这使得成员测试在操作`range`对象时非常有效。在这种情况下，你会说他们的时间复杂度是 *O(1)* 。

**注意:**列表、元组和`range`对象有一个`.index()`方法，返回给定值在底层序列中第一次出现的索引。此方法对于在序列中定位值非常有用。

有些人可能认为他们可以使用方法来确定一个值是否在一个序列中。但是，如果值不在序列中，那么`.index()`会引发一个 [`ValueError`](https://realpython.com/python-traceback/#valueerror) :

>>>

```py
>>> (2, 3, 5, 9, 7).index(8)
Traceback (most recent call last):
    ...
ValueError: tuple.index(x): x not in tuple
```

您可能不想通过引发异常来判断一个值是否在一个序列中，因此您应该使用成员操作符而不是`.index()`来达到这个目的。

请记住，成员测试中的目标值可以是任何类型。测试将检查该值是否在目标集合中。例如，假设您有一个假想的应用程序，其中用户使用用户名和密码进行身份验证。你可以有这样的东西:

```py
# users.py

username = input("Username: ")
password = input("Password: ")

users = [("john", "secret"), ("jane", "secret"), ("linda", "secret")]

if (username, password) in users:
    print(f"Hi {username}, you're logged in!")
else:
    print("Wrong username or password")
```

这是一个幼稚的例子。不太可能有人会这样处理他们的用户和密码。但是该示例显示目标值可以是任何数据类型。在这种情况下，您使用一个字符串元组来表示给定用户的用户名和密码。

下面是代码在实践中的工作方式:

```py
$ python users.py
Username: john
Password: secret
Hi john, you're logged in!

$ python users.py
Username: tina
Password: secret
Wrong username or password
```

在第一个例子中，用户名和密码是正确的，因为它们在`users`列表中。在第二个示例中，用户名不属于任何注册用户，因此身份验证失败。

在这些例子中，重要的是要注意数据在登录元组中的存储顺序是至关重要的，因为在元组比较中像`("john", "secret")`这样的东西不等于`("secret", "john")`，即使它们有相同的条目。

在本节中，您已经探索了一些示例，这些示例展示了带有常见 Python 内置序列的成员运算符的核心行为。然而，还有一个内置序列。是的，弦乐！在下一节中，您将了解在 Python 中成员运算符如何处理这种数据类型。

[*Remove ads*](/account/join/)

### 字符串

Python 字符串是每个 Python 开发者工具箱中的基本工具。像元组、列表和范围一样，字符串也是序列，因为它们的项或字符是顺序存储在内存中的。

当需要判断目标字符串中是否存在给定的字符时，可以对字符串使用`in`和`not in`操作符。例如，假设您使用字符串来设置和管理给定资源的用户权限:

>>>

```py
>>> class User:
...     def __init__(self, username, permissions):
...         self.username = username
...         self.permissions = permissions
...

>>> admin = User("admin", "wrx")
>>> john = User("john", "rx")

>>> def has_permission(user, permission):
...     return permission in user.permissions
...

>>> has_permission(admin, "w")
True
>>> has_permission(john, "w")
False
```

`User`类有两个参数，一个用户名和一组权限。为了提供权限，您使用一个字符串，其中`w`表示用户拥有*写*权限，`r`表示用户拥有*读*权限，`x`表示*执行*权限。注意，这些字母与您在 Unix 风格的[文件系统权限](https://en.wikipedia.org/wiki/File-system_permissions)中找到的字母相同。

`has_permission()`中的成员测试检查当前`user`是否有给定的`permission`，相应地返回`True`或`False`。为此，`in`操作符搜索权限字符串来查找单个字符。在这个例子中，您想知道用户是否有*写*权限。

但是，您的权限系统有一个隐藏的问题。如果用空字符串调用函数会发生什么？这是你的答案:

>>>

```py
>>> has_permission(john, "")
True
```

因为空字符串总是被认为是任何其他字符串的子字符串，所以类似于`"" in user.permissions`的表达式将返回`True`。根据谁有权访问您的用户权限，这种成员资格测试行为可能意味着您的系统存在安全漏洞。

您还可以使用成员运算符来确定一个[字符串是否包含一个子字符串](https://realpython.com/python-string-contains-substring/):

>>>

```py
>>> greeting = "Hi, welcome to Real Python!"

>>> "Hi" in greeting
True
>>> "Hi" not in greeting
False

>>> "Hello" in greeting
False
>>> "Hello" not in greeting
True
```

对于字符串数据类型，如果`substring`是`string`的一部分，类似于`substring in string`的表达式就是`True`。否则，表情就是`False`。

**注意:**与列表、元组和`range`对象等其他序列不同，字符串提供了一个`.find()`方法，您可以在现有字符串中搜索给定的子字符串时使用这个方法。

例如，您可以这样做:

>>>

```py
>>> greeting.find("Python")
20

>>> greeting.find("Hello")
-1
```

如果子串存在于底层字符串中，那么`.find()`返回子串在字符串中开始的索引。如果目标字符串不包含子字符串，那么结果是得到`-1`。因此，像`string.find(substring) >= 0`这样的表达式相当于一个`substring in string`测试。

然而，成员测试可读性更强，也更明确，这使得它在这种情况下更可取。

在字符串上使用成员资格测试时要记住的重要一点是，字符串比较是区分大小写的:

>>>

```py
>>> "PYTHON" in greeting
False
```

这个成员测试返回`False`，因为字符串比较是区分大小写的，大写的`"PYTHON"`在`greeting`中不存在。要解决这种区分大小写的问题，您可以使用 [`.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper) 或 [`.lower()`](https://docs.python.org/3/library/stdtypes.html?highlight=lower#str.lower) 方法来规范化所有字符串:

>>>

```py
>>> "PYTHON".lower() in greeting.lower()
True
```

在这个例子中，您使用`.lower()`将目标子字符串和原始字符串转换成小写字母。这种转换在隐式字符串比较中不区分大小写。

### 发电机

[生成器函数](https://realpython.com/introduction-to-python-generators/)和[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)创建内存高效的[迭代器](https://docs.python.org/3/glossary.html#term-iterator)，称为[生成器迭代器](https://docs.python.org/3/glossary.html#term-generator-iterator)。为了提高内存效率，这些迭代器按需生成条目，而不需要在内存中保存完整的值序列。

实际上，生成器函数是一个[函数](https://realpython.com/defining-your-own-python-function/)，它在函数体中使用了 [`yield`](https://realpython.com/python-keywords/#the-yield-keyword) 语句。例如，假设您需要一个生成器函数，它接受一组数字并返回一个迭代器，该迭代器从原始数据中生成平方值。在这种情况下，您可以这样做:

>>>

```py
>>> def squares_of(values):
...     for value in values:
...         yield value ** 2
...

>>> squares = squares_of([1, 2, 3, 4])

>>> next(squares)
1
>>> next(squares)
4
>>> next(squares)
9
>>> next(squares)
16
>>> next(squares)
Traceback (most recent call last):
    ...
StopIteration
```

这个函数返回一个生成器迭代器，根据需要生成平方数。可以使用内置的 [`next()`](https://docs.python.org/3/library/functions.html#next) 函数从迭代器中检索连续值。当生成器迭代器被完全消耗完时，它会引发一个`StopIteration`异常，告知不再有剩余的值。

您可以在生成器函数上使用成员操作符，如`squares_of()`:

>>>

```py
>>> 4 in squares_of([1, 2, 3, 4])
True
>>> 9 in squares_of([1, 2, 3, 4])
True
>>> 5 in squares_of([1, 2, 3, 4])
False
```

当您将`in`操作符与生成器迭代器一起使用时，它将按预期工作，如果值出现在迭代器中，则返回`True`，否则返回`False`。

然而，在检查生成器的成员资格时，需要注意一些事情。一个生成器迭代器将只产生每个项目一次。如果你消耗了所有的条目，那么迭代器将被耗尽，你将无法再次迭代它。如果您只使用生成器迭代器中的一些项，那么您只能迭代剩余的项。

当您在生成器迭代器上使用`in`或`not in`时，操作符将在搜索目标值时消耗它。如果值存在，那么操作符将消耗所有值，直到目标值。其余的值在生成器迭代器中仍然可用:

>>>

```py
>>> squares = squares_of([1, 2, 3, 4])

>>> 4 in squares
True

>>> next(squares)
9
>>> next(squares)
16
>>> next(squares)
Traceback (most recent call last):
    ...
StopIteration
```

在这个例子中，`4`在生成器迭代器中，因为它是`2`的平方。因此，`in`返回`True`。当你使用`next()`从`square`中检索一个值时，你得到`9`，它是`3`的平方。该结果确认您不再能够访问前两个值。您可以继续调用`next()`，直到当生成器迭代器用尽时，您得到一个`StopIteration`异常。

同样，如果值不在生成器迭代器中，那么操作符将完全消耗迭代器，您将无法访问它的任何值:

>>>

```py
>>> squares = squares_of([1, 2, 3, 4])

>>> 5 in squares
False

>>> next(squares)
Traceback (most recent call last):
    ...
StopIteration
```

在这个例子中，`in`操作符完全消耗了`squares`,返回了`False`,因为目标值不在输入数据中。因为生成器迭代器现在已经用完了，所以用`squares`作为参数调用`next()`会引发`StopIteration`。

还可以使用生成器表达式创建生成器迭代器。这些表达式使用与[列表理解](https://realpython.com/list-comprehension-python/)相同的语法，但是用圆括号(`()`)代替了方括号(`[]`)。您可以将`in`和`not in`操作符用于生成器表达式的结果:

>>>

```py
>>> squares = (value ** 2 for value in [1, 2, 3, 4])
>>> squares
<generator object <genexpr> at 0x1056f20a0>

>>> 4 in squares
True

>>> next(squares)
9
>>> next(squares)
16
>>> next(squares)
Traceback (most recent call last):
    ...
StopIteration
```

`squares` [变量](https://realpython.com/python-variables/)现在保存由生成器表达式产生的迭代器。这个迭代器从输入的数字列表中产生平方值。来自生成器表达式的生成器迭代器与来自生成器函数的生成器迭代器工作方式相同。因此，当您在成员资格测试中使用它们时，同样的规则也适用。

当您在生成器迭代器中使用`in`和`not in`操作符时，会出现另一个关键问题。当您使用无限迭代器时，这个问题可能会出现。下面的函数返回一个产生无限整数的迭代器:

>>>

```py
>>> def infinite_integers():
...     number = 0
...     while True:
...         yield number
...         number += 1
...

>>> integers = infinite_integers()
>>> integers
<generator object infinite_integers at 0x1057e8c80>

>>> next(integers)
0
>>> next(integers)
1
>>> next(integers)
2
>>> next(integers)
3
>>> next(integers)
```

`infinite_integers()`函数返回一个生成器迭代器，存储在`integers`中。这个迭代器按需产生值，但是记住，将会有无限个值。因此，在这个迭代器中使用成员操作符不是一个好主意。为什么？好吧，如果目标值不在生成器迭代器中，那么你会遇到一个无限循环，这将使你的执行[挂起](https://en.wikipedia.org/wiki/Hang_(computing))。

[*Remove ads*](/account/join/)

### 字典和集合

Python 的成员操作符也可以处理字典和集合。如果您直接在字典上使用`in`或`not in`操作符，那么它将检查字典是否有给定的键。你也可以使用 [`.keys()`](https://realpython.com/python-dicts/#dkeys) 的方法来做这个检查，它更明确地表达了你的意图。

您还可以检查给定值或键值对是否在字典中。要做这些检查，可以分别使用 [`.values()`](https://realpython.com/python-dicts/#dvalues) 和 [`.items()`](https://realpython.com/python-dicts/#ditems) 方法:

>>>

```py
>>> likes = {"color": "blue", "fruit": "apple", "pet": "dog"}

>>> "fruit" in likes
True
>>> "hobby" in likes
False
>>> "blue" in likes
False

>>> "fruit" in likes.keys()
True
>>> "hobby" in likes.keys()
False
>>> "blue" in likes.keys()
False

>>> "dog" in likes.values()
True
>>> "drawing" in likes.values()
False

>>> ("color", "blue") in likes.items()
True
>>> ("hobby", "drawing") in likes.items()
False
```

在这些例子中，您直接在您的`likes`字典上使用`in`操作符来检查`"fruit"`、`"hobby"`和`"blue"`键是否在字典中。注意，即使`"blue"`是`likes`中的一个值，测试也返回`False`，因为它只考虑键。

接下来，使用`.keys()`方法得到相同的结果。在这种情况下，显式的方法名称会让阅读您代码的其他程序员更清楚您的意图。

要检查像`"dog"`或`"drawing"`这样的值是否出现在`likes`中，您可以使用`.values()`方法，该方法返回一个带有底层字典中的值的[视图对象](https://docs.python.org/3/library/stdtypes.html#dict-views)。类似地，要检查一个键值对是否包含在`likes`中，可以使用`.items()`。请注意，目标键-值对必须是两项元组，键和值按此顺序排列。

如果使用的是集合，那么成员运算符就像处理列表或元组一样工作:

>>>

```py
>>> fruits = {"apple", "banana", "cherry", "orange"}

>>> "banana" in fruits
True
>>> "banana" not in fruits
False

>>> "grape" in fruits
False
>>> "grape" not in fruits
True
```

这些例子表明，您还可以通过使用成员运算符`in`和`not in`来检查一个给定值是否包含在一个集合中。

现在您已经知道了`in`和`not in`操作符是如何处理不同的内置数据类型的，是时候通过几个例子将这些操作符付诸实践了。

## 将 Python 的`in`和`not in`操作符付诸实施

用`in`和`not in`进行成员测试是编程中非常常见的操作。您将在许多现有的 Python 代码库中找到这类测试，并且也将在您的代码中使用它们。

在接下来的小节中，您将学习如何用成员测试替换基于 [`or`](https://realpython.com/python-or-operator/) 操作符的布尔表达式。因为成员测试在您的代码中很常见，所以您还将学习如何使这些测试更有效。

### 替换连锁的`or`操作符

使用成员测试来用几个`or`操作符替换一个复合布尔表达式是一种有用的技术，它允许您简化代码并使其更具可读性。

要了解这项技术的实际应用，假设您需要编写一个函数，该函数将颜色名称作为一个字符串，并确定它是否是一种原色。为了解决这个问题，您将使用 [RGB(红、绿、蓝)](https://en.wikipedia.org/wiki/RGB_color_model)颜色模型:

>>>

```py
>>> def is_primary_color(color):
...     color = color.lower()
...     return color == "red" or color == "green" or color == "blue"
...

>>> is_primary_color("yellow")
False

>>> is_primary_color("green")
True
```

在`is_primary_color()`中，您使用一个复合布尔表达式，该表达式使用`or`操作符来检查输入颜色是红色、绿色还是蓝色。即使该功能如预期的那样工作，情况可能会令人困惑，难以阅读和理解。

好消息是你可以用一个简洁易读的成员测试来代替上面的条件:

>>>

```py
>>> def is_primary_color(color):
...     primary_colors = {"red", "green", "blue"}
...     return color.lower() in primary_colors ...

>>> is_primary_color("yellow")
False

>>> is_primary_color("green")
True
```

现在，您的函数使用`in`操作符来检查输入颜色是红色、绿色还是蓝色。将一组原色分配给一个适当命名的变量，如`primary_colors`，也有助于提高代码的可读性。最后的检查现在很清楚了。任何阅读您的代码的人都会立即理解您正试图根据 RGB 颜色模型来确定输入颜色是否是原色。

如果你再看一下这个例子，你会注意到原色已经被存储在一个集合中。为什么？你会在下一节找到你的答案。

[*Remove ads*](/account/join/)

### 编写高效的成员测试

Python 使用一种叫做[哈希表](https://realpython.com/python-hash-table/)的[数据结构](https://en.wikipedia.org/wiki/Data_structure)来实现字典和集合。哈希表有一个显著的特性:在数据结构中寻找任何给定的值需要大约相同的时间，不管表中有多少个值。使用大 O 符号，你会说哈希表中的值查找的时间复杂度为 *O(1)* ，这使得它们非常快。

现在，哈希表的这个特性与字典和集合上的成员测试有什么关系呢？事实证明，`in`和`not in`操作符在操作这些类型时工作非常快。这个细节允许您通过在成员测试中优先使用字典和集合而不是列表和其他序列来优化代码的性能。

要了解集合的效率比列表高多少，请继续创建以下脚本:

```py
# performance.py

from timeit import timeit

a_list = list(range(100_000))
a_set = set(range(100_000))

list_time = timeit("-1 in a_list", number=1, globals=globals())
set_time = timeit("-1 in a_set", number=1, globals=globals())

print(f"Sets are {(list_time / set_time):.2f} times faster than Lists")
```

这个脚本创建了一个包含十万个值的整数列表和一个包含相同数量元素的集合。然后，脚本计算确定数字`-1`是否在列表和集合中所需的时间。你预先知道`-1`不会出现在列表或集合中。因此，在得到最终结果之前，成员操作符必须检查所有的值。

正如您已经知道的，当`in`操作符在一个列表中搜索一个值时，它使用一个时间复杂度为 *O(n)* 的算法。另一方面，当`in`操作符在集合中搜索一个值时，它使用哈希表查找算法，该算法的时间复杂度为 *O(1)* 。这一事实可以在性能方面产生很大的差异。

使用以下命令从命令行运行您的脚本:

```py
$ python performance.py
Sets are 1563.33 times faster than Lists
```

尽管您的命令输出可能略有不同，但在这个特定的成员测试中，当您使用集合而不是列表时，它仍然会显示出显著的性能差异。有了列表，处理时间将与值的数量成正比。有了集合，对于任何数量的值，时间都差不多。

该性能测试表明，当您的代码对大型值集合进行成员资格检查时，您应该尽可能使用集合而不是列表。当您的代码在执行过程中执行几个成员测试时，您也将受益于 set。

但是，请注意，仅仅为了执行一些成员测试而将现有列表转换为集合并不是一个好主意。记住把链表转换成集合是一个时间复杂度为 *O(n)* 的操作。

## 使用`operator.contains()`进行成员资格测试

`in`操作符在 [`operator`](https://docs.python.org/3/library/operator.html#module-operator) 模块中有一个等价的函数，它来自[标准库](https://docs.python.org/3/library/index.html)。这个功能叫做 [`contains()`](https://docs.python.org/3/library/operator.html#operator.contains) 。它有两个参数——一组值和一个目标值。如果输入集合包含目标值，则返回`True`:

>>>

```py
>>> from operator import contains

>>> contains([2, 3, 5, 9, 7], 5)
True

>>> contains([2, 3, 5, 9, 7], 8)
False
```

`contains()`的第一个参数是值的集合，第二个参数是目标值。请注意，参数的顺序不同于常规的成员资格操作，在常规操作中，目标值排在第一位。

当您使用 [`map()`](https://realpython.com/python-map-function/) 或 [`filter()`](https://realpython.com/python-filter-function/) 等工具来处理代码中的可重复项时，这个函数就派上了用场。例如，假设你有一堆笛卡尔坐标点作为元组存储在一个列表中。您想要创建一个只包含不在坐标轴上的点的新列表。使用`filter()`功能，您可以得出以下解决方案:

>>>

```py
>>> points = [
...     (1, 3),
...     (5, 0),
...     (3, 7),
...     (0, 6),
...     (8, 3),
...     (2, 0),
... ]

>>> list(filter(lambda point: not contains(point, 0), points))
[(1, 3), (3, 7), (8, 3)]
```

在这个例子中，您使用`filter()`来检索不包含`0`坐标的点。为此，在 [`lambda`](https://realpython.com/python-lambda/) 函数中使用`contains()`。因为`filter()`返回一个迭代器，所以您将所有内容都包装在对`list()`的调用中，将迭代器转换成一个点列表。

尽管上面例子中的结构可以工作，但它相当复杂，因为它意味着导入`contains()`，在它上面创建一个`lambda`函数，并调用几个函数。您可以直接使用`contains()`或`not in`操作符使用列表理解得到相同的结果:

>>>

```py
>>> [point for point in points if not contains(point, 0)]
[(1, 3), (3, 7), (8, 3)]

>>> [point for point in points if 0 not in point]
[(1, 3), (3, 7), (8, 3)]
```

上面的列表理解比前一个例子中对应的`filter()`调用更短，并且更具可读性。它们也不太复杂，因为你不需要创建一个`lambda`函数或者调用`list()`，所以你减少了知识需求。

[*Remove ads*](/account/join/)

## 支持用户定义类中的成员测试

提供一个 [`.__contains__()`](https://docs.python.org/3/reference/datamodel.html#object.__contains__) 方法是在您自己的类中支持成员测试的最显式和首选的方式。当你在成员测试中使用你的类的一个实例作为右操作数时，Python 会自动调用这个[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)。

您可能只向作为值集合的类添加一个`.__contains__()`方法。这样，类的用户将能够确定给定值是否存储在类的特定实例中。

举例来说，假设您需要创建一个最小的[堆栈](https://realpython.com/how-to-implement-python-stack/)数据结构来存储遵循 [LIFO(后进先出)](https://realpython.com/queue-in-python/#stack-last-in-first-out-lifo)原则的值。定制数据结构的一个要求是支持成员测试。因此，您最终编写了下面的类:

```py
# stack.py

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

 def __contains__(self, item): return item in self.items
```

您的`Stack`类支持堆栈数据结构的两个核心功能。你可以*将一个值*推到栈顶，*从栈顶弹出一个值*。请注意，您的数据结构使用了一个`list`对象来存储和操作实际数据。

您的类也支持使用`in`和`not in`操作符的成员测试。为此，该类实现了一个依赖于`in`操作符本身的`.__contains__()`方法。

要测试您的类，请继续运行以下代码:

>>>

```py
>>> from stack import Stack

>>> stack = Stack()
>>> stack.push(1)
>>> stack.push(2)
>>> stack.push(3)

>>> 2 in stack
True
>>> 42 in stack
False
>>> 42 not in stack
True
```

您的类完全支持`in`和`not in`操作符。干得好！现在，您知道了如何在自己的类中支持成员测试。

注意，如果一个给定的类有一个`.__contains__()`方法，那么这个类不必是可迭代的，成员操作符也能工作。在上面的例子中，`Stack`是不可迭代的，操作符仍然工作，因为它们从`.__contains__()`方法中检索结果。

除了提供一个`.__contains__()`方法，至少还有两种方法支持用户定义类中的成员测试。如果你的类有一个 [`.__iter__()`](https://docs.python.org/3/reference/datamodel.html#object.__iter__) 或者一个 [`.__getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) 方法，那么`in`和`not in`操作符也可以工作。

考虑下面这个`Stack`的替代版本:

```py
# stack.py

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

 def __iter__(self): yield from self.items
```

这个`.__iter__()`特殊方法使得你的类[是可迭代的](https://docs.python.org/3/glossary.html#term-iterable)，这足以让成员测试工作。来吧，试一试！

支持成员测试的另一种方法是实现一个`.__getitem__()`方法，该方法在类中使用从零开始的整数索引来处理索引操作:

```py
# stack.py

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

 def __getitem__(self, index): return self.items[index]
```

当您对底层对象执行[索引操作](https://realpython.com/python-lists-tuples/#list-elements-can-be-accessed-by-index)时，Python 会自动调用`.__getitem__()`方法。在本例中，当您执行`stack[0]`时，您将获得`Stack`实例中的第一项。Python 利用`.__getitem__()`让成员操作符正常工作。

## 结论

现在您知道了如何使用 Python 的 **`in`** 和 **`not in`** 操作符来执行成员测试。这种类型的测试允许您检查给定的值是否存在于值集合中，这是编程中非常常见的操作。

**在本教程中，您已经学会了如何:**

*   使用 Python 的 **`in`** 和 **`not in`** 操作符运行**成员测试**
*   使用具有不同**数据类型**的`in`和`not in`运算符
*   与`operator.contains()`、**一起工作，相当于`in`操作员的功能**
*   支持**自己班级**中的`in`和`not in`

有了这些知识，您就可以在代码中使用 Python 的`in`和`not in`操作符进行成员测试了。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-in-operator-code/)，你将使用它们用`in`和`not in`在 Python 中执行成员测试。*****