# Python 的空等价物:None

> 原文：<https://www.pythoncentral.io/python-null-equivalent-none/>

## 什么是 null 或 None 关键字

`null`关键字常用于许多编程语言，如 Java、C++、C#和 Javascript。它是赋给变量的一个值。也许你见过这样的东西:

#### Javascript 中为空

```py

var null_variable = null;

```

#### PHP 中为空

```py

$null_variable = NULL;

```

#### Java 中为空

```py

SomeObject null_object = null;

```

`null`关键字的概念是，它给变量一个*中性*，或“空”行为。注意，从技术上讲，`null`的行为在高级和低级语言之间是变化的，所以为了简单起见，我们将引用面向对象语言中的概念。

## Python 的空等价物:None

Python 中`null`关键字的对等词是`None`。这样设计有两个原因:

*   许多人会认为“null”这个词有点深奥。对于编程新手来说，这并不是最友好的词语。同样，“无”指的是预期的功能——它什么也不是，也没有行为。
*   在大多数面向对象语言中，对象的命名倾向于使用 camel-case 语法。`ThisIsMyObject`例。您很快就会看到，Python 的`None`类型是一个对象，并且表现得像一个对象。

将`None`类型赋给变量的语法非常简单。如下所示:

```py

my_none_variable = None

```

## 为什么要用 Python 的 None 类型？

有很多例子可以说明为什么你会使用`None`。

通常你会想要执行一个可能有效也可能无效的动作。使用`None`是稍后检查动作状态的一种方式。这里有一个例子:

```py

# We'd like to connect to a database. We don't know if the authentication

# details are correct, so we'll try. If the database connection fails,

# it will throw an exception. Note that MyDatabase and DatabaseException

# are not real classes, we're just using them as examples.
数据库连接=无
#尝试连接
尝试:
 database = MyDatabase(db_host，db_user，db_password，db _ database)
database _ connect = database . connect()
除数据库例外:
通过
如果 database_connection 为 None: 
打印('数据库无法连接')
否则:
打印('数据库可以连接')

```

另一个场景是你可能需要实例化一个类，这取决于一个条件。您可以将一个变量分配给`None`，然后可选地稍后将它分配给一个对象实例。然后你可能需要检查这个类是否已经被实例化了。这样的例子数不胜数——欢迎在评论中提供一些！

## Python 的 None 是面向对象的

Python 是非常面向对象的，你很快就会明白为什么。请注意，`Null`关键字是一个对象，其行为就像一个对象。如果我们检查`None`对象是什么类型，我们得到如下结果:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

```py
>>> type(None)
 <class>
```

*   [Python 2.x](#)

```py
>>> type(None)
 <type>
```

从中我们可以发现三件事:

*   `None`是一个对象——一个类。非基本类型，如数字，或`True`和`False`。
*   在 Python 3.x 中，`type`对象被更改为新的样式类。然而`None`的行为是相同的。
*   因为`None`是一个对象，我们不能用它来检查变量是否存在。它是一个值/对象，而不是用于检查条件的运算符。

## 检查变量是否为 None

有两种方法可以检查一个变量是否为`None`。一种方法是使用`is`关键字。另一种是使用`==`语法。这两种比较方法是不同的，稍后您会看到原因:

```py

null_variable = None

not_null_variable = 'Hello There!'
# is 关键字
如果 null_variable 是 None: 
 print('null_variable 是 None ')
else:
print(' null _ variable 不是 None ')
如果非空变量为无:
打印('非空变量为无')
否则:
打印('非空变量为无')
# The = = operator
if null _ variable = = None:
print(' null _ variable is None ')
else:
print(' null _ variable is not None ')
if not _ null _ variable = = None:
print(' not _ null _ variable is None ')
else:
print(' not _ null _ variable is not None ')

```

这段代码会给我们以下输出:
【shell】
null _ variable is None
not _ null _ variable is not None
null _ variable is None
not _ null _ variable is not None

太好了，所以他们是一样的！嗯，算是吧。它们是基本类型。然而，上课时你需要小心。Python 为类/对象提供了*覆盖*比较运算符的能力。所以可以比较类，比如`MyObject == MyOtherObject`。本文不会深入讨论如何在类中覆盖比较操作符，但是它应该提供了为什么应该避免使用`==`语法检查变量是否为`None`的见解。

```py

class MyClass:

    def __eq__(self, my_object):

        # We won't bother checking if my_object is actually equal

        # to this class, we'll lie and return True. This may occur

        # when there is a bug in the comparison class.
返回 True
my_class = MyClass()
如果我的类是无:
打印('我的类是无，使用 is 关键字')
否则:
打印('我的类不是无，使用 is 关键字')
if my _ class == None:
print(' my _ class is None，使用==语法')
else:
print(' my _ class is not None，使用= =语法')

```

这为我们提供了以下输出:

```py

my_class is not None, using the is keyword

my_class is None, using the == syntax

```

有意思！所以你可以看到，`is`关键字检查两个对象是否完全相同。而`==`操作符首先检查类是否覆盖了操作符。对于 PHP 程序员来说，使用`==`语法与 Python 中的`==`相同，其中使用`is`关键字等同于`===`语法。

因此，使用`is`关键字来检查两个变量是否完全相同总是明智的。