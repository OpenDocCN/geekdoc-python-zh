# 使用 Python getattr()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-getattr-function>

又见面了！在今天的文章中，我们将看看 Python 中的 Python getattr()函数。

这个函数试图获取 Python 对象的属性。这与 [setattr()函数](https://www.askpython.com/python/python-setattr-function)非常相似，但是它不修改对象。

让我们通过一些简单的例子来了解如何使用这个函数。

* * *

## Python getattr()的基本语法

getattr()函数的基本语法如下:

```py
value = getattr(object, attribute_name[, default_value])

```

在这里，我们将`object`传递给`getattr()`。它试图获取这个对象的`attribute_name`(必须是一个字符串)。如果属性存在，就会给我们对应的`value`。

这相当于以下语法:

```py
value = object.attribute_name

```

否则，有两种情况:

*   如果提供了默认值(`default_value`)，它将简单地使用它并返回这个默认值的值。
*   否则，它将简单地抛出一个`AttributeError`异常，因为没有找到我们的属性。

`getattr()`的目的不仅仅是获取属性值，还要检查它是否存在！

现在让我们来看几个例子，以便更好地理解。

* * *

## 使用 getattr()–一些例子

让我们先举一个简单的例子，你有一个学生类，有`name`和`roll_num`属性。我们会用`getattr()`干掉他们。

```py
class Student():
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

student = Student('Amit', 8)

print(f"Name: {getattr(student, 'name')}")
print(f"Roll No: {getattr(student, 'roll_no')}")

```

**输出**

```py
Name: Amit
Roll No: 8

```

事实上，我们确实设置了正确的属性，并且我们使用`getattr()`来取回它们！

现在，在这种情况下，因为两个属性都在那里，所以没有错误发生。然而，让我们试着得到一个没有放入类中的属性；比如——`age`。

```py
class Student():
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

student = Student('Amit', 8)

print(f"Name: {getattr(student, 'name')}")
print(f"Roll No: {getattr(student, 'roll_no')}")

# Will raise 'AttributeError' Exception since the attribute 'age' is not defined for our instance
print(f"Age: {getattr(student, 'age')}")

```

**输出**

```py
Name: Amit
Roll No: 8
Traceback (most recent call last):
  File "getattr_example.py", line 12, in <module>
    print(f"Age: {getattr(student, 'age')}")
AttributeError: 'Student' object has no attribute 'age'

```

这里，我们试图检索一个未定义的属性。由于没有默认选项，Python 直接引发了一个异常。

如果我们想将默认选项设置为`'100'`，那么在这种情况下，尽管我们试图获取`age`，但由于它不在那里，它将返回`100` 。

让我们来测试一下。

```py
class Student():
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

student = Student('Amit', 8)

print(f"Name: {getattr(student, 'name')}")
print(f"Roll No: {getattr(student, 'roll_no')}")

# Will not raise AttributeError, since a default value is provided
print(f"Age: {getattr(student, 'age', 100)}")

```

**输出**

```py
Name: Amit
Roll No: 8
Age: 100

```

* * *

## 为什么应该使用 getattr()？

既然我们前面提到这个函数相当于`object.attribute_name`，那么这个函数的意义是什么呢？

虽然上面使用点符号的语句是正确的，但它只有在调用时实际定义了属性名时才有效！

因此，如果你的类的对象结构不断变化，你可以使用`getattr()`来检查对象是否处于某种特定的状态！

它还为我们提供了一个非常好的方法来轻松处理异常，并提供后备默认值。如果出现问题，我们可以捕获`AttributeError`异常或者检查默认值。

希望这能让你更清楚地了解为什么这对你有用！

* * *

## 结论

我们学习了如何使用`getattr()`函数在运行时检索对象的属性。

## 参考

*   getattr()上的 [Python 官方文档](https://docs.python.org/3/library/functions.html#getattr)
*   关于 Python getattr()的 JournalDev 文章

* * *