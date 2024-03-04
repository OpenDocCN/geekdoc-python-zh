# 在 Python 中删除对象的属性

> 原文：<https://www.pythonforbeginners.com/basics/delete-attribute-from-an-object-in-python>

Python 是一种面向对象的编程语言。我们经常在编程时使用用自定义类定义的对象。在本文中，我们将讨论如何在 Python 中删除对象的属性。

## 使用 Python 中的 del 语句从对象中删除属性

[del 语句](https://www.pythonforbeginners.com/basics/del-statement)可以用来删除任何对象及其属性。`del`语句的语法如下。

```py
del object_name
```

为了查看如何从对象中删除属性，让我们首先创建一个定制类`Person`，它具有属性`name`、`age`、`SSN`和`weight`。

```py
class Person:
    def __init__(self, name, age, SSN, weight):
        self.name = name
        self.age = age
        self.SSN = SSN
        self.weight = weight 
```

现在我们将创建一个名为`Person`类的对象`person1`。之后，我们将使用如下所示的`del`语句从`person1`对象中删除属性`weight`。

```py
class Person:
    def __init__(self, name, age, SSN, weight):
        self.name = name
        self.age = age
        self.SSN = SSN
        self.weight = weight

    def __str__(self):
        return "Name:" + str(self.name) + " Age:" + str(self.age) + " SSN: " + str(self.SSN) + " weight:" + str(
            self.weight)

person1 = Person(name="Will", age="40", SSN=1234567890, weight=60)
print(person1)
del person1.weight
print(person1)
```

输出:

```py
Name:Will Age:40 SSN: 1234567890 weight:60
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 16, in <module>
    print(person1)
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 10, in __str__
    self.weight)
AttributeError: 'Person' object has no attribute 'weight'
```

在上面的例子中，你可以看到我们可以在执行`del`语句之前打印属性`weight`。当我们在执行完`del`语句后试图打印属性`weight`时，程序遇到了`AttributeError`异常，称对象中没有名为`weight`的属性。因此，我们已经使用 python 中的`del`语句成功地从对象中删除了属性。

## 使用 Python 中的 delattr()函数从对象中删除属性

我们还可以使用 delattr()函数从对象中删除属性。delattr()函数接受一个对象作为它的第一个输入参数，属性名作为它的第二个输入参数。执行后，它从给定对象中删除属性。您可以在下面的示例中观察到这一点。

```py
class Person:
    def __init__(self, name, age, SSN, weight):
        self.name = name
        self.age = age
        self.SSN = SSN
        self.weight = weight

    def __str__(self):
        return "Name:" + str(self.name) + " Age:" + str(self.age) + " SSN: " + str(self.SSN) + " weight:" + str(
            self.weight)

person1 = Person(name="Will", age="40", SSN=1234567890, weight=60)
print(person1)
delattr(person1, "weight")
print(person1) 
```

输出:

```py
Name:Will Age:40 SSN: 1234567890 weight:60
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 16, in <module>
    print(person1)
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 10, in __str__
    self.weight)
AttributeError: 'Person' object has no attribute 'weight' 
```

您可以看到，我们能够在执行`delattr()` 函数之前打印出`person1`对象的`weight`属性。在执行了`delattr()`函数后，当我们试图打印`person1`对象的`weight`属性时，程序引发了`AttributeError`异常，表示该属性已被删除。

如果我们传递一个对象中不存在的属性名，它会引发如下所示的`AttributeError`异常。

```py
class Person:
    def __init__(self, name, age, SSN, weight):
        self.name = name
        self.age = age
        self.SSN = SSN
        self.weight = weight

    def __str__(self):
        return "Name:" + str(self.name) + " Age:" + str(self.age) + " SSN: " + str(self.SSN) + " weight:" + str(
            self.weight)

person1 = Person(name="Will", age="40", SSN=1234567890, weight=60)
print(person1)
delattr(person1, "BMI")
print(person1)
```

输出:

```py
Name:Will Age:40 SSN: 1234567890 weight:60
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 15, in <module>
    delattr(person1, "BMI")
AttributeError: BMI
```

在这里，您观察到我们试图从对象中不存在的`person1`对象中删除`BMI`属性。因此，程序遇到了`AttributeError`异常。

## 结论

在本文中，我们讨论了在 python 中从对象中删除属性的两种方法。要了解关于对象和类的更多信息，您可以阅读这篇关于 python 中的[类的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/basics/classes-in-python)[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。

建议阅读:

*   [用 Python 开发聊天应用，源代码](https://codinginfinite.com/python-chat-application-tutorial-source-code/)
*   [使用 Python 中的 sklearn 模块进行多项式回归](https://codinginfinite.com/polynomial-regression-using-sklearn-module-in-python/)