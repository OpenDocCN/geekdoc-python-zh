# Python 中的可调用对象

> 原文：<https://www.pythonforbeginners.com/basics/callable-objects-in-python>

你可能听说过 python 中的[函数是可调用对象。在本文中，我们将讨论术语“可调用对象”的确切含义。我们将讨论可调用对象实现背后的概念，并将实现程序来演示 python 中可调用对象的使用。](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)

## 调用对象是什么意思？

我们通过在任何对象后面加上圆括号来称呼它们。例如，当我们必须调用一个函数时，我们在它们后面放上圆括号，如下所示。

```py
def add(num1, num2):
    value = num1 + num2
    return value

val = add(10, 20)
print("The sum of {} and {} is {}".format(10, 20, val))
```

输出:

```py
The sum of 10 and 20 is 30
```

这里，我们调用了 add()函数，将 10 和 20 作为输入参数。该函数在执行后返回输入数字的总和。

同理，我们也可以调用其他可调用对象。但是，如果我们调用一个不可调用的对象，python 解释器将抛出一个 TypeError 异常，并给出一条消息，说明该对象不可调用。这可以通过以下方式观察到。

```py
val = 10
val()
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/webscraping.py", line 2, in <module>
    val()
TypeError: 'int' object is not callable 
```

在这里，你可以看到我们已经定义了一个整型变量，然后我们调用了它。在执行时，它会引发 TypeError 异常，并显示一条消息，说明“int”对象不可调用。

调用一个函数可以，但是调用一个整型变量会引发异常，这是什么原因？让我们找出答案。

## Python 中什么是可调用对象？

python 中的可调用对象是这样一种对象，它在被调用时执行一些代码，而不是引发 TypeError。

每个可调用对象都在其类定义中实现了 __call__()方法。如果我们使用这个细节来定义可调用对象，那么 python 中的可调用对象就是那些在类定义中实现了 __call__()方法的对象。

如果对象在其类定义中没有 __call__()方法的实现，则每当调用该对象时，它都会引发 TypeError 异常。这可以从下面的例子中看出。

```py
class Website:
    def __init__(self):
        self.name = "Python For Beginners"

myWebsite = Website()
myWebsite() 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/webscraping.py", line 7, in <module>
    myWebsite()
TypeError: 'Website' object is not callable
```

这里，对象 **myWebsite** 在其类 **Website** 的定义中没有 __call__()方法的实现。因此，它会引发 TypeError 异常，并显示一条消息，说明在调用“Website”对象时，它是不可调用的。

现在让我们在打印网站地址的**网站**类中实现 __call__()方法。观察这里的输出。

```py
class Website:
    def __init__(self):
        self.name = "Python For Beginners"
    def __call__(self, *args, **kwargs):
        print("Called me?")
        print("I am available at pythonforbeginners.com")

myWebsite = Website()
myWebsite() 
```

输出:

```py
Called me?
I am available at pythonforbeginners.com 
```

现在，您可能很清楚，我们可以调用任何在其类定义中实现了 __call__()方法的对象。

## 如何在 python 中创建可调用对象？

我们在上面已经看到，所有可调用对象在它们的类定义中都有 __call__()方法的实现。因此，要在 python 中创建一个可调用的对象，我们将在对象的函数定义中实现 __call__()方法，如 abve 给出的示例所示。

```py
class Website:
    def __init__(self):
        self.name = "Python For Beginners"

    def __call__(self, *args, **kwargs):
        print("Called me?")
        print("I am available at pythonforbeginners.com")

myWebsite = Website()
myWebsite() 
```

输出:

```py
Called me?
I am available at pythonforbeginners.com 
```

## 结论

在本文中，我们讨论了 python 中的可调用对象。我们还讨论了如何使用 __call__()方法创建可调用对象。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)