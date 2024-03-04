# 使用 Python 中的类创建装饰者

> 原文：<https://www.pythonforbeginners.com/basics/create-decorators-using-classes-in-python>

您可能知道如何使用函数在 python 中创建装饰器。在本文中，我们将讨论使用 Python 中的类来创建 decorators 的方法。首先我们将讨论什么是可调用对象，然后我们将通过为它们定义类来实现使用这些可调用对象的装饰器。

## 什么是可调用对象？

python 中任何可以像函数和方法一样调用的对象都称为可调用对象。在 python 中，每个可调用对象在其类定义中都有 __call__()方法的实现。我们可以说，任何在类定义中有***_ _ call _ _*_**()方法的对象都称为可调用对象。

要调用一个对象，我们需要在对象的类定义中实现***_ _ call _ _*_**()方法。

例如，看看下面的源代码。

```py
class Car:
   def __init__(self):
       self.brand = "Tesla"
       self.speed = "100mph"

   def __call__(self, *args, **kwargs):
       print("Called me? I am coming at {} speed.".format(self.speed))
```

在这里，我们定义了一个名为 ***汽车*** 的类。除了它的构造函数定义，我们还在它的类定义中定义了 ***__call__*** ()方法。现在我们可以调用 ***Car*** 类的任何实例，下面将执行 ***__call__*** ()方法。

```py
class Car:
   def __init__(self):
       self.brand = "Tesla"
       self.speed = "100mph"

   def __call__(self, *args, **kwargs):
       print("Called me? I am coming at {} speed.".format(self.speed))

# create callable object
myCar = Car()
# call myCar
myCar()
```

输出:

```py
Called me? I am coming at 100mph speed. 
```

我们可以使用任何一个接受函数或可调用对象作为输入并返回可调用对象或函数的可调用对象来实现 python 装饰器。

## 如何使用类创建 decorators？

通过使用 __call__()方法定义可调用对象，我们可以使用类来创建 decorators，这一点我们将在本节中讨论。

首先，我们将定义一个函数 ***add*** ()，它将两个数字作为输入，并打印它们的和。

```py
def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

# execute
add(10, 20) 
```

输出:

```py
The sum of 10 and 20 is 30.
```

现在，我们必须以这样的方式定义一个装饰器，即 ***add*** ()函数还应该打印数字的乘积以及总和。为此，我们可以创建一个装饰类。

我们可以在一个类内部实现***_ _*_ _ call _ _**()方法来实现 decorators。

首先，我们将定义 **decorator_class** 的构造函数，它接受 ***add*** ()函数作为输入参数，并将其赋给一个类变量 ***func*** 。

然后我们将实现***_ _ call _ _*_**()方法。 ***__call__*** ()方法中的代码将计算输入数字的乘积并打印出来。之后，它将使用给定的输入数字调用输入函数 add()。最后，它将返回 add()函数返回的值。

```py
class decorator_class:
   def __init__(self, func):
       self.func = func

   def __call__(self, *args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       return self.func(args[0], args[1])
```

在 ***__call__*** ()方法中，我们实现了打印作为输入给出的数字的乘积，然后在调用 ***add*** ()函数后返回其输出的代码。*()函数打印输入数字的和。我们已经定义了函数和装饰类，让我们看看如何使用它们。*

### *通过将函数作为参数传递给类构造函数来创建装饰器*

*要创建装饰函数，我们可以将 ***add*** ()函数作为输入参数传递给装饰器类的构造函数。将 ***add*** ()函数赋给***decorator _ class***中的 ***func*** 变量。一旦***decorator _ class***的实例被调用，它将执行 ***__call__*** ()方法内的代码来打印输入数字的乘积。然后它调用后会返回函数 ***func*** 的输出。由于 ***加上*** ()的功能已经赋值给 ***func*** ，它将打印出数字的总和。*

```py
*`class decorator_class:
   def __init__(self, func):
       self.func = func

   def __call__(self, *args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       return self.func(args[0], args[1])

def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

# execute
decorator_object = decorator_class(add)
decorator_object(10, 20)`*
```

*输出:*

```py
*`Product of 10 and 20 is 200
The sum of 10 and 20 is 30.`*
```

### *使用@符号创建装饰者*

*我们可以在定义*()函数之前，用 **@** 符号指定 ***decorator_class*** 名称，而不是将 add 函数传递给类构造函数。此后，每当调用 ***add*** ()函数时，它总是打印输入数字的乘积和总和。**

```py
**`class decorator_class:
   def __init__(self, func):
       self.func = func

   def __call__(self, *args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       return self.func(args[0], args[1])

@decorator_class
def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

# execute
add(10, 20)`** 
```

**输出:**

```py
**`Product of 10 and 20 is 200
The sum of 10 and 20 is 30.`** 
```

**这种方法有一个缺点，就是在用@ sign 定义修饰函数的时候，不能用 ***add*** ()函数仅仅把数字相加。它总是打印数字的乘积以及它们的总和。因此，如果您在任何其他地方以原始形式使用 add()函数，将会导致错误。**

## **结论**

**在本文中，我们讨论了使用 python 中的类创建 Python 装饰器的方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)**