# Python 装饰者

> 原文：<https://www.pythonforbeginners.com/basics/python-decorators>

Python 为我们提供了许多执行不同任务的结构。在编程时，有时我们可能需要修改函数的工作方式。但是我们可能不被允许改变函数的源代码，因为它可能在程序中以其原始形式被使用。在这种情况下，可以使用 Python decorators。

在本文中，我们将研究什么是 Python decorators，我们如何创建 decorator，以及我们如何使用它们来修改 Python 中其他函数的功能。

## 什么是 Python 装饰者？

Python decorators 是函数或其他可调用的对象，可用于向另一个函数添加功能，而无需修改其源代码。python 中的 decorator 接受一个函数作为输入参数，向它添加一些功能，然后返回一个包含修改后的功能的新函数。

在 python 中实现 decorators 需要了解不同的概念，比如第一类对象和嵌套函数。首先，我们将看看这些概念，这样我们在理解 python decorators 的实现时就不会遇到问题。

## 理解装饰者所需的概念

### 第一类对象

在 Python 中，第一类对象是那些

*   可以作为参数传递给函数。
*   可以从函数中返回。
*   可以赋给一个变量。

我们在程序中使用的所有变量都是第一类对象，无论它是原始数据类型、集合对象还是使用类定义的对象。

这里我想强调一下，python 中的**函数也是第一类对象，我们可以传递一个函数作为输入参数，也可以从一个函数返回一个函数。**

例如，让我们看看下面的源代码。

这里，我们定义了一个函数 ***add()*** ，它将两个数字作为输入，并打印它们的和。我们定义了另一个函数 ***random_adder*** ()，它以一个函数为输入，随机生成两个数字，并调用输入函数 ***add*** ()，将随机生成的数字作为输入。

```py
import random

def add(num1, num2):
   value = num1 + num2
   print("In the add() function. The sum of {} and {} is {}.".format(num1, num2, value))

def random_adder(func):
   val1 = random.randint(0, 10)
   val2 = random.randint(0, 100)
   print("In the random_adder. Values generated are {} and {}".format(val1, val2))
   func(val1, val2)

# execute
random_adder(add) 
```

输出:

```py
In the random_adder. Values generated are 1 and 14
In the add() function. The sum of 1 and 14 is 15. 
```

从代码和输出中，您可以观察到函数 ***add()*** 已经作为输入传递给了 ***random_adder*** ()函数，并且 ***random_adder*** ()函数调用了打印输出的*函数。*

*我们也可以从另一个函数或可调用对象返回一个函数。例如，我们可以修改上面的源代码，在 ***random_adder*** ()函数中定义一个函数 ***operate*** ()。 ***operate*** ()函数执行前面源代码中 ***random_adder*** ()函数完成的全部操作。*

*现在我们可以从 ***random_adder*** ()函数中返回 ***operate*** ()函数，并赋给一个名为 ***do_something*** 的变量。这样，我们就可以在 ***random_adder*** ()函数之外，通过调用变量 ***do_something*** 来执行 ***operate*** ()函数，如下所示。*

```py
*`import random

def add(num1, num2):
   value = num1 + num2
   print("In the add() function. The sum of {} and {} is {}.".format(num1, num2, value))

def random_adder(func):
   print("In the random_adder.")

   def operate():
       val1 = random.randint(0, 10)
       val2 = random.randint(0, 100)
       print("In the operate() function. Values generated are {} and {}".format(val1, val2))
       func(val1, val2)

   print("Returning the operate() function.")
   return operate

# execute
do_something = random_adder(add)
do_something()`* 
```

*输出:*

```py
*`In the random_adder.
Returning the operate() function.
In the operate() function. Values generated are 3 and 25
In the add() function. The sum of 3 and 25 is 28.`* 
```

### *嵌套函数*

*嵌套函数是定义在另一个函数内部的函数。例如，看看下面的源代码。*

*这里，我们定义了一个函数 ***add*** ()，它将两个数字作为输入，并计算它们的和。此外，我们在*()中定义了函数 ***square*** ()来打印在 ***add*** ()函数中计算的“ ***值*** ”的平方。**

```py
**`def add(num1, num2):
   value = num1 + num2
   print("In the add() function. The sum of {} and {} is {}.".format(num1, num2, value))

   def square():
       print("I am in square(). THe square of {} is {}.".format(value, value ** 2))

   print("calling square() function inside add().")
   square()

# execute
add(10, 20)`**
```

**输出:**

```py
**`In the add() function. The sum of 10 and 20 is 30.
calling square() function inside add().
I am in square(). THe square of 30 is 900.`** 
```

### **自由变量**

**我们知道一个变量可以在它被定义的范围内被访问。但是，在嵌套函数的情况下，当我们在内部函数中时，我们可以访问封闭函数的元素。**

**在上面的例子中，你可以看到我们在 ***add*** ()函数中定义了变量“ ***value*** ”，但是我们在 ***square*** ()函数中访问了它。这些类型的变量被称为自由变量。**

****但是为什么叫自由变量呢？****

**因为即使定义它的函数已经完成了它的执行，也可以访问它。例如，看看下面给出的源代码。**

```py
**`def add(num1, num2):
   value = num1 + num2
   print("In the add() function. The sum of {} and {} is {}.".format(num1, num2, value))

   def square():
       print("I am in square(). THe square of {} is {}.".format(value, value ** 2))

   print("returning square() function.")
   return square

# execute
do_something = add(10, 20)
print("In the outer scope. Calling do_something.")
do_something()`** 
```

**输出:**

```py
**`In the add() function. The sum of 10 and 20 is 30.
returning square() function.
In the outer scope. Calling do_something.
I am in square(). THe square of 30 is 900.`** 
```

**这里，一旦 ***add*** ()函数返回 ***square*** ()函数，它就完成了它的执行，并从内存中被清除。尽管如此，我们还是可以通过调用已经赋给变量 ***do_something*** 的 ***square*** ()函数来访问变量 ***value*** 。**

**既然我们已经讨论了实现 python decorators 所需的概念，那么让我们深入研究一下如何实现 decorator。**

## **如何创建 Python Decorators？**

**我们可以使用任何可调用对象来创建 python decorators，这些对象可以接受一个可调用对象作为输入参数，并且可以返回一个可调用对象。在这里，我们将使用 Python 中的函数创建 decorators。**

**对于要成为装饰器的函数，它应该遵循以下属性。**

1.  **它必须接受一个函数作为输入。**
2.  **它必须包含嵌套函数。**
3.  **它必须返回一个函数。**

**首先，我们将定义一个函数 ***add*** ()，它将两个数字作为输入，并打印它们的和。**

```py
**`def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

# execute
add(10, 20)`** 
```

**输出:**

```py
**`The sum of 10 and 20 is 30.`**
```

**现在，我们必须以这样的方式定义一个装饰器，即 ***add*** ()函数还应该打印数字的乘积以及总和。为此，我们可以创建一个装饰函数。**

**让我们首先定义一个函数，它将 ***add*** ()函数作为输入，并用附加的需求来修饰它。**

```py
**`def decorator_function(func):
   def inner_function(*args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       func(args[0], args[1])

   return inner_function`** 
```

**在***decorator _ function***()，我们定义了 ***inner_function*** ()，它打印作为输入给出的数字的乘积，然后调用 ***add*** ()函数。***decorator _ function***()返回***inner _ function***()。**

**既然我们已经定义了***decorator _ function***()和 ***add*** ()函数，那么让我们看看如何使用***decorator _ function***()来修饰 ***add*** ()函数。**

### **通过将函数作为参数传递给另一个函数来创建 Python 装饰器**

**修饰 ***add*** ()函数的第一种方法是将其作为输入参数传递给***decorator _ function***()。一旦***decorator _ function***()被调用，它将返回***inner _ function***()并赋给变量 ***do_something*** 。之后，变量 ***do_something*** 将变为可调用，并在被调用时执行***inner _ function***()内的代码。由此，我们可以调用 ***do_something*** 来打印输入数字的乘积和之和。**

```py
**`def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

def decorator_function(func):
   def inner_function(*args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       func(args[0], args[1])

   return inner_function

# execute
do_something = decorator_function(add)
do_something(10, 20)`**
```

**输出:**

```py
**`Product of 10 and 20 is 200
The sum of 10 and 20 is 30.`** 
```

### **使用@ sign 创建 Python 装饰器**

**执行相同操作的一个更简单的方法是使用“@”符号。我们可以指定***decorator _ function***的名称，在 ***@*** 符号之前定义 ***添加*** ()函数。此后，每当调用 ***add*** ()函数时，它总是打印输入数字的乘积和总和。**

```py
**`def decorator_function(func):
   def inner_function(*args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       return func(args[0], args[1])
   return inner_function

@decorator_function
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

**这种方法有一个缺点，就是不能用 ***加*** ()函数只是把数字相加。它总是打印数字的乘积以及它们的总和。因此，通过正确地分析您的需求，选择您将要用来实现装饰器的方法。**

## **结论**

**在本文中，我们讨论了什么是 python 装饰器，以及我们如何使用 Python 中的函数来实现它们。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)**