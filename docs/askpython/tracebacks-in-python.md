# 理解 Python 中的回溯

> 原文：<https://www.askpython.com/python/examples/tracebacks-in-python>

Traceback 是 Python 提供的消息或信息或带有一些数据的一般报告，帮助我们了解程序中发生的错误。专业术语也叫 **[养异常](https://www.askpython.com/python/examples/exceptions-in-python)** 。对于任何开发工作，错误处理都是编写程序时的关键部分之一。因此，处理错误的第一步是了解我们在代码中会遇到的最常见的错误。

回溯为我们提供了大量的信息和一些关于运行程序时发生的错误的消息。因此，对最常见的错误有一个大致的了解是非常重要的。

***也读作:[Python 中更容易调试的技巧](https://www.askpython.com/python/tricks-for-easier-debugging-in-python)***

回溯通常被称为某些其他名称，如 ****栈跟踪**** 、**回溯**或**栈回溯**。堆栈是所有编程语言中的一个抽象概念，它只是指系统内存或处理器内核中的一个位置，在那里指令被逐个执行。每当在检查代码时出现错误，回溯会试图告诉我们位置以及在执行这些指令时遇到的错误类型。

## Python 中一些最常见的回溯

这里列出了我们在 Python 中遇到的最常见的回溯。随着本文的深入，我们还将尝试理解这些错误的一般含义。

*   句法误差
*   NameError
*   索引错误
*   TypeError
*   属性错误
*   KeyError
*   ValueError
*   module 找不到和导入错误

## Python 中回溯的概述

在讨论最常见的回溯类型之前，让我们试着了解一下一般堆栈跟踪的结构。

```py
# defining a function
def multiply(num1, num2):
    result = num1 * num2
    print(results)

# calling the function
multiply(10, 2)

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 6, in <module>
    multiply(10, 2)
  File "d:\Python\traceback.py", line 3, in multiply
    print(results)
NameError: name 'results' is not defined. Did you mean: 'result'?

```

解释:

Python 正试图帮助我们，给我们关于在执行程序时发生的错误的所有信息。输出的最后一行说这应该是一个**名称错误**，甚至建议我们一个解决方案**。** Python 也试图告诉我们可能是错误来源的行号。

我们可以看到我们的代码中有一个变量名不匹配。我们没有使用 **"result"** ，正如我们之前在代码中声明的那样，我们编写了 **"results "，**，在执行程序时抛出一个错误。

所以，这是 Python 中**回溯**的一般结构层次，这也意味着 Python 回溯应该自底向上读取，这在大多数其他编程语言中不是这样。

## 了解回溯

### 1.语法错误

所有编程语言都有其特定的语法。如果我们错过了那个语法，程序将抛出一个错误。代码必须首先被解析，然后它将给出我们想要的输出。因此，我们必须确保它正确运行的正确语法。

让我们试着看看 Python 引发的 **SyntaxError** 异常。

```py
# defining a function
def multiply(num1, num2):
    result = num1 * num2
    print "result"

# calling the function
multiply(10, 2)

```

输出:

```py
File "d:\Python\traceback.py", line 4
    print "result"
    ^^^^^^^^^^^^^^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?

```

解释:

当我们试图运行上面的代码时，我们看到 Python 引发了一个 **SyntaxError** 异常。要在 Python3.x 中打印输出，我们需要用括号将它括起来。我们也可以看到错误的位置，在我们的错误下面显示**“^”**符号。

### 2\. NameError

在编写任何程序时，我们都要声明变量、函数和类，还要将模块导入其中。当在我们的程序中使用这些时，我们需要确保声明的东西应该被正确引用。相反，如果我们犯了某种错误，Python 会抛出一个错误并引发一个异常。

让我们看一个 Python 中的 **NameError** 的例子。

```py
# defining a function
def multiply(num1, num2):
    result = num1 * num2
    print(result)

# calling the function
multipli(10, 2)

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 8, in <module>
    multipli(10, 2)
NameError: name 'multipli' is not defined. Did you mean: 'multiply'?

```

解释:

我们的回溯表明名字**“multiplie”**没有定义，是一个**名字错误**。我们没有定义变量**、【乘数】、**，因此出现了错误。

### 3.索引错误

使用索引是 Python 中一种非常常见的模式。我们必须迭代 Python 中的各种数据结构来对它们执行操作。索引表示数据结构的顺序，例如列表或元组。每当我们试图从数据结构中不存在的系列或序列中检索某种索引数据时，Python 都会抛出一个错误，指出我们的代码中有一个 **IndexError** 。

让我们看一个例子。

```py
# declaring a list
my_list = ["apple", "orange", "banana", "mango"]

# Getting the element at the index 5 from our list
print(my_list[5])

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 5, in <module>
    print(my_list[5])
IndexError: list index out of range

```

解释:

我们的回溯表明在第 5 行有一个 **IndexError** 。从堆栈跟踪中可以明显看出，我们的列表在索引 5 处不包含任何元素，因此超出了范围。

### 4\. TypeError

当试图执行一个操作或使用一个函数时，Python 抛出一个 **TypeError** 错误类型的对象被一起用于该操作。

让我们看一个例子。

```py
# declaring variables
first_num = 10
second_num = "15"

# Printing the sum
my_sum = first_num + second_num
print(my_sum)

```

**输出:**

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 6, in <module>
    my_sum = first_num + second_num
TypeError: unsupported operand type(s) for +: 'int' and 'str'

```

解释:

在我们的代码中，我们试图计算两个数的和。但是 Python 抛出了一个异常，说第 6 行的操作数**“+”**有一个**类型错误**。堆栈跟踪告诉我们添加一个**整数**和一个**字符串**是无效的，因为它们的类型不匹配。

### 5.属性错误

每当我们试图访问一个特定对象上不可用的属性时，Python 就会抛出一个**属性错误。**

我们来看一个例子。

```py
# declaring a tuple
my_tuple = (1, 2, 3, 4)

# Trying to append an element to our tuple
my_tuple.append(5)

# Print the result
print(my_tuple)

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 5, in <module>    
    my_tuple.append(5)
AttributeError: 'tuple' object has no attribute 'append'

```

解释:

Python 说第 5 行的对象**“元组”**有一个 **AttributeError** 。由于 **[元组是不可变的数据结构](https://www.askpython.com/python/tuple/python-tuple)** ，我们试图对其使用“追加”方法。因此，这里有一个由 Python 引发的异常。Tuple 对象没有属性“append ”,因为我们试图改变它，这在 Python 中是不允许的。

### 6\. KeyError

[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)是 Python 中的另一种[数据结构。我们在程序中一直使用它。它由**键:值**对组成，我们需要在任何需要的时候访问这些键和值。但是，如果我们试图在字典中搜索一个不存在的键，会发生什么呢？](https://www.askpython.com/python/data-structures-in-python)

让我们尝试使用一个不存在的键，看看 Python 对此有什么说法。

```py
# dictionary
my_dict = {"name": "John", "age": 54, "job": "Programmer"}

# Trying to access info from our dictionary object
get_info = my_dict["email"]

# Print the result
print(get_info)

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 5, in <module>
    get_info = my_dict["email"]
KeyError: 'email'

```

解释:

在上面的例子中，我们试图访问关键字" **email** "的值。嗯，Python 在我们的 dictionary 对象中搜索了关键字“email ”,并使用堆栈跟踪引发了一个异常。回溯说，在我们程序的第 5 行有一个**键错误**。在指定的对象中找不到提供的键，因此出现错误。

### 7\. ValueError

每当指定的数据类型中有不正确的值时，Python 就会引发 **ValueError** 异常。提供的参数的数据类型可能是正确的，但是如果它不是一个合适的值，Python 将会抛出一个错误。

让我们看一个例子。

```py
import math

# Variable declaration
my_num = -16

# Check the data type
print(f"The data type is: {type(my_num)}") # The data type is: <class 'int'>

# Trying to get the square root of our number
my_sqrt = math.sqrt(my_num)

# Print the result
print(my_sqrt)

```

输出:

```py
The data type is: <class 'int'>
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 10, in <module>
    my_sqrt = math.sqrt(my_num)
ValueError: math domain error

```

解释:

在上面的例子中，我们试图使用 Python 中内置的[数学模块来得到一个数的平方根。我们使用正确的数据类型**“int”**作为函数的参数，但是 Python 抛出了一个带有 **ValueError** 的回溯作为例外。](https://www.askpython.com/python-modules/python-math-module)

这是因为我们不能得到一个负数的平方根，因此，这是一个不正确的参数值，Python 在第 10 行告诉我们这个错误是一个 **ValueError** 。

## 8.ImportError 和 ModuleNotFoundError

当导入不存在的特定模块出错时，Python 会引发 ImportError 异常。**当模块的指定路径无效或不正确时，ModuleNotFound** 作为异常出现。

让我们来看看这些错误是如何发生的。

**导入错误示例:**

```py
# Import statement
from math import addition

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 2, in <module>
    from math import addition
ImportError: cannot import name 'addition' from 'math' (unknown location)

```

**ModuleNotFoundError Example:**

```py
import addition

```

输出:

```py
Traceback (most recent call last):
  File "d:\Python\traceback.py", line 1, in <module>
    import addition
ModuleNotFoundError: No module named 'addition'

```

解释:

ModuleNotFoundError 是 ImportError 的子类，因为它们输出相似类型的错误，并且可以使用 Python 中的 try 和 except 块来避免。

## 摘要

在本文中，我们讨论了在编写 Python 代码时遇到的最常见的错误或回溯类型。对于所有级别的开发人员来说，在我们编写的任何程序中犯错误或引入 bug 都是很常见的。Python 是一种非常流行、用户友好且易于使用的语言，它有一些很棒的内置工具，可以在我们开发东西时尽可能地帮助我们。回溯是这些工具中的一个很好的例子，也是学习 Python 时需要理解的一个基本概念。

## 参考

[追溯文件](https://docs.python.org/3/library/traceback.html)