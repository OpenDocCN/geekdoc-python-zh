# 在 Python 中使用 len()函数

> 原文：<https://realpython.com/len-python-function/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 len()函数**](/courses/pythons-len-function/)

在许多情况下，您需要找到存储在数据结构中的项目数量。Python 的内置函数`len()`是帮助你完成这项任务的工具。

有些情况下使用`len()`很简单。然而，在其他情况下，您需要更详细地了解这个函数的工作原理，以及如何将它应用于不同的数据类型。

**在本教程中，您将学习如何:**

*   使用`len()`找到**内置数据类型**的长度
*   将`len()`与**第三方数据类型**一起使用
*   用**用户自定义类**为`len()`提供支持

到本文结束时，您将知道何时使用`len()` Python 函数以及如何有效地使用它。您将知道哪些内置数据类型是`len()`的有效参数，哪些不能使用。你还将了解如何使用第三方类型的`len()`，比如 [NumPy](https://realpython.com/numpy-tutorial/) 中的`ndarray`和 [pandas](https://realpython.com/pandas-python-explore-dataset/) 中的`DataFrame`，以及你自己的类。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 的`len()` 入门

[`len()`](https://docs.python.org/3.9/library/functions.html?highlight=len#len) 函数是 Python 的内置函数之一。它返回一个对象的长度。例如，它可以返回列表中的项目数。您可以将函数用于许多不同的数据类型。然而，并不是所有的数据类型都是`len()`的有效参数。

您可以从查看此功能的帮助开始:

>>>

```py
>>> help(len)
Help on built-in function len in module builtins:
len(obj, /)
 Return the number of items in a container.
```

该函数将一个对象作为参数，并返回该对象的长度。`len()`的[文档](https://docs.python.org/3/library/functions.html?highlight=len#len)更进一步:

> 返回一个对象的长度(项目数)。参数可以是序列(如字符串、字节、元组、列表或范围)或集合(如字典、集合或冻结集)。([来源](https://docs.python.org/3/library/functions.html?highlight=len#len))

当你用`len()`使用内置数据类型和很多第三方类型时，函数不需要迭代数据结构。容器对象的长度存储为对象的属性。每次在数据结构中添加或删除项目时，都会修改该属性的值，并且`len()`会返回长度属性的值。这确保了`len()`的高效运作。

在接下来的部分中，您将了解如何对序列和集合使用`len()`。您还将了解一些不能用作`len()` Python 函数参数的数据类型。

[*Remove ads*](/account/join/)

### 使用内置序列的`len()`

序列是一个包含有序项目的容器。[列表、元组](https://realpython.com/python-lists-tuples/)和[字符串](https://realpython.com/python-strings/)是 Python 中三个基本的内置序列。您可以通过调用`len()`找到序列的长度:

>>>

```py
>>> greeting = "Good Day!"
>>> len(greeting)
9

>>> office_days = ["Tuesday", "Thursday", "Friday"]
>>> len(office_days)
3

>>> london_coordinates = (51.50722, -0.1275)
>>> len(london_coordinates)
2
```

当找到字符串`greeting`、列表`office_days`和元组`london_coordinates`的长度时，可以用同样的方式使用`len()`。所有这三种数据类型都是`len()`的有效参数。

函数`len()`总是返回一个整数，因为它在计算你传递给它的对象中的项数。如果参数是空序列，函数返回`0`:

>>>

```py
>>> len("")
0
>>> len([])
0
>>> len(())
0
```

在上面的例子中，您找到了一个空字符串、一个空列表和一个空元组的长度。该函数在每种情况下都返回`0`。

`range`对象也是可以使用 [`range()`](https://realpython.com/python-range/) 创建的序列。一个`range`对象并不存储所有的值，而是在需要的时候生成它们。然而，你仍然可以使用`len()`找到一个`range`物体的长度:

>>>

```py
>>> len(range(1, 20, 2))
10
```

这个数字范围包括从`1`到`19`的整数，增量为`2`。`range`对象的长度可由起始值、停止值和步长值决定。

在本节中，您已经将`len()` Python 函数用于字符串、列表、元组和`range`对象。但是，您也可以将该函数与任何其他内置序列一起使用。

### 将`len()`与内置集合一起使用

在某些时候，您可能需要在一个列表或另一个序列中找到唯一项的数量。您可以使用[设置](https://realpython.com/python-sets/)和`len()`来实现这一点:

>>>

```py
>>> import random

>>> numbers = [random.randint(1, 20) for _ in range(20)]

>>> numbers
[3, 8, 19, 1, 17, 14, 6, 19, 14, 7, 6, 1, 17, 10, 8, 14, 17, 10, 2, 5]

>>> unique_numbers = set(numbers)
{1, 2, 3, 5, 6, 7, 8, 10, 14, 17, 19}

>>> len(unique_numbers)
11
```

你使用一个[列表理解](https://realpython.com/list-comprehension-python/)生成列表`numbers`，它包含二十个范围在`1`和`20`之间的随机数。每次代码运行时，输出都是不同的，因为你生成的是随机数。在这个特定的运行中，在 20 个随机生成的数字列表中有 11 个唯一的数字。

另一个经常使用的内置数据类型是[字典](https://realpython.com/python-dicts/)。在字典中，每个条目由一个键值对组成。当使用字典作为`len()`的参数时，该函数返回字典中的项目数:

>>>

```py
>>> len({"James": 10, "Mary": 12, "Robert": 11})
3

>>> len({})
0
```

第一个示例的输出显示了这个字典中有三个键值对。与序列的情况一样，当参数是空字典或空集时，`len()`将返回`0`。这导致空字典和空集是虚假的。

### 探索其他内置数据类型

不能使用所有内置数据类型作为`len()`的参数。对于不存储多项的数据类型，长度的概念是不相关的。数字和布尔类型就是这种情况:

>>>

```py
>>> len(5)
Traceback (most recent call last):
    ...
TypeError: object of type 'int' has no len()

>>> len(5.5)
Traceback (most recent call last):
     ...
TypeError: object of type 'float' has no len()

>>> len(True)
Traceback (most recent call last):
     ...
TypeError: object of type 'bool' has no len()

>>> len(5 + 2j)
Traceback (most recent call last):
     ...
TypeError: object of type 'complex' has no len()
```

[整数、](https://realpython.com/python-numbers/)、[布尔、](https://realpython.com/python-boolean/)和[复杂](https://realpython.com/python-complex-numbers/)类型是不能与`len()`一起使用的内置数据类型的例子。当参数是没有长度的数据类型的对象时，该函数会引发一个`TypeError`。

您还可以探索是否有可能使用迭代器和生成器作为`len()`的参数:

>>>

```py
>>> import random

>>> numbers = [random.randint(1, 20) for _ in range(20)]
>>> len(numbers)
20

>>> numbers_iterator = iter(numbers)
>>> len(numbers_iterator)
Traceback (most recent call last):
     ...
TypeError: object of type 'list_iterator' has no len()

>>> numbers_generator = (random.randint(1, 20) for _ in range(20))
>>> len(numbers_generator)
Traceback (most recent call last):
     ...
TypeError: object of type 'generator' has no len()
```

您已经看到列表有长度，这意味着您可以在`len()`中将它用作参数。使用内置函数 [`iter()`](https://docs.python.org/3/library/functions.html#iter) 从列表中创建一个迭代器。在迭代器中，每一项都是在需要的时候获取的，比如使用函数`next()`或者在循环中。但是，你不能在`len()`中使用迭代器。

当您试图使用迭代器作为`len()`的参数时，您会得到一个`TypeError`。当迭代器在需要时获取每一项时，测量其长度的唯一方法是用尽迭代器。迭代器也可以是无限的，比如 [`itertools.cycle()`](https://realpython.com/python-itertools/) 返回的迭代器，因此其长度无法定义。

出于同样的原因，你不能将[发电机](https://realpython.com/introduction-to-python-generators/)与`len()`一起使用。这些物体的长度不用完是测不出来的。

[*Remove ads*](/account/join/)

## 通过一些示例进一步探索`len()`

在本节中，您将了解一些常见的`len()`用例。这些示例将帮助您更好地理解何时使用该函数以及如何有效地使用它。在一些例子中，您还会看到`len()`是一种可能的解决方案，但是可能有更多的 Pythonic 方式来实现相同的输出。

### 验证用户输入的长度

`len()`的一个常见用例是验证用户输入的序列长度:

```py
# username.py

username = input("Choose a username: [4-10 characters] ")

if 4 <= len(username) <= 10:
    print(f"Thank you. The username {username} is valid")
else:
    print("The username must be between 4 and 10 characters long")
```

在这个例子中，您使用一个`if`语句来检查由`len()`返回的整数是否大于或等于`4`并且小于或等于`10`。您可以运行这个脚本，您将得到类似于下面的输出:

```py
$ python username.py
Choose a username: [4-10 characters] stephen_g
Thank you. The username stephen_g is valid
```

在这种情况下，用户名有九个字符长，因此`if`语句中的条件计算结果为`True`。您可以再次运行脚本并输入无效的用户名:

```py
$ python username.py
Choose a username: [4-10 characters] sg
The username must be between 4 and 10 characters long
```

在这种情况下，`len(username)`返回`2`，并且`if`语句中的条件评估为`False`。

### 根据物体的长度结束循环

如果您需要检查一个可变序列(比如一个列表)的长度何时达到一个特定的数字，您将使用`len()`。在下面的示例中，您要求用户输入三个用户名选项，并将它们存储在一个列表中:

```py
# username.py

usernames = []

print("Enter three options for your username")

while len(usernames) < 3:
    username = input("Choose a username: [4-10 characters] ")
    if 4 <= len(username) <= 10:
        print(f"Thank you. The username {username} is valid")
        usernames.append(username)
    else:
        print("The username must be between 4 and 10 characters long")

print(usernames)
```

您现在在 [`while`](https://realpython.com/python-while-loop/) 语句中使用来自`len()`的结果。如果用户输入了无效的用户名，您不会保留输入。当用户输入一个有效的字符串时，您将它添加到列表`usernames`中。循环重复，直到列表中有三个项目。

您甚至可以使用`len()`来检查序列何时为空:

>>>

```py
>>> colors = ["red", "green", "blue", "yellow", "pink"]

>>> while len(colors) > 0:
...     print(f"The next color is {colors.pop(0)}")
...
The next color is red
The next color is green
The next color is blue
The next color is yellow
The next color is pink
```

您使用 list 方法`.pop()`在每次迭代中从列表中移除第一个项目，直到列表为空。如果你在大的列表上使用这个方法，你应该从列表的末尾删除项目，因为这样更有效。您还可以使用来自`collections`内置模块的[队列](https://realpython.com/python-deque/)数据类型，这允许您有效地从左侧弹出。

通过使用序列的[真值](https://realpython.com/python-boolean/),有一种更 Pythonic 化的方法来实现相同的输出:

>>>

```py
>>> colors = ["red", "green", "blue", "yellow", "pink"]

>>> while colors:
...    print(f"The next color is {colors.pop(0)}")
...
The next color is red
The next color is green
The next color is blue
The next color is yellow
The next color is pink
```

空列表是虚假的。这意味着`while`语句将空列表解释为`False`。非空列表是真实的，`while`语句将其视为`True`。由`len()`返回的值决定了一个序列的真实性。当`len()`返回任何非零整数时，序列为真，当`len()`返回`0`时，序列为假。

[*Remove ads*](/account/join/)

### 查找序列最后一项的索引

假设您想要生成一个范围为`1`到`10`的随机数序列，并且您想要不断地向该序列添加数字，直到所有数字的总和超过`21`。下面的代码创建一个空列表，并使用一个`while`循环来填充列表:

>>>

```py
>>> import random

>>> numbers = []
>>> while sum(numbers) <= 21:
...    numbers.append(random.randint(1, 10))

>>> numbers
[3, 10, 4, 7]

>>> numbers[len(numbers) - 1]
7

>>> numbers[-1]  # A more Pythonic way to retrieve the last item
7

>>> numbers.pop(len(numbers) - 1)  # You can use numbers.pop(-1)
7

>>> numbers
[3, 10, 4]
```

您将随机数添加到列表中，直到总和超过`21`。当你生成随机数时，你得到的输出会有所不同。要显示列表中的最后一个数字，可以使用`len(numbers)`并从中减去`1`，因为列表的第一个索引是`0`。Python 中的索引允许您使用索引`-1`来获取列表中的最后一项。所以，虽然在这种情况下可以用`len()`，但是不需要。

您想要删除列表中的最后一个数字，以便列表中所有数字的总和不超过`21`。再次使用`len()`来计算列表中最后一项的索引，并将其用作 list 方法`.pop()`的参数。即使在这种情况下，您也可以使用`-1`作为`.pop()`的参数，从列表中删除最后一项并将其返回。

### 将列表分成两半

如果需要将一个序列分成两半，就需要使用代表序列中点的索引。可以用`len()`来求这个值。在以下示例中，您将创建一个随机数列表，然后将其拆分为两个较小的列表:

>>>

```py
>>> import random

>>> numbers = [random.randint(1, 10) for _ in range(10)]
>>> numbers
[9, 1, 1, 2, 8, 10, 8, 6, 8, 5]

>>> first_half = numbers[: len(numbers) // 2]
>>> second_half = numbers[len(numbers) // 2 :]

>>> first_half
[9, 1, 1, 2, 8]
>>> second_half
[10, 8, 6, 8, 5]
```

在定义`first_half`的赋值语句中，使用切片来表示从`numbers`开始到中点的项目。通过分解切片表达式中使用的步骤，可以计算出切片表示的内容:

1.  首先，`len(numbers)`返回整数`10`。
2.  接下来，`10 // 2`使用[整数除法](https://realpython.com/python-numbers/#integer-division)运算符返回整数`5`。
3.  最后，`0:5`是表示前五个项目的切片，其索引为`0`到`4`。注意，端点被排除在外。

在下一个赋值中，您定义了`second_half`，在切片中使用了相同的表达式。然而，在这种情况下，整数`5`代表范围的开始。切片现在是`5:`，表示从索引`5`到列表末尾的项目。

如果您的原始列表包含奇数个项目，那么它的一半长度将不再是一个整数。当你使用整数除法时，你得到该数的[底数](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions)。列表`first_half`现在将比`second_half`少包含一个项目。

你可以通过创建一个包含 11 个数字而不是 10 个数字的初始列表来尝试一下。得到的列表将不再是两半，而是代表了分割奇数序列的最接近的选择。

## 通过第三方库使用`len()`功能

您还可以将 Python 的`len()`与第三方库中的几个自定义数据类型一起使用。在本教程的最后一部分，您将了解到`len()`的行为如何依赖于类定义。在这一节中，您将看到对两个流行的第三方库中的数据类型使用`len()`的例子。

### NumPy 的`ndarray`

NumPy 模块是 Python 中所有定量编程应用的基石。该模块介绍了 [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) 的数据类型。这种数据类型以及 NumPy 中的函数非常适合于数值计算，并且是其他模块中数据类型的构建块。

在开始使用 NumPy 之前，您需要安装这个库。您可以使用 Python 的标准包管理器 [`pip`](https://realpython.com/what-is-pip/) ，并在控制台中运行以下命令:

```py
$ python -m pip install numpy
```

您已经安装了 NumPy，现在您可以从一个列表中创建一个 NumPy 数组，并在数组上使用`len()`:

>>>

```py
>>> import numpy as np

>>> numbers = np.array([4, 7, 9, 23, 10, 6])
>>> type(numbers)
<class 'numpy.ndarray'>

>>> len(numbers)
6
```

NumPy 函数 [`np.array()`](https://numpy.org/doc/stable/reference/generated/numpy.array.html?highlight=array#numpy.array) 从您作为参数传递的列表中创建一个类型为`numpy.ndarray`的对象。

但是，NumPy 数组可以有多个维度。您可以通过将列表转换为数组来创建二维数组:

>>>

```py
>>> import numpy as np

>>> numbers = [
 [11, 1, 10, 10, 15],
 [14, 9, 16, 4, 4],
]

>>> numbers_array = np.array(numbers)
>>> numbers_array
array([[11,  1, 10, 10, 15],
 [14,  9, 16,  4,  4]])

>>> len(numbers_array)
2

>>> numbers_array.shape
(2, 5)

>>> len(numbers_array.shape)
2

>>> numbers_array.ndim
2
```

列表`numbers`由两个列表组成，每个列表包含五个整数。当您使用此列表创建 NumPy 数组时，结果是一个两行五列的数组。当您将这个二维数组作为参数在`len()`中传递时，该函数返回数组中的行数。

要获得两个维度的大小，可以使用属性`.shape`，这是一个显示行数和列数的元组。您可以通过使用`.shape`和`len()`或者使用属性`.ndim`来获得 NumPy 数组的维数。

一般来说，当你有一个任意维数的数组时，`len()`返回第一维的大小:

>>>

```py
>>> import numpy as np

>>> array_3d = np.random.randint(1, 20, [2, 3, 4])
>>> array_3d
array([[[14,  9, 15, 14],
 [17, 11, 10,  5],
 [18,  1,  3, 12]],
 [[ 1,  5,  6, 10],
 [ 6,  3,  1, 12],
 [ 1,  4,  4, 17]]])

>>> array_3d.shape
(2, 3, 4)

>>> len(array_3d)
2
```

在本例中，您创建了一个形状为`(2, 3, 4)`的三维数组，其中每个元素都是介于`1`和`20`之间的随机整数。这次您使用函数`np.random.randint()`来创建一个数组。函数`len()`返回`2`，是第一维度的大小。

查看 [NumPy 教程:Python 中数据科学的第一步](https://realpython.com/numpy-tutorial/)，了解更多关于使用 NumPy 数组的信息。

[*Remove ads*](/account/join/)

### 熊猫的`DataFrame`

[熊猫](https://pandas.pydata.org)库中的 [`DataFrame`](https://realpython.com/pandas-dataframe/) 类型是另一种在许多应用程序中广泛使用的数据类型。

在使用 pandas 之前，您需要在控制台中使用以下命令安装它:

```py
$ python -m pip install pandas
```

您已经安装了 pandas 包，现在您可以从字典创建一个数据框架:

>>>

```py
>>> import pandas as pd

>>> marks = {
 "Robert": [60, 75, 90],
 "Mary": [78, 55, 87],
 "Kate": [47, 96, 85],
 "John": [68, 88, 69],
}

>>> marks_df = pd.DataFrame(marks, index=["Physics", "Math", "English"])

>>> marks_df
 Robert  Mary  Kate  John
Physics      60    78    47    68
Math         75    55    96    88
English      90    87    85    69

>>> len(marks_df)
3

>>> marks_df.shape
(3, 4)
```

字典的关键字是代表班级中学生姓名的字符串。每个键的值是一个列表，其中包含三个主题的标记。当您从这个字典创建一个数据框架时，您使用一个包含主题名称的列表来定义索引。

数据帧有三行四列。函数`len()`返回数据帧中的行数。`DataFrame`类型也有一个`.shape`属性，可以用来显示 DataFrame 的第一维代表行数。

您已经看到了`len()`如何处理许多内置数据类型，以及来自第三方模块的一些数据类型。在下一节中，您将学习如何定义任何类，以便它可以用作`len()` Python 函数的参数。

您可以在[熊猫数据框架:让数据工作变得愉快](https://realpython.com/pandas-dataframe/)中进一步探索熊猫模块。

## 在用户定义的类上使用`len()`

定义类时，可以定义的一个特殊方法是 [`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 。这些[特殊方法](https://docs.python.org/3/reference/datamodel.html#special-method-names)被称为 dunder 方法，因为它们在方法名的开头和结尾都有双下划线。Python 的内置`len()`函数调用其参数的`.__len__()`方法。

在上一节中，您已经看到了当参数是 pandas `DataFrame`对象时`len()`的行为。这个行为是由`DataFrame`类的`.__len__()`方法决定的，你可以在`pandas.core.frame`中模块的[源代码](https://github.com/pandas-dev/pandas/blob/master/pandas/core/frame.py)中看到:

```py
class DataFrame(NDFrame, OpsMixin):
    # ...
    def __len__(self) -> int:
        """
 Returns length of info axis, but here we use the index.
 """
        return len(self.index)
```

该方法使用`len()`返回 DataFrame 的`.index`属性的长度。这个 dunder 方法将数据帧的长度定义为等于数据帧中的行数，如`.index`所示。

您可以通过下面的玩具示例进一步探索`.__len__()` dunder 方法。您将定义一个名为`YString`的类。这种数据类型基于内置的字符串类，但是`YString`类型的对象赋予字母 Y 比其他所有字母更重要的地位:

```py
# ystring.py

class YString(str):
    def __init__(self, text):
        super().__init__()

    def __str__(self):
        """Display string as lowercase except for Ys that are uppercase"""
        return self.lower().replace("y", "Y")

    def __len__(self):
        """Returns the number of Ys in the string"""
        return self.lower().count("y")
```

`YString`的 [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) 方法使用父`str`类的`.__init__()`方法初始化对象。你可以使用 [`super()`](https://realpython.com/python-super/) 功能来实现。 [`.__str__()`](https://docs.python.org/3/reference/datamodel.html#object.__str__) 方法定义了对象的显示方式。函数`str()`、`print()`和`format()`都调用这个方法。对于这个类，您将对象表示为一个全小写的字符串，字母 Y 除外，它显示为大写。

对于这个 toy 类，您将对象的长度定义为字母 Y 在字符串中出现的次数。因此，`.__len__()`方法返回字母 y 的计数。

你可以创建一个类为`YString`的对象，并找出它的长度。上例中使用的模块名是`ystring.py`:

>>>

```py
>>> from ystring import YString

>>> message = YString("Real Python? Yes! Start reading today to learn Python")

>>> print(message)
real pYthon? Yes! start reading todaY to learn pYthon

>>> len(message)  # Returns number of Ys in message
4
```

您从类型为`str`的对象创建一个类型为`YString`的对象，并使用`print()`显示该对象的表示。然后使用对象`message`作为`len()`的参数。这调用了类的`.__len__()`方法，结果是字母 Y 在`message`中出现的次数。在这种情况下，字母 Y 出现了四次。

`YString`类不是一个非常有用的类，但是它有助于说明如何定制`len()`的行为来满足您的需求。`.__len__()`方法必须返回一个非负整数。否则，它会引发错误。

另一个特殊的方法是`.__bool__()`方法，它决定了如何将一个对象转换成布尔值。通常不为序列和集合定义`.__bool__()` dunder 方法。在这些情况下，`.__len__()`方法决定了物体的真实性:

>>>

```py
>>> from ystring import YString

>>> first_test = "tomorrow"
>>> second_test = "today"

>>> bool(first_test)
True

>>> bool(YString(first_test))
False

>>> bool(second_test)
True

>>> bool(YString(second_test))
True
```

变量`first_string`中没有 Y。如来自`bool()`的输出所示，字符串是真实的，因为它不是空的。然而，当您从这个字符串创建一个类型为`YString`的对象时，这个新对象是 falsy，因为字符串中没有 Y 字母。因此，`len()`返回`0`。相反，变量`second_string`包含字母 Y，因此字符串和类型`YString`的对象都是真的。

你可以在 Python 3 的[面向对象编程(OOP)中阅读更多关于使用面向对象编程和定义类的内容。](https://realpython.com/python3-object-oriented-programming/)

[*Remove ads*](/account/join/)

## 结论

您已经探索了如何使用`len()`来确定序列、集合和其他一次包含几个项目的数据类型中的项目数量，比如 NumPy 数组和 pandas 数据帧。

Python 函数是许多程序中的关键工具。它的一些用法很简单，但是正如你在本教程中看到的，这个函数有比它最基本的用例更多的内容。知道什么时候可以使用这个函数，以及如何有效地使用它，将有助于你写出更整洁的代码。

**在本教程中，您已经学会了如何:**

*   使用`len()`找到**内置数据类型**的长度
*   将`len()`与**第三方数据类型**一起使用
*   用**用户自定义类**为`len()`提供支持

您现在已经为理解`len()`函数打下了良好的基础。了解更多关于`len()`的知识有助于您更好地理解数据类型之间的差异。您已经准备好在您的算法中使用`len()`，并通过用`.__len__()`方法增强它们来改进您的一些类定义的功能。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 len()函数**](/courses/pythons-len-function/)*******