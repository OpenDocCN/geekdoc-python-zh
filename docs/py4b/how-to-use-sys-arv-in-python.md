# 如何在 Python 中使用 sys.arv

> 原文：<https://www.pythonforbeginners.com/argv/how-to-use-sys-arv-in-python>

刚接触 Python 的学生很早就学会了如何使用 **input()** 接受来自用户的数据。然而，还有另一种与程序交互的方法。

使用 **sys 模块**，可以直接从命令提示符或终端向 Python 程序传递参数。

这个特性允许在编写和运行程序时有更大的灵活性。例如，通过向 Python 程序提供不同的参数，可以改变代码的行为。

我们可以告诉 Python 程序根据通过命令行传递的参数来执行不同的功能。

## sys.argv 是什么？

sys 模块让我们影响 Python 解释器的行为。使用 sys.argv，Python 将在命令行中查找参数，并将它们作为列表提供给我们的程序。

这意味着我们可以在用户运行程序时寻找用户提供的输入。然后，我们可以使用该信息来更改输出。

## 我如何使用它？

在使用 sys 模块之前，您需要导入它。我们可以用 *import* 关键字，后跟模块名来实现。

导入语法:

```py
import *module_name*
```

有大量的 Python 库可供选择。熟悉 sys 之类的常用模块是个不错的主意。

### 示例 1:导入 sys 模块

```py
import sys

print(sys.version) 
```

运行这段代码将告诉您正在使用的 Python 版本，以及关于您的计算机系统的一些其他相关信息。

通过 sys 模块，我们可以操纵 Python 运行时环境。例如，我们可以从命令行改变与 Python 交互的方式。

## 命令行参数简介

sys 模块让我们从命令行向 python 提供额外的参数。这些参数作为列表包含在内。它们可以用来从命令行改变程序的行为方式。

这些附加参数位于 python 文件的名称之后。以下是在 Python 中使用命令行参数的语法:

```py
python python_file.py arg1 arg2 arg3
```

在下面的例子中，我们将传递三个参数。使用 **len()** 方法，我们可以找到并显示 sys.argv 列表的长度。

### 示例 2:使用 sys.argv

```py
# use len() to find the number of arguments passed to Python
import sys
total_args = len(sys.argv)

print("You passed: {} arguments.".format(total_args))
print("First argument: {}".format(sys.argv[0])) 
```

在命令提示符中，键入“python”和文件名后，包括一些附加参数，如下所示:

#### 投入

```py
python pfb_args.py one two three
```

#### 输出

```py
You passed: 4 arguments.
First argument: argv.py 
```

如您所见，Python 将文件名作为第一个参数。其他参数按照给出的顺序排列。

默认情况下， **len()** 方法在参数列表中包含文件名。如果我们想知道传递的参数的数量并且不包括文件名，我们可以使用 *len(sys.argv)-1* 。

```py
import sys
# exclude the file name from the list of arguments
total_args = len(sys.argv)-1

print("You passed: {} arguments.".format(total_args)) 
```

#### 投入

```py
python pfb_args.py one two
```

#### 输出

```py
You passed: 2 arguments.
```

## 打印 sys.argv 中的所有元素

因为命令行参数是作为列表传递的，所以我们可以用 for 循环遍历它们。

### 示例 3:使用 for 循环打印 sys.argv

```py
import sys

total_args = len(sys.argv)

for i in range(total_args):
    print("First argument: {}".format(sys.argv[i])) 
```

#### 投入

```py
python pfb_args.py one two three
```

#### 输出

```py
First argument: argv.py
First argument: one
First argument: two
First argument: three 
```

## 使用 str()方法打印 sys.argv

然而，有一种更简单的方法来打印 sys.argv 的内容。 **str()** 方法让我们将 sys.argv 的内容打印为一个字符串。

### 示例 5:对 sys.argv 使用 str()方法

```py
import sys
print(str(sys.argv)) 
```

在命令提示符下:

```py
python pfb_args.py one two three
```

#### 输出

```py
['argv.py', 'one', 'two', 'three']
```

## 用 sys.argv 求一个数的幂

我们可以在终端使用命令行参数来计算一个数的幂。

在对输入执行任何计算之前，我们需要转换参数。默认情况下，Python 假设我们正在传递字符串。

要将字符串转换成数字，使用 **int()** 方法。在下面的例子中， **int()** 方法用于将用户输入转换成数字。

在 Python 中， ****** 运算符用于计算给定数字的幂。

### 示例 6:寻找能力

```py
number = int(sys.argv[1])
power = int(sys.argv[2])

print("{} to the power of {} is {}.".format(number, power, number**power)) 
```

#### 投入

```py
python pfb_args.py 9 2
```

#### 输出

```py
9 to the power of 2 is 81.
```

## 求一组的平均值

再比如，我们可以用 sys.argv 来求一组数字的平均值。我们将使用一个 [python string split](https://www.pythonforbeginners.com/dictionary/python-split) 方法从列表中删除第一个元素。请记住，sys.argv 列表中的第一个元素将是 python 文件的名称。我们想忽略这一点。

从命令行传递给 Python 的任何参数都将被读取为字符串。我们必须使用 **float()** 将它们转换成浮点数。

得到总和后，我们需要用它除以集合的长度。确定传递了多少个数字的一种方法是从 sys.argv 的长度中减去 1。

### 示例 7:使用 sys.argv 查找一组数字的平均值

```py
# find the average of a set of numbers 
import sys

nums = sys.argv[1:]
sum = 0

for num in nums:
    sum += float(num)

average = sum/(len(sys.argv)-1)

print("The average of the set is: {}".format(average)) 
```

#### 投入

```py
python sum_of_set.py 5 100 14 25
```

#### 输出

```py
The average of the set is: 36.0
```

## 使用 sys.argv 读取文件

sys 模块通常用于处理多个文件。例如，如果我们编写一个 Python 程序来编辑文本文件，如果有一种方法可以轻松地告诉程序要读取什么文件，这将是非常有用的。

根据我们已经了解的 sys 模块和命令行参数的信息，我们可以告诉 Python 读取文本文件。此外，我们可以一次提供几个文件，Python 会读取所有文件。

在下一个示例中，我们将使用 sys.argv 读取多个文件。每本书都包含了莎士比亚的《哈姆雷特》第二幕中的几句台词。

这是文本文件。将它们保存在与 Python 文件相同的文件夹中:

#### 哈姆雷特 _part1.txt

克劳迪斯国王:乌云怎么还笼罩着你？哈姆雷特
不是这样的，大人；我太晒太阳了。

#### 哈姆雷特 _part2.txt

好哈姆雷特，脱掉你的睡衣，让你的眼睛看起来像一个丹麦的朋友。
不要永远蒙着面纱
在尘土中寻找你高贵的父亲:
你知道这很平常；所有的生命都会死去，穿过自然走向永恒。是的，夫人，这很平常。

使用 sys.argv，我们将告诉 Python 打开每个文件并将其内容打印到控制台。我们还将利用 **readline()** 方法来读取文本文件的行。

### 示例 8:使用 sys 模块读取多个文件

```py
import sys
# read files with sys.argv

def read_file(filename):
    with open(filename) as file:
        while True:
            line = file.readline()

            if len(line) == 0:
                break

            print(line)

files = sys.argv[1:]

for filename in files:
    read_file(filename) 
```

一旦程序有了 sys.argv 提供的文件名，就会使用 for 循环打开每个文件。在终端中，我们将向 Python 提供文件的名称。

#### 投入

```py
python sys_read_file.py hamlet_part1.txt hamlet_part2.txt
```

#### 输出

```py
KING CLAUDIUS

How is it that the clouds still hang on you?

HAMLET

Not so, my lord; I am too much i' the sun.

QUEEN GERTRUDE

Good Hamlet, cast thy nighted colour off,

And let thine eye look like a friend on Denmark.

Do not for ever with thy vailed lids

Seek for thy noble father in the dust:

Thou know'st 'tis common; all that lives must die,

Passing through nature to eternity.

HAMLET

Ay, madam, it is common. 
```

## 额外资源

概括地说，sys 模块为我们提供了对变量和函数的访问，这些变量和函数可用于操纵 Python 运行时环境。

我们可以使用 sys.argv 从命令行或终端向 Python 程序传递参数。在我们的示例中，sys 模块用于将文件名传递给我们的程序，但是您会发现有多种方式可以使用这个工具。

如果你想学习更多的 Python，请点击下面的链接，在那里你可以找到关于 Python 数据类型和捕捉异常的信息。

*   学习如何使用 [Python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。
*   使用 [Python 尝试 catch](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 来避免常见错误。