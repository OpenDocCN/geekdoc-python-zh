# Python 中的计算器程序

> 原文：<https://www.askpython.com/python/examples/calculator-program-python>

Python 编程是评估和进行操作的一个很好的工具。在本文中，我们将学习 Python 3 中一个简单的命令行计算器程序。

我们将使用[数学运算符](https://www.askpython.com/python/python-operators)、条件语句、函数和[处理用户输入](https://www.askpython.com/python/examples/python-user-input)来制作我们的计算器。

## 先决条件

系统应该在本地计算机上安装 Python 3，并在机器上设置一个编程环境。

* * *

## 接受/提示用户输入

我们将接受用户的输入。为此，我们将使用 Python 的 input()函数。对于这个程序，我们将让用户输入两个数字，所以让程序提示这两个数字。

```py
num_1 = input('Enter your first number: ')
num_2 = input('Enter your second number: ')

```

**输出**:

```py
Enter your first number: 10
Enter your second number: 5

```

我们应该在运行程序之前保存它。您应该能够在终端窗口中键入内容来响应每个提示。

* * *

## 定义和使用运算符

现在，让我们在计算器程序中加入加、乘、除、减等运算符。

```py
num_1 = int(input('Enter your first number: '))
num_2 = int(input('Enter your second number: '))

# Addition
print('{} + {} = '.format(num_1, num_2))
print(num_1 + num_2)

# Subtraction
print('{} - {} = '.format(num_1, num_2))
print(num_1 - num_2)

# Multiplication
print('{} * {} = '.format(num_1, num_2))
print(num_1 * num_2)

# Division
print('{} / {} = '.format(num_1, num_2))
print(num_1 / num_2)
# The format() will help out output look descent and formatted.

```

**输出**:

```py
Enter your first number: 15
Enter your second number: 10
15 + 10 = 
25
15 - 10 =
05
15 * 10 =
150
15 / 10 =
1.5

```

如果你看一下上面的输出，我们可以注意到，只要用户输入`num_1`作为`15`和`num_2`作为`10`，计算器的所有操作都会被执行。

如果我们想限制程序一次只执行一个操作，我们就必须使用条件语句，使整个计算器程序成为基于用户选择的操作程序。

* * *

## 包括使程序成为用户选择的条件语句

因此，我们将开始在程序的顶部添加一些信息，以及要做出的选择，以使用户理解他/她应该选择什么。

```py
choice = input('''
Please select the type of operation you want to perform:
+ for addition
- for subtraction
* for multiplication
/ for division
''')

num_1 = int(input('Enter your first number: '))
num_2 = int(input('Enter your second number: '))

if choice == '+':
    print('{} + {} = '.format(num_1, num_2))
    print(num_1 + num_2)

elif choice == '-':
    print('{} - {} = '.format(num_1, num_2))
    print(num_1 - num_2)

elif choice == '*':
    print('{} * {} = '.format(num_1, num_2))
    print(num_1 * num_2)

elif choice == '/':
    print('{} / {} = '.format(num_1, num_2))
    print(num_1 / num_2)

else:
    print('Enter a valid operator, please run the program again.')

```

**输出**:

```py
Please select the type of operation you want to perform:
+ for addition
- for subtraction
* for multiplication
/ for division

* 

Please enter the first number: 10
Please enter the second number: 40
10 * 40 = 
400

```

* * *

## 参考

*   Python 计算器简单程序
*   [Python if else elif 语句](https://www.askpython.com/python/python-if-else-elif-statement)