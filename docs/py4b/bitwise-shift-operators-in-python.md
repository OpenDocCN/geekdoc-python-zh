# Python 中的按位移位运算符

> 原文：<https://www.pythonforbeginners.com/basics/bitwise-shift-operators-in-python>

Python 中有各种类型的运算符，如算术运算符、比较运算符和位运算符。在我们的程序中，我们使用这些操作符来控制执行顺序和操作数据。在本文中，我们将研究不同的 python 按位移位操作符，它们的功能和例子。

## 什么是按位移位运算符？

按位移位运算符是二元运算符。这些运算符用于将一个数的二进制表示向左或向右移动一定的位置。按位移位运算符通常用于需要将整数乘以或除以 2 的幂的运算。这里，按位左移运算符用于将一个数乘以 2 的幂，而 python 中的按位右移运算符用于将一个数除以 2 的幂。

## Python 中的按位右移运算符

python 中的按位右移运算符将输入数字的二进制表示形式的位向右移动指定的位数。移位产生的空位由 0 填充。

按位右移的语法是**a>n。**这里的“a”是其位将向右移动“n”位的数字。

从下图可以理解按位右移操作的工作原理。

假设我们必须将 14 的位移动 2 位。我们首先将它转换成二进制格式。

*   二进制格式的 14 写成 1110。

移位后，最右边的两位 1 和 0 将被丢弃，最左边的空位将被填充 0。14 >> 2 的输出将是二进制的 0011，它转换为整数格式的值 3。

在这里，您可以观察到我们已经将位移动了 2 位，因此输入数被 2²除，即 4。同样，如果我们将数字右移 n 位，数字的整数值将除以 2^n 。我们可以使用下面的程序在 python 中使用右移位操作符来验证这个输出。

```py
myNum1 = 14
myNum2 = 2
shiftNum = myNum1 >> myNum2
print("Operand 1 is:", myNum1)
print("operand 2 is:", myNum2)
print("Result of the right shift operation on {} by {} bits is {}.".format(myNum1, myNum2, shiftNum)) 
```

输出:

```py
Operand 1 is: 14
operand 2 is: 2
Result of the right shift operation on 14 by 2 bits is 3. 
```

## Python 中的按位左移运算符

Python 中的按位左移运算符将输入数字的二进制表示形式的位向左移动指定的位数。移位产生的空位由 0 填充。

按位左移的语法是**a<n。**这里的“a”是其位将向左移动“n”位的数字。

从下图可以理解按位左移操作的工作原理。

假设我们必须将 14 的位移动 2 位。我们首先将它转换成二进制格式。

*   二进制格式的 14 写成 1110。

移位后，最右边的空位将用 0 填充。14 << 2 的输出将是二进制的 111000，转换为整数格式的值 56。

在这里，您可以观察到，我们将位移动了 2 位，因此输入数乘以了 2²，即 4。同样，如果我们将该数左移 n 位，该数的整数值将乘以 2^n 。我们可以使用 python 中的左移位操作符，通过下面的程序来验证这个输出。

```py
myNum1 = 14
myNum2 = 2
shiftNum = myNum1 << myNum2
print("Operand 1 is:", myNum1)
print("operand 2 is:", myNum2)
print("Result of the left shift operation on {} by {} bits is {}.".format(myNum1, myNum2, shiftNum)) 
```

输出:

```py
Operand 1 is: 14
operand 2 is: 2
Result of the left shift operation on 14 by 2 bits is 56. 
```

## 结论

在本文中，我们讨论了按位移位运算符、它们的语法以及 Python 中的例子。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)