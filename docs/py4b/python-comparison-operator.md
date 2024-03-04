# Python 比较运算符

> 原文：<https://www.pythonforbeginners.com/basics/python-comparison-operator>

Python 中有各种类型的运算符，如算术运算符、比较运算符和位运算符。在我们的程序中，我们使用这些操作符来控制执行顺序和操作数据。在本文中，我们将研究不同 python 比较操作符、它们的功能和例子。

## 什么是 Python 比较运算符？

比较运算符是用于比较程序中两个对象的二元运算符。这些操作符也称为关系操作符，因为它们决定了 python 中两个对象之间的关系。

比较运算符在比较操作数后返回布尔值“真”或“假”。

## Python 等于运算符

Python 等于运算符用于检查两个对象是否相等。

python 中等于运算符的语法是 a == b。这里 a 和 b 是正在进行相等性检查的操作数。变量 a 和 b 可以包含具有诸如整数、浮点或字符串之类的原始数据类型的任何对象，或者它们可以包含对诸如列表、元组、集合和字典之类的容器对象的引用。如果两个操作数相等，则输出为真。否则，输出将为假。

您可以使用下面的程序来理解等于运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 == myNum2
compare13 = myNum1 == myNum3
compare14 = myNum1 == myNum4
print("{} is equal to {}?: {}".format(myNum1, myNum2, compare12))
print("{} is equal to {}?: {}".format(myNum1, myNum3, compare13))
print("{} is equal to {}?: {}".format(myNum1, myNum4, compare14))
```

输出:

```py
14 is equal to 14?: True
14 is equal to 10?: False
14 is equal to 5?: False
```

## Python 不等于运算符

Python 不等于运算符用于检查两个对象是否不相等。

python 中等于运算符的语法是！= b。这里 a 和 b 是被检查是否不相等的操作数。变量 a 和 b 可以包含具有诸如整数、浮点或字符串之类的原始数据类型的任何对象，或者它们可以包含对诸如列表、元组、集合和字典之类的容器对象的引用。如果两个操作数相等，则输出为假。否则，输出将为真。

使用下面的程序，你可以理解不等于运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 != myNum2
compare13 = myNum1 != myNum3
compare14 = myNum1 != myNum4
print("{} is not equal to {}?: {}".format(myNum1, myNum2, compare12))
print("{} is not equal to {}?: {}".format(myNum1, myNum3, compare13))
print("{} is not equal to {}?: {}".format(myNum1, myNum4, compare14))
```

输出:

```py
14 is not equal to 14?: False
14 is not equal to 10?: True
14 is not equal to 5?: True
```

## Python 小于运算符

Python 小于运算符用于检查一个对象是否小于另一个对象。

python 中小于运算符的语法是 a < b。变量 a 和 b 可以包含任何具有整数、浮点或字符串等原始数据类型的对象，也可以包含对列表、元组、集和字典等容器对象的引用。如果 a 小于 b，则输出为真，否则输出为假。

使用下面的程序，您可以理解小于运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 < myNum2
compare13 = myNum3 < myNum1
compare14 = myNum1 < myNum4
print("{} is less than {}?: {}".format(myNum1, myNum2, compare12))
print("{} is less than {}?: {}".format(myNum3, myNum1, compare13))
print("{} is less than {}?: {}".format(myNum1, myNum4, compare14))
```

输出:

```py
14 is less than 14?: False
10 is less than 14?: True
14 is less than 5?: False
```

小于运算符不支持字符串和数字数据类型(如 float 或 int)之间的比较。如果我们尝试执行这样的比较，将会出现 TypeError。你可以使用除了块之外的 [python try 来避免它。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

## Python 大于运算符

Python 大于运算符用于检查一个对象是否大于另一个对象。

python 中大于运算符的语法是 a > b。变量 a 和 b 可以包含任何具有整数、浮点或字符串等原始数据类型的对象，也可以包含对列表、元组、集和字典等容器对象的引用。如果 a 大于 b，则输出为真，否则输出为假。

使用下面的程序，您可以理解大于号运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 > myNum2
compare13 = myNum1 > myNum3
compare14 = myNum1 > myNum4
print("{} is greater than {}?: {}".format(myNum1, myNum2, compare12))
print("{} is greater than {}?: {}".format(myNum1, myNum3, compare13))
print("{} is greater than {}?: {}".format(myNum1, myNum4, compare14))
```

输出:

```py
14 is greater than 14?: False
14 is greater than 10?: True
14 is greater than 5?: True
```

大于运算符不支持字符串和数字数据类型(如 float 或 int)之间的比较。如果我们尝试执行这样的比较，将会出现 TypeError。

## Python 小于或等于

Python 小于或等于运算符用于检查一个对象是否小于或等于另一个对象。

python 中小于或等于运算符的语法是 a <= b。变量 a 和 b 可以包含任何具有整数、浮点或字符串等原始数据类型的对象，也可以包含对列表、元组、集和字典等容器对象的引用。如果 a 小于或等于 b，则输出为真，否则输出为假。

您可以使用下面的程序来理解小于或等于运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 <= myNum2
compare13 = myNum3 <= myNum1
compare14 = myNum1 <= myNum4
print("{} is less than or equal to {}?: {}".format(myNum1, myNum2, compare12))
print("{} is less than or equal to {}?: {}".format(myNum3, myNum1, compare13))
print("{} is less than or equal to {}?: {}".format(myNum1, myNum4, compare14)) 
```

输出:

```py
14 is less than or equal to 14?: True
10 is less than or equal to 14?: True
14 is less than or equal to 5?: False
```

## Python 大于或等于

Python 大于或等于运算符用于检查一个对象是否大于或等于另一个对象。

python 中大于或等于运算符的语法是 a >= b。变量 a 和 b 可以包含任何具有整数、浮点或字符串等原始数据类型的对象，也可以包含对列表、元组、集和字典等容器对象的引用。如果 a 大于或等于 b，则输出为真，否则输出为假。

您可以使用下面的程序理解大于或等于运算符的工作原理。

```py
myNum1 = 14
myNum2 = 14
myNum3 = 10
myNum4 = 5
compare12 = myNum1 >= myNum2
compare13 = myNum1 >= myNum3
compare14 = myNum1 >= myNum4
print("{} is greater than or equal to {}?: {}".format(myNum1, myNum2, compare12))
print("{} is greater than or equal to {}?: {}".format(myNum1, myNum3, compare13))
print("{} is greater than or equal to {}?: {}".format(myNum1, myNum4, compare14)) 
```

输出:

```py
14 is greater than or equal to 14?: True
14 is greater than or equal to 10?: True
14 is greater than or equal to 5?: True
```

## 结论

在本文中，我们讨论了每一个 python 比较操作符。我们还讨论了一些示例以及在使用每个 python 比较运算符执行比较时可能出现的错误。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)