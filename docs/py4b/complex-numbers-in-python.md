# Python 中的复数

> 原文：<https://www.pythonforbeginners.com/data-types/complex-numbers-in-python>

在进行数据科学、机器学习或科学计算时，我们经常需要对包括复数在内的[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)进行计算。在本文中，我们将学习如何在 python 中定义和使用复数。

## 什么是复数？

复数是可以写成(a+b j)形式的数，其中 a 和 b 是实数。这里 j 是一个虚数，定义为-1 的平方根。复数成对出现，主要用于计算负数的平方根。

## Python 中如何定义复数？

我们可以在 python 中定义复数，只需声明一个变量并指定一个(a+bj)形式的表达式。这里的“a”和“b”应该是一个数字文字，而“j”可以是任何字母字符。我们还可以使用 type()函数检查已定义变量的数据类型，如下所示。

```py
myNum= 3+2j
print("The Number is:")
print(myNum)
print("Data type of Number is:")
print(type(myNum))
```

输出:

```py
The Number is:
(3+2j)
Data type of Number is:
<class 'complex'>
```

我们也可以使用 complex()函数定义一个复数。复函数将强制输入作为表示复数的实部的第一参数，将可选输入作为表示复数的虚部的第二参数。我们可以使用 complex()函数定义一个复数，如下所示。

```py
myNum= complex(3,2)
print("The Number is:")
print(myNum)
print("Data type of Number is:")
print(type(myNum))
```

输出:

```py
The Number is:
(3+2j)
Data type of Number is:
<class 'complex'>
```

## 提取复数的实部和虚部

在一个复数(a+b j)中，“a”称为实部，“b”称为虚部。我们可以使用包含值“a”的名为“real”的属性提取复数的实部，并使用包含值“b”的属性“imag”提取虚部，如下所示。

```py
myNum= complex(3,2)
print("The Complex Number is:")
print(myNum)
print("Real part of the complex Number is:")
print(myNum.real)
print("Imaginary part of the complex Number is:")
print(myNum.imag)
```

输出:

```py
 The Complex Number is:
(3+2j)
Real part of the complex Number is:
3.0
Imaginary part of the complex Number is:
2.0
```

## Python 中复数的共轭

对于一个复数(a+ b j)，它的共轭定义为复数(a- b j)。我们可以在 python 中使用 conjugate()方法获得任意复数的共轭。在复数上调用 conjugate()方法时，会返回该数字的复共轭。这可以如下进行。

```py
myNum= complex(3,2)
print("The Complex Number is:")
print(myNum)
print("Conjugate of the complex Number is:")
print(myNum.conjugate())
```

输出:

```py
The Complex Number is:
(3+2j)
Conjugate of the complex Number is:
(3-2j)
```

## 复数的大小

复数(a+b j)的大小是(a，b)点到(0，0)的距离。它是代表复数的向量的长度。在 python 中，可以按如下方式计算复数的大小。

```py
import math
def magnitude(num):
    x=num.real
    y=num.imag
    mag=x*x+y*y
    return math.sqrt(mag)
myNum= complex(3,2)
print("The Complex Number is:")
print(myNum)
print("Magnitude of the complex Number is:")
print(magnitude(myNum))
```

输出:

```py
The Complex Number is:
(3+2j)
Magnitude of the complex Number is:
3.605551275463989
```

## 结论

在本文中，我们学习了 python 中复数及其属性的基础知识。我们还推导了复数的实部、虚部、共轭和幅度。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。