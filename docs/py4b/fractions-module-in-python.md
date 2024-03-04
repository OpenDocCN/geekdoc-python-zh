# Python 中的分数模块

> 原文：<https://www.pythonforbeginners.com/basics/fractions-module-in-python>

你一定在 python 中使用过整数和浮点数这样的数字数据类型。但是你用过分数的实际形式吗？在本文中，我们将学习分数，并使用 python 中的分数模块对分数进行运算。

## 如何使用 Python 中的分数模块？

我们可以使用 python 中分数模块的分数方法来创建分数形式的有理数。我们可以像下面这样导入模块。

```py
import fractions
```

在 python 中，我们可以将整数、浮点数或字符串转换成分数。为了将两个整数的比率转换为分数，我们使用了分数模块的 fraction()方法，并将分子作为第一个参数传递，将分母作为第二个参数传递。该函数返回一个分数对象，如下所示。

```py
 import fractions
myInt1=1
myInt2=2
print("Integer 1(numerator) is:",myInt1)
print("Integer 2(denominator) is:",myInt2)
myFraction=fractions.Fraction(myInt1,myInt2)
print("Fraction value is:",myFraction)
```

输出:

```py
Integer 1(numerator) is: 1
Integer 2(denominator) is: 2
Fraction value is: 1/2
```

我们可以使用 fractions 模块的 fraction 方法从浮点数中获得一个分数值。当我们将一个浮点数作为输入传递给 Fraction()方法时，它返回相应的分数值，如下所示。

```py
import fractions
myFloat=0.5
print("Floating point number is:",myFloat)
myFraction=fractions.Fraction(myFloat)
print("Fraction value is:",myFraction)
```

输出:

```py
Floating point number is: 0.5
Fraction value is: 1/2
```

我们也可以使用 fraction()方法将字符串转换成分数。我们可以将分数的字符串表示或字符串中的浮点文字作为输入传递给 fraction 方法，该方法返回相应的分数值，如下所示。

```py
import fractions
myStr1="0.5"
print("String literal is:",myStr1)
myFraction1=fractions.Fraction(myStr1)
print("Fraction value is:",myFraction1)
myStr2="1/2"
print("String literal is:",myStr2)
myFraction2=fractions.Fraction(myStr2)
print("Fraction value is:",myFraction2)
```

输出:

```py
String literal is: 0.5
Fraction value is: 1/2
String literal is: 1/2
Fraction value is: 1/2
```

## 分数怎么四舍五入？

我们可以在 python 中使用 round()方法，根据分数分母所需的位数，对分数进行四舍五入。round()方法把要舍入的分数作为第一个参数，把分母要舍入到的位数作为第二个参数。该函数返回分母中有所需位数的分数。这可以这样理解。

```py
 import fractions
myInt1=50
myInt2=3
myFraction=fractions.Fraction(myInt1,myInt2)
print("Fraction value is:",myFraction)
rounded=round(myFraction,2)
print("Rounded value is:",rounded)
```

输出:

```py
Fraction value is: 50/3
Rounded value is: 1667/100
```

当我们没有将分母四舍五入的位数作为第二个参数传递时，round()方法会将分数转换为最接近的整数。这可以看如下。

```py
import fractions
myInt1=50
myInt2=3
myFraction=fractions.Fraction(myInt1,myInt2)
print("Fraction value is:",myFraction)
rounded=round(myFraction)
print("Rounded value is:",rounded)
```

输出:

```py
Fraction value is: 50/3
Rounded value is: 17
```

## 从分数中获取分子和分母

我们也可以从分数中提取分子和分母。为了提取分子，我们使用 fraction 对象的“分子”字段。类似地，为了提取分母，我们使用 fraction 对象的“分母”字段。这可以从下面的例子中理解。

```py
import fractions
myInt1=50
myInt2=3
myFraction=fractions.Fraction(myInt1,myInt2)
print("Fraction value is:",myFraction)
print("Numerator is:",myFraction.numerator)
print("Denominator is:",myFraction.denominator)
```

输出:

```py
Fraction value is: 50/3
Numerator is: 50
Denominator is: 3
```

## 分数的算术运算

我们可以使用 python 中的分数模块对分数执行加减乘除等算术运算，就像我们对整数和浮点数等其他[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)执行这些运算一样。

我们可以在 python 中对两个给定的分数执行算术运算，如下所示。

```py
 import fractions
myFraction1=fractions.Fraction(50,3)
myFraction2=fractions.Fraction(1,2)
print("First Fraction value is:",myFraction1)
print("Second Fraction value is:",myFraction2)
print("Fraction1 + Fraction2 is:",myFraction1 + myFraction2)
print("Fraction1 - Fraction2 is:",myFraction1 - myFraction2)
print("Fraction1 * Fraction2 is:",myFraction1 * myFraction2)
print("Fraction1 / Fraction2 is:",myFraction1 / myFraction2)
```

输出:

```py
First Fraction value is: 50/3
Second Fraction value is: 1/2
Fraction1 + Fraction2 is: 103/6
Fraction1 - Fraction2 is: 97/6
Fraction1 * Fraction2 is: 25/3
Fraction1 / Fraction2 is: 100/3
```

## 使用分数模块从浮点数中获取近似有理值

我们可以从任何浮点或十进制值中获得分数形式的有理数。为了从十进制数中获得分数，我们可以将十进制数传递给 fraction()方法，该方法会将它们转换为有理数，如下所示。

```py
 import fractions
myFloat= 22/7
print("Floating point value is:",myFloat)
myFraction=fractions.Fraction(myFloat)
print("Fraction value is:",myFraction)
```

输出:

```py
Floating point value is: 3.142857142857143
Fraction value is: 7077085128725065/2251799813685248
```

在获得上述格式的分数后，我们可以使用 limit_denominator()方法限制分母的最高值。当对分数调用 limit_denominator()方法时，该方法将分母的最大允许值作为输入，并返回相应的分数。这可以从下面的例子中理解。

```py
import fractions
myFloat= 22/7
print("Floating point value is:",myFloat)
myFraction=fractions.Fraction(myFloat)
print("Fraction value is:",myFraction)
myFraction1=myFraction.limit_denominator(100)
print("Approximate Fraction value with denominator limited to 100 is:",myFraction1)
```

输出:

```py
Floating point value is: 3.142857142857143
Fraction value is: 7077085128725065/2251799813685248
Approximate Fraction value with denominator limited to 100 is: 22/7
```

## 结论

在本文中，我们研究了分数数据类型，并使用 python 中的分数模块实现了它。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。