# Python 中的十进制模块

> 原文：<https://www.pythonforbeginners.com/basics/decimal-module-in-python>

Python 有[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)，如 int、float 和复数，但由于浮点数的机器依赖性，我们需要更精确的数据类型来进行要求高精度的计算。在本文中，我们将研究 python 中的十进制模块，它实现了精度高达 28 位的十进制数。

## Python 中什么时候应该用 decimal 代替 float？

Python 将十进制数实现为依赖于机器的双精度浮点数。对于精度因业务原因而至关重要的计算，浮点数在不同的机器上运行时可能会导致错误。因此，对于这样的应用程序，我们需要一个独立于机器的数据类型来实现十进制数，这已经使用 python 中的 decimal 模块实现了。此外，十进制模块实现了精度高达 28 位十进制数的十进制数，而浮点数的精度只有 18 位。这可以在下面的例子中观察到。

```py
import decimal
float_division=4/3
decimal_devision=decimal.Decimal(4)/decimal.Decimal(3) 
print("Result for float division of 4 by 3:")
print(float_division)
print("Result for decimal division of 4 by 3:")
print(decimal_devision)
```

输出:

```py
Result for float division of 4 by 3:
1.3333333333333333
Result for decimal division of 4 by 3:
1.333333333333333333333333333
```

使用小数而不是浮点数的第二个原因是，在 python 中不能使用数字的精确值来表示数字，而只能使用近似值，这对关键程序来说是危险的。

由于近似值，浮点值对于不同的计算会产生不同的结果。例如，如果我们使用浮点值将 1.2 和 2.2 相加，答案应该等于 3.4。但是当我们把加出来的数和 3.4 比较，就不相等了。出现这种错误是因为浮点数的近似值，1.2 和 2.2 之和不等于 3.4。因此，在需要比较十进制值的情况下，我们应该使用十进制模块，而不是浮点数。从下面的例子可以更清楚地看出这一点。

```py
x=1.2
y=2.2
z=3.4
a=x+y
print("x:",x)
print("y:",y)
print("z:",z)
print("x+y:",a)
print("x+y==z?:",a==z)
```

输出:

```py
x: 1.2
y: 2.2
z: 3.4
x+y: 3.4000000000000004
x+y==z?: False
```

## Python 中的十进制模块怎么用？

要在 python 中使用 decimal 模块，我们可以如下导入它。

```py
import decimal
```

导入的模块具有预定义的上下文，其中包含精度、舍入、标志、允许的最小和最大数值的默认值。我们可以使用 getcontext()方法查看上下文的值，如下所示。

```py
 import decimal
print(decimal.getcontext())
```

输出:

```py
Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[Inexact, Rounded], traps=[InvalidOperation, DivisionByZero, Overflow])
```

我们还可以设置上下文的精度和其他参数。要将精度从 28 位改为 3 位，我们可以使用下面的程序。

```py
import decimal
decimal.getcontext().prec=3
print(decimal.getcontext())
```

输出:

```py
Context(prec=3, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[Inexact, Rounded], traps=[InvalidOperation, DivisionByZero, Overflow])
```

默认情况下，在使用小数模块对小数进行舍入时，数字会被均匀地舍入。我们可以通过改变上下文中的“舍入”值来改变这种行为，如下所示。

```py
import decimal
decimal.getcontext().rounding="ROUND_HALF_DOWN"
print(decimal.getcontext())
```

输出:

```py
Context(prec=3, rounding=ROUND_HALF_DOWN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[Inexact, Rounded], traps=[InvalidOperation, DivisionByZero, Overflow])
```

由十进制模块定义的十进制数的所有算术运算都类似于浮点数。差异在于由于实现的不同而导致的值的精度。

## 舍入值

我们可以使用 round()函数将数字四舍五入到特定的位数。round 函数将需要舍入的十进制数作为第一个参数，将需要舍入的位数作为第二个参数，并返回舍入后的十进制值，如下所示。

```py
import decimal
num1=decimal.Decimal(4)
num2=decimal.Decimal(3)
print("First number is:",num1)
print("Second number is:",num2)
num3=num1/num2
print("num1 divided by num2 is:",num3)
num4=round(num3,2)
print("Rounded value upto two decimal points is:",num4)
```

输出:

```py
First number is: 4
Second number is: 3
num1 divided by num2 is: 1.333333333333333333333333333
Rounded value upto two decimal points is: 1.33
```

## 使用十进制模块比较数字

正如在上面的一节中所看到的，比较浮点数会导致不正确的结果，但是十进制数是精确的，它们的比较总是会得到预期的结果。这可以看如下。

```py
import decimal
x=decimal.Decimal("1.2")
y=decimal.Decimal("2.2")
z=decimal.Decimal("3.4")
a=x+y
print("x:",x)
print("y:",y)
print("z:",z)
print("x+y:",a)
print("x+y==z?:",a==z)
```

输出:

```py
x: 1.2
y: 2.2
z: 3.4
x+y: 3.4
x+y==z?: True
```

在上面的代码中，你可以看到我们将字符串转换成了小数，而不是将浮点数转换成了小数。这样做是因为如果我们将浮点数转换为十进制数，近似误差会传播到十进制数，我们将不会得到所需的输出。

## 结论

在本文中，我们研究了使用浮点数进行算术运算的缺点，并使用十进制模块在 python 中实现了相同的运算，并且没有错误。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。