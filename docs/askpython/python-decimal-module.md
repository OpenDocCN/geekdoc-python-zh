# Python 十进制模块–你需要知道的 7 个函数！

> 原文：<https://www.askpython.com/python-modules/python-decimal-module>

嘿，大家好！在本文中，我们将看看其中一个有趣的模块——Python 十进制模块。

无论在哪个领域，我们都需要搜索函数来执行数学运算。Python 十进制模块为我们提供了我们需要的所有数学函数。

所以，让我们开始吧！

* * *

## 了解 Python 十进制模块

Python decimal 模块包含各种函数来处理数字数据并对其执行不同的数学运算。使用十进制模块，我们可以在整个程序中有效地处理十进制数。

`The decimal module`为我们提供了控制和克服小数值精度问题的功能。

理解了十进制模块的需要后，现在让我们看看该模块提供的一些重要功能。

为了使用这些功能，我们需要导入如下所示的模块:

```py
import decimal

```

* * *

## 十进制模块的功能及实现

可以对十进制或数字数据执行不同的算术运算来提高结果。

我们可以使用如下所示的`decimal.Decimal() function`来定义小数点数字

**语法:**

```py
import decimal
variable = decimal.Decimal(decimal-number)

```

此外，我们可以通过十进制模块的内置函数— `decimal.getcontext().prec`函数来控制小数点位数结果的精度值。

**语法:**

```py
decimal.getcontext().prec = precision value

```

下面解释的函数帮助我们高效快速地执行小数点算术运算。

* * *

### 1.exp()函数–指数计算

`exp() function`计算传递给它的特定十进制数的**指数值，即 e^x** 。

**语法:**

```py
decimal.Decimal(decimal-number).exp()

```

**举例:**

```py
import decimal as d

d.getcontext().prec = 5

#Intializing with an addition operation
val = d.Decimal(12.201) + d.Decimal(12.20)

#Calculating exponential of the decimal value
exp = val.exp()

#variable with no calculations
no_math = d.Decimal(1.131231)

print("Sum: ",val)
print("Exponential: ", exp)
print(no_math)

```

**输出:**

```py
Decimal Number:  24.401                                                                                                       
3.9557E+10                                                                                                                    
1.131231000000000097571728474576957523822784423828125   

```

注意我们输出的数字总数是 5？这是因为我们在这里设置了精度值。

需要记住的一点是，当您对两位小数执行数学运算时，精度值适用，而不是当您使用上面的“no_math”变量中显示的值直接初始化变量时。

* * *

### 2.sqrt()函数——平方根

sqrt()函数计算并返回传递给它的十进制数的平方根值。

**语法:**

```py
decimal.Decimal(decimal-number).sqrt()

```

**举例:**

```py
import decimal as d

d.getcontext().prec = 3

val = d.Decimal(122.20)
sqrt = val.sqrt()
print("Decimal Number: ",val)
print("Square root of the decimal number: ",sqrt)

```

**输出:**

```py
Decimal Number:  122.2000000000000028421709430404007434844970703125
Square root of the decimal number:  11.1

```

同样，请注意声明的值如何包含完整的十进制数，而计算的值遵循我们的 3 位数精度集。

要找到更多的数学运算，请阅读我们关于 Python 中的[数学模块的文章](https://www.askpython.com/python-modules/python-math-module)

* * *

### 3.对数函数

十进制模块为我们提供了以下函数来计算小数点数字的对数值

*   **decimal.ln()**
*   **decimal.log10()**

`decimal.ln() function` 返回十进制数的自然对数值，如下所示

```py
decimal.Decimal(decimal-number).ln()

```

decimal.log10()函数用于计算传递给它的十进制数的以 10 为底的对数值。

```py
decimal.Decimal(decimal-number).log10()

```

**举例:**

```py
import decimal as d

d.getcontext().prec = 2

val = d.Decimal(122.20)

log = val.ln()
print("Natural log value of the decimal number: ",log)

log_10 = val.log10()
print("Log value with base 10 of the decimal number: ",log_10)

```

**输出:**

```py
Natural log value of the decimal number:  4.8
Log value with base 10 of the decimal number:  2.1

```

* * *

### 4.compare()函数

`decimal.compare() function`比较两个小数点，并根据以下条件返回值

*   如果第一个十进制数小于第二个十进制数，则返回-1。
*   如果第一个十进制数大于第二个十进制数，则返回 1。
*   如果两个小数点的值相等，则返回 0。

**举例:**

```py
import decimal as d

valx = d.Decimal(122.20)
valy = d.Decimal(123.01)

print("Value 1: ",valx)
print("Value 2: ",valy)

compare = valx.compare(valy)
print(compare)

```

**输出:**

```py
Value 1:  122.2000000000000028421709430404007434844970703125
Value 2:  123.0100000000000051159076974727213382720947265625
-1

```

* * *

### 5.函数的作用是

`decimal.copy_abs() function`返回传递给它的有符号十进制数的绝对值。

**语法:**

```py
decimal.Decimal(signed decimal number).copy_abs()

```

**举例:**

```py
import decimal as d

valx = d.Decimal(-122.20)
print("Value 1: ",valx)

absolute = valx.copy_abs()
print("Absolute value of the given decimal number: ",absolute)

```

**输出:**

```py
Value 1:  -122.2000000000000028421709430404007434844970703125
Absolute value of the given decimal number:  122.2000000000000028421709430404007434844970703125

```

* * *

### 6.最大和最小函数

Python decimal 模块包含以下函数来计算小数点数字的最小值和最大值。

*   **min()函数:返回两个十进制值中的最小值。**
*   **max()函数:返回两个小数值中的最大值。**

```py
#Syntax for min() function-
decimal1.min(decimal2)

#Syntax for max() function-
decimal1.max(decimal2)

```

**举例:**

```py
import decimal as d

valx = d.Decimal(122.20)
valy = d.Decimal(123.01)

print("Value 1: ",valx)
print("Value 2: ",valy)

min_val = valx.min(valy)
print("The minimum of the two values: ",min_val)

max_val = valx.max(valy)
print("The maximum of the two values: ",max_val)

```

**输出:**

```py
Value 1:  122.2000000000000028421709430404007434844970703125
Value 2:  123.0100000000000051159076974727213382720947265625
The minimum of the two values:  122.2000000000000028421709430
The maximum of the two values:  123.0100000000000051159076975

```

* * *

### 7.十进制模块的逻辑运算

Decimal 模块包含一组内置函数，用于对十进制数执行[逻辑运算](https://www.askpython.com/python/python-logical-operators)，如 AND、OR、XOR 等。

*   **logical_and()函数:**对两个十进制数进行逻辑与运算，并返回结果。
*   **logical_or()函数:**对两个十进制数进行逻辑或运算，并返回结果。
*   **logical_xor()函数:**对两个十进制数进行逻辑异或运算，并返回结果。

```py
#Syntax for logical_and() function-
decimal1.logical_and(decimal2)

#Syntax for logical_or() function-
decimal1.logical_or(decimal2)

#Syntax for logical_xor() function-
decimal1.logical_xor(decimal2)

```

**举例:**

```py
import decimal as d

valx = d.Decimal(1001)
valy = d.Decimal(1111)

print("Value 1: ",valx)
print("Value 2: ",valy)

AND = valx.logical_and(valy)
print("The logical AND value of the two decimals: ",AND)

OR = valx.logical_or(valy)
print("The logical OR value of the two decimals: ",OR)

XOR = valx.logical_xor(valy)
print("The logical XOR value of the two decimals: ",XOR)

```

**输出:**

```py
Value 1:  1001
Value 2:  1111
The logical AND value of the two decimals:  1001
The logical OR value of the two decimals:  1111
The logical XOR value of the two decimals:  110

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   了解 Python 十进制模块— JournalDev