# Python 中的 cmath 模块简介

> 原文：<https://www.pythonforbeginners.com/basics/introduction-to-cmath-module-in-python>

在进行数据科学、机器学习或科学计算时，我们经常需要对包括复数在内的[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)进行计算。在本文中，我们将使用 python 中的 cmath 模块，使用该模块中提供的不同方法对复数执行操作。

## 计算复数的相位

复数的相位定义为实轴和代表复数的向量之间的角度。使用 cmath 模块，我们可以使用 phase()方法找到一个复数的相位。phase 方法将一个复数作为输入，并返回一个表示该复数相位的浮点数，如下所示。

```py
import cmath
myNum=3+2j
print("The complex number is:",myNum)
myPhase= cmath.phase(myNum)
print("Phase of  the complex number is:",myPhase)
```

输出:

```py
The complex number is: (3+2j)
Phase of  the complex number is: 0.5880026035475675
```

## 复数的极坐标

在极坐标中，复数被定义为由作为第一元素的复数的模和作为第二元素的复数的相位组成的元组。我们可以用 python 中的 polar()方法求一个复数的极坐标。polar()方法将一个复数作为输入，并返回一个表示极坐标的元组，如下所示。

```py
import cmath
myNum=3+2j
print("The complex number is:",myNum)
myPol= cmath.polar(myNum)
print("Polar coordinates of  the complex number are:",myPol)
```

输出:

```py
The complex number is: (3+2j)
Polar coordinates of  the complex number are: (3.605551275463989, 0.5880026035475675)
```

如果我们知道一个复数的模数和相位，也就是说，如果我们知道一个复数的极坐标，我们可以使用 rect()方法获得该复数。rect()方法将模数作为第一个参数，将复数的相位作为第二个参数，并返回相应的复数，如下所示。

```py
import cmath
myPol= (3.605551275463989, 0.5880026035475675)
print("Polar coordinates of  the complex number are:",myPol)
print("Modulus of the complex number is:",myPol[0])
print("Phase of the complex number is:",myPol[1])
myRec=cmath.rect(myPol[0],myPol[1])
print("Complex number in rectangular form is:",myRec)
```

输出:

```py
Polar coordinates of  the complex number are: (3.605551275463989, 0.5880026035475675)
Modulus of the complex number is: 3.605551275463989
Phase of the complex number is: 0.5880026035475675
Complex number in rectangular form is: (3+1.9999999999999996j)
```

## cmath 模块中的常数

cmath 模块还提供某些数学常数，如无穷大、NaN 和 pi，它们在数学计算中很有用。下面的示例中给出了一些常数。

```py
import cmath
print("Given below are some of the constants defined in cmath module.")
print("e:",cmath.e)
print("Infinity (real axis):",cmath.inf)
print("Infinity (Imaginary axis):",cmath.infj)
print("NaN (real):",cmath.nan)
print("NaN (imaginary):",cmath.nanj)
print("Pi:",cmath.pi)
print("Tau:",cmath.tau)
```

输出:

```py
Given below are some of the constants defined in cmath module.
e: 2.718281828459045
Infinity (real axis): inf
Infinity (Imaginary axis): infj
NaN (real): nan
NaN (imaginary): nanj
Pi: 3.141592653589793
Tau: 6.283185307179586
```

## cmath 模块中的三角函数

对于复数的数学计算，cmath 模块提供了一组三角函数。所有三角函数都将复数作为输入，并返回一个表示三角函数相应输出的复数。下面讨论一些例子。

```py
import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Sine of the complex number is:",cmath.sin(myNum))
print("Cosine of the complex number is:",cmath.cos(myNum))
print("Tangent of the complex number is:",cmath.tan(myNum))
print("Inverse Sine of the complex number is:",cmath.asin(myNum))
print("Inverse Cosine of the complex number is:",cmath.acos(myNum))
print("Inverse Tangent of the complex number is:",cmath.atan(myNum)) 
```

输出:

```py
Complex number is: (3+2j)
Sine of the complex number is: (0.5309210862485197-3.59056458998578j)
Cosine of the complex number is: (-3.7245455049153224-0.5118225699873846j)
Tangent of the complex number is: (-0.009884375038322495+0.965385879022133j)
Inverse Sine of the complex number is: (0.9646585044076028+1.9686379257930964j)
Inverse Cosine of the complex number is: (0.6061378223872937-1.9686379257930964j)
Inverse Tangent of the complex number is: (1.3389725222944935+0.14694666622552977j)
```

## cmath 模中的双曲函数

就像三角函数一样，cmath 模块也为 python 中的数学计算提供了双曲函数和反双曲三角函数。所有这些函数都将一个复数作为输入，并根据其性质返回一个表示双曲或反双曲三角函数输出的复数。下面是一些例子。

```py
import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Hyperbolic Sine of the complex number is:",cmath.sinh(myNum))
print("Hyperbolic Cosine of the complex number is:",cmath.cosh(myNum))
print("Hyperbolic Tangent of the complex number is:",cmath.tanh(myNum))
print("Inverse Hyperbolic Sine of the complex number is:",cmath.asinh(myNum))
print("Inverse Hyperbolic Cosine of the complex number is:",cmath.acosh(myNum))
print("Inverse Hyperbolic Tangent of the complex number is:",cmath.atanh(myNum)) 
```

输出:

```py
Complex number is: (3+2j)
Hyperbolic Sine of the complex number is: (-4.168906959966565+9.15449914691143j)
Hyperbolic Cosine of the complex number is: (-4.189625690968807+9.109227893755337j)
Hyperbolic Tangent of the complex number is: (1.00323862735361-0.003764025641504249j)
Inverse Hyperbolic Sine of the complex number is: (1.9833870299165355+0.5706527843210994j)
Inverse Hyperbolic Cosine of the complex number is: (1.9686379257930964+0.6061378223872937j)
Inverse Hyperbolic Tangent of the complex number is: (0.22907268296853878+1.4099210495965755j)
```

## cmath 模块中的对数函数

cmath 模块提供了两种方法，即 log()和 log10()，用于复数的对数计算。log()函数将一个复数作为第一个输入，并将一个可选参数作为对数函数的底数。当我们只将复数作为输入传递给 log()函数时，它返回以“e”为基数的复数的自然对数。当我们还将第二个参数(即 base)传递给 log()函数时，它将使用提供的底数计算复数的对数。这可以从下面的例子中看出。

```py
import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Natural log of the complex number is:",cmath.log(myNum))
print("Logarithm of the complex number with base 5 is:",cmath.log(myNum,5))
```

输出:

```py
Complex number is: (3+2j)
Natural log of the complex number is: (1.2824746787307684+0.5880026035475675j)
Logarithm of the complex number with base 5 is: (0.7968463205835412+0.36534655919610926j)
```

log10()方法计算以 10 为底的复数的对数，如下所示。

```py
import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Logarithm of the complex number with base 10 is:",cmath.log10(myNum))
```

输出:

```py
Complex number is: (3+2j)
Logarithm of the complex number with base 10 is: (0.5569716761534184+0.255366286065454j)
```

## cmath 模块中的幂函数

cmath 模块为 python 中的计算提供了两个幂函数，即 exp()和 sqrt()。exp()函数接受一个复数作为输入，并返回一个表示输入的指数值的复数。这可以从下面的例子中看出。

```py
import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Exponential of the complex number is:",cmath.exp(myNum)) 
```

输出:

```py
Complex number is: (3+2j)
Exponential of the complex number is: (-8.358532650935372+18.263727040666765j)
```

sqrt()函数也接受一个复数作为输入，并返回表示输入平方根的复数，如下所示。

```py
 import cmath
myNum=3+2j
print("Complex number is:",myNum)
print("Square root of the complex number is:",cmath.sqrt(myNum)) 
```

输出:

```py
Complex number is: (3+2j)
Square root of the complex number is: (1.8173540210239707+0.5502505227003375j)
```

## 结论

在本文中，我们研究了 cmath 模块中的函数和方法，以便在 Python 中对复数执行数学运算。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。