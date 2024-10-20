# Numpy fabs-按元素计算绝对值。

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-fabs>

Python 中的 NumPy fabs 函数是处理数字的有用工具。它本质上与数学中的模函数相同，用于计算特定数字或数值数组的绝对值。它对 NumPy 特别有用，因为它与 ndarrays 一起工作。

基本上，numpy fabs()函数返回数值数据的正数值。但是，它不能处理复杂的值——对于这些值，您可以使用 abs()函数。总而言之，这是掌握你的数字的好方法！

## numpy.fabs()的语法

numpy fabs 函数如下所示:

```py
numpy.fabs(a, /, out=None, *, where=True, casting='same_kind', 
 order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'fabs'>

```

上面的函数返回数组中元素的正/绝对值。

**参数**

*   *a*:***array _ like***->要计算绝对值的数值的数组。如果 a 是一个标量值，那么返回的数组，即 b，也将是一个标量值。
*   *out*:***ndarray*， *None，(此参数可选*** )- >存储结果时，可以在此参数中指定一个位置。形状必须与输入数组相同。如果没有提供它，或者我们把它赋值为 NONE，那么就创建并返回一个新的数组。
*   *其中***:*****array _ like，(此参数可选** )* - >对于其中的位置，where==TRUE，out 数组将返回 ufunc 结果。在其他任何地方，out 数组将保持其原始值。如果 where 参数为 true，则通用函数值将被更改，如果为 false 或未指定，则只有输出保留返回值。要了解 python 中通用函数(ufunc)的更多信息，[点击这里](https://numpy.org/doc/stable/reference/ufuncs.html)。****
*   *****【kwargs】*:***(该参数也是可选的)*** - >涉及关键字的自变量在该参数中指定。****

******返回值******

****numpy.fabs()函数返回一个数组，例如“b”，在标量输入的情况下，它是一个标量。它是一个包含输入数组中所有给定数字数据的正数值的数组。返回类型总是 [float](https://docs.python.org/3/library/stdtypes.html) 。****

## ****Numpy.fabs()的示例****

****让我们看一些关于如何实现和使用 numpy fabs()函数的例子。****

### ****示例 1–使用 Numpy 计算绝对值****

****第一个是返回单个元素的绝对值。这段代码使用 numpy 模块计算数字-9 的绝对值，并将结果存储在变量 n 中。fabs()函数用于计算任意数字的绝对值。最后，使用 print()函数打印结果。****

```py
**#importing required module
import numpy as py
n=py.fabs(-9) #using the fab function and storing the result in a variable
print(n) #printing the result** 
```

****输出:****

```py
**9.0** 
```

### ****示例 2–将现有数组传递给 Numpy.fabs()****

****现在，让我们取一个包含现有值的数组。该代码使用 numpy 模块来导入 fabs 函数。该函数将一个数字列表作为输入，并返回一个数字数组，其中包含列表中每个数字的绝对值。然后，代码打印出数字列表，并显示列表中每个数字的绝对值。****

```py
**#importing required module
import numpy as py
n=[-1.3,-8.6,50.0,-4,-67.55,69.1,0] #pre-defined array
s=py.fabs(n) #using the fabs function
print(s) #printing the result** 
```

****上述代码的输出类似于下面所示:****

```py
**[1.3 ,8.6 ,50\. ,4\. ,67.55 ,69.1 ,0.]** 
```

### ****示例 3–传递用户输入数组****

****现在，我们来看另一个例子，数组将成为用户输入。因此，我们需要从用户那里提取输入，然后找到用户定义的数组中所有元素的绝对值。下面给出的是我们将如何去做。****

```py
**import numpy as py #importing required modules
n=eval(input("enter required values separated by a comma=")) #take user input
n=py.array(n) #convert user input into an array
s=py.fabs(n) #using the fabs() function
print(s) #displaying the result** 
```

****上述代码的输出将如下所示:****

```py
**enter required values separated by a comma=-1.9,-5.4,-8.0,-33.33     #enter required input               
[ 1.9   5.4   8\.   33.33]** 
```

### ****示例 4–2D 阵列上的 Numpy fabs 功能****

****您也可以通过以下方式在 2d 阵列上使用 fabs 功能。这段代码导入 NumPy 模块，创建一个名为“n”的负数 2D 数组，然后使用 numpy fabs 函数计算数组中元素的绝对值，并将它们存储在数组“s”中。最后，它打印数组中元素的绝对值。****

```py
**import numpy as py #import required module
n=[[-1.2,-6.7],[-4.6,-9.1],[-6.9,-2.2]] #initializing 2D array
s=py.fabs(n) #compute absolute value
print("the absolute value of the 2D array is")
print(s) #display the output** 
```

****输出如下所示:****

```py
**the absolute value of the 2D array is
[[1.2 6.7]
 [4.6 9.1]
 [6.9 2.2]]** 
```

## ****结论:****

****numpy fabs()函数对于 numpy 用户来说是一个非常有用的工具，可以快速方便地找到任何数字、数组或矩阵的绝对值。它对于处理大型数值数组特别有用，因为它可以快速计算数组中每一项的绝对值。这使得处理复杂的数字和数据集更加容易。总而言之，这是一个很好的方法来掌握你的数字，并让他们在检查中！****