# NumPy real 和 NumPy imag 完全指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-real-numpy-imag

嘿大家，欢迎回到另一个令人兴奋的 NumPy 教程。

你们一定都熟悉复数，对吧？如果没有，让我给你一个快速回顾。所以，复数是由两部分组成的特殊数字，实部和虚部。形式为 **z = x+yi 的数 **z** ，其中 x 和 y 是实数，那么数 z，**被称为复数。

这个简短的介绍是本教程所必需的，因为它是进一步理解这两个函数的基础。现在，我们可以仅仅通过使用简单的 NumPy 函数来提取复数的实部和虚部吗？是的，我们可以使用 NumPy 库的两个函数来实现，即 **`numpy.real()`** 和 **`numpy.imag()`** 。

这就是我们将在文章中讨论的内容。

## 理解数字真实

在 NumPy 库的众多函数中， **numpy real** 是一个**提取复杂参数的实数部分**的函数。

让我们来看看函数的语法。

### NumPy real 的语法

```py
numpy.real(val)

```

输入 **`val`** 可以是单个复数，也可以是复数的 NumPy 数组。

我们将在代码片段部分讨论它的返回类型，因为通过观察输出，您会更容易理解它。

### 使用 NumPy real

让我们写一些代码，让我们对函数的理解更清晰。

#### 1.单个复数的 NumPy 实数

```py
import numpy as np

# Passing Complex Numbers as argument to the fucntion
print("The real part of 1+3j is:",np.real(1+3j))
print("The real part of 3j is:",np.real(3j))

print("\n")

# Passing real numbers as argument to the function
print("The real part of 1 is:",np.real(1))
print("The real part of -1.1 is:",np.real(-1.1))

```

#### 输出

```py
The real part of 1+3j is: 1.0
The real part of 3j is: 0.0

The real part of 1 is: 1
The real part of -1.1 is: -1.1

```

**注:**每一个实数都可以表示为复数。例如，1 可以表示为 1+0i，其中虚部变为 0。

在上面的例子中，我们已经将复数和实数作为输入传递给了 **`np.real()`** 函数。注意到这两种情况下输出的差异真的很有趣。

当一个复数作为参数传递给函数时，输出是类型 **float** 。然而，当一个实数作为参数传递给函数时，那么输出的**类型**就是与输入数字相同的**。这只是函数返回类型的不同。**

#### 2.NumPy 复数数组的 NumPy 实数

```py
import numpy as np

a = np.array((1+3j , 3j , 1 , 0.5))

b = np.real(a)

print("The input array:\n",a)
print("The real part of the numbers:\n",b)

```

#### 输出

```py
The input array:
 [1\. +3.j 0\. +3.j 1\. +0.j 0.5+0.j]
The real part of the numbers:
 [1\.  0\.  1\.  0.5]

```

我们来理解一下上面的例子。这里，我们创建了一个变量 **`a`** 来存储四个元素的 NumPy 数组，其中两个是复数，另外两个是实数。

在下一行中，我们使用了 **`np.real()`** 函数来提取输入数组中元素的实部。该函数的输出是一个 NumPy 数组，存储在变量 **`b`** 中。

在接下来的两行中，我们使用了两个 print 语句来分别打印输入数组和输出数组。

这就是使用 NumPy 实数函数的全部内容。现在，我们将了解 NumPy imag 函数。

## 关于 NumPy imag

NumPy imag 也是 NumPy 库的数学函数之一，**提取复数实参的虚部**。

它的语法非常类似于 NumPy 实函数。

### NumPy 图像的语法

```py
numpy.imag(val)

```

输入 **`val`** 可以是单个复数，也可以是复数的 NumPy 数组。

它的返回类型**与我们在上一节中讨论的 NumPy 实函数的返回类型**完全相似。

### 使用 NumPy 图像

让我们对不同类型的输入值使用这个函数。

#### 1.单个复数的 NumPy 图像

```py
import numpy as np

# Passing Complex Numbers as argument to the fucntion
print("The imaginary part of 1+3j is:",np.imag(1+3j))
print("The imaginary part of 3j is:",np.imag(3j))

print("\n")

# Passing imag numbers as argument to the function
print("The imaginary part of 1 is:",np.imag(1))
print("The imaginary part of -1.1 is:",np.imag(-1.1))

```

#### 输出

```py
The imaginary part of 1+3j is: 3.0
The imaginary part of 3j is: 3.0

The imaginary part of 1 is: 0
The imaginary part of -1.1 is: 0.0

```

这里，输出的类型取决于函数的输入类型。我们可以观察到，如果一个复数作为输入被传递，那么函数的输出是一个浮点数，而如果输入是一个实数，输出数的类型取决于输入数的类型。

函数 **`np.imag()`** 提取复数的虚部。与复数中的术语“j”相关联的数字是复数的虚部**。**

#### **2.复数 NumPy 数组的 NumPy 图像**

```py
import numpy as np

a = np.array((1+3j , 3j , 1 , 0.5))

b = np.imag(a)

print("The input array:\n",a)
print("The imaginary part of the numbers:\n",b) 
```

#### **输出**

```py
The input array:
 [1\. +3.j 0\. +3.j 1\. +0.j 0.5+0.j]
The imaginary part of the numbers:
 [3\. 3\. 0\. 0.] 
```

**在上面的代码片段中，前两个输出是清楚的，但是为什么输出数组中的另外两个值是 0 呢？**

**实际上，输入数组的后两个元素是实数，可以分别写成 **1+0j** 和 **0.5+0j** 。所以，很明显，两个数的虚部都等于 0。这就是输出数组中最后两个值等于 0 的原因。**

**所以，这就是使用 NumPy real 和 NumPy imag 函数的全部内容。**

## **摘要**

**在本文中，我们学习了 NumPy real 和 imag 函数以及不同类型的示例。我们还了解了这两个函数的返回类型，这是本文最有趣的部分🙂**

**不断学习，不断探索更多这样的文章。**

## **参考**

*   **[NumPy 文档–NumPy real](https://numpy.org/doc/stable/reference/generated/numpy.real.html)**
*   **[num py documentation–num py imag](https://numpy.org/doc/stable/reference/generated/numpy.imag.html)**