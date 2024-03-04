# 在 Python 中堆叠和连接 Numpy 数组

> 原文：<https://www.pythonforbeginners.com/basics/stack-and-concatenate-numpy-arrays-in-python>

Numpy 数组是数字数据最有效的数据结构之一。您可以使用内置函数对 numpy 数组执行不同的数学运算[。本文将讨论如何在 Python 中使用不同的函数连接 numpy 数组。](https://www.pythonforbeginners.com/basics/create-numpy-array-in-python)

## Concatenate()函数

您可以使用 concatenate()函数沿现有轴连接一维和二维 numpy 数组。它具有以下语法。

np.concatenate((arr1，arr2，…，arrN)，axis=0)

这里，concatenate()函数将一组 numpy 数组作为它的第一个输入参数。执行后，它返回连接的数组。

*   使用 axis 参数确定连接输入数组的轴。它的默认值为 0。
*   对于轴=0，不同阵列的行垂直连接，即不同阵列的行成为输出阵列的行。
*   对于轴=1，数组水平连接，即输入数组的列成为输出数组的列。
*   对于 axis=None，所有输入数组都被展平，输出是一维 numpy 数组。

### 使用 Python 中的 Concatenate()函数连接一维数组

您可以使用 concatenate()函数连接一维 numpy 数组，方法是将包含 numpy 数组的元组作为输入参数传递，如下所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,8)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7]
Concatenated array is:
[0 1 2 3 4 5 6 7]
```

这里，我们水平连接了两个 numpy 数组。因此，输入数组的所有元素都被转换为输出数组的元素。

不能使用 concatenate()函数和 axis=1 参数垂直连接一维 numpy 数组。这样做会导致 numpy。AxisError 异常，显示消息“numpy。AxisError:轴 1 超出了 1 维数组的界限。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,8)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
AxisError: axis 1 is out of bounds for array of dimension 1
```

在这里，您可以观察到我们试图使用 concatenate()函数垂直连接 numpy 数组，这导致了 AxisError 异常。

### 使用 Python 中的 Concatenate()函数连接二维数组

我们可以使用 concatenate()函数水平和垂直连接二维数组。

要水平连接 numpy 数组，可以将数组元组作为第一个输入参数，将 axis=1 作为第二个输入参数传递给 concatenate()函数。执行后，concatenate()函数将返回一个 numpy 数组，如下所示。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4 15 16 17 18 19]
 [ 5  6  7  8  9 20 21 22 23 24]
 [10 11 12 13 14 25 26 27 28 29]]
```

在上面的例子中，我们水平连接了二维数组。您可以观察到，输入数组的行被组合起来创建输出数组的行。

在水平连接 numpy 数组时，需要确保所有输入数组都有相同的行数。水平串联具有不同行数的数组将导致 ValueError 异常，并显示消息“ValueError:串联轴的所有输入数组维度必须完全匹配”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,35).reshape(4,5)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 3 and the array at index 1 has size 4
```

在上面的例子中，我们试图连接 3 行和 4 行的数组。因为输入数组具有不同的行数，所以程序会遇到 ValueError 异常。

还可以使用 concatenate()函数垂直连接 numpy 数组。为此，您需要将参数 axis=0 作为输入与数组元组一起传递。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]] 
```

在本例中，我们垂直连接了二维 numpy 数组。您可以观察到输入数组的列被组合在一起，以创建输出数组的列。

在垂直连接 numpy 数组时，您需要确保所有输入数组都有相同数量的列。否则，程序将遇到 ValueError 异常，并显示消息“ValueError:串联轴的所有输入数组维数必须完全匹配”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,36).reshape(3,7)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 7
```

在这个例子中，我们试图连接两个分别有 5 列和 7 列的数组。由于列数不同，程序在连接数组时遇到 ValueError 异常。

您也可以连接所有二维 numpy 数组来创建一维数组。为此，您必须将参数 axis=None 与数组元组一起传递给 concatenate()函数。在这种情况下，所有输入数组首先被展平为一维数组。之后，它们被连接起来。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,36).reshape(3,7)
print("Second array is:")
print(arr2)
arr3=np.concatenate([arr1,arr2],axis=None)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19 20 21]
 [22 23 24 25 26 27 28]
 [29 30 31 32 33 34 35]]
Concatenated array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35]
```

在这个例子中，我们在连接二维数组后创建了一个一维数组。您可以观察到输入二维数组的元素以行主顺序包含在输出数组中。

concatenate()函数沿现有轴组合输入数组。因此，组合二维数组只能得到二维数组作为输出。如果希望堆叠 numpy 数组以形成数据立方体类型的结构，则不能使用 concatenate()函数来实现。为此，我们将使用 stack()函数。

## 使用 Python 中的 hstack()函数连接 Numpy 数组

hstack()函数的工作方式类似于参数 axis=1 的 concatenate()函数。当我们将数组元组传递给 hstack()函数时，它会水平堆叠输入数组的列，并返回一个新数组。对于一维输入数组，它返回一维数组，如下所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,8)
print("Second array is:")
print(arr2)
arr3=np.hstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7]
Concatenated array is:
[0 1 2 3 4 5 6 7]
```

在这里，您可以观察到输入数组被连接起来形成一个输出一维数组。输入数组的元素包含在输出数组中的顺序与它们作为 hstack()函数的输入的顺序相同。

对于二维数组输入，hstack()函数返回一个二维数组，如下所示。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.hstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4 15 16 17 18 19]
 [ 5  6  7  8  9 20 21 22 23 24]
 [10 11 12 13 14 25 26 27 28 29]]
```

在上面的例子中，输入数组的行已经被组合以创建输出数组的行。

在使用具有 hstack()函数的二维数组时，需要确保输入数组具有相等的行数。否则，您将得到一个 ValueError 异常，并显示消息“ValueError:串联轴的所有输入数组维度必须完全匹配”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,35).reshape(4,5)
print("Second array is:")
print(arr2)
arr3=np.hstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 3 and the array at index 1 has size 4
```

在上面的例子中，我们试图使用 hstack()函数连接两个行数不同的二维数组。因此，程序会遇到 ValueError 异常。因此，我们可以说 hstack()函数的工作方式类似于参数 axis=1 的 concatenate()函数。

## 使用 Python 中的 vstack()函数连接 Numpy 数组

不能使用 concatenate()函数垂直连接 numpy 数组。但是，您可以使用 vstack()函数垂直连接一维数组来创建二维 numpy 数组。当我们将 1 维 numpy 数组的元组或列表传递给 vstack()函数时，它返回一个 2 维数组，其中所有输入的 numpy 数组都被转换为行。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.vstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Concatenated array is:
[[0 1 2 3 4]
 [5 6 7 8 9]]
```

在本例中，我们将两个一维数组垂直连接在一起，创建了一个二维数组。使用 concatenate()函数不可能做到这一点。

使用 vstack()函数连接一维 numpy 数组时，需要确保所有数组的长度相等。否则，程序将遇到 ValueError 异常，如下例所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,8)
print("Second array is:")
print(arr2)
arr3=np.vstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 3
```

在这里，您可以观察到我们试图连接两个长度分别为 5 和 3 的数组。因此，程序会遇到 ValueError 异常。

还可以使用 vstack()函数垂直连接二维 numpy 数组。在这种情况下，vstack()函数的工作方式类似于参数 axis=0 的 concatenate()函数。

当我们将一个 2-D numpy 数组的列表或元组传递给 vstack()函数时，我们得到一个 2-D 数组作为输出。输入数组的行被转换为输出 numpy 数组的行。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.vstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
```

在本例中，您可以看到我们使用 vstack()函数垂直连接了两个二维数组。在这里，输入数组的列组合起来创建输出数组的列。

在使用带有 vstack()函数的二维数组时，需要确保输入数组具有相等的列数。否则，您将得到一个 ValueError 异常，并显示消息“ValueError:串联轴的所有输入数组维度必须完全匹配”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,36).reshape(3,7)
print("Second array is:")
print(arr2)
arr3=np.vstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 7
```

在这个例子中，我们试图连接两个分别有 5 列和 7 列的数组。因此，程序会遇到 ValueError 异常。

对于 vstack()函数，您可以观察到它的行为与 concatenate()函数不同。对于一维数组，concatenate()函数的工作方式与 concatenate()函数不同。但是，对于二维数组，vstack()函数的工作方式与 concatenate()函数类似。

## 使用 Stack()函数堆叠数组

stack()函数可用于使用 N 维数组创建 N+1 维数组。例如，您可以使用 stack()函数从二维数组创建三维数组，或者从一维数组创建二维数组。

stack()函数将一组输入数组作为它的第一个输入参数，并将我们用来堆叠数组的轴作为它的第二个输入参数。

### 使用 Python 中的 Stack()函数堆叠一维数组

您可以使用 stack()函数跨行和列堆叠一维数组。要跨行堆叠一维数组，可以传递参数 axis=0 和输入数组元组，如下所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Concatenated array is:
[[0 1 2 3 4]
 [5 6 7 8 9]]
```

在本例中，我们使用 stack()函数垂直堆叠了两个一维数组，以创建一个二维数组。这里，输入数组被转换成输出数组的行。

还可以使用 stack()函数跨列堆叠两个一维数组。为此，您需要将参数 axis=1 和输入数组元组一起传递给 stack()函数。在输出中，您将获得一个二维数组，其中输入数组被转换为输出数组的列。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Concatenated array is:
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]]
```

在上面的例子中，您可以观察到我们堆叠了两个一维 numpy 数组来创建一个二维数组。这里，一维输入数组被转换成二维数组的列。

使用 stack()函数只能堆叠长度相等的一维 numpy 数组。否则，程序将遇到 ValueError 异常，并显示消息“ValueError:所有输入数组必须具有相同的形状”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,15)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all input arrays must have the same shape
```

在上面的例子中，我们试图水平堆叠两个一维数组。因为数组的长度不同，程序会遇到 ValueError 异常，指出所有输入数组的形状必须相同。

### 使用 Python 中的 Stack()函数堆叠二维数组

您可以使用 stack()函数跨行、列和深度堆叠二维数组。

要跨长度堆叠两个 numpy 数组，我们可以将 axis=0 和输入数组元组一起传递给 stack()函数。如果输入元组中有 K 个形状为 MxN 的输入数组，则输出数组的形状将为 KxMxN。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=0)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]]

 [[15 16 17 18 19]
  [20 21 22 23 24]
  [25 26 27 28 29]]]
```

在本例中，我们使用 stack()函数堆叠了两个形状为 3×5 的 numpy 数组。得到的数组是 2x3x5 的形状。因此，在使用 stack()函数堆叠二维数组后，我们得到了三维数组。

要跨宽度堆叠 numpy 数组，可以将 axis=1 与输入数组元组一起传递给 stack()函数。如果输入元组中有 K 个形状为 MxN 的输入数组，则输出数组的形状将为 MxKxN。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=1)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[[ 0  1  2  3  4]
  [15 16 17 18 19]]

 [[ 5  6  7  8  9]
  [20 21 22 23 24]]

 [[10 11 12 13 14]
  [25 26 27 28 29]]]
```

在本例中，您可以看到我们得到了形状为 3x2x5 的输出数组。因此，stack()函数在参数 axis=1 的情况下执行时，会在宽度方向上堆叠输入数组。

为了在深度上堆叠 numpy 数组，我们可以将 axis=2 和输入数组的元组一起传递给 stack()函数。如果输入元组中有 K 个形状为 MxN 的输入数组，则输出数组的形状将为 MxNxK。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=2)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[[ 0 15]
  [ 1 16]
  [ 2 17]
  [ 3 18]
  [ 4 19]]

 [[ 5 20]
  [ 6 21]
  [ 7 22]
  [ 8 23]
  [ 9 24]]

 [[10 25]
  [11 26]
  [12 27]
  [13 28]
  [14 29]]]
```

在本例中，我们在深度方向上堆叠了 2 个形状为 3×5 的数组后，得到了形状为 3x5x2 的输出数组。

您应该记住，传递给输入元组中的 stack()函数的输入数组应该具有相同的形状。否则，程序将遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,35).reshape(4,5)
print("Second array is:")
print(arr2)
arr3=np.stack([arr1,arr2],axis=2)
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all input arrays must have the same shape
```

在上面的例子中，我们将不同形状的输入数组传递给了 stack()函数。因此，该函数会引发 ValueError 异常。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用 Python 中的 column_stack()函数堆栈 Numpy 数组

numpy 模块为我们提供了 column_stack()函数来将一维数组作为二维数组的列进行堆叠。当我们将一个一维数组的列表或元组传递给 column_stack()函数时，它返回一个二维数组，其中包含作为其列的输入一维数组。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Concatenated array is:
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]] 
```

在这里，所有的输入一维数组都被堆叠成输出数组的列。我们可以说 column_stack()函数的工作方式类似于 axis=1 的 stack()函数。

传递给 column_stack()函数的一维数组必须具有相等的长度。否则，程序将会出错。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,12)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 5 and the array at index 1 has size 7
```

这里，我们尝试使用 column_stack()函数分别堆叠大小为 5 和 7 的两个数组。因此，该函数会引发 ValueError 异常。

还可以使用 column_stack()函数水平堆叠二维数组，如下所示。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4 15 16 17 18 19]
 [ 5  6  7  8  9 20 21 22 23 24]
 [10 11 12 13 14 25 26 27 28 29]] 
```

当我们使用 column_stack()函数堆叠二维数组时，输入数组的行被组合起来以创建输出数组的行。因此，该函数的工作方式类似于 axis=1 时的 hstack()函数或 vstack()函数。

当使用 column_stack()函数堆叠二维数组时，需要确保每个输入数组中的行数相同。否则，程序将会遇到如下所示的 ValueError 异常。

```py
import numpy as np
arr1=np.arange(15).reshape((3,5))
print("First array is:")
print(arr1)
arr2=np.arange(15,35).reshape(4,5)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 3 and the array at index 1 has size 4
```

还可以使用 column_stack()函数将一维和二维数组作为新数组的列进行堆叠，如下所示。

```py
import numpy as np
arr1=np.arange(3)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0 15 16 17 18 19]
 [ 1 20 21 22 23 24]
 [ 2 25 26 27 28 29]]
```

在这里，您可以看到一维数组和二维数组的列构成了输出数组的列。同样，一维数组的长度和二维数组的行数应该相同，这一点很重要。否则，程序将会遇到如下所示的异常。

```py
import numpy as np
arr1=np.arange(4)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.column_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 4 and the array at index 1 has size 3
```

## 使用 Python 中的 row_stack()函数堆栈 Numpy 数组

numpy 模块为我们提供了 row_stack()函数来将一维数组作为二维数组的行进行堆栈。当我们将一个一维数组的列表或元组传递给 row_stack()函数时，它返回一个二维数组，其中包含作为其行的输入一维数组。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Concatenated array is:
[[0 1 2 3 4]
 [5 6 7 8 9]]
```

在上面的例子中，我们堆叠了两个长度为 5 的一维数组，以创建一个大小为 2×5 的二维数组。通过观察输出，我们可以说 row_stack()函数的工作方式类似于 vstack()函数。

传递给 row_stack()函数的一维数组必须具有相等的长度。否则，程序将会出错。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,11)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 6
```

还可以使用 row_stack()函数垂直堆叠二维数组，如下所示。

```py
import numpy as np
arr1=np.arange(15).reshape(3,5)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
```

在这个例子中，我们使用 row_stack()函数垂直堆叠了两个二维数组。您可以观察到输入数组的列已经被组合在一起，以创建输出数组的列。

当使用 row_stack()函数堆叠二维数组时，需要确保每个输入数组中的列数是相同的。否则，程序将会遇到如下所示的 ValueError 异常。

```py
import numpy as np
arr1=np.arange(15).reshape(3,5)
print("First array is:")
print(arr1)
arr2=np.arange(15,33).reshape(3,6)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 6
```

您还可以使用 row_stack()函数将一维和二维数组堆叠为新数组的行，如下所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[ 0  1  2  3  4]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
```

在这里，您可以看到一维数组和二维数组的列构成了输出数组的行。同样，一维数组的长度和输入二维数组中的列数应该相同，这一点很重要。否则，程序将会遇到如下所示的异常。

```py
import numpy as np
arr1=np.arange(6)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.row_stack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 6 and the array at index 1 has size 5
```

## 使用 dstack()函数跨深度堆叠 Numpy 数组

函数的作用是:在深度上堆叠 numpy 数组。当我们将一个一维数组的列表或元组传递给 dstack()函数时，它会返回一个三维数组，如下所示。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,10)
print("Second array is:")
print(arr2)
arr3=np.arange(11,16)
print("Third array is:")
print(arr3)
arr4=np.dstack([arr1,arr2,arr3])
print("Concatenated array is:")
print(arr4)
```

输出:

```py
First array is:
[0 1 2 3 4]
Second array is:
[5 6 7 8 9]
Third array is:
[11 12 13 14 15]
Concatenated array is:
[[[ 0  5 11]
  [ 1  6 12]
  [ 2  7 13]
  [ 3  8 14]
  [ 4  9 15]]]
```

如果给定 K 个长度为 N 的一维数组作为 dstack()函数的输入，则输出数组的形状为 1xNxK。

作为 dstack()函数的输入给出的一维数组的长度应该相同。如果输入数组的长度不同，程序会遇到如下所示的 ValueError 异常。

```py
import numpy as np
arr1=np.arange(5)
print("First array is:")
print(arr1)
arr2=np.arange(5,11)
print("Second array is:")
print(arr2)
arr3=np.arange(11,16)
print("Third array is:")
print(arr3)
arr4=np.dstack([arr1,arr2,arr3])
print("Concatenated array is:")
print(arr4)
```

输出:

```py
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 6
```

对于二维数组，dstack()函数的工作方式类似于参数 axis=2 的 stack()函数。如果输入元组中有 K 个形状为 MxN 的输入数组，则输出数组的形状将为 MxNxK。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(15).reshape(3,5)
print("First array is:")
print(arr1)
arr2=np.arange(15,30).reshape(3,5)
print("Second array is:")
print(arr2)
arr3=np.dstack([arr1,arr2])
print("Concatenated array is:")
print(arr3)
```

输出:

```py
First array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
Second array is:
[[15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]]
Concatenated array is:
[[[ 0 15]
  [ 1 16]
  [ 2 17]
  [ 3 18]
  [ 4 19]]

 [[ 5 20]
  [ 6 21]
  [ 7 22]
  [ 8 23]
  [ 9 24]]

 [[10 25]
  [11 26]
  [12 27]
  [13 28]
  [14 29]]]
```

在本例中，我们在深度方向上堆叠了两个形状为 3×5 的二维阵列。输出数组的形状为 3x5x2。因此，我们可以说 dstack()函数作为参数 axis=2 的 stack()函数工作。

## 结论

在本文中，我们讨论了如何在 Python 中连接 1 维和 2 维 NumPy 数组。我们还讨论了如何水平、垂直和横向堆叠 1-D 和 2-D numpy 数组。

要了解更多关于数据帧和 numpy 数组的信息，您可以阅读这篇关于 [pandas 数据帧索引](https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python)的文章。您可能也会喜欢这篇关于用 Python 进行[文本分析的文章。](https://www.pythonforbeginners.com/basics/text-analysis-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！