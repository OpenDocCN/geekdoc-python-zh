# 在 Python 中拆分 Numpy 数组

> 原文：<https://www.pythonforbeginners.com/basics/split-a-numpy-array-in-python>

Numpy 数组是处理数字数据的最有效的数据结构之一。您可以使用内置函数对 numpy 数组执行不同的数学运算。在本文中，我们将讨论如何使用不同的函数在 Python 中分割 numpy 数组。

numpy 模块为我们提供了各种函数来将 Numpy 数组分割成不同的子数组。让我们逐一讨论。

## 使用 Split()函数拆分 numpy 数组

split()方法可用于根据索引将 numpy 数组分成相等的部分。它具有以下语法。

```py
numpy.split(myArr, index_array_or_parts, axis)
```

这里，

*   myArr 是我们必须拆分的数组。
*   index_array_or_parts 确定如何将数组拆分成子数组。如果它是一个数值，数组被分成相等的部分。如果 index_array_or_parts 是索引数组，则子数组基于索引数组中的索引。
*   参数“轴”确定阵列拆分的轴。默认情况下，它的值为 0。您可以使用此参数来拆分二维数组。
*   为了理解 split()函数的工作原理，请考虑以下示例。

```py
import numpy as np
myArr=np.arange(9)
print("The array is:")
print(myArr)
arr=np.split(myArr,3)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[0 1 2 3 4 5 6 7 8]
The split array is:
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])] 
```

在这里，我们将 myArr 分成 3 个相等的部分。您可以观察到 split()方法返回一个 numpy 数组，其中包含原始数组的子数组。

split()函数返回的子数组是原始数组的视图。因此，对 split()函数返回的子数组所做的任何更改都将反映在 myArr 中。

```py
import numpy as np
myArr=np.arange(9)
print("The array is:")
print(myArr)
arr=np.split(myArr,3)
print("The split array is:")
print(arr)
arr[0][1]=999
print("The original arrays is:")
print(myArr)
```

输出:

```py
The array is:
[0 1 2 3 4 5 6 7 8]
The split array is:
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
The original arrays is:
[  0 999   2   3   4   5   6   7   8] 
```

在上面的例子中，您可以观察到我们对 split()方法返回的一个子数组进行了更改。但是，这种变化反映在原始数组中。这表明拆分的数组只是原始 numpy 数组的视图。

在将数组分割成相等的部分时，需要确保数值 index_array_or_parts 必须是数组长度的一个因子。否则，程序将遇到 ValueError 异常，并显示消息“数组拆分不会导致等分”。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(9)
print("The arrays is:")
print(myArr)
arr=np.split(myArr,4)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

在上面的例子中，我们试图将一个包含 9 个元素的数组分成 4 部分。因此，程序会遇到 ValueError 异常。

您也可以在其索引处拆分数组。为此，您需要将一个索引数组传递给 split()函数，如以下语法所示。

```py
numpy.split(myArr, [index1,index2,index3,….., indexN])
```

如果使用上面的语法，myArr 被分成不同的子数组。

*   第一个子数组由从索引 0 到索引 1-1 的元素组成。
*   第二子数组由从索引 index1 到索引 index2-1 的元素组成。
*   第三个子数组由从索引 index2 到索引 index3-1 的元素组成。
*   如果 indexN 小于数组的长度，则最后一个子数组由从索引 indexN-1 到最后一个元素的元素组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(12)
print("The array is:")
print(myArr)
arr=np.split(myArr,[1,4,7])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11]
The split array is:
[array([0]), array([1, 2, 3]), array([4, 5, 6]), array([ 7,  8,  9, 10, 11])]
```

在本例中，我们在索引 1、4 和 7 处拆分了输入数组。因此，第一子数组包含索引 0 处的元素，第二子数组包含从索引 1 到 3 的元素，第三子数组包含从索引 4 到 6 的元素，最后一个子数组包含从索引 7 到原始数组的最后一个元素的元素。

如果 indexN 大于数组的长度，您可以观察到最后的子数组将只是一个空的 numpy 数组。

```py
import numpy as np
myArr=np.arange(12)
print("The array is:")
print(myArr)
arr=np.split(myArr,[1,4,11,14,17,20])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11]
The split array is:
[array([0]), array([1, 2, 3]), array([ 4,  5,  6,  7,  8,  9, 10]), array([11]), array([], dtype=int64), array([], dtype=int64), array([], dtype=int64)]
```

在本例中，我们尝试在索引 1、4、11、14、17 和 20 处拆分输入数组。由于数组只包含 12 个元素，split()方法返回的最后三个子数组是空的 numpy 数组。

### 将二维 numpy 数组垂直和水平分割成相等的部分

还可以使用 split()函数垂直拆分二维 numpy 数组。当我们将一个二维数组传递给 split()函数时，数组的行被分组为子数组。

例如，您可以将一个二维 numpy 数组拆分成行数相等的不同子数组，如下所示。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,3)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23, 24, 25, 26]]), array([[27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49, 50, 51, 52, 53]]), array([[54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])]
```

在本例中，我们将形状为 9×9 的二维 numpy 数组垂直拆分为 3 个子数组。因此，split()方法返回一个包含 3 个形状为 3×9 的数组的数组。

这里，所需的子数组数量应该是原始数组中行数的一个因子。否则，split()方法将无法分割原始数组，程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,4)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

这里，我们试图将一个有 9 行的数组分成 4 个子数组。因此，程序会遇到 ValueError 异常。

如果您想要水平分割一个二维数组，即沿着列，您可以在 split()方法中使用值为 1 的参数轴，如下所示。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,3, axis=1)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2],
       [ 9, 10, 11],
       [18, 19, 20],
       [27, 28, 29],
       [36, 37, 38],
       [45, 46, 47],
       [54, 55, 56],
       [63, 64, 65],
       [72, 73, 74]]), array([[ 3,  4,  5],
       [12, 13, 14],
       [21, 22, 23],
       [30, 31, 32],
       [39, 40, 41],
       [48, 49, 50],
       [57, 58, 59],
       [66, 67, 68],
       [75, 76, 77]]), array([[ 6,  7,  8],
       [15, 16, 17],
       [24, 25, 26],
       [33, 34, 35],
       [42, 43, 44],
       [51, 52, 53],
       [60, 61, 62],
       [69, 70, 71],
       [78, 79, 80]])]
```

在本例中，我们将一个有 9 列的数组拆分为每个有 3 列的子数组。这里，所需子阵列的数量应该是列数的一个因子。否则，程序会遇到 ValueError 异常。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,4,axis=1)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

这里，我们试图将一个有 9 列的数组分割成 4 个子数组。因此，程序会遇到 ValueError 异常。

### 使用行和列索引拆分二维 numpy 数组

还可以使用下面的语法，根据 numpy 数组的行索引垂直拆分它们。

```py
numpy.split(myArr, [rowindex1,rowindex2,rowindex3,….., rowindexN])
```

如果使用上面的语法，myArr 被垂直分割成不同的子数组。

*   第一个子数组由从行索引 0 到行索引 rowindex1-1 的行组成。
*   第二个子数组由从行索引 rowindex1 到行索引 rowindex2-1 的行组成。
*   第三个子数组由从行索引 rowindex2 到行索引 rowindex3-1 的行组成。
*   如果 indexN 小于数组中的行数，则最后一个子数组由从行索引 rowindexN-1 到最后一行的行组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,[2,5])
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44]]), array([[45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])]
```

在上面的例子中，我们在行索引 2 和 5 处分割了输入数组。因此，我们得到 3 个子数组。第一个子数组包含从开始到索引 1 的行。第二行子数组包含从索引 2 到索引 4 的行。最后一个子数组包含从索引 5 到最后的行。

如果 rowindexN 大于数组的长度，您可以看到最后一个子数组将只是一个空的 numpy 数组。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,[2,5,10,12,15])
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44]]), array([[45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]]), array([], shape=(0, 9), dtype=int64), array([], shape=(0, 9), dtype=int64), array([], shape=(0, 9), dtype=int64)]
```

在上面的例子中，我们试图在行索引 2、5、10、12 和 15 处分割原始数组。然而，输入数组只有 9 行。因此，输出数组包含 3 个空的 numpy 数组。

要水平拆分 numpy 数组，即根据列索引对列进行分组，可以使用以下语法。

```py
numpy.split(myArr, [columnindex1,columnindex2,columnindex3,….., columnindexN], axis=1)
```

如果使用上面的语法，myArr 被分成不同的子数组。

*   第一个子数组由从列索引 0 到列索引 columnindex1-1 的列组成。
*   第二个子数组由从列索引 columnindex1 到列索引 columnindex2-1 的列组成。
*   第三个子数组由从列索引 columnindex2 到列索引 columnindex3-1 的列组成。
*   如果 indexN 小于数组中的列数，则最后一个子数组由从列索引 columnindexN-1 到最后一行的列组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,[2,5],axis=1)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1],
       [ 9, 10],
       [18, 19],
       [27, 28],
       [36, 37],
       [45, 46],
       [54, 55],
       [63, 64],
       [72, 73]]), array([[ 2,  3,  4],
       [11, 12, 13],
       [20, 21, 22],
       [29, 30, 31],
       [38, 39, 40],
       [47, 48, 49],
       [56, 57, 58],
       [65, 66, 67],
       [74, 75, 76]]), array([[ 5,  6,  7,  8],
       [14, 15, 16, 17],
       [23, 24, 25, 26],
       [32, 33, 34, 35],
       [41, 42, 43, 44],
       [50, 51, 52, 53],
       [59, 60, 61, 62],
       [68, 69, 70, 71],
       [77, 78, 79, 80]])]
```

在上面的例子中，我们在列索引 2 和 5 处分割了输入数组。因此，输出数组包含三个子数组。第一个子数组包含索引为 0 和 1 的列。第二子数组包含从索引 2 到 4 的列。最后一个子数组包含从索引 5 到最后一列的列。

如果 columnindexN 大于数组的长度，您可以看到最后一个子数组将只是一个空的 numpy 数组。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.split(myArr,[2,5,10,12,15],axis=1)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1],
       [ 9, 10],
       [18, 19],
       [27, 28],
       [36, 37],
       [45, 46],
       [54, 55],
       [63, 64],
       [72, 73]]), array([[ 2,  3,  4],
       [11, 12, 13],
       [20, 21, 22],
       [29, 30, 31],
       [38, 39, 40],
       [47, 48, 49],
       [56, 57, 58],
       [65, 66, 67],
       [74, 75, 76]]), array([[ 5,  6,  7,  8],
       [14, 15, 16, 17],
       [23, 24, 25, 26],
       [32, 33, 34, 35],
       [41, 42, 43, 44],
       [50, 51, 52, 53],
       [59, 60, 61, 62],
       [68, 69, 70, 71],
       [77, 78, 79, 80]]), array([], shape=(9, 0), dtype=int64), array([], shape=(9, 0), dtype=int64), array([], shape=(9, 0), dtype=int64)]
```

在上面的示例中，我们尝试在列索引 2、5、10、12 和 15 处拆分原始数组。然而，输入数组只有 9 列。因此，输出数组包含 3 个空的 numpy 数组。

split()函数有一个缺点。当它无法将数组拆分为等长的子数组时，会引发 ValueError 异常。为了避免遇到 ValueError 异常，可以使用 array_split()函数。

## 使用 array_split()函数拆分 Numpy 数组

array_split()函数的工作方式与 split()函数相似。唯一的区别是，当它不能将数组拆分成相等的元素时，它不会引发 ValueError 异常。相反，对于应该分成 n 部分的长度为 l 的数组，它返回 l % n 个大小为 l//n + 1 的子数组，其余的大小为 l//n。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(12)
print("The array is:")
print(myArr)
arr=np.array_split(myArr,5)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11]
The split array is:
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7]), array([8, 9]), array([10, 11])]
```

在上面的例子中，array_split()函数不能将输入数组分成 5 等份。但是，它不会引发 ValueError 异常。现在，array_split()函数返回 12%5，即大小为(12/5)+1 的 2 个子数组，即 3。其余子阵列的大小为 12//5，即 2。

在输出数组中，您可以看到有 2 个子数组，每个子数组有 3 个元素，还有 3 个子数组，每个子数组有 2 个元素。

还可以使用 array_split()函数沿行或列拆分二维数组，如下所示。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.array_split(myArr,4)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23, 24, 25, 26]]), array([[27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44]]), array([[45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62]]), array([[63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])]
```

在这个例子中，我们试图将一个 9 行的 numpy 数组垂直分割成 4 部分。因此，我们将得到 9%4，即 1 个大小为(9//4)+1 的子数组，即 3。其余子阵列的大小将为 9//4，即 2。

在输出数组中，您可以看到一个子数组包含三行，其余的子数组各包含两行。

## 使用 hsplit()函数拆分 Numpy 数组

函数的作用是:水平分割一个二维数组。要将一个二维数组拆分成列数相等的子数组，可以将原始数组和所需的子数组数传递给 hsplit()函数。执行后，它返回一个包含所有子数组的 numpy 数组，如下所示。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.hsplit(myArr,3)
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2],
       [ 9, 10, 11],
       [18, 19, 20],
       [27, 28, 29],
       [36, 37, 38],
       [45, 46, 47],
       [54, 55, 56],
       [63, 64, 65],
       [72, 73, 74]]), array([[ 3,  4,  5],
       [12, 13, 14],
       [21, 22, 23],
       [30, 31, 32],
       [39, 40, 41],
       [48, 49, 50],
       [57, 58, 59],
       [66, 67, 68],
       [75, 76, 77]]), array([[ 6,  7,  8],
       [15, 16, 17],
       [24, 25, 26],
       [33, 34, 35],
       [42, 43, 44],
       [51, 52, 53],
       [60, 61, 62],
       [69, 70, 71],
       [78, 79, 80]])]
```

在本例中，我们使用 hsplit()函数将一个 9×9 的数组分割成三个 9×3 形状的子数组。

如果原始数组中的列数不是所需子数组数的倍数，则 hsplit()函数将无法将原始数组平均分成多个子数组。在这种情况下，程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.hsplit(myArr,4)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

在上面的例子中，我们试图将一个有 9 列的数组水平分割成 4 个子数组。因为 4 不是 9 的因子，所以 hsplit()函数不能平分数组，程序运行时出现 ValueError 异常。

要使用列索引水平拆分 numpy 数组，可以使用以下语法。

```py
numpy.hsplit(myArr, [columnindex1,columnindex2,columnindex3,….., columnindexN])
```

如果使用上面的语法，myArr 被水平分割成不同的子数组。

*   第一个子数组由从列索引 0 到列索引 columnindex1-1 的列组成。
*   第二个子数组由从列索引 columnindex1 到列索引 columnindex2-1 的列组成。
*   第三个子数组由从列索引 columnindex2 到列索引 columnindex3-1 的列组成。

如果 indexN 小于数组中的列数，则最后一个子数组由从列索引 columnindexN-1 到最后一行的列组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.hsplit(myArr,[2,5,8])
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1],
       [ 9, 10],
       [18, 19],
       [27, 28],
       [36, 37],
       [45, 46],
       [54, 55],
       [63, 64],
       [72, 73]]), array([[ 2,  3,  4],
       [11, 12, 13],
       [20, 21, 22],
       [29, 30, 31],
       [38, 39, 40],
       [47, 48, 49],
       [56, 57, 58],
       [65, 66, 67],
       [74, 75, 76]]), array([[ 5,  6,  7],
       [14, 15, 16],
       [23, 24, 25],
       [32, 33, 34],
       [41, 42, 43],
       [50, 51, 52],
       [59, 60, 61],
       [68, 69, 70],
       [77, 78, 79]]), array([[ 8],
       [17],
       [26],
       [35],
       [44],
       [53],
       [62],
       [71],
       [80]])]
```

在本例中，我们在列索引 2、5 和 8 处拆分了一个包含 9 列的数组。因此，该阵列被分成 4 个子阵列。第一个子数组包含从索引 0 到 1 的列，第二个子数组包含从索引 2 到 4 的列，第三个子数组包含从索引 5 到 7 的列，第四个子数组包含原始数组中索引 8 处的列。

如果 columnindexN 大于数组的长度，您可以看到最后一个子数组将只是一个空的 numpy 数组。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The arrays is:")
print(myArr)
arr=np.hsplit(myArr,[2,5,8,12,20])
print("The split array is:")
print(arr)
```

输出:

```py
The arrays is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1],
       [ 9, 10],
       [18, 19],
       [27, 28],
       [36, 37],
       [45, 46],
       [54, 55],
       [63, 64],
       [72, 73]]), array([[ 2,  3,  4],
       [11, 12, 13],
       [20, 21, 22],
       [29, 30, 31],
       [38, 39, 40],
       [47, 48, 49],
       [56, 57, 58],
       [65, 66, 67],
       [74, 75, 76]]), array([[ 5,  6,  7],
       [14, 15, 16],
       [23, 24, 25],
       [32, 33, 34],
       [41, 42, 43],
       [50, 51, 52],
       [59, 60, 61],
       [68, 69, 70],
       [77, 78, 79]]), array([[ 8],
       [17],
       [26],
       [35],
       [44],
       [53],
       [62],
       [71],
       [80]]), array([], shape=(9, 0), dtype=int64), array([], shape=(9, 0), dtype=int64)]
```

在上面的例子中，我们在索引 2、5、8、12 和 20 处分割了原始数组。因为输入数组只包含 9 列，所以输出数组包含两个空子数组。

本质上，hsplit()函数的工作方式与参数 axis=1 的 split()函数完全一样。

## 使用 vsplit()函数拆分数组

您可以使用 vsplit()函数垂直拆分 numpy 数组，即沿行拆分。要将二维数组拆分成行数相等的子数组，可以将原始数组和所需的子数组数传递给 vsplit()函数。执行后，它返回一个包含所有子数组的 numpy 数组，如下所示。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The array is:")
print(myArr)
arr=np.vsplit(myArr,3)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23, 24, 25, 26]]), array([[27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49, 50, 51, 52, 53]]), array([[54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])]
```

在本例中，我们使用 vsplit()函数将一个 9×9 的数组垂直分割成三个大小为 3×9 的子数组。

如果原始数组中的行数不是所需子数组数的倍数，vsplit()函数将无法将原始数组平均划分为多个子数组。在这种情况下，程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The array is:")
print(myArr)
arr=np.vsplit(myArr,4)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

在这个例子中，您可以观察到我们试图将一个 9 行的 numpy 数组垂直分割成 4 个子数组。由于 4 不是 9 的因子，vsplit()函数无法拆分原始值，程序运行时出现 ValueError 异常。

若要根据行索引垂直拆分二维 numpy 数组，可以使用以下语法。

```py
numpy.vsplit(myArr, [rowindex1,rowindex2,rowindex3,….., rowindexN])
```

如果使用上面的语法，myArr 被垂直分割成不同的子数组。

*   第一个子数组由从行索引 0 到行索引 rowindex1-1 的行组成。
*   第二个子数组由从行索引 rowindex1 到行索引 rowindex2-1 的行组成。
*   第三个子数组由从行索引 rowindex2 到行索引 rowindex3-1 的行组成。

如果 indexN 小于数组中的行数，则最后一个子数组由从行索引 rowindexN-1 到最后一行的行组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The array is:")
print(myArr)
arr=np.vsplit(myArr,[2,5,8])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44]]), array([[45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71]]), array([[72, 73, 74, 75, 76, 77, 78, 79, 80]])]
```

在本例中，我们在第 2、5 和 8 行索引处垂直拆分了一个包含 9 行的数组。因此，该阵列被分成 4 个子阵列。第一个子数组包含从索引 0 到 1 的行，第二个子数组包含从索引 2 到 4 的行，第三个子数组包含从索引 5 到 7 的行，第四个子数组包含原始数组中索引 8 处的行。

如果 rowindexN 大于数组的长度，您可以看到最后一个子数组将只是一个空的 numpy 数组。

```py
import numpy as np
myArr=np.arange(81).reshape((9,9))
print("The array is:")
print(myArr)
arr=np.vsplit(myArr,[2,5,8,12,20])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
The split array is:
[array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17]]), array([[18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44]]), array([[45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71]]), array([[72, 73, 74, 75, 76, 77, 78, 79, 80]]), array([], shape=(0, 9), dtype=int64), array([], shape=(0, 9), dtype=int64)]
```

在上面的例子中，我们在索引 2、5、8、12 和 20 处垂直分割了原始数组。因为输入数组只包含 9 列，所以输出数组包含两个空子数组。

vsplit()函数的工作方式类似于参数 axis=0 的 split()函数。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在 Python 中沿深度分割三维数组

如果您有一个三维 numpy 数组，您可以使用 split()函数、array_split()函数或 split()函数沿深度拆分数组。

### 使用 Split()函数沿深度方向拆分三维数组

要将一个三维数组拆分成具有相同深度的子数组，可以将原始数组和所需子数组的数量传递给 split()或 array_split()函数，其中参数 axis=2。执行后，这些函数返回一个 numpy 数组，其中包含所有子数组，如下所示。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.split(myArr,2, axis=2)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]]

 [[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]]

 [[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]
The split array is:
[array([[[ 0,  1],
        [ 4,  5],
        [ 8,  9],
        [12, 13]],

       [[16, 17],
        [20, 21],
        [24, 25],
        [28, 29]],

       [[32, 33],
        [36, 37],
        [40, 41],
        [44, 45]],

       [[48, 49],
        [52, 53],
        [56, 57],
        [60, 61]]]), array([[[ 2,  3],
        [ 6,  7],
        [10, 11],
        [14, 15]],

       [[18, 19],
        [22, 23],
        [26, 27],
        [30, 31]],

       [[34, 35],
        [38, 39],
        [42, 43],
        [46, 47]],

       [[50, 51],
        [54, 55],
        [58, 59],
        [62, 63]]])]
```

在上面的示例中，我们将一个 4x4x4 的三维 numpy 数组沿深度方向拆分为两个 4x4x2 形状的数组。

如果原始数组的深度不是所需子数组数量的倍数，split()函数将无法将原始数组平均分成多个子数组。在这种情况下，程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.split(myArr,3, axis=2)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

在上面的例子中，我们试图将一个 4x4x4 的数组分成 3 部分。由于 3 不是 4 的因子，程序运行时遇到 ValueError 异常。

### 使用 array_split()函数沿深度方向拆分三维数组

当所需子数组的数量不是数组深度的一个因素时，array_split()函数不会抛出任何数组。array_split()函数将返回深度为(depth//number_of _sub-arrays)的深度为%number_of _sub-arrays 的数组，其余的子数组的深度为(depth//number_of _sub-arrays)。例如，考虑下面的例子。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.array_split(myArr,3, axis=2)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]]

 [[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]]

 [[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]
The split array is:
[array([[[ 0,  1],
        [ 4,  5],
        [ 8,  9],
        [12, 13]],

       [[16, 17],
        [20, 21],
        [24, 25],
        [28, 29]],

       [[32, 33],
        [36, 37],
        [40, 41],
        [44, 45]],

       [[48, 49],
        [52, 53],
        [56, 57],
        [60, 61]]]), array([[[ 2],
        [ 6],
        [10],
        [14]],

       [[18],
        [22],
        [26],
        [30]],

       [[34],
        [38],
        [42],
        [46]],

       [[50],
        [54],
        [58],
        [62]]]), array([[[ 3],
        [ 7],
        [11],
        [15]],

       [[19],
        [23],
        [27],
        [31]],

       [[35],
        [39],
        [43],
        [47]],

       [[51],
        [55],
        [59],
        [63]]])]
```

在这里，我们尝试将一个 4x4x4 的数组在深度上分成 3 个部分。因此，输出数组包含 4%3，即 1 个深度为(4//3)+1，即 2 的子数组，其余子数组的深度为 4//3，即 1。

在输出中，您可以看到我们有一个 4x4x2 形状的子数组和两个 4x4x1 形状的子数组。

## 使用 dsplit()函数沿深度分割三维数组

除了 split()函数之外，还可以使用 dsplit()函数沿深度方向拆分 numpy 数组。为此，您只需将原始数组和所需子数组的数量传递给 dsplit()函数。执行后，dsplit()函数返回一个包含所有子数组的 numpy 数组。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.dsplit(myArr,2)
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]]

 [[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]]

 [[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]
The split array is:
[array([[[ 0,  1],
        [ 4,  5],
        [ 8,  9],
        [12, 13]],

       [[16, 17],
        [20, 21],
        [24, 25],
        [28, 29]],

       [[32, 33],
        [36, 37],
        [40, 41],
        [44, 45]],

       [[48, 49],
        [52, 53],
        [56, 57],
        [60, 61]]]), array([[[ 2,  3],
        [ 6,  7],
        [10, 11],
        [14, 15]],

       [[18, 19],
        [22, 23],
        [26, 27],
        [30, 31]],

       [[34, 35],
        [38, 39],
        [42, 43],
        [46, 47]],

       [[50, 51],
        [54, 55],
        [58, 59],
        [62, 63]]])]
```

在上面的示例中，我们使用 dsplit()函数将一个 4x4x4 的三维 numpy 数组沿深度方向拆分为两个 4x4x2 形状的数组。

如果原始数组的深度不是所需子数组数量的倍数，dsplit()函数将无法将原始数组平均分成多个子数组。在这种情况下，程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.dsplit(myArr,3)
print("The split array is:")
print(arr)
```

输出:

```py
ValueError: array split does not result in an equal division
```

在上面的例子中，我们试图将一个 4x4x4 的数组分成 3 部分。由于 3 不是 4 的因子，程序运行时遇到 ValueError 异常。

要使用索引跨深度拆分三维 numpy 数组，可以使用以下语法。

```py
numpy.dsplit(myArr, [depthindex1,depthindex2,depthindex3,….., depthindexN])
```

如果使用上面的语法，myArr 将在深度上被分割成不同的子数组。

*   第一子阵列由从深度索引 0 到深度索引 depthindex1-1 的元素组成。
*   第二子阵列由从深度索引 depthindex1 到深度索引 depthindex2-1 的元素组成。
*   第三个子阵列由从深度索引 depthindex2 到深度索引 depthindex3-1 的元素组成。

如果 depthindexN 小于数组中深度方向的元素数，则最后一个子数组由深度索引 depthindexN 到最后一行的元素组成。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.dsplit(myArr,[1,3])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]]

 [[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]]

 [[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]
The split array is:
[array([[[ 0],
        [ 4],
        [ 8],
        [12]],

       [[16],
        [20],
        [24],
        [28]],

       [[32],
        [36],
        [40],
        [44]],

       [[48],
        [52],
        [56],
        [60]]]), array([[[ 1,  2],
        [ 5,  6],
        [ 9, 10],
        [13, 14]],

       [[17, 18],
        [21, 22],
        [25, 26],
        [29, 30]],

       [[33, 34],
        [37, 38],
        [41, 42],
        [45, 46]],

       [[49, 50],
        [53, 54],
        [57, 58],
        [61, 62]]]), array([[[ 3],
        [ 7],
        [11],
        [15]],

       [[19],
        [23],
        [27],
        [31]],

       [[35],
        [39],
        [43],
        [47]],

       [[51],
        [55],
        [59],
        [63]]])]
```

在上面的例子中，我们使用 dsplit()函数在深度索引 1 和 3 处分割输入数组。因此，输出中的第一个子数组包含直到索引 0 的元素，第二个子数组包含从深度索引 1 到深度索引 2 的元素。第三个子数组包含深度索引为 3 的元素。

您可以验证输出子数组的形状是(4，4，1)、(4，4，2)和(4，4，1)。

如果 depthindexN 大于深度上的元素数，您可以观察到最后的子数组将只是一个空的 numpy 数组，如下所示。

```py
import numpy as np
myArr=np.arange(64).reshape((4,4,4))
print("The array is:")
print(myArr)
arr=np.dsplit(myArr,[1,3,5,8])
print("The split array is:")
print(arr)
```

输出:

```py
The array is:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]]

 [[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]]

 [[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]
The split array is:
[array([[[ 0],
        [ 4],
        [ 8],
        [12]],

       [[16],
        [20],
        [24],
        [28]],

       [[32],
        [36],
        [40],
        [44]],

       [[48],
        [52],
        [56],
        [60]]]), array([[[ 1,  2],
        [ 5,  6],
        [ 9, 10],
        [13, 14]],

       [[17, 18],
        [21, 22],
        [25, 26],
        [29, 30]],

       [[33, 34],
        [37, 38],
        [41, 42],
        [45, 46]],

       [[49, 50],
        [53, 54],
        [57, 58],
        [61, 62]]]), array([[[ 3],
        [ 7],
        [11],
        [15]],

       [[19],
        [23],
        [27],
        [31]],

       [[35],
        [39],
        [43],
        [47]],

       [[51],
        [55],
        [59],
        [63]]]), array([], shape=(4, 4, 0), dtype=int64), array([], shape=(4, 4, 0), dtype=int64)]
```

在本例中，我们尝试在索引 1、3、5 和 8 处拆分输入数组。因为输入数组的深度为 4，所以输出数组包含两个空的 numpy 数组。

## 结论

在本文中，我们讨论了如何在 Python 中分割 numpy 数组。要了解更多关于数据帧和 numpy 数组的信息，您可以阅读这篇关于 [pandas 数据帧索引](https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python)的文章。您可能也会喜欢这篇关于用 Python 进行[文本分析的文章。](https://www.pythonforbeginners.com/basics/text-analysis-in-python)