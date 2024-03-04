# Python 中的 Numpy 数组操作

> 原文：<https://www.pythonforbeginners.com/basics/numpy-array-operations-in-python>

Numpy 数组是 Python 中处理和分析数字数据的一个很好的工具。在上一篇文章中，我们讨论了创建 numpy 数组的不同方法。在本文中，我们将讨论各种 numpy 数组操作，使用这些操作可以在 Python 中分析数值数据。

## Python 中 Numpy 数组的算术运算

我们可以用最简单的方式在 Python 中对 numpy 数组执行算术运算。如果你想给一个 numpy 数组的所有元素加一个数，你可以简单地把这个数加到数组本身。python 解释器将算术运算传播给数组中的所有元素，给定的数字被添加到所有数组元素中。请记住，在算术运算过程中，原始数组不会被修改。因此，您需要将算术运算的结果存储在所需的变量中。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The original array is:",arr1)
number=10
arr=arr1+10
print("The modified array is:",arr)
```

输出:

```py
The original array is: [1 2 3 4 5 6 7]
The modified array is: [11 12 13 14 15 16 17]
```

在这里，您可以看到我们在数组中添加了数字 10。执行加法语句后，结果被广播到每个元素，值增加 10。

您还可以使用相同的语法对数组元素执行减法、乘法和除法运算。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The original array is:",arr1)
number=10
arr=arr1*10
print("The modified array is:",arr)
```

输出:

```py
The original array is: [1 2 3 4 5 6 7]
The modified array is: [10 20 30 40 50 60 70]
```

Numpy 还允许您在两个数组的元素之间执行算术运算。您可以对一个数组的元素和另一个数组的元素执行算术运算，就像对单个元素执行相同的运算一样。

例如，如果您想将一个数组的元素添加到另一个数组的元素中，可以按如下方式进行。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The first array is:",arr1)
arr2=np.array([8,9,10,11,12,13,14])
print("The second array is:",arr2)
arr=arr1+arr2
print("The output array is:",arr)
```

输出:

```py
The first array is: [1 2 3 4 5 6 7]
The second array is: [ 8  9 10 11 12 13 14]
The output array is: [ 9 11 13 15 17 19 21]
```

这里，我们添加了两个数组。相加后，一个数组中的元素被添加到另一个数组中相同位置的元素中。

如果要添加的数组中有不同数量的元素，程序可能会出错。因此，您需要确保数组具有相同的形状。否则，程序将会遇到如下所示的 ValueError 异常。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The first array is:",arr1)
arr2=np.array([8,9,10,11,12,13])
print("The second array is:",arr2)
arr=arr1+arr2
print("The output array is:",arr)
```

输出:

```py
The first array is: [1 2 3 4 5 6 7]
The second array is: [ 8  9 10 11 12 13]
ValueError: operands could not be broadcast together with shapes (7,) (6,) 
```

就像加法一样，您可以在两个 numpy 数组的元素之间执行减法、乘法和除法，如下所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The first array is:",arr1)
arr2=np.array([8,9,10,11,12,13,14])
print("The second array is:",arr2)
arr=arr1*arr2
print("The output array is:",arr)
```

输出:

```py
The first array is: [1 2 3 4 5 6 7]
The second array is: [ 8  9 10 11 12 13 14]
The output array is: [ 8 18 30 44 60 78 98]
```

这里，我们执行了两个数组之间的乘法。您还可以根据需要执行其他算术运算。

## Python 中使用 Numpy 数组的比较操作

我们可以一次比较一个 numpy 数组的元素和另一个元素。这看起来就像比较两个数字一样简单。例如，您可以比较 numpy 数组的数组元素是否大于某个数字，如下所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The array is:",arr1)
number=5
arr=arr1>number
print("The output array is:",arr)
```

输出:

```py
The array is: [1 2 3 4 5 6 7]
The output array is: [False False False False False  True  True]
```

在这里，我们在比较之后得到一个布尔值的 numpy 数组。首先，将输入数组的元素与给定元素进行比较。对于 numpy 数组中的每个元素，比较操作的输出存储在输出 numpy 数组中。对于大于 5 的元素，我们在输出数组中得到 True。与原始数组中小于或等于 5 的元素相对应的输出数组中的元素为假。

还可以使用比较运算符对两个 numpy 数组进行元素比较。在这种情况下，将一个数组中的元素与另一个数组中相同位置的元素进行比较。比较的输出在另一个数组中返回，如下所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The first array is:",arr1)
arr2=np.array([8,9,10,11,12,13,14])
print("The second array is:",arr2)
arr=arr1>arr2
print("The output array is:",arr)
```

输出:

```py
The first array is: [1 2 3 4 5 6 7]
The second array is: [ 8  9 10 11 12 13 14]
The output array is: [False False False False False False False]
```

这里，将第一个数组中的每个元素与第二个数组中的相应元素进行比较。然后将结果存储在输出数组中。

同样，您需要确保数组长度相等。否则，程序将会遇到 ValueError 异常。

您还可以用与一维数组类似的方式对二维 numpy 数组执行算术和比较操作。

当我们对一个带有数字的二维 numpy 数组执行算术或比较操作时，结果会传播到每个元素。

当我们在二维数组上执行元素方式的 numpy 数组操作时，操作是按元素方式执行的。在这里，您应该确保在对二维 numpy 数组执行元素算术或比较操作时，数组的形状应该相同。

## 找出 Numpy 数组中的最小和最大元素

要查找 numpy 数组中的最大元素，可以使用 max()方法。当在 numpy 数组上调用 max()方法时，返回数组的最大元素。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The array is:",arr1)
max_element=arr1.max()
print("The maximum element is:",max_element)
```

输出:

```py
The array is: [1 2 3 4 5 6 7]
The maximum element is: 7
```

这里，max()方法返回数组中最大的元素。还可以使用 max()方法来查找二维 numpy 数组中的最大元素，如下所示。

```py
import numpy as np
arr1=np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print("The array is:",arr1)
max_element=arr1.max()
print("The maximum element is:",max_element)
```

输出:

```py
The array is: [[ 1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14]]
The maximum element is: 14
```

要查找 numpy 数组中的最小元素，可以使用 min()方法。在 numpy 数组上调用 min()方法时，将返回最小元素，如下例所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The array is:",arr1)
min_element=arr1.min()
print("The minimum element is:",min_element)
```

输出:

```py
The array is: [1 2 3 4 5 6 7]
The minimum element is: 1
```

您可以获取最小和最大元素的位置，而不是获取 numpy 数组中的最小值和最大值。为此，您可以使用 argmax()和 argmin()方法。

## 找出 Numpy 数组中最小和最大元素的索引

在 numpy 数组上调用 argmax()方法时，将返回数组中最大元素的位置，如下所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The array is:",arr1)
max_element=arr1.argmax()
print("The index of maximum element is:",max_element)
```

输出:

```py
The array is: [1 2 3 4 5 6 7]
The index of maximum element is: 6
```

在这里，您可以看到最大的元素是索引为 6 的 7。因此，argmax()方法返回值 6。

我们可以使用 argmin()方法找到 numpy 数组中最小元素的索引。在 numpy 数组上调用 argmin()方法时，将返回数组中最小元素的位置，如下所示。

```py
import numpy as np
arr1=np.array([1,2,3,4,5,6,7])
print("The array is:",arr1)
min_element=arr1.argmin()
print("The index of minimum element is:",min_element)
```

输出:

```py
The array is: [1 2 3 4 5 6 7]
The index of minimum element is: 0
```

这里，最小的数字是索引为 0 的 1。因此，argmin()方法返回 0。

对于二维数组，argmax()和 argmin()方法不会返回最大元素的确切位置。相反，如果二维数组已经按行主顺序展平，则它们返回最大元素的索引。

例如，14 是下例所示数组中最大的元素。它的位置是(0，6)。但是，argmax()方法返回值 6。这是因为 argmax()方法返回元素的索引，如果它在一个扁平的数组中。argmin()方法的工作方式类似于 argmax()方法。

```py
import numpy as np
arr1=np.array([[8,9,10,11,12,13,14],[1,2,3,4,5,6,7]])
print("The array is:",arr1)
max_element=arr1.argmax()
print("The index of maximum element is:",max_element)
```

输出:

```py
The array is: [[ 8  9 10 11 12 13 14]
 [ 1  2  3  4  5  6  7]]
The index of maximum element is: 6 
```

## 在 Python 中对 Numpy 数组进行排序

可以使用 sort()方法对 numpy 数组进行排序。对 numpy 数组调用 sort()方法时，将对 numpy 数组中的元素进行排序，如下例所示。

```py
import numpy as np
arr1=np.array([1,2,17,4,21,6,7])
print("The original array is:",arr1)
arr1.sort()
print("The sorted array is:",arr1)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7]
The sorted array is: [ 1  2  4  6  7 17 21]
```

在这里，您可以看到原始数组已经排序。sort()方法在执行后返回值 None。

如果我们在一个二维 numpy 数组上使用 sort()方法，所有的内部数组都按升序排序，如下所示。

```py
import numpy as np
arr1=np.array([[8,9,12,11,27,34,14],[55,2,15,4,22,6,7]])
print("The original array is:",arr1)
arr1.sort()
print("The sorted array is:",arr1)
```

输出:

```py
The original array is: [[ 8  9 12 11 27 34 14]
 [55  2 15  4 22  6  7]]
The sorted array is: [[ 8  9 11 12 14 27 34]
 [ 2  4  6  7 15 22 55]]
```

在上面的例子中，当我们对一个二维 numpy 数组调用 sort()方法时，您可以观察到内部的一维数组是按升序排序的。

## 在 Python 中切分 Numpy 数组

分割一个 numpy 数组类似于在 Python 中[分割一个列表。您可以使用索引运算符对 numpy 数组执行切片操作。分割 numpy 数组的语法如下。](https://www.pythonforbeginners.com/dictionary/python-slicing)

```py
mySlice=myArray[start_index:end_index:interval]
```

这里，

*   myArray 是现有的 numpy 数组。
*   mySlice 是新创建的 myArray 切片。
*   start_index 是 myArray 的索引，我们必须从中选择元素。
*   end_index 是 myArray 的索引，在此之前我们必须选择元素。
*   间隔是 myArray 中必须包含在 mySlice 中的两个连续元素之间的间隔。

您可以使用上面的语法分割 numpy 数组，如下所示。

```py
import numpy as np
myArr=np.array([1,2,17,4,21,6,7,12,13,14])
print("The original array is:",myArr)
mySlice=myArr[2:6:1]
print("The slice is:",mySlice)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7 12 13 14]
The slice is: [17  4 21  6] 
```

在上面的例子中，我们将原始数组中索引 2 到 5 的元素切片到 mySlice 中。当对连续元素进行切片时，也可以省略语法中的最后一个冒号和间隔。结果会是一样的。您可以在下面的示例中观察到这一点。

```py
import numpy as np
myArr=np.array([1,2,17,4,21,6,7,12,13,14])
print("The original array is:",myArr)
mySlice=myArr[2:6]
print("The slice is:",mySlice)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7 12 13 14]
The slice is: [17  4 21  6]
```

在上面的例子中，我们从切片语法中省略了最后一个冒号和间隔。但是，结果与前面的示例相似。

如果您想选择从开始到特定索引的元素，您可以将起始值留空，如下所示。

```py
import numpy as np
myArr=np.array([1,2,17,4,21,6,7,12,13,14])
print("The original array is:",myArr)
mySlice=myArr[:6:1]
print("The slice is:",mySlice)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7 12 13 14]
The slice is: [ 1  2 17  4 21  6]
```

在上面的例子中，我们让开始索引为空。因此，输出切片包含从索引 0 到索引 5 的元素。

要选择从特定索引到末尾的数组元素，可以将末尾索引留空，如下所示。

```py
import numpy as np
myArr=np.array([1,2,17,4,21,6,7,12,13,14])
print("The original array is:",myArr)
mySlice=myArr[2::1]
print("The slice is:",mySlice)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7 12 13 14]
The slice is: [17  4 21  6  7 12 13 14]
```

在这里，我们将结束索引留空。因此，输出片段包含从索引 2 到最后的元素。

在上面的例子中，元素选自连续的索引。但是，您可以使用如下所示的间隔值以特定间隔选择元素。

```py
import numpy as np
myArr=np.array([1,2,17,4,21,6,7,12,13,14])
print("The original array is:",myArr)
mySlice=myArr[2::2]
print("The slice is:",mySlice)
```

输出:

```py
The original array is: [ 1  2 17  4 21  6  7 12 13 14]
The slice is: [17 21  7 13]
```

在上面的例子中，我们将 interval 的值作为 2 传递。因此，输出切片包含来自输入数组的替换元素。

## 在 Python 中分割二维 Numpy 数组

还可以使用索引运算符对二维 numpy 数组执行切片操作。为了对二维 numpy 数组进行切片，我们使用以下语法。

```py
mySlice=myArray[start_row:end_row:row_interval,start_column:end_column:column_interval]
```

这里，术语 start_row、end_row 和 row_interval 分别表示要包括在切片中的起始行的索引、要包括在切片中的最后一行的索引以及要包括在切片中的 myArray 的两个连续行之间的间隔。

类似地，术语 start_column、end_column 和 column_interval 分别表示要包括在片中的起始列的索引、要包括在片中的最后一列的索引以及要包括在片中的 myArray 的两个连续列之间的间隔。

例如，您可以将数组的第一行切为第三行，第二列切为第四列，如下所示。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[0:4:1,1:5:1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is: 
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is: 
[[ 1  2  3  4]
 [11 12 13 14]
 [21 22 23 24]
 [31 32 33 34]]
```

在上面的例子中，我们对从索引 0 到 3 的行和从索引 1 到 4 的列进行了切片。因此，我们从 10×10 的数组中得到一个 4×4 的数组。

如果希望包括所有行，但只选择切片中的某些列，可以将 start_row 和 end_row 值留空。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[::1,1:5:1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[ 1  2  3  4]
 [11 12 13 14]
 [21 22 23 24]
 [31 32 33 34]
 [41 42 43 44]
 [51 52 53 54]
 [61 62 63 64]
 [71 72 73 74]
 [81 82 83 84]
 [91 92 93 94]]
```

在这个例子中，我们选择了所有的行，但是从索引 1 到 4 中选择了列。为此，我们在切片语法中将开始和结束行索引留空。

类似地，如果您希望包含所有列，但选择切片中的某些行，可以将 start_column 和 end_column 值留空，如下所示。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[0:4:1,::1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]]
```

在本例中，我们选择了所有的列，但是从索引 0 到 3 中选择了行。为此，我们在切片语法中将开始和结束列索引留空。

如果希望在切片中包含从开头到某个行索引的行，可以将 start_row 留空。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[:4:1,2:5:1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[ 2  3  4]
 [12 13 14]
 [22 23 24]
 [32 33 34]]
```

如果您想将某个行索引中的行包含到最后，可以将 end_row 留空，如下所示。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[3::1,2:5:1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[32 33 34]
 [42 43 44]
 [52 53 54]
 [62 63 64]
 [72 73 74]
 [82 83 84]
 [92 93 94]]
```

如果希望在切片中包含从开始到某个列索引的列，可以将 start_column 留空。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[1:4:1,:5:1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[10 11 12 13 14]
 [20 21 22 23 24]
 [30 31 32 33 34]] 
```

如果您想包含从某个列索引到最后的列，可以将 end_column 留空，如下所示。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[3::1,4::1]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[34 35 36 37 38 39]
 [44 45 46 47 48 49]
 [54 55 56 57 58 59]
 [64 65 66 67 68 69]
 [74 75 76 77 78 79]
 [84 85 86 87 88 89]
 [94 95 96 97 98 99]] 
```

您还可以在选定的行和列之间引入间隔，如下所示。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[3::2,4::3]
print("The slice is:")
print(mySlice)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[34 37]
 [54 57]
 [74 77]
 [94 97]]
```

切片是原始 numpy 数组的视图。因此，对切片所做的任何更改也会反映在原始数组中。例如，看看下面的例子。

```py
import numpy as np
myArr=np.arange(100).reshape((10,10))
print("The original array is:")
print(myArr)
mySlice=myArr[3::2,4::3]
print("The slice is:")
print(mySlice)
mySlice+=100
print("The original array is:")
print(myArr)
```

输出:

```py
The original array is:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
The slice is:
[[34 37]
 [54 57]
 [74 77]
 [94 97]]
The original array is:
[[  0   1   2   3   4   5   6   7   8   9]
 [ 10  11  12  13  14  15  16  17  18  19]
 [ 20  21  22  23  24  25  26  27  28  29]
 [ 30  31  32  33 134  35  36 137  38  39]
 [ 40  41  42  43  44  45  46  47  48  49]
 [ 50  51  52  53 154  55  56 157  58  59]
 [ 60  61  62  63  64  65  66  67  68  69]
 [ 70  71  72  73 174  75  76 177  78  79]
 [ 80  81  82  83  84  85  86  87  88  89]
 [ 90  91  92  93 194  95  96 197  98  99]]
```

这里，我们只对切片进行了更改。但是，这些更改也会反映在原始数组中。这表明切片只是原始阵列的视图。因此，如果您想要对切片进行任何更改，您应该首先使用 copy()方法将其复制到另一个变量。

## 在 Python 中重塑 Numpy 数组

您可以使用 shape()方法重塑 numpy 数组。shape()方法有两种用法。

首先，您可以在想要整形的数组上调用 shape()方法。在 numpy 数组上调用 reshape()方法时，该方法将输出数组中的行数和列数作为其第一个和第二个输入参数。执行后，它将返回经过整形的数组，如下例所示。

```py
import numpy as np
arr1=np.arange(20)
print("The first array is:")
print(arr1)
arr2=arr1.reshape((4,5))
print("The reshaped array is:")
print(arr2)
```

输出:

```py
The first array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The reshaped array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
```

在上面的例子中，我们将一个有 20 个元素的一维数组改造成了一个形状为(4，5)的二维数组。

还可以将原始数组传递给 shape()方法。在这种方法中，shape()方法将原始数组作为其第一个输入参数，将包含行数和列数的元组作为其第二个输入参数。执行后，它返回如下所示的整形后的数组。

```py
import numpy as np
arr1=np.arange(20)
print("The first array is:")
print(arr1)
arr2=np.reshape(arr1,(4,5))
print("The reshaped array is:")
print(arr2)
```

输出:

```py
The first array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The reshaped array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
```

只要原始数组中的元素总数等于所需的整形数组中的元素数，shape()方法就能很好地工作。但是，如果原始数组中的元素总数小于或大于所需输出数组中的元素数，则程序会遇到 ValueError 异常。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(20)
print("The first array is:")
print(arr1)
arr2=np.reshape(arr1,(4,7))
print("The reshaped array is:")
print(arr2)
```

输出:

```py
ValueError: cannot reshape array of size 20 into shape (4,7)
```

由 shape()方法返回的数组是原始数组的视图。因此，对新数组的任何更改都会导致对原始数组的修改。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(20)
print("The first array is:")
print(arr1)
arr2=np.reshape(arr1,(4,5))
print("The reshaped array is:")
print(arr2)
arr2+=100
print("The first array is:")
print(arr1)
```

输出:

```py
The first array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The reshaped array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
The first array is:
[100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
 118 119]
```

在这里，您可以观察到在重新整形的二维数组中所做的更改在原始的一维数组中也是可见的。这证明了整形后的数组只是原始数组的视图。因此，如果您想要对整形后的数组进行更改，如果您不想对原始数组进行更改，您应该首先考虑将其复制到一个新变量中。

### 在 Python 中展平 Numpy 数组

要展平 numpy 数组，可以将值-1 作为维度传递给 shape()方法，如下所示。

```py
import numpy as np
arr1=np.arange(20).reshape(4,5)
print("The first array is:")
print(arr1)
arr2=np.reshape(arr1,-1)
print("The flattened array is:")
print(arr2)
```

输出:

```py
The first array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
The flattened array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

可以使用 flatten()方法来展平 numpy 数组，而不使用 shape()方法。flatten()方法在二维 numpy 数组上调用时，会返回数组的平面一维视图。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(20).reshape(4,5)
print("The first array is:")
print(arr1)
arr2=arr1.flatten()
print("The flattened array is:")
print(arr2)
```

输出:

```py
The first array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
The flattened array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

flatten()方法返回一个副本，而不是一个视图。要理解这一点，请看下面的例子。

```py
import numpy as np
arr1=np.arange(20).reshape(4,5)
print("The first array is:")
print(arr1)
arr2=arr1.flatten()
print("The flattened array is:")
print(arr2)
arr2+=50
print("The first array is:")
print(arr1)
```

输出:

```py
The first array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
The flattened array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The first array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
```

在这里，您可以看到对展平数组所做的更改没有反映在原始数组中。在 reshape()方法中不会发生这种情况。

reshape()方法返回原始数组的视图，对经过整形的数组所做的任何更改在原始数组中都是可见的。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(20).reshape(4,5)
print("The first array is:")
print(arr1)
arr2=arr1.reshape(-1)
print("The flattened array is:")
print(arr2)
arr2+=50
print("The first array is:")
print(arr1)
```

输出:

```py
The first array is:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
The flattened array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The first array is:
[[50 51 52 53 54]
 [55 56 57 58 59]
 [60 61 62 63 64]
 [65 66 67 68 69]]
```

因此，如果希望在修改展平的数组时保持原始数组不变，应该使用 flatten()方法而不是 shape()方法来展平 numpy 数组。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在 Python 中复制 Numpy 数组

因为 shape()方法和 flatten()方法返回原始 NumPy 数组的视图，所以如果不想更改原始数组，就不能修改视图。在这种情况下，可以使用 copy()方法将视图复制到另一个数组中。

当在 numpy 数组上调用 copy()方法时，它返回原始数组的副本。如果对复制阵列进行任何更改，这些更改不会反映在原始阵列中。您可以在下面的示例中观察到这一点。

```py
import numpy as np
arr1=np.arange(20)
print("The first array is:")
print(arr1)
arr2=arr1.copy()
print("The copied array is:")
print(arr2)
```

输出:

```py
The first array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
The copied array is:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

## 结论

在本文中，我们讨论了 Python 中不同的 numpy 数组操作。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何[创建熊猫数据框架](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)的文章。您可能也会喜欢这篇关于 Python 中的[字符串操作的文章。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)