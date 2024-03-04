# 在 Python 中创建 Numpy 数组

> 原文：<https://www.pythonforbeginners.com/basics/create-numpy-array-in-python>

Numpy 数组在 python 中使用，特别是在数据分析、[机器学习](https://codinginfinite.com/machine-learning-an-introduction/)和数据科学中，用来操作数字数据。在本文中，我们将讨论使用示例和工作代码在 Python 中创建 numpy 数组的不同方法。

Python 中有各种创建 numpy 数组的函数。让我们逐一讨论。

## Python 中 Numpy 数组的列表

我们可以使用`numpy.array()`函数从 python 列表中创建一个 numpy 数组。`array()`函数将一个列表作为其输入参数，并返回一个 numpy 数组。在这种情况下，数组元素的数据类型与列表中元素的数据类型相同。

```py
myList=[1,2,3,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList)
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 2, 3, 4, 5]
The array is:
[1 2 3 4 5]
The data type of array is:
int64
```

在上面的代码中，我们首先使用`array()`函数创建了一个 numpy 数组。之后，我们使用 NumPy 数组的`dtype`属性来获取数组中元素的数据类型。在这里，您可以看到一个整数列表为我们提供了一个数据类型为`int64`的元素数组。

还可以使用`array()` 函数中的`dtype`参数显式指定数组元素的数据类型，如下例所示。

```py
myList=[1,2,3,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="float")
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 2, 3, 4, 5]
The array is:
[1\. 2\. 3\. 4\. 5.]
The data type of array is:
float64
```

在上面的代码中，我们给出了一个整数列表作为输入参数。但是，输出数组包含数据类型为`float64`的元素，因为我们在创建数组时已经指定了。

要创建一个二维 numpy 数组，您可以将一个列表的列表[传递给函数`array()`，如下所示。](https://www.pythonforbeginners.com/basics/list-of-lists-in-python)

```py
myList=[[1,2,3,4,5],[6,7,8,9,10]]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="float")
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
The array is:
[[ 1\.  2\.  3\.  4\.  5.]
 [ 6\.  7\.  8\.  9\. 10.]]
The data type of array is:
float64
```

## 使用不同数据类型的元素创建 Numpy 数组

如果输入列表包含不同但兼容的数据类型的元素，则 numpy 数组中的元素会自动转换为更广泛的数据类型。例如，如果我们有一个 float 和 int 的列表，那么得到的 numpy 数组元素将是 float64 数据类型。您可以在下面的例子中观察到这一点。

```py
myList=[1,3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList)
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 3.14, 4, 5]
The array is:
[1\.   3.14 4\.   5\.  ]
The data type of array is:
float64
```

这里，我们给出了一个包含整数和浮点数的列表。因此，numpy 数组的所有元素都被转换为浮点数。

如果输入列表包含诸如 str 和 int 之类的数据类型，则结果 numpy 数组元素将具有由< u32 或< u64 表示的字符串数据类型。它显示元素分别以 4 字节或 8 字节存储为 unicode 对象。

```py
myList=[1,"Aditya", 3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList)
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 'Aditya', 3.14, 4, 5]
The array is:
['1' 'Aditya' '3.14' '4' '5']
The data type of array is:
<U32
```

当输入列表元素具有不同的数据类型时，您还可以选择指定数组元素的数据类型。例如，如果您有一个 floats 和 int 的列表，并且您希望数组元素的数据类型是 int，那么您可以在如下所示的`dtype`参数中指定它。

```py
myList=[1, 3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="int")
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 3.14, 4, 5]
The array is:
[1 3 4 5]
The data type of array is:
int64
```

在这里，您可以观察到具有 float 数据类型的列表元素已经被转换为 int。因此，3.14 已转换为 3。

您还可以将元素转换为其他数据类型，如字符串，如下所示。

```py
myList=[1, 3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="str")
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 3.14, 4, 5]
The array is:
['1' '3.14' '4' '5']
The data type of array is:
<U4 
```

在指定数组元素的数据类型时，需要确保输入列表中的所有元素都可以转换为 numpy 数组中指定的数据类型。如果没有发生，程序就会出错。例如，我们不能将字母表转换成整数。因此，如果我们将 numpy 数组的目标数据类型指定为 int，用于包含带有字母的字符串的输入列表，程序将运行到如下所示的 [TypeError 异常](https://www.pythonforbeginners.com/basics/typeerror-in-python)。

```py
myList=[1,"Aditya", 3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="int")
print("The array is:")
print(myArr)
print("The data type of array is:")
print(myArr.dtype)
```

输出:

```py
The list is:
[1, 'Aditya', 3.14, 4, 5]
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_4810/3347589872.py in <module>
      2 print("The list is:")
      3 print(myList)
----> 4 myArr = np.array(myList,dtype="int")
      5 print("The array is:")
      6 print(myArr)

ValueError: invalid literal for int() with base 10: 'Aditya'
```

在这个例子中，我们将数组元素的数据类型指定为整数。但是，输入列表包含无法转换为整数的字符串。因此，程序运行到[value error int()的无效文字，错误基数为 10](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10) 。

## 在 Python 中检查 Numpy 数组的属性

您可以对 Numpy 数组执行许多操作来检查其属性。例如，您可以使用 numpy 数组的`ndim`属性来检查数组的维度，如下所示。

```py
myList=[1,"Aditya", 3.14,4,5]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="str")
print("The array is:")
print(myArr)
print("The dimension of array is:")
print(myArr.ndim)
```

输出:

```py
The list is:
[1, 'Aditya', 3.14, 4, 5]
The array is:
['1' 'Aditya' '3.14' '4' '5']
The dimension of array is:
1
```

这里，我们创建了一个一维数组。因此，它的维数是 1。

对于二维列表输入，数组的维数将是 2，如下所示。

```py
myList=[[1,2,3,4,5],[6,7,8,9,10]]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="str")
print("The array is:")
print(myArr)
print("The dimension of array is:")
print(myArr.ndim)
```

输出:

```py
The list is:
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
The array is:
[['1' '2' '3' '4' '5']
 ['6' '7' '8' '9' '10']]
The dimension of array is:
2
```

要检查 numpy 数组中数组元素的数据类型，可以使用`dtype`属性。属性给出一个`dtype`对象作为输出。要获得数据类型的名称，您可以使用前面几节已经讨论过的 `dtype.name`属性。

您还可以使用`shape`属性找到 numpy 数组的形状。numpy 数组的`shape`属性返回一个元组，该元组将行数和列数分别作为其第一个和第二个元素。您可以在下面的示例中观察到这一点。

```py
myList=[[1,2,3,4,5],[6,7,8,9,10]]
print("The list is:")
print(myList)
myArr = np.array(myList,dtype="str")
print("The array is:")
print(myArr)
print("The shape of array is:")
print(myArr.shape)
```

输出:

```py
The list is:
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
The array is:
[['1' '2' '3' '4' '5']
 ['6' '7' '8' '9' '10']]
The shape of array is:
(2, 5)
```

在这个例子中，我们创建了一个 numpy 数组，它有一个 2-D 列表，有两个内部列表，每个列表包含 5 个元素。因此，numpy 数组的形状是(2，5)。

## 创建由 0、1 和特定序列组成的 Numpy 数组

使用内置的数组操作，我们可以创建不同类型的 numpy 数组。让我们讨论其中的一些。

### 在 Python 中创建包含 1 的 Numpy 数组

您可以使用`ones()` 函数创建包含 1 的 numpy 数组。要使用`ones()`函数创建一个包含 1 的一维数组，需要将数组中所需元素的数量作为输入参数进行传递。执行后，`ones()`函数将返回一个一维 numpy 数组，其中包含所需数量的元素，如下所示。

```py
myArr = np.ones(5)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[1\. 1\. 1\. 1\. 1.]
```

默认情况下，数组中元素的数据类型是 float64。要创建整数数组，可以使用`ones()`函数的`dtype`参数来指定元素的数据类型，如下例所示。

```py
myArr = np.ones(5,dtype="int")
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[1 1 1 1 1]
```

在这里，您可以看到数组包含整数作为元素，而不是浮点数。

要使用`ones()` 函数创建二维数组，可以将一个包含行数和列数的元组以 `(number_of_rows, number_of_columns)`的格式传递给`ones()`函数。执行后，`ones()`函数将返回期望的数组。

```py
myArr = np.ones((2,3),dtype="int")
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[[1 1 1]
 [1 1 1]]
```

同样，数组中元素的默认数据类型是`float64`。因此，您可以使用`ones()`函数的`dtype`参数来改变数组元素的数据类型。

### 在 Python 中创建包含零的 Numpy 数组

就像 1 的数组一样，您也可以使用`zeros()`函数创建包含 0 的 numpy 数组。

要使用`zeros()`函数创建一个包含 1 的一维数组，需要将数组中所需元素的数量作为输入参数进行传递。执行后，`zeros()`函数将返回一个一维 numpy 数组，其中包含所需数量的零，如下所示。

```py
myArr = np.zeros(5)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[0\. 0\. 0\. 0\. 0.]
```

默认情况下，数组中元素的数据类型是`float64`。要创建整数数组，可以使用`zeros()`函数的`dtype`参数来指定元素的数据类型，如下例所示。

```py
myArr = np.zeros(5,dtype="int")
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[0 0 0 0 0]
```

这里，您可以看到数组包含整数作为其元素，而不是浮点数。

要使用`zeros()`函数创建二维数组，可以将一个包含行数和列数的元组以`(number_of_rows, number_of_columns)`的格式传递给`zeros()`函数。执行后，`zeros()`函数将返回所需的带零的数组，如下所示。

```py
myArr = np.zeros((2,3),dtype="int")
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[[0 0 0]
 [0 0 0]]
```

同样，数组中元素的默认数据类型是`float64`。因此，您可以使用`zeros()`函数的`dtype`参数来改变数组元素的数据类型。

### 用 0 到 1 之间的随机数创建 Numpy 数组

您可以使用`numpy.random.rand()`函数创建 numpy 数组，元素范围从 0 到 1。

要创建一维 numpy 数组，可以将所需元素的数量作为输入参数传递给`rand()`函数。执行后，`rand()`函数返回一个 numpy 数组，其中包含 0 到 1 之间的指定数量的浮点数。

```py
myArr = np.random.rand(5)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[0.693256   0.26514033 0.86510414 0.52163653 0.1188453 ]
```

要创建一个二维随机数数组，可以将行数作为第一个输入参数，将所需数组中的列数作为第二个参数。执行后，`rand()`函数将返回具有所需形状的 numpy 数组，如下所示。

```py
myArr = np.random.rand(2,3)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[[0.03166493 0.06408176 0.73167115]
 [0.49889714 0.34302884 0.9395251 ]]
```

建议阅读:如果你对机器学习感兴趣，并想在这方面进行更多探索，你可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于带有数字示例的](https://codinginfinite.com/regression-in-machine-learning-with-examples/) [k-means 聚类的文章。](https://codinginfinite.com/k-means-clustering-explained-with-numerical-example/)

### 用随机整数创建 Numpy 数组

要用一个范围内的随机整数创建一个 numpy 数组，可以使用`random.randint()` 函数。`random.randint()`函数的语法如下。

```py
random.randint(start, end, number_of_elements)
```

这里，

*   参数`start`表示范围的起始编号。
*   参数`end`表示范围的最后一个数字。
*   参数`number_of_elements`表示所需数组中元素的数量。

默认情况下，`random.randint()`函数返回的数组元素的数据类型是`int64`。

```py
myArr = np.random.randint(2,100,5)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[22 10 30 87 96]
```

您不能通过指定`dtype`参数来使用`randint()`函数创建浮点数数组。如果你试图这样做，程序将会遇到一个`TypeError`异常。

### 用 Python 中的一个范围内的元素创建 Numpy 数组

如果您想创建一个包含某个范围内的元素的 numpy 数组，您可以使用`numpy.arange()` 函数。

要创建一个元素从 0 到 N 的数组，可以将 N 作为输入参数传递给 `arange()` 函数。在由`arange()`函数返回的数组中，你将得到直到 N-1 的数字。这是因为 N 在该范围内是唯一的。

```py
myArr = np.arange(10)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[0 1 2 3 4 5 6 7 8 9]
```

这里，我们将 10 传递给了`arange()`函数。因此，它返回了一个包含从 0 到 9 的元素的数组。

要创建一个元素在 M 和 N 之间的 numpy 数组，可以将 M 作为第一个输入参数，将 N 作为第二个输入参数传递给`arange()`函数。执行后，您将获得一个 numpy 数组，其中包含从 M 到 N-1 的数字。

```py
myArr = np.arange(3,10)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[3 4 5 6 7 8 9]
```

这里，N 必须大于 m。否则，您将得到一个空数组，如下所示。

```py
myArr = np.arange(13,10)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[]
```

您还可以决定数组中两个连续元素之间的差异。为此，您可以将所需的数字之差作为第三个输入参数传递给`arange()` 函数。执行后，它将返回一个 numpy 数组，该数组包含某个范围内的元素以及它们之间的常数差。

```py
myArr = np.arange(3,10,2)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[3 5 7 9]
```

如果您想获得一个元素降序排列的数组，您可以将一个较大的数字作为第一个输入，将一个较小的数字作为第二个输入参数传递给`arange()` 函数。作为第三个输入参数，您需要传递一个负数作为两个连续元素之间的差值。这样，您将获得一个 numpy 数组，其中的元素按降序排列，如下例所示。

```py
myArr = np.arange(23,10,-2)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[23 21 19 17 15 13 11]
```

默认情况下， `arange()` 函数返回一个以整数为元素的数组。要获得浮点数数组，可以在`arange()`函数的`dtype`参数中指定数据类型，如下所示。

```py
myArr = np.arange(23,10,-2,dtype="float")
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[23\. 21\. 19\. 17\. 15\. 13\. 11.]
```

## 一个范围内所需元素数量的 Numpy 数组

除了决定 numpy 数组中元素的范围之外，还可以指定数组中给定范围内要包含的元素数量。为此，您可以使用`linspace()`功能。

`linspace()`函数的语法如下。

```py
linspace(start, end, number_of_elements)
```

这里，

*   参数`start`表示范围的起始编号。
*   参数`end`表示范围的最后一个数字。
*   参数`number_of_elements`表示所需数组中元素的数量。

默认情况下，`linspace()`函数返回的数组元素的数据类型是`float64`。

```py
myArr = np.linspace(2,50,10)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[ 2\.          7.33333333 12.66666667 18\.         23.33333333 28.66666667
 34\.         39.33333333 44.66666667 50\.        ] 
```

还可以通过向 end 参数传递一个较小的数字，向 start 参数传递一个较大的数字，以逆序获取数组中的元素，如下所示。

```py
myArr = np.linspace(122,50,10)
print("The array is:")
print(myArr)
```

输出:

```py
The array is:
[122\. 114\. 106\.  98\.  90\.  82\.  74\.  66\.  58\.  50.]
```

`linspace()` 函数返回一个浮点数数组。然而，您可以通过使用`linspace()`函数的`dtype`参数来创建一个带有整数元素的 numpy 数组。为此，您只需要将文字“`int`”和其他输入参数一起传递给`linspace()` 方法中的`dtype`参数。

## 在 Python 中将文件加载到 Numpy 数组中

还可以通过加载文本文件来创建 numpy 数组。为此，您可以使用`loadtxt()` 功能。

`loadtxt()`函数将文件名作为它的第一个输入参数，将元素的数据类型作为 dtype 参数的输入参数。执行后，它返回一个 numpy 数组，如下所示。

```py
File_data = np.loadtxt("1d.txt", dtype=int)
print(File_data)
```

输出:

```py
[1 2 3 5 6]
```

如果输入文件包含除空格和数字以外的字符，程序将会出错。您可以在下面的示例中观察到这一点。

```py
File_data = np.loadtxt("1d.txt", dtype=int)
print(File_data)
```

输出:

```py
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_4810/1616002059.py in <module>
----> 1 File_data = np.loadtxt("1d.txt", dtype=int)
      2 print(File_data)

ValueError: invalid literal for int() with base 10: '1,2,3,5,6'
```

您可以通过读取包含二维数据的文件来创建二维 numpy 数组，如下所示。

```py
File_data = np.loadtxt("2d.txt", dtype=int)
print(File_data)
```

输出:

```py
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

通过观察输出，我们可以推断出`loadtxt()`函数首先将文件作为字符串读取。之后，它会在新行处分割文件。换行符后的每个字符串都被视为一个新行。然后，它在空格处拆分每一行，以获得 numpy 数组的各个元素。

## 结论

在本文中，我们讨论了使用 Python 中的 numpy 模块创建数组的不同方法。要了解更多 python 主题，可以阅读这篇关于 Python 中的[列表理解的文章。你可能也会喜欢这篇关于如何用 Python](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 创建聊天应用程序的文章[。](https://codinginfinite.com/python-chat-application-tutorial-source-code/)