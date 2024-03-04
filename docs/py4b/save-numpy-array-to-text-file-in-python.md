# 用 Python 将 Numpy 数组保存到文本文件

> 原文：<https://www.pythonforbeginners.com/basics/save-numpy-array-to-text-file-in-python>

python 中的数据分析广泛使用 Numpy 数组。在本文中，我们将讨论如何在 python 中将 numpy 数组保存到文本文件中。

## 使用 str()函数将 Numpy 数组保存到文本文件中

我们可以使用`str()`函数和文件处理将 numpy 数组保存到文本文件中。在这种方法中，我们将首先使用`str()`函数将 numpy 数组转换成一个字符串。`str()` 函数将 numpy 数组作为输入参数，并返回它的字符串表示。将 numpy 数组转换成字符串后，我们将把字符串保存到一个文本文件中。

为了将 numpy 数组保存到一个文本文件中，我们将首先使用`open()` 函数在 append 模式下打开一个文件。`open()`函数将文件名作为第一个输入参数，将文字“`a`”作为第二个输入参数，以表示文件是以追加模式打开的。执行后，它返回一个包含文本文件的 file 对象。

获得 file 对象后，我们将使用`write()`方法将包含 numpy 数组的字符串保存到文件中。在 file 对象上调用`write()`方法时，该方法将字符串作为其输入参数，并将该字符串追加到文件中。将字符串写入文件后，不要忘记使用`close()`方法关闭文件。

使用 `str()`函数将 numpy 数组保存到文本文件的完整代码如下。

```py
import numpy as np

myFile = open('sample.txt', 'r+')
myArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("The array is:", myArray)
print("The content of the file before saving the array is:")
text = myFile.read()
print(text)
myFile.write(str(myArray))
myFile.close()
myFile = open('sample.txt', 'r')
print("The content of the file after saving the array is:")
text = myFile.read()
print(text)
```

输出:

```py
The array is: [1 2 3 4 5 6 7 8 9]
The content of the file before saving the array is:
I am a sample text file.
I was created by Aditya.
You are reading me at Pythonforbeginners.com.

The content of the file after saving the array is:
I am a sample text file.
I was created by Aditya.
You are reading me at Pythonforbeginners.com.
[1 2 3 4 5 6 7 8 9]
```

推荐机器学习文章:[机器学习中的回归与实例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)

## 使用 numpy.savetxt()函数将 Numpy 数组保存到文本文件中

不使用`str()`函数，我们可以使用`numpy.savetxt()`函数将 numpy 数组保存到 python 中的文本文件中。在这种方法中，我们首先使用前面例子中讨论的`open()`函数在追加模式下打开文本文件。打开文件后，我们将使用 `numpy.savetxt()`函数将数组保存到文本文件中。这里，`numpy.savetxt()`函数将 file 对象作为它的第一个输入参数，将 numpy 数组作为它的第二个输入参数。执行后，它将 numpy 数组保存到文本文件中。您可以在下面的示例中观察到这一点。

```py
import numpy as np

myFile = open('sample.txt', 'r+')
myArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("The array is:", myArray)
np.savetxt(myFile, myArray)
myFile.close()
myFile = open('sample.txt', 'r')
print("The content of the file after saving the array is:")
text = myFile.read()
print(text)
```

输出:

```py
The array is: [1 2 3 4 5 6 7 8 9]
The content of the file after saving the array is:
1.000000000000000000e+00
2.000000000000000000e+00
3.000000000000000000e+00
4.000000000000000000e+00
5.000000000000000000e+00
6.000000000000000000e+00
7.000000000000000000e+00
8.000000000000000000e+00
9.000000000000000000e+00
```

在执行`savetxt()`函数后，必须使用`close()` 对象关闭文件对象。否则，更改将不会写入文件。

## 结论

在本文中，我们讨论了用 python 将 numpy 数组保存到文本文件的两种方法。想了解更多关于 Python 编程的知识，可以阅读这篇关于 Python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。