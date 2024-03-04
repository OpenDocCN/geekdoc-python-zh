# Python 中的字符串切片

> 原文：<https://www.pythonforbeginners.com/strings/string-slicing-in-python>

字符串是 Python 中最常用的数据结构之一。我们使用 Python 中的[字符串处理所有文本数据。在本文中，我们将研究使用切片从给定的字符串中提取子字符串的方法。我们还将看一些字符串切片的例子，以便更好地理解它。](https://www.pythonforbeginners.com/basics/strings)

## 什么是字符串切片？

字符串切片是取出字符串的一部分的过程。提取部分中的字符可以是连续的，或者它们可以以规则的间隔出现在原始字符串中。

您可以使用下面的类比来理解字符串切片。

假设你正在把一条面包切成片。所有的面包片，不管有多厚，都构成了一片面包。类似地，我们可以从字符串创建切片。这种类比的唯一区别是，在切片面包的情况下，原始面包在切片形成后被破坏。相反，在对字符串进行切片时，即使创建了新的切片，原始字符串仍保持原样。

让我们举一个例子。假设我们有一个字符串“ **Pythonforbeginners** ”。

字符串的不同部分如下:

1.  " **Python** ":这里，我们取了字符串的前六个字符。您可以从任何索引开始的原始字符串中提取任意数量的连续字符，这些字符将构成字符串的一部分。
2.  " **nigeb** ":这里，我们取了一些逆序的字符。您可以从原始字符串中以逆序从任何索引开始提取任意数量的连续字符，这些字符将构成字符串的一部分。
3.  " **Ptof** ":这里，我们取了一些间隔 1 个位置出现的字符。您可以从任何索引开始，以固定的间隔从原始字符串中提取任意数量的字符，这些字符将构成字符串的一部分。
4.  " **sngr** ":这里，我们取了一些以相反顺序出现在两个位置之间的字符。您可以从任何索引开始，以相反的顺序，以固定的间隔从原始字符串中提取任意数量的字符，这些字符将构成字符串的一部分。

在 python 中，有两种方法可以创建字符串片段。我们可以使用索引和内置的 slice()方法从字符串创建切片。让我们逐一讨论这两种方法。

## 使用字符串索引进行切片

我们可以使用字符的索引从字符串中创建一个片段。使用索引进行字符串切片的语法是 **`string_name [ start_index:end_index:step_size ]`** 。这里，

*   start_index 是字符串中开始分段的字符的索引。
*   end_index 是字符串中字符的索引，字符串的切片在该处终止。这里，end_index 是排他的，end_index 处的字符将不包括在分片字符串中。
*   step_size 用于确定原始字符串中将要包含在分片字符串中的索引。
    *   step_size 为 1 意味着将从原始字符串的 start_index 开始到 end_index-1 结束的连续字符创建切片。
    *   step_size 为 2 意味着我们将使用备用字符创建原始字符串的一部分，从原始字符串的 start_index 开始，到 end_index-1 结束。
    *   step_size 为 3 意味着我们将在原始字符串的每个字符之间留出 2 个位置之后选择字符，这些字符必须包括在从原始字符串的 start_index 开始到 end_index-1 结束的分片字符串中。
*   如果 start_index 大于 end_index，并且 step_size 为负值，则以相反的顺序进行切片。

我们可以通过下面的例子来理解上述语法的工作原理。

```py
myString = "Pythonforbeginners"
mySlice = myString[0:6:1]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 0, 5, 1, mySlice))
mySlice = myString[13:8:-1]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 13, 9, -1, mySlice))
mySlice = myString[0:8:2]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 0, 8, 2, mySlice))
mySlice = myString[18:7:-3]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 18, 7, -3, mySlice))
```

输出:

```py
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 5 and step size 1 is 'Python'
Slice of string 'Pythonforbeginners' starting at index 13, ending at index 9 and step size -1 is 'nigeb'
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 8 and step size 2 is 'Ptof'
Slice of string 'Pythonforbeginners' starting at index 18, ending at index 7 and step size -3 is 'sngr'
```

字符串切片的另一种语法是我们只指定 start_index 和 end_index，如 **`string_name [ start_index:end_index]`** 中所示。这里，步长取为 1，字符从 start_index 到 end_index-1 连续选择，如下所示。

```py
myString = "Pythonforbeginners"
mySlice = myString[0:6]
print("Slice of string '{}' starting at index {}, ending at index {} is '{}'".format(myString, 0, 5, mySlice))
```

输出:

```py
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 5 is 'Python'
```

我们也可以选择不指定 start_index 和 end_index。在这种情况下，start_index 的默认值取为 0，end_index 的默认值取为字符串的长度。您可以在下面的示例中观察到这些变化。

```py
myString = "Pythonforbeginners"
mySlice = myString[:6]
print("Slice of string '{}' starting at index {}, ending at index {} is '{}'".format(myString, 0, 5, mySlice))
mySlice = myString[13:]
print("Slice of string '{}' starting at index {}, ending at last index is '{}'".format(myString, 13, mySlice))
```

输出:

```py
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 5 is 'Python'
Slice of string 'Pythonforbeginners' starting at index 13, ending at last index is 'nners'
```

## 使用内置函数进行切片

我们可以使用 slice()方法，而不是直接使用字符的索引。slice()方法将 start_index、end_index 和 step_size 作为输入，并创建一个 slice 对象。然后将 slice 对象作为 index 传递给原始字符串，然后创建原始字符串的切片，如下所示。

```py
myString = "Pythonforbeginners"
slice_obj = slice(0, 6, 1)
mySlice = myString[slice_obj]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 0, 5, 1,
                                                                                                      mySlice))
slice_obj = slice(13, 8, -1)
mySlice = myString[slice_obj]
print(
    "Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 13, 9, -1,
                                                                                                    mySlice))
slice_obj = slice(0, 8, 2)
mySlice = myString[slice_obj]
print("Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 0, 8, 2,
                                                                                                      mySlice))
slice_obj = slice(18, 7, -3)
mySlice = myString[slice_obj]
print(
    "Slice of string '{}' starting at index {}, ending at index {} and step size {} is '{}'".format(myString, 18, 7, -3,
                                                                                                    mySlice)) 
```

输出:

```py
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 5 and step size 1 is 'Python'
Slice of string 'Pythonforbeginners' starting at index 13, ending at index 9 and step size -1 is 'nigeb'
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 8 and step size 2 is 'Ptof'
Slice of string 'Pythonforbeginners' starting at index 18, ending at index 7 and step size -3 is 'sngr'
```

您可以看到，slice 对象的工作方式几乎与我们使用字符索引从字符串创建切片的方式相同。使用下面的例子可以更清楚地理解这一点。

```py
myString = "Pythonforbeginners"
# specify only start and end index
slice_obj = slice(5, 16)
mySlice = myString[slice_obj]
print("Slice of string '{}' starting at index {}, ending at index {} is '{}'".format(myString, 0, 5, mySlice))
# specify only end index
slice_obj = slice(12)
mySlice = myString[slice_obj]
print("Slice of string '{}' starting at index {}, ending at index {} is '{}'".format(myString, 0, 12, mySlice))
```

输出:

```py
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 5 is 'nforbeginne'
Slice of string 'Pythonforbeginners' starting at index 0, ending at index 12 is 'Pythonforbeg'
```

## 结论

在本文中，我们讨论了 python 中的字符串切片。我们还研究了从给定字符串创建切片的不同方法。要学习更多关于 python 中字符串的知识，你可以阅读这篇关于[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。