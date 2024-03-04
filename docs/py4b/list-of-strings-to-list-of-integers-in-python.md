# Python 中的字符串列表到整数列表

> 原文：<https://www.pythonforbeginners.com/basics/list-of-strings-to-list-of-integers-in-python>

列表是 python 编程中最常用的容器对象之一。在本文中，我们将讨论在 python 中将字符串列表转换为整数列表的各种方法。

## 使用 For 循环将字符串列表转换为整数列表

我们可以使用 for 循环和`int()`函数将字符串列表转换成整数列表。为此，我们将首先创建一个名为`output_list`的空列表。之后，我们将使用 for 循环遍历字符串列表。在遍历时，我们将使用`int()`函数将每个字符串元素转换成一个整数。之后，我们将使用`append()` 方法将该整数添加到`output_list`中。在列表上调用`append()`方法时，它接受一个元素作为输入参数，并将该元素追加到列表中。

在执行 for 循环后，我们将得到一个整数列表，如`output_list`。您可以在下面的示例中观察到这一点。

```py
myList = ["1", "2", "3", "4", "5"]
output_list = []
for element in myList:
    value = int(element)
    output_list.append(value)
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
The input list is: ['1', '2', '3', '4', '5']
The output list is: [1, 2, 3, 4, 5]
```

如果列表中有不能转换成整数的元素，程序将运行到如下所示的`ValueError`异常。

```py
myList = ["1", "2", "3", "4", "5", "PFB"]
output_list = []
for element in myList:
    value = int(element)
    output_list.append(value)
print("The input list is:", myList)
print("The output list is:", output_list) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 4, in <module>
    value = int(element)
ValueError: invalid literal for int() with base 10: 'PFB'
```

在这里，您可以看到字符串“PFB”无法转换为整数。因此，程序引发了`ValueError`异常。

要处理该错误，可以在 python 中使用 try-except 块来进行异常处理。在 for 循环中，我们将使用 try 块中的`int()` 函数将元素转换为整数，然后将其追加到`output_list`中。在 except 块中，我们将打印每个不能转换为整数的元素。这样，我们将只使用输入列表中那些可以使用 `int()`函数直接转换成整数的元素来获得一个整数列表。

您可以在下面的示例中观察到这一点。

```py
myList = ["1", "2", "3", "4", "5", "PFB"]
output_list = []
for element in myList:
    try:
        value = int(element)
        output_list.append(value)
    except ValueError:
        print("{} cannot be converted to integer.".format(element))
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
PFB cannot be converted to integer.
The input list is: ['1', '2', '3', '4', '5', 'PFB']
The output list is: [1, 2, 3, 4, 5]
```

## 使用列表理解将字符串列表转换为整数列表

代替 for 循环，我们可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)和`int()`函数将字符串列表转换为整数列表，如下所示。

```py
myList = ["1", "2", "3", "4", "5"]
output_list = [int(element) for element in myList]
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
The input list is: ['1', '2', '3', '4', '5']
The output list is: [1, 2, 3, 4, 5]
```

使用列表理解有一个限制，如果输入列表的任何元素没有被转换成整数，我们将不能处理错误，因为我们不能在列表理解语法中使用异常处理。

```py
myList = ["1", "2", "3", "4", "5", "PFB"]
output_list = [int(element) for element in myList]
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 2, in <module>
    output_list = [int(element) for element in myList]
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 2, in <listcomp>
    output_list = [int(element) for element in myList]
ValueError: invalid literal for int() with base 10: 'PFB' 
```

## 使用 map()函数将字符串列表转换为整数列表

`map()`函数用于将一个函数应用于一个可迭代对象的所有元素。`map()`函数将一个函数作为第一个输入参数，将一个 iterable 对象作为第二个输入参数。它对输入 iterable 对象的所有元素逐一执行输入参数中给定的函数，并返回一个包含输出值的 iterable 对象。

为了将字符串列表转换成整数列表，我们将把 `int()`函数作为第一个输入参数传递给`map()`函数，并将字符串列表作为第二个输入参数。执行后，我们将得到如下所示的整数列表。

```py
myList = ["1", "2", "3", "4", "5"]
output_list = list(map(int, myList))
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
The input list is: ['1', '2', '3', '4', '5']
The output list is: [1, 2, 3, 4, 5]
```

如果我们使用这种方法，如果输入列表包含不能转换为整数的字符串，我们将无法避免或处理异常。

```py
myList = ["1", "2", "3", "4", "5", "PFB"]
output_list = list(map(int, myList))
print("The input list is:", myList)
print("The output list is:", output_list)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 2, in <module>
    output_list = list(map(int, myList))
ValueError: invalid literal for int() with base 10: 'PFB'
```

## 结论

在本文中，我们讨论了在 python 中将字符串列表转换为整数列表的三种方法。要了解更多关于字符串的知识，可以阅读这篇关于 python 中[字符串格式化的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/basics/strings-formatting)[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。