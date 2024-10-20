# 用 Python 将列表转换成 JSON 字符串[简单的一步一步]

> 原文：<https://www.askpython.com/python/list/convert-list-to-json>

在本教程中，我们将讨论如何将 Python 列表转换成 JSON。JSON 是表示结构化数据的最流行的数据格式之一，它使用 JavaScript 符号来存储和交换文本数据。我们开始吧！

***也读:[如何用 Python 读一个 JSON 文件](https://www.askpython.com/python/examples/read-a-json-file-in-python)***

* * *

## 导入 JSON 模块

我们需要用来处理 JSON 对象和文件的`json` Python 模块。它包含了`dumps()`函数，我们将在这里使用它将 Python 列表转换成 JSON。Python 中的`json`模块是一个标准的 Python 包，它附带了普通的 Python 解释器安装。因此，我们不必在本地系统上手动安装它。`dumps()`函数将 Python 列表作为其参数，将其转换为 JSON 字符串，然后返回该 JSON 字符串。使用`dumps()`函数将 Python 列表转换成 JSON 字符串的语法:

```py
# Import the json Python module
import json

# Call the dumps() function and pass the Python list
json_str = json.dumps(list)

```

现在，让我们讨论如何将几种类型的 [Python 列表](https://www.askpython.com/python/difference-between-python-list-vs-array)即简单列表、列表列表和字典列表转换成 JSON 字符串。

## 将列表转换为 JSON

首先，我们将看到如何将一个简单的 Python 列表转换成 JSON 字符串。我们将创建一个简单的偶数 Python 列表，然后将其传递给上面讨论的`json.dumps()`函数，该函数将 Python 对象(列表)转换为 JSON 字符串。让我们看看如何使用 Python 代码实现这一点。

```py
# Import json Python module
import json

# Create a Python list
list_1 = [2, 4, 6, 8, 10]

# Convert the above Python list to JSON string
json_str_1 = json.dumps(list_1)

# Check the type of value returned by json.dumps()
print(type(json_str_1))

# Print the result
print(json_str_1)

```

**输出:**

```py
<class 'str'>
[2, 4, 6, 8, 10]

```

## 将列表列表转换为 JSON 字符串

其次，我们将讨论如何将一个列表的 [Python 列表转换成 JSON 字符串。我们将创建一个包含大写英文字母字符及其对应的 ASCII 码的 Python 列表。然后我们将它传递给`json.dumps()`函数，该函数将把 Python 列表转换成 JSON 字符串。让我们编写 Python 代码来实现这一点。](https://www.askpython.com/python/examples/linked-lists-in-python)

```py
# Import json Python module
import json

# Create a Python list of lists
list_2 = [['A', 'B', 'C', 'D', 'E'],
          [65, 66, 67, 68, 69]]

# Convert the above Python list of lists to JSON string
json_str_2 = json.dumps(list_2)

# Check the type of value returned by json.dumps()
print(type(json_str_2))

# Print the result
print(json_str_2)

```

**输出:**

```py
<class 'str'>
[["A", "B", "C", "D", "E"], [65, 66, 67, 68, 69]]

```

## 将字典列表转换为 JSON 字符串

第三，我们将看到如何将 Python 字典列表转换成 JSON 字符串。我们将创建一个包含奇数、偶数和质数列表的字典 Python 列表，这些列表对应于它们的键(标签)。然后我们将它传递给`json.dumps()`函数，该函数将把 Python 字典列表转换成 JSON 字符串。让我们通过 Python 代码来实现这一点。

```py
# Import json Python module
import json

# Create a Python list of dictionaries
list_3 = [{'Odd': [1, 3, 5, 7, 9]},
          {'Even': [2, 4, 6, 8, 12]},
          {'Prime': [2, 3, 5, 7, 11]}]

# Convert the above Python list of dictionaries to JSON string
json_str_3 = json.dumps(list_3)

# Check the type of value returned by json.dumps()
print(type(json_str_3))

# Print the result
print(json_str_3)

```

**输出:**

```py
<class 'str'>
[{"Odd": [1, 3, 5, 7, 9]}, {"Even": [2, 4, 6, 8, 12]}, {"Prime": [2, 3, 5, 7, 11]}]

```

## 结论

在本教程中，我们学习了如何使用`json` Python 模块及其`dumps()`函数将几种类型的 Python 列表转换成 JSON 字符串。希望你已经理解了上面讨论的概念，并准备好自己尝试一下。感谢阅读这篇文章！请继续关注我们，了解更多与 Python 编程相关的令人惊叹的学习内容。