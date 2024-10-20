# 如何用 Python 把 JSON 转换成字典？

> 原文：<https://www.askpython.com/python/dictionary/convert-json-to-a-dictionary>

大家好！在本教程中，我们将讨论如何将 JSON 转换成 Python 中的字典。

* * *

## JSON 是什么？

JSON 代表 **JavaScript 对象符号**。它是表示结构化数据的最流行和被广泛接受的数据格式之一。它是一种轻量级格式，用于存储和交换用 JavaScript 符号编写的文本数据。包含 JSON 数据的文件必须以扩展名`.json`保存。

## Python 中的 JSON

JSON 文件中 JSON 数据的表示类似于 Python 字典。这意味着 JSON 数据也是一组 **name: value** 对的集合，就像 Python 字典一样。

在 Python 中，我们有一个内置模块叫做 **[json](https://www.askpython.com/python-modules/python-json-module)** 。让我们在 Python 程序中导入`json`模块来处理 JSON 数据。

## 将 JSON 转换成字典的先决条件

*   导入 Python json 模块。
*   如果 JSON 文件不在同一个目录中，请提供它的完整路径
*   所有的 JSON 数据(字符串)都应该用双引号括起来，以避免 JSONDecodeError。

## 创建一个示例 JSON 文件

让我们创建一个包含一些 JSON 字符串的样本 JSON 文件。我们将在我们的 Python 程序中使用这个 JSON 文件来演示`json`模块在 Python 中处理 JSON 数据的工作。

```py
{
    "Linux": ["Ubuntu", "Fedora", "CentOS", "Linux Mint", 
              "Debian", "Kali Linux"],
    "Windows": ["Windows 2000", "Windows XP", "Windows Vista", 
                "Windows 7", "Windows 8", "Windows 10"],
    "MacOS": ["OS X 10.8", "OS X 10.9", "OS X 10.10", "OS X 10.11",
              "MacOS 10.12", "MacOS 10.13", "MacOS 10.14"]
}

```

## 将 JSON 转换成字典

我们已经创建了一个包含 JSON 数据(字符串)的样本 JSON 文件。现在，让我们将这个 JSON 数据转换成一个 [Python 对象](https://www.askpython.com/python/built-in-methods/python-object-method)。我们将按照下面给出的步骤将 JSON 转换成 Python 中的字典

1.  在程序中导入`json`模块。
2.  打开我们在上面创建的样本 JSON 文件。
3.  使用 **`json.load()`** 功能将文件数据转换成字典。
4.  检查 **`json.load()`** 函数返回的值的类型。
5.  使用 for 循环打印 Python 字典中的键:值对。
6.  关闭打开的示例 JSON 文件，这样它就不会被篡改。

让我们通过 Python 代码实现所有这些步骤。

```py
# Import json Python module
import json

# Open the sample JSON file
# Using the open() function
file = open("C:\path\sample_file.json", 'r')

# Convert the JSON data into Python object
# Here it is a dictionary
json_data = json.load(file)

# Check the type of the Python object
# Using type() function 
print(type(json_data))

# Iterate through the dictionary
# And print the key: value pairs
for key, value in json_data.items():
    print(f"\nKey: {key}")
    print(f"Value: {value}\n")

# Close the opened sample JSON file
# Using close() function
file.close()

```

**输出:**

```py
<class 'dict'>

Key: Linux
Value: ['Ubuntu', 'Fedora', 'CentOS', 'Linux Mint', 'Debian', 'Kali Linux']

Key: Windows
Value: ['Windows 2000', 'Windows XP', 'Windows Vista', 'Windows 7', 'Windows 8', 'Windows 10']

Key: MacOS
Value: ['OS X 10.8', 'OS X 10.9', 'OS X 10.10', 'OS X 10.11', 'MacOS 10.12', 'MacOS 10.13', 'MacOS 10.14']

```

## **总结**

在本教程中，我们学习了如何读取一个 JSON 文件，然后使用 [json.load()函数](https://www.askpython.com/python/examples/serialize-deserialize-json)将其转换为 Python 字典。希望您已经清楚这个主题，并准备好自己执行这些操作。感谢您阅读本文，请继续关注我们，了解更多关于 Python 编程的精彩内容。