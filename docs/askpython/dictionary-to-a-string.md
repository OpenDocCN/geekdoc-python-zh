# 如何用 Python 把字典转换成字符串？

> 原文：<https://www.askpython.com/python/string/dictionary-to-a-string>

在本教程中，我们将讨论在 Python 中将字典转换成字符串的不同方法。

* * *

## Python 中的字典是什么？

**字典**是一个 Python 对象，用于以**键:值**格式存储数据。键及其对应的值由冒号(:)分隔。字典中的每个键:值对由逗号(，)分隔。Python 中的字典总是用大括号{}括起来。

Python 字典是一种**无序的**数据结构，不像 Python 中的列表或元组。我们可以通过使用相应的键直接访问 Python 字典的任何值。让我们看看如何通过 Python 代码创建字典。

```py
# Create a Python dictionary
dc = {'py': "PYTHON", 'cpp': "C++", 'mat': "MATLAB"}
print(type(dc))
print(dc)

```

**输出:**

```py
<class 'dict'> 
{'py': 'PYTHON', 'cpp': 'C++', 'mat': 'MATLAB'}

```

## Python 中的字符串是什么？

一个**字符串**也是一个 Python 对象，它是最常用的数据结构，用于存储一系列字符。在 Python 中，包含在单引号、双引号或三引号中的任何内容都是字符串。在 Python 中，单引号和双引号可以互换使用来表示单行字符串，而三引号用于存储多行字符串。让我们通过 Python 代码创建一个字符串。

```py
# Create a Python dictionary
sr = "py: PYTHON cpp: C++ mat: MATLAB"
print(type(sr))
print(sr)

```

**输出:**

```py
<class 'str'> 
py: PYTHON cpp: C++ mat: MATLAB

```

## Python 中把字典转换成字符串的不同方法

在 Python 中，有多种方法可以将字典转换成字符串。让我们来讨论一些最常用的方法/途径。

### 1.使用 str()函数

在这个将字典转换为字符串的方法中，我们将简单地将字典对象传递给`str()`函数。

```py
# Create a Python dictionary
dc = {'A': 'Android', 'B': 'Bootstrap', 'C': 'C Programming', 'D': 'Dart'}
print(type(dc))
print(f"Given dictionary: {dc}")

# Convert the dictionary to a string
# using str() function
sr = str(dc)
print(type(sr))
print(f"Converted string: {sr}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary: {'A': 'Android', 'B': 'Bootstrap', 'C': 'C Programming', 'D': 'Dart'} 
<class 'str'> 
Converted string: {'A': 'Android', 'B': 'Bootstrap', 'C': 'C Programming', 'D': 'Dart'}

```

### 2.使用 json.dumps()函数

在这个将字典转换成字符串的方法中，我们将把字典对象传递给`json.dumps()`函数。为了使用`json.dumps()`函数，我们必须导入 [JSON 模块](https://www.askpython.com/python-modules/python-json-module)，这是 Python 中的一个内置包。

```py
# Import Python json module
import json

# Create a Python dictionary
dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
print(type(dict))
print(f"Given dictionary: {dict}")

# Convert the dictionary to a string
# using json.dumps() function
str = json.dumps(dict)
print(type(str))
print(f"Converted string: {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary: {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5} 
<class 'str'> 
Converted string: {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

```

### 3.使用空字符串和 for 循环

在这个将字典转换成字符串的方法中，我们将简单地通过使用一个 [`for`循环](https://www.askpython.com/course/python-course-for-loop)遍历字典对象来访问字典的键。然后我们访问对应于每个键的值，并将 key: value 对添加到一个空字符串中。

```py
# Create a Python dictionary
dict = {'D': "Debian", 'U': "Ubuntu", 'C': "CentOS"}
print(type(dict))
print(f"Given dictionary- {dict}")

# Create an empty string
str = ""

# Convert the dictionary to a string
# using for loop only
for item in dict:
    str += item + ':' + dict[item] + ' '
print(type(str))
print(f"Converted string- {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary- {'D': 'Debian', 'U': 'Ubuntu', 'C': 'CentOS'} 
<class 'str'> 
Converted string- D:Debian U:Ubuntu C:CentOS

```

### 4.使用空字符串、for 循环和 items()函数

在这个将字典转换成字符串的方法中，我们将首先通过使用一个`for`循环和`items()`函数遍历字典对象来访问 key:value 对。然后，我们将每个键:值对添加到一个空字符串中。

```py
# Create a Python dictionary
dict = {'Google': "GCP", 'Microsoft': "Azure", 'Amazon': "AWS"}
print(type(dict))
print(f"Given dictionary- {dict}")

# Create an empty string
str = ""

# Convert the dictionary to a string
# using for loop and items() function
for key, value in dict.items():
    str += key + ':' + value + ' '
print(type(str))
print(f"Converted string- {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary- {'Google': 'GCP', 'Microsoft': 'Azure', 'Amazon': 'AWS'} 
<class 'str'> 
Converted string- Google:GCP Microsoft:Azure Amazon:AWS  

```

### 5.使用 for 循环、items()和 str.join()函数

在这个将字典转换为字符串的方法中，我们将使用一个`for`循环遍历字典对象，并添加键&的值。然后我们将使用`str.join()`函数将所有的键:值对连接在一起作为一个单独的字符串。

```py
# Create a Python dictionary
dict = {'Google': " Chrome", 'Microsoft': " Edge", 'Apple': " Safari"}
print(type(dict))
print(f"Given dictionary: {dict}")

# Convert the dictionary to a string
# using str.join() and items() function
str = ', '.join(key + value for key, value in dict.items())
print(type(str))
print(f"Converted string: {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary: {'Google': '-Chrome', 'Microsoft': '-Edge', 'Apple': '-Safari'} 
<class 'str'> 
Converted string: Google-Chrome, Microsoft-Edge, Apple-Safari

```

在 Python 中，我们还可以将 dictionary 对象的键和值转换成两个独立的字符串，而不是将键:值对转换成一个字符串。我们一个一个来讨论吧。

## 将字典的关键字转换为字符串

首先，我们将讨论将 dictionary 对象的键转换成字符串的不同方法。

### 1.使用空字符串和 for 循环

```py
# Create a Python dictionary
dict = {'OS': "Operating System", 'DBMS': "Database Management System", 'CN': "Computer Network"}
print(type(dict))
print(f"Given dictionary- {dict}")

# Create an empty string
str = ""

# Convert the dictionary keys into a string
# using for loop only
for item in dict:
    str += item + " "
print(type(str))
print(f"Keys in string- {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary- {'OS': 'Operating System', 'DBMS': 'Database Management System', 'CN': 'Computer Network'} 
<class 'str'> 
Keys in string- OS DBMS CN

```

### 2.使用 for 循环和 str.join()函数

```py
# Create a Python dictionary
dict = {'gcc': "C program", 'g++': "C++ program", 'py': "Python Program"}
print(type(dict))
print(f"Given dictionary: {dict}")

# Convert the dictionary keys into a string
# using str.join()
str = ', '.join(key for key in dict)
print(type(str))
print(f"Keys in string: {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary: {'gcc': 'C program', 'g++': 'C++ program', 'py': 'Python Program'} 
<class 'str'> 
Keys in string: gcc, g++, py

```

## 将字典的值转换为字符串

现在让我们讨论将 dictionary 对象的值转换为字符串的不同方法。

### 1.使用空字符串和 for 循环

```py
# Create a Python dictionary
dict = {'OS': "Operating_System", 'DBMS': "Database_Management_System", 'CN': "Computer_Network"}
print(type(dict))
print(f"Given dictionary- {dict}")

# Create an empty string
str = ""

# Convert the dictionary values into a string
# using for loop only
for item in dict:
    str += dict[item] + " "
print(type(str))
print(f"Values in string- {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary- {'OS': 'Operating_System', 'DBMS': 'Database_Management_System', 'CN': 'Computer_Network'} 
<class 'str'> 
Values in string- Operating_System Database_Management_System Computer_Network 

```

### 2.使用 for 循环和 str.join()函数

```py
# Create a Python dictionary
dict = {'gcc': "C program", 'g++': "C++ program", 'py': "Python Program"}
print(type(dict))
print(f"Given dictionary: {dict}")

# Convert the dictionary values into a string
# using str.join()
str = ', '.join(dict[item] for item in dict)
print(type(str))
print(f"Values in string: {str}")

```

**输出:**

```py
<class 'dict'> 
Given dictionary: {'gcc': 'C program', 'g++': 'C++ program', 'py': 'Python Program'} 
<class 'str'> 
Values in string: C program, C++ program, Python Program

```

## 结论

在本教程中，我们学习了用 Python 将字典转换成字符串的不同方法。我们还学习了如何将 dictionary 对象的键和值转换成两个独立的字符串。希望你已经理解了上面讨论的东西，并将自己尝试这些。