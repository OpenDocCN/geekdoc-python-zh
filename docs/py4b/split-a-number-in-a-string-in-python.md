# 在 Python 中拆分字符串中的数字

> 原文：<https://www.pythonforbeginners.com/python-strings/split-a-number-in-a-string-in-python>

在处理文本数据时，可能需要从文本数据中提取数字。在 python 中，我们使用字符串处理文本数据。所以，我们要做的任务就是在一个字符串中找到并拆分一个数字。在提取数字时，我们可以将字符串分为两种类型。第一种类型将只包含空格分隔的数字，第二种类型的字符串也将包含字母和标点符号以及数字。在本文中，我们将看到如何从这两种类型的字符串中逐一提取数字。所以，让我们深入研究一下。

## 当字符串只包含空格分隔的数字时，拆分字符串中的数字。

当字符串格式中只包含空格分隔的数字时，我们可以简单地使用 [python 字符串分割](https://www.pythonforbeginners.com/dictionary/python-split)操作在空格处分割字符串。在任何字符串上调用 split 方法时，都会返回一个子字符串列表，在我们的例子中，这些子字符串是字符串格式的数字。在得到列表中字符串格式的数字后，我们可以使用`int()` 函数将所有的字符串转换成整数。这可以如下进行。

```py
 num_string="10 1 23 143 234 108 1117"
print("String of numbers is:")
print(num_string)
str_list=num_string.split()
print("List of numbers in string format is:")
print(str_list)
num_list=[]
for i in str_list:
    num_list.append(int(i))    
print("Output List of numbers is:")
print(num_list)
```

输出:

```py
String of numbers is:
10 1 23 143 234 108 1117
List of numbers in string format is:
['10', '1', '23', '143', '234', '108', '1117']
Output List of numbers is:
[10, 1, 23, 143, 234, 108, 1117]
```

我们也可以如下使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)执行上述操作。

```py
 num_string="10 1 23 143 234 108 1117"
print("String of numbers is:")
print(num_string)
str_list=num_string.split()
print("List of numbers in string format is:")
print(str_list)
num_list=[int(i) for i in str_list]   
print("Output List of numbers is:")
print(num_list)
```

输出:

```py
String of numbers is:
10 1 23 143 234 108 1117
List of numbers in string format is:
['10', '1', '23', '143', '234', '108', '1117']
Output List of numbers is:
[10, 1, 23, 143, 234, 108, 1117]
```

我们还可以使用`map()`函数将列表中的字符串转换成整数。`map()`函数将一个函数和一个 iterable 作为输入参数，对 iterable 对象的每个元素执行该函数，并返回可以转换为任何 iterable 的输出 map 对象。这里我们将提供 `int()`函数作为第一个参数，字符串列表作为第二个参数，这样字符串可以被转换成整数。这可以如下进行。

```py
num_string="10 1 23 143 234 108 1117"
print("String of numbers is:")
print(num_string)
str_list=num_string.split()
print("List of numbers in string format is:")
print(str_list)
num_list=list(map(int,str_list))
print("Output List of numbers is:")
print(num_list)
```

输出:

```py
String of numbers is:
10 1 23 143 234 108 1117
List of numbers in string format is:
['10', '1', '23', '143', '234', '108', '1117']
Output List of numbers is:
[10, 1, 23, 143, 234, 108, 1117]
```

## 当字符串包含字母时，拆分字符串中的数字。

当字符串包含字母时，我们将首先使用正则表达式从字符串中提取数字，然后将数字转换为整数形式。

为了提取数字，我们将使用来自`re`模块的 `findall()`方法。`re.findall()`将模式(在我们的例子中是一个或多个数字)和字符串作为输入，并返回与模式匹配的子字符串列表。提取字符串形式的数字列表后，我们可以将字符串转换为整数，如下所示。

```py
 import re
num_string="I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117"
print("Given String is:")
print(num_string)
pattern="\d+"
str_list=re.findall(pattern,num_string)
print("List of numbers in string format is:")
print(str_list)
num_list=[]
for i in str_list:
    num_list.append(int(i)) 
print("Output List of numbers is:")
print(num_list) 
```

输出:

```py
Given String is:
I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117
List of numbers in string format is:
['20', '102', '23', '120', '1117']
Output List of numbers is:
[20, 102, 23, 120, 1117]
```

我们可以使用列表理解来执行上述操作，如下所示。

```py
import re
num_string="I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117"
print("Given String is:")
print(num_string)
pattern="\d+"
str_list=re.findall(pattern,num_string)
print("List of numbers in string format is:")
print(str_list)
num_list=[int(i) for i in str_list]
print("Output List of numbers is:")
print(num_list)
```

输出:

```py
Given String is:
I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117
List of numbers in string format is:
['20', '102', '23', '120', '1117']
Output List of numbers is:
[20, 102, 23, 120, 1117]
```

我们可以如下使用`map()`函数执行相同的操作。

```py
 import re
num_string="I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117"
print("Given String is:")
print(num_string)
pattern="\d+"
str_list=re.findall(pattern,num_string)
print("List of numbers in string format is:")
print(str_list)
num_list=list(map(int,str_list))
print("Output List of numbers is:")
print(num_list)
```

输出:

```py
Given String is:
I have solved 20 ques102tions in last 23 days and have scored 120 marks with rank 1117
List of numbers in string format is:
['20', '102', '23', '120', '1117']
Output List of numbers is:
[20, 102, 23, 120, 1117]
```

## 结论

在本文中，我们看到了如何使用不同的方法，如列表理解和正则表达式，将一个字符串中的数字拆分并提取到另一个列表中。请继续关注更多内容丰富的文章。