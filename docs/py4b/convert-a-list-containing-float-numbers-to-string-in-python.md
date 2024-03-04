# 在 Python 中将包含浮点数的列表转换为字符串

> 原文：<https://www.pythonforbeginners.com/lists/convert-a-list-containing-float-numbers-to-string-in-python>

在 python 中工作时，我们可能需要将包含浮点数的列表转换为字符串。在本文中，我们将研究不同的方法来将包含浮点数的列表转换为包含列表元素的字符串，以空格分隔子字符串的形式。

## 将包含浮点数的列表转换为字符串的任务的重要函数

要将浮点数列表转换为字符串，我们需要几个字符串方法，我们将首先讨论这些方法，然后执行操作。

**str()函数**

`str()`函数将整数文字、浮点文字或任何其他给定的输入转换为字符串文字，并在转换后返回输入的字符串文字。这可以如下进行。

```py
float_num=10.0
str_num=str(float_num)
```

**join()方法**

在分隔符上调用方法，字符串的可迭代对象作为输入传递给方法。它用分隔符连接 iterable 对象中的每个字符串，并返回一个新字符串。

`join()`方法的语法与`separator.join(iterable)`相同，其中分隔符可以是任何字符，iterable 可以是列表或元组等。这可以这样理解。

```py
 str_list=["I","am","Python","String"]
print("List of strings is:")
print(str_list)
separator=" "
str_output=separator.join(str_list)
print("String output is:")
print(str_output)
```

输出:

```py
List of strings is:
['I', 'am', 'Python', 'String']
String output is:
I am Python String
```

**map()函数**

`map()`函数将一个函数(函数是第一类对象，在 python 中可以作为参数传递)和一个 iterable 作为参数，对 iterable 对象的每个元素执行函数，并返回可以转换为任何 iterable 的输出 map 对象。

`map()`函数的语法是`map(function,iterable)`。这可以这样理解。

```py
 float_list=[10.0,11.2,11.7,12.1]
print("List of floating point numbers is:")
print(float_list)
str_list=list(map(str,float_list))
print("List of String numbers is:")
print(str_list) 
```

输出:

```py
List of floating point numbers is:
[10.0, 11.2, 11.7, 12.1]
List of String numbers is:
['10.0', '11.2', '11.7', '12.1']
```

现在我们将看到如何使用上述函数将包含浮点数的列表转换为字符串。

## 使用 for 循环将包含浮点数的列表转换为字符串

我们可以将浮点数列表转换为字符串，方法是声明一个空字符串，然后执行[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)将列表元素添加到字符串中，如下所示。

```py
 float_list=[10.0,11.2,11.7,12.1]
print("List of floating point numbers is:")
print(float_list)
float_string=""
for num in float_list:
    float_string=float_string+str(num)+" "

print("Output String is:")
print(float_string.rstrip())
```

输出:

```py
List of floating point numbers is:
[10.0, 11.2, 11.7, 12.1]
Output String is:
10.0 11.2 11.7 12.1
```

在上面的程序中，输出字符串在末尾包含一个额外的空格，必须使用`rstrip()`方法将其删除。为了避免这个额外的操作，我们可以使用 `join()`方法，而不是通过添加字符串来创建输出字符串来执行连接操作，如下所示。

```py
 float_list=[10.0,11.2,11.7,12.1]
print("List of floating point numbers is:")
print(float_list)
float_string=""
for num in float_list:
    float_string=" ".join([float_string,str(num)])
print("Output String is:")
print(float_string.lstrip())
```

输出

```py
List of floating point numbers is:
[10.0, 11.2, 11.7, 12.1]
Output String is:
10.0 11.2 11.7 12.1
```

在上面的方法中，在输出字符串的左侧添加了一个额外的空格，必须使用`lstrip()`方法将其删除。为了避免这种情况，我们可以使用 map()函数将浮点数列表转换为字符串列表，然后使用`join()`方法执行字符串连接，得到如下输出字符串，而不是对列表的每个元素应用`str()`函数。

```py
float_list=[10.0,11.2,11.7,12.1]
print("List of floating point numbers is:")
print(float_list)
str_list=list(map(str,float_list))
print("List of String numbers is:")
print(str_list)
float_string=""
for num in float_list:
    float_string=" ".join(str_list)

print("Output String is:")
print(float_string)
```

输出:

```py
List of floating point numbers is:
[10.0, 11.2, 11.7, 12.1]
List of String numbers is:
['10.0', '11.2', '11.7', '12.1']
Output String is:
10.0 11.2 11.7 12.1
```

## 使用列表理解将包含浮点数的列表转换为字符串

代替 for 循环，我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 来执行浮点数列表到字符串的转换，如下所示。我们可以使用 join()方法来理解列表，如下所示。

```py
float_list=[10.0,11.2,11.7,12.1]
print("List of floating point numbers is:")
print(float_list)
str_list=[str(i) for i in float_list]
print("List of floating string numbers is:")
print(str_list)
float_string=" ".join(str_list)
print("Output String is:")
print(float_string)
```

输出:

```py
List of floating point numbers is:
[10.0, 11.2, 11.7, 12.1]
List of floating string numbers is:
['10.0', '11.2', '11.7', '12.1']
Output String is:
10.0 11.2 11.7 12.1
```

## 结论

在本文中，我们看到了如何使用不同的字符串方法(如`str()`、`join()`)将包含浮点数的列表转换为字符串，以及 python 中的循环或列表理解。请继续关注更多内容丰富的文章。