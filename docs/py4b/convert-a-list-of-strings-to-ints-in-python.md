# 在 Python 中将字符串列表转换为整型

> 原文：<https://www.pythonforbeginners.com/basics/convert-a-list-of-strings-to-ints-in-python>

在 python 中，我们使用列表来存储不同的元素。在本文中，我们将讨论将字符串列表转换为整型的不同方法。我们还将讨论如何在 python 中将字符串列表转换成整型。

## int()函数

`int()` 函数将一个字符串或浮点文字作为其输入参数，并返回一个整数，如下所示。

```py
myStr = "11"
print("data type of {} is {}".format(myStr, type(myStr)))
myInt = int(myStr)
print("data type of {} is {}".format(myInt, type(myInt)))
```

输出:

```py
data type of 11 is <class 'str'>
data type of 11 is <class 'int'>
```

如果`int()`函数的输入不能转换成整数，程序运行到`ValueError`异常，并以消息“`[ValueError: invalid literal for int() with base 10](ValueError: invalid literal for int() with base 10)`终止。您可以在下面的示例中观察到这一点。

```py
myStr = "Aditya"
print("data type of {} is {}".format(myStr, type(myStr)))
myInt = int(myStr)
print("data type of {} is {}".format(myInt, type(myInt)))
```

输出:

```py
data type of Aditya is <class 'str'>
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    myInt = int(myStr)
ValueError: invalid literal for int() with base 10: 'Aditya' 
```

为了避免异常，我们可以首先检查输入是否可以转换为整数。浮点数将总是被转换成整数。然而，当`int()`函数的输入是一个字符串并且由字符而不是数字组成时，就会发生`ValueError`异常。因此，要将字符串转换为整数，我们首先要检查它是否只由数字组成。为此，我们将使用`isdigit()`方法。

## isdigit()方法

在字符串上调用`isdigit()`方法时，如果字符串仅由十进制数字组成，则返回 true，如下所示。

```py
myStr = "11"
is_digit = myStr.isdigit()
print("{} consists of only digits:{}".format(myStr, is_digit)) 
```

输出:

```py
11 consists of only digits:True
```

如果字符串包含除十进制数字之外的任何其他字符，`isdigit()`方法将返回`False`。您可以在下面的示例中观察到这一点。

```py
myStr = "Aditya"
is_digit = myStr.isdigit()
print("{} consists of only digits:{}".format(myStr, is_digit))
```

输出:

```py
Aditya consists of only digits:False
```

在将字符串转换为整数时，我们可以使用`isdigit()`方法来避免 ValueError 异常。为此，我们将首先调用字符串上的`isdigit()`方法。如果它返回`True`，我们将使用`int()`函数将字符串转换为整数。否则，我们会说字符串不能转换成整数。您可以在下面的示例中观察到这一点。

```py
myStr = "Aditya"
is_digit = myStr.isdigit()
if is_digit:
    myInt=int(myStr)
    print("The integer is:",myInt)
else:
    print("'{}' cannot be converted into integer.".format(myStr))
```

输出:

```py
'Aditya' cannot be converted into integer.
```

既然我们已经讨论了`int()`函数和`isdigit() method, let's`函数的工作原理，现在我们来讨论在 python 中将字符串列表转换成整型的不同方法。

## 使用 Python 中的 for 循环将字符串列表转换为整数

要在 python 中将字符串列表转换为整型，我们将使用以下步骤。

*   首先，我们将创建一个名为`new_list`的空列表来存储整数。
*   之后，我们将使用 for 循环遍历字符串列表。
*   在迭代时，我们将首先检查当前字符串是否可以使用`isdigit()` 方法转换为 int。
*   如果字符串可以转换成 int，我们就用`int()`函数把字符串转换成 int。否则，我们会说当前字符串不能转换成整数。
*   在将字符串转换成 int 之后，我们将使用 `append()` 方法将它附加到`new_list`中。在`new_list`上调用`append()`方法时，该方法将新创建的整数作为其输入参数，并将其添加到`new_list`。
*   最后，我们将移动到输入列表中的下一个字符串。

在执行 for 循环后，我们将在`new_list`中获得 int 的列表。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
new_list = []
for string in myList:
    is_digit = string.isdigit()
    if is_digit:
        myInt = int(string)
        new_list.append(myInt)
    else:
        print("'{}' cannot be converted into an integer.".format(string))
print("The list of strings is:")
print(myList)
print("The list of ints is:")
print(new_list) 
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

### 使用 Python 中的 for 循环将字符串列表转换为整型

要使用 for 循环、`int()`函数和`isdigit()`函数将字符串列表的[列表转换为整型，我们将使用以下步骤。](https://www.pythonforbeginners.com/basics/list-of-lists-in-python)

*   首先，我们将创建一个名为`new_list`的空列表来存储列表的输出列表。
*   之后，我们将使用 for 循环遍历输入列表的内部列表。
*   在 for 循环中，我们将创建一个名为`temp_list`的空列表来存储从内部列表中获得的 int 列表。
*   然后，我们将使用另一个 for 循环迭代每个内部列表的元素。
*   在迭代内部列表的元素时，我们将首先检查当前字符串是否可以使用`isdigit()`方法转换为 int。
*   如果字符串可以转换成 int，我们就用`int()`函数把字符串转换成 int。否则，我们会说当前字符串不能转换成整数。然后，我们将移动到当前内部列表中的下一个字符串。
*   在将当前内部列表的所有字符串转换成整数后，我们还将使用 `append()`方法将它们附加到`temp_list`中。
*   迭代完每个内部循环后，我们将使用`append()`方法将`temp_list`添加到`new_list`中。然后，我们将移动到列表列表中的下一个内部列表。

执行 for 循环后，我们将获得包含整数元素而不是字符串的列表，如下例所示。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', "Aditya"]]
print("The list of lists of strings is:")
print(myList)
new_list = []
for inner_list in myList:
    temp_list = []
    for element in inner_list:
        is_digit = element.isdigit()
        if is_digit:
            myInt = int(element)
            temp_list.append(myInt)
        else:
            print("'{}' cannot be converted into an integer.".format(element))
    new_list.append(temp_list)

print("The list of lists of ints is:")
print(new_list)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', 'Aditya']]
'Aditya' cannot be converted into an integer.
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

在上面的示例中，字符串“Aditya”不能转换为 int。因此，它已从输出中省略。

## 使用列表理解将字符串列表转换为整数

[python 中的 List comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 用于从现有的容器对象创建新的列表。您可以使用 list comprehension 而不是 for 循环将字符串列表转换为 int 列表，如下所示。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
new_list = [int(element) for element in myList]
print("The list of strings is:")
print(myList)
print("The list of ints is:")
print(new_list)
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

这种方法在调用`int()`函数之前不检查字符串是否可以转换成 int。因此，如果我们在列表中发现一个不能转换成整数的元素，程序可能会遇到`ValueError`异常。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
new_list = [int(element) for element in myList]
print("The list of strings is:")
print(myList)
print("The list of ints is:")
print(new_list)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    new_list = [int(element) for element in myList]
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <listcomp>
    new_list = [int(element) for element in myList]
ValueError: invalid literal for int() with base 10: 'Aditya' 
```

在上面的例子中，字符串'`Aditya`'不能转换成 int。因此，程序会遇到 ValueError 异常。

异常导致程序突然终止。这可能会导致数据或程序中已完成的工作丢失。

为了处理异常，使程序不会突然终止，您可以使用如下所示的 [python try-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
print("The list of strings is:")
print(myList)
try:
    new_list = [int(element) for element in myList]
    print("The list of ints is:")
    print(new_list)
except ValueError:
    print("Some values in the input list can't be converted to int.")
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', 'Aditya']
Some values in the input list can't be converted to int.
```

## 使用 map()函数将字符串列表转换为整数

`map()` 函数用于使用一条 python 语句将函数应用于容器对象的所有元素。它将一个函数作为第一个输入参数，将一个容器对象作为第二个输入参数。执行后，它返回一个包含输出元素的 map 对象。

要将字符串列表转换为 int，我们将首先通过传递 int 函数作为第一个输入参数和字符串列表作为第二个参数来获得 map 对象。一旦我们获得了地图对象，我们将使用`list()`构造函数把它转换成一个列表。列表将包含整数形式的所有元素。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
new_list = list(map(int,myList))
print("The list of strings is:")
print(myList)
print("The list of ints is:")
print(new_list)
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21']
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

同样，这种方法在调用`int()`函数之前不检查字符串是否可以转换成 int。因此，如果我们在列表中发现一个不能转换成整数的元素，程序可能会遇到`ValueError`异常。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
new_list = list(map(int, myList))
print("The list of strings is:")
print(myList)
print("The list of ints is:")
print(new_list)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    new_list = list(map(int, myList))
ValueError: invalid literal for int() with base 10: 'Aditya' 
```

为了处理异常，使程序不会突然终止，您可以使用 python try-except 块，如下所示。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
print("The list of strings is:")
print(myList)
try:
    new_list = list(map(int, myList))
    print("The list of ints is:")
    print(new_list)
except ValueError:
    print("Some values in the input list couldn't be converted to int.")
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', 'Aditya']
Some values in the input list couldn't be converted to int.
```

### 使用 map()函数将字符串列表转换为整型

要使用 python 中的`map()`函数将字符串列表转换为整型，我们将使用以下步骤。

*   首先，我们将创建一个名为`new_list`的空列表来存储输出列表。
*   之后，我们将使用 for 循环遍历列表列表。
*   在迭代期间，我们将首先通过传递 int 函数作为其第一个输入参数和内部字符串列表作为其第二个参数来获得每个内部列表的 map 对象。
*   一旦我们获得了地图对象，我们将使用`list()`构造函数把它转换成一个列表。
*   该列表将包含整数形式的内部列表的元素。我们将使用`append()`方法把这个列表附加到`new_list`中。然后，我们将进入下一个内部列表。

在执行 for 循环后，我们将获得包含整数的列表列表作为内部列表的元素，如下面的代码所示。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21']]
print("The list of lists of strings is:")
print(myList)
new_list = []
for inner_list in myList:
    temp_list = list(map(int, inner_list))
    new_list.append(temp_list)

print("The list of lists of ints is:")
print(new_list)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21']]
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

同样，在这种情况下，程序可能会运行到`ValueError`异常。因此，不要忘记在程序中使用 try-except 块，如下所示。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21']]
print("The list of lists of strings is:")
print(myList)
new_list = []
for inner_list in myList:
    try:
        temp_list = list(map(int, inner_list))
        new_list.append(temp_list)
    except ValueError:
        print("Some values couldn't be converted to int.")

print("The list of lists of ints is:")
print(new_list)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21']]
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

## 使用 eval()函数将字符串列表转换为整数

`eval()` 函数用于解析和评估 python 语句。`eval()`函数将一个字符串作为其输入参数，解析该字符串，计算值，并返回输出。

例如，我们可以将字符串`“2+2”` 传递给 eval 函数，它将返回 4 作为输出。您可以在下面的示例中观察到这一点。

```py
x = eval('2+2')
print(x)
```

输出:

```py
4
```

类似地，当我们将一个只包含十进制数字的字符串传递给`eval()`函数时，它会返回一个整数，如下所示。

```py
x = eval('11')
print(x)
```

输出:

```py
11
```

如果输入字符串包含字母字符而不是数字，程序将运行到`NameError`异常。您可以在下面的示例中观察到这一点。

```py
x = eval('Aditya')
print(x)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 1, in <module>
    x = eval('Aditya')
  File "<string>", line 1, in <module>
NameError: name 'Aditya' is not defined
```

这里，术语“`Aditya`”被评估为变量名。因此，`eval()`函数试图获取与变量“`Aditya`”关联的对象的值。但是，该程序不包含任何名为“`Aditya`”的变量。因此，程序遇到了一个`NameError`异常。

要使用`eval()` 函数将字符串列表转换为整数，我们将使用以下步骤。

*   首先，我们将创建一个名为`new_list`的空列表来存储整数。
*   之后，我们将使用 for 循环遍历字符串列表。
*   在迭代时，我们将首先检查当前字符串是否可以使用`isdigit()`方法转换为 int。
*   如果字符串可以转换成 int，我们就用`eval()`函数把字符串转换成 int。否则，我们会说当前字符串不能转换成整数。
*   在将字符串转换成 int 之后，我们将使用`append()`方法将它附加到`new_list`中。
*   最后，我们将移动到输入列表中的下一个字符串。

在执行 for 循环后，我们将在`new_list`中获得 int 的列表。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
print("The list of strings is:")
print(myList)
new_list = []
for string in myList:
    is_digit = string.isdigit()
    if is_digit:
        myInt = eval(string)
        new_list.append(myInt)
    else:
        print("'{}' cannot be converted into int.".format(string))
print("The list of ints is:")
print(new_list)
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', 'Aditya']
'Aditya' cannot be converted into int.
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

### 使用 eval()函数将字符串列表转换为整型

要使用`eval()`函数将字符串列表转换为整型，我们将使用以下步骤。

*   首先，我们将创建一个名为`new_list`的空列表来存储列表的输出列表。
*   之后，我们将使用 for 循环遍历输入列表的内部列表。
*   在 for 循环中，我们将创建一个名为`temp_list`的空列表来存储从内部列表中获得的 int 值。
*   然后，我们将使用另一个 for 循环迭代当前内部列表的元素。
*   在迭代内部列表的元素时，我们将首先检查当前字符串是否可以使用`isdigit()`方法转换为 int。
*   如果字符串可以转换成 int，我们就用`eval()`函数把字符串转换成 int。否则，我们会说当前字符串不能转换成整数。最后，我们将移动到当前内部列表中的下一个字符串。
*   在将当前内部列表的所有字符串转换成整数后，我们将使用`append()`方法将它们附加到`temp_list`中。
*   迭代完每个内部循环后，我们将使用`append()`方法将`temp_list`添加到`new_list`中。然后，我们将移动到列表列表中的下一个内部列表。

执行 for 循环后，我们将获得包含整数元素而不是字符串的列表，如下例所示。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', "Aditya"]]
print("The list of lists of strings is:")
print(myList)
new_list = []
for inner_list in myList:
    temp_list = []
    for element in inner_list:
        is_digit = element.isdigit()
        if is_digit:
            myInt = eval(element)
            temp_list.append(myInt)
        else:
            print("'{}' cannot be converted into an integer.".format(element))
    new_list.append(temp_list)

print("The list of lists of ints is:")
print(new_list)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', 'Aditya']]
'Aditya' cannot be converted into an integer.
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

## 在 Python 中将字符串列表就地转换为整型

前面几节中讨论的所有方法都创建了一个新的输出列表。如果我们想将输入列表的元素从 string 转换成 int，我们可以使用`int()`方法或`eval()`方法，而不是创建一个新的输出列表。

### 使用 int()函数将字符串列表就地转换为整数

要使用`int()`函数将字符串列表就地转换为整数，我们将使用以下步骤。

*   首先，我们将使用`len()`函数找到字符串列表的长度。`len()`函数将列表作为其输入参数，并返回列表的长度。我们将把长度存储在变量`list_len`中。
*   获得列表长度后，我们将使用`range()`函数创建一个从 0 到`list_len`的数字序列。`range()`函数将`list_len`作为输入参数，并返回序列。
*   获得序列后，我们将使用 for 循环遍历序列。迭代时，我们将使用索引来访问列表中的每个元素。
*   获得元素后，我们将检查它是否可以转换为整数。为此，我们将使用`isdigit()`方法。
*   如果字符串可以转换成 int，我们就用`int()`函数把字符串转换成 int。否则，我们会说当前字符串不能转换成整数。最后，我们将移动到当前内部列表中的下一个字符串。
*   在将当前内部列表的所有字符串转换成整数后，我们将把它们重新分配到它们在输入列表中的原始位置。

执行 for 循环后，输入列表的所有元素都将被转换为整数。您可以在下面的示例中观察到这一点。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
list_len = len(myList)
sequence = range(list_len)
print("The list of strings is:")
print(myList)
for index in sequence:
    string = myList[index]
    is_digit = string.isdigit()
    if is_digit:
        myInt = int(string)
        myList[index] = myInt
    else:
        myList.remove(string)
        print("'{}' cannot be converted into int.".format(string))
print("The list of ints is:")
print(myList)
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', 'Aditya']
'Aditya' cannot be converted into int.
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

在上面的示例中，字符串“Aditya”不能转换为整数。因此，我们使用 remove()方法删除了字符串。

### 使用 eval()函数将字符串列表就地转换为整型

除了使用`int()`函数，您还可以使用`eval()`函数将字符串列表转换为 int，如下所示。

```py
myList = ['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', "Aditya"]
list_len = len(myList)
sequence = range(list_len)
print("The list of strings is:")
print(myList)
for index in sequence:
    string = myList[index]
    is_digit = string.isdigit()
    if is_digit:
        myInt = eval(string)
        myList[index] = myInt
    else:
        myList.remove(string)
        print("'{}' cannot be converted into int.".format(string))
print("The list of ints is:")
print(myList)
```

输出:

```py
The list of strings is:
['1', '2', '23', '32', '12', '44', '34', '55', '46', '21', 'Aditya']
'Aditya' cannot be converted into int.
The list of ints is:
[1, 2, 23, 32, 12, 44, 34, 55, 46, 21]
```

## 在 Python 中就地将字符串列表转换为整型

### 使用 int()函数将字符串列表转换为整数

我们还可以使用 `int()`函数将字符串列表就地转换成整型。为此，我们将使用以下步骤。

*   我们将使用 for 循环遍历列表的内部列表。
*   对于每个内部列表，我们将使用`len()`函数找到字符串内部列表的长度。我们将把长度存储在变量`list_len`中。
*   获得内部列表的长度后，我们将使用`range()`函数创建一个从 0 到`list_len`的数字序列。`range()` 函数将`list_len`作为输入参数，并返回序列。
*   获得序列后，我们将使用 for 循环遍历序列。迭代时，我们将使用索引访问内部列表的每个元素。
*   获得元素后，我们将检查它是否可以转换为整数。为此，我们将使用`isdigit()`方法。
*   如果字符串可以转换为 int，我们将使用`int()`函数将字符串转换为 int。否则，我们会说当前字符串不能转换成整数。最后，我们将移动到当前内部列表中的下一个字符串。
*   在将当前内部列表的所有字符串转换成整数后，我们将把它们重新分配到它们在输入内部列表中的原始位置。

执行上述步骤后，原始列表的元素将被转换为整数。您可以在下面的示例中观察到这一点。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', "Aditya"]]
print("The list of lists of strings is:")
print(myList)
for inner_list in myList:
    list_len = len(inner_list)
    sequence = range(list_len)
    for index in sequence:
        string = inner_list[index]
        is_digit = string.isdigit()
        if is_digit:
            myInt = int(string)
            inner_list[index] = myInt
        else:
            print("'{}' cannot be converted into int.".format(string))
            inner_list.remove(string)

print("The list of lists of ints is:")
print(myList)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', 'Aditya']]
'Aditya' cannot be converted into int.
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

### 使用 eval()函数将字符串列表转换为整数

除了使用`int()`函数，你还可以使用`eval()`函数将字符串列表转换成整数，如下所示。

```py
myList = [['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', "Aditya"]]
print("The list of lists of strings is:")
print(myList)
for inner_list in myList:
    list_len = len(inner_list)
    sequence = range(list_len)
    for index in sequence:
        string = inner_list[index]
        is_digit = string.isdigit()
        if is_digit:
            myInt = eval(string)
            inner_list[index] = myInt
        else:
            print("'{}' cannot be converted into int.".format(string))
            inner_list.remove(string)

print("The list of lists of ints is:")
print(myList)
```

输出:

```py
The list of lists of strings is:
[['1', '2', '23'], ['32', '12'], ['44', '34', '55', '46'], ['21', 'Aditya']]
'Aditya' cannot be converted into int.
The list of lists of ints is:
[[1, 2, 23], [32, 12], [44, 34, 55, 46], [21]]
```

## 结论

在本文中，我们讨论了在 python 中将字符串列表转换为整型的不同方法。如果需要将字符串列表转换成整数，可以使用使用`map()` 函数的方法。如果需要将字符串列表就地转换为 int，可以使用本文最后两节中讨论的方法。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。

快乐学习！