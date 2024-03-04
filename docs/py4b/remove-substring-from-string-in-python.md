# 在 Python 中从字符串中移除子字符串

> 原文：<https://www.pythonforbeginners.com/basics/remove-substring-from-string-in-python>

在 python 中处理文本数据时，我们有时需要从文本中移除特定的子串。在本文中，我们将讨论在 Python 中从字符串中移除子串的不同方法。

## 使用 split()方法从 Python 中的字符串中移除子字符串

Python 中的`split()` 方法用于在分隔符处将字符串分割成子字符串。当在字符串上调用`split()`方法时，它以分隔符的形式接受一个字符串作为它的输入参数。执行后，它从原始字符串中返回一个子字符串列表，该子字符串在分隔符处拆分。

要使用`split()`方法从 Python 中的字符串中删除子串，我们将使用以下步骤。

*   首先，我们将创建一个名为`output_string`的空字符串来存储输出字符串。
*   然后，我们将使用`split()`方法从需要移除特定子串的位置将字符串分割成子串。为此，我们将调用输入字符串上的`split()`方法，将需要删除的子字符串作为输入参数。执行后，`split()`方法将返回一串子字符串。我们将把这个列表分配给一个变量`str_list`。
*   一旦我们得到了字符串列表，我们将使用 for 循环遍历`str_list`中的子字符串。在迭代过程中，我们将使用[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)操作将当前子字符串添加到`output_string`中。

在执行 for 循环后，我们将在变量`output_string`中获得所需的输出字符串。您可以在下面的代码中观察到这一点。

```py
myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
output_string = ""
str_list = myStr.split(substring)
for element in str_list:
    output_string += element

print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

在输出中，您可以观察到子字符串`python`已经从输入字符串中删除。

## 使用 join()方法在 Python 中移除字符串中的子字符串

多次执行字符串连接需要不必要的存储和时间。因此，我们可以通过使用 `join()` 方法来避免这种情况。

当在分隔符字符串上调用时，`join()`方法将 iterable 对象作为其输入参数。执行后，它返回一个字符串，该字符串由分隔符字符串分隔的 iterable 对象的元素组成。

要使用`join()`方法从 python 中的字符串中删除 substring，我们将使用以下步骤。

*   首先，我们将使用`split()`方法将输入字符串从需要删除特定子字符串的位置分割成子字符串。为此，我们将调用输入字符串上的`split()`方法，将需要删除的子字符串作为输入参数。执行后，`split()`方法将返回一串子字符串。我们将把这个列表分配给一个变量`str_list`。
*   接下来，我们将调用空字符串上的`join()`方法，将`str_list`作为其输入参数。

在执行了 `join()`方法之后，我们将得到所需的字符串输出，如下所示。

```py
myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
str_list = myStr.split(substring)
output_string = "".join(str_list)
print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

在这里，您可以观察到我们已经使用 `join()`方法将由`split()`方法返回的列表转换成了一个字符串。因此，我们避免了重复的字符串连接，就像我们在前面的例子中所做的那样。

## 使用 replace()方法从 Python 中的字符串中移除子字符串

在 python 中，`replace()`方法用于替换字符串中的一个或多个字符。当在一个字符串上调用时，`replace()`方法将两个子字符串作为它的输入参数。执行后，它将第一个参数中的子字符串替换为第二个输入参数中的子字符串。然后它返回修改后的字符串。

为了使用`replace()`方法从字符串中删除子串，我们将调用原始字符串上的`replace()` 方法，将要删除的子串作为第一个输入参数，一个空字符串作为第二个输入参数。

在执行了`replace()`方法之后，我们将得到如下例所示的输出字符串。

```py
myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
output_string = myStr.replace(substring, "")
print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

这里，我们使用`replace()`方法在一个语句中从输入字符串中删除了所需的子字符串。

## 使用正则表达式从 python 化的字符串中删除子串

正则表达式为我们提供了在 Python 中操作字符串的有效方法。在 python 中，我们还可以使用正则表达式从字符串中删除子串。为此，我们可以使用`re.split()` 方法和`re.sub()`方法。

### 使用 re.split()方法在 Python 中移除字符串中的子字符串

方法用于在指定的分隔符处分割文本。`re.split()`方法将分隔符字符串作为第一个输入参数，将文本字符串作为第二个输入参数。执行后，它返回由分隔符分隔的原始字符串列表。

要使用`re.split()`方法从 Python 中的字符串中删除子串，我们将使用以下步骤。

*   首先，我们将创建一个名为`output_string`的空字符串来存储输出字符串。
*   然后，我们将使用`re.split()`方法从需要移除特定子串的位置将字符串分割成子串。为此，我们将执行`re.split()`方法，将需要删除的子字符串作为第一个输入参数，将文本字符串作为第二个输入参数。执行后，`re.split()`方法将返回一串子字符串。我们将把这个列表分配给一个变量`str_list`。
*   一旦我们得到了字符串列表，我们将使用 for 循环遍历`str_list`中的子字符串。在迭代过程中，我们将使用字符串连接操作将当前子字符串添加到`output_string`。

在执行 for 循环后，我们将在变量`output_string`中获得所需的输出字符串。您可以在下面的代码中观察到这一点。

```py
import re

myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
output_string = ""
str_list = re.split(substring, myStr)
for element in str_list:
    output_string += element

print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

您可以观察到使用`re.split()`方法的方法与使用 string `split()`方法的方法几乎相似。但是，这两种方法的执行速度不同。如果输入字符串非常大，那么应该首选`re.split()`方法来拆分输入字符串。

多次执行字符串连接需要不必要的内存和时间。因此，我们可以通过使用 `join()`方法来避免这种情况。

要使用`join()`方法从 python 中的字符串中删除 substring，我们将使用以下步骤。

*   首先，我们将使用`re.split()`方法将输入字符串从需要删除特定子字符串的位置分割成子字符串。为此，我们将执行`re.split()`方法，将需要删除的子字符串作为第一个输入参数，将文本字符串作为第二个输入参数。执行后，`re.split()`方法将返回一串子字符串。我们将把这个列表分配给一个变量`str_list`。

*   接下来，我们将调用空字符串上的`join()`方法，将`str_list`作为其输入参数。

在执行了`join()`方法之后，我们将得到所需的字符串输出，如下所示。

```py
import re

myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
str_list = re.split(substring, myStr)
output_string = "".join(str_list)
print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

在这种方法中，我们仅用两条 python 语句就获得了输出字符串。此外，我们没有做重复的字符串连接，这需要不必要的时间。

### 使用 re.sub()方法从 Python 中的字符串中移除子字符串

在 python 中，`re.sub()`方法用于替换字符串中的一个或多个字符。`re.sub()`方法有三个输入参数。第一个输入参数是需要替换的子字符串。第二个输入参数是替代子串。原始字符串作为第三个输入字符串传递。

执行后，`re.sub()`方法用第二个输入参数的子字符串替换第一个参数中的子字符串。然后它返回修改后的字符串。

为了使用`re.sub()` 方法从字符串中删除子串，我们将执行`re.sub()` 方法，将要删除的子串作为第一个输入参数，一个空字符串作为第二个输入参数，原始字符串作为第三个输入参数。

在执行了`re.sub()`方法之后，我们将得到如下例所示的输出字符串。

```py
import re

myStr = "I am PFB. I provide free python tutorials for you to learn python."
substring = "python"
output_string = re.sub(substring, "", myStr)
print("The input string is:", myStr)
print("The substring is:", substring)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The substring is: python
The output string is: I am PFB. I provide free  tutorials for you to learn .
```

`re.sub()`方法的工作方式类似于`replace()` 方法。但是，它比后者更快，应该是首选。

## 通过索引移除 Python 中字符串的子串

有时，当我们知道子串在字符串中的位置时，我们可能需要从字符串中移除子串。为了在 python 中通过索引从字符串中移除子串，我们将使用字符串切片。

如果我们必须从索引 I 到 j 中移除子串，我们将制作两个字符串片段。第一个片段将从索引 0 到 i-1，第二个片段将从索引 j+1 到最后一个字符。

获得切片后，我们将连接切片以获得输出字符串，如下例所示。

```py
import re

myStr = "I am PFB. I provide free python tutorials for you to learn python."
output_string = myStr[0:5]+myStr[11:]
print("The input string is:", myStr)
print("The output string is:", output_string)
```

输出:

```py
The input string is: I am PFB. I provide free python tutorials for you to learn python.
The output string is: I am  provide free python tutorials for you to learn python.
```

## 结论

在本文中，我们讨论了用 Python 从字符串中删除子串的不同方法。在所有方法中，使用`re.sub()`方法和`replace()`方法的方法具有最好的时间复杂度。因此，我建议你在你的程序中使用这些方法。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。