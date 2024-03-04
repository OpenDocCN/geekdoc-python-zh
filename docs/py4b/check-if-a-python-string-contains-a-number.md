# 检查 Python 字符串是否包含数字

> 原文：<https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number>

在 python 编程语言中，字符串用于处理文本数据。有时，在我们必须验证用户输入或数据的情况下，我们需要检查字符串是否包含数字。在本 python 教程中，我们将讨论检查给定字符串是否包含数字的不同方法。

## 使用 order()函数检查 Python 字符串是否包含数字

在 ASCII 编码中，数字用于表示字符。每个字符都被分配了一个 0 到 127 之间的特定数字。我们可以使用`ord()`函数在 python 中找到任意数字的 ASCII 值。

order()函数将一个字符作为其输入参数，并返回其 ASCII 值。您可以在下面的 python 代码中观察到这一点。

```py
zero = "0"
nine = "9"
print("The ASCII value of the character \"0\" is:", ord(zero))
print("The ASCII value of the character \"9\" is:", ord(nine))
```

输出:

```py
The ASCII value of the character "0" is: 48
The ASCII value of the character "9" is: 57
```

正如您在上面的 python 程序中看到的，字符“0”的 ASCII 值是 48。此外，字符“9”的 ASCII 值是 57。因此，任何数字字符的 [ASCII 值](https://www.pythonforbeginners.com/basics/ascii-value-in-python)都在 48 到 57 之间。

在 python 中，我们可以使用数字字符的 ASCII 值来检查字符串是否包含数字。为此，我们将字符串视为一个字符序列。

之后，我们将遵循下面提到的步骤。

*   我们将使用一个 [for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)和一个`flag`变量遍历原始字符串对象的字符。最初，我们将把变量`flag`初始化为布尔值`False`。
*   之后，我们将遍历给定输入字符串的字符。迭代时，我们将把每个字符转换成它的 ASCII 数值。
*   之后，我们将检查数值是否在 48 和 57 之间。如果是，字符代表一个数字，因此我们可以说字符串包含一个整数值。
*   一旦我们找到一个字符，它的 ASCII 值在 48 到 57 之间，我们将把布尔值`True`赋给`flag`变量，表明该字符串包含数字字符。在这里，我们已经发现字符串包含一个数字。因此，我们将使用[中断语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)退出 for 循环。
*   如果我们在执行 for 循环时没有找到代表一个数字的字符，`flag`变量将包含值`False`。

在 for 循环执行之后，我们将检查`flag`变量是否包含值`True`。如果是，我们将打印该字符串包含一个数字。否则，我们将打印字符串不包含任何数字。您可以在下面的代码中观察到这一点。

```py
myStr = "I am 23 but it feels like I am 15."
flag = False
for character in myStr:
    ascii_val = ord(character)
    if 48 <= ascii_val <= 57:
        flag = True
        break
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

如果您以一个空字符串作为输入来运行上例中给出的代码，程序将按预期运行。因此，在执行 for 循环之前，将变量`flag`初始化为`False`非常重要。否则，程序将无法正常工作。

## 使用 isnumeric()方法检查 Python 字符串是否为数字

Python 为我们提供了不同的[字符串方法](https://www.pythonforbeginners.com/basics/strings-built-in-methods)，用它们我们可以检查一个字符串是否包含数字。当在字符串上调用`isnumeric()`方法时，如果字符串仅由数字字符组成，则返回`True`。如果字符串包含非数字字符，则返回`False`。

我们可以使用如下例所示的`isnumeric()`方法检查一个字符串是否只包含数字字符。

```py
myStr1 = "123"
isNumber = myStr1.isnumeric()
print("{} is a number? {}".format(myStr1, isNumber))
myStr2 = "PFB"
isNumber = myStr2.isnumeric()
print("{} is a number? {}".format(myStr2, isNumber)) 
```

输出:

```py
123 is a number? True
PFB is a number? False
```

如果我们得到一个包含字母数字字符的字符串，您也可以使用`isnumeric()` 方法检查该字符串是否包含数字。为此，我们将使用 For 循环遍历原始字符串对象的字符。

*   迭代时，我们将对每个字符调用`isnumeric()`方法。如果字符代表一个数字， `isnumeric()`方法将返回`True`。因此，我们可以说字符串包含一个数字。
*   我们将把由`isnumeric()`方法返回的布尔值赋给`flag`变量。
*   如果`flag`变量的值为`True`，则表明该字符串包含数字字符。一旦我们发现字符串包含一个数字，我们将使用 [break 语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)退出 for 循环。
*   如果我们在执行 for 循环时没有找到代表一个数字的字符，`flag`变量将包含值`False`。

如果在 for 循环执行之后,`flag`变量包含值`False`,我们就说这个字符串包含数字。否则，我们会说字符串不包含数字。您可以在下面的代码中观察到这一点。

```py
myStr = "I am 23 but it feels like I am 15."
flag = False
for character in myStr:
    flag = character.isnumeric()
    if flag:
        break
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

## 使用 isdigit()方法验证 Python 字符串是否包含数字

我们可以使用`isdigit()`方法来检查一个字符串是否是数字，而不是上一节中使用的`isnumeric()`。在字符串上调用`isdigit()`方法时，如果字符串中的所有字符都是十进制数字，则返回`True`。否则，它返回`False`。您可以在下面的示例中观察到这一点。

```py
myStr1 = "1"
isDigit = myStr1.isdigit()
print("{} is a digit? {}".format(myStr1, isDigit))
myStr2 = "A"
isDigit = myStr2.isnumeric()
print("{} is a digit? {}".format(myStr2, isDigit))
```

输出:

```py
1 is a digit? True
A is a digit? False
```

我们还可以使用 `isdigit()`方法检查一个字符串是否包含数字。为此，我们将使用 For 循环遍历原始字符串对象的字符。

*   迭代时，我们将对每个字符调用`isdigit()`方法。如果字符代表一个数字，`isdigit()`方法将返回`True`。因此，我们可以说字符串包含一个数字。
*   我们将把由`isdigit()`方法返回的布尔值赋给`flag`变量。如果`flag`变量的值为`True`，则表明该字符串包含数字字符。
*   一旦我们发现字符串包含一个数字，我们将使用 break 语句退出 for 循环。
*   如果我们在执行 for 循环时没有找到代表一个数字的字符，`flag`变量将包含值`False`。

如果在 for 循环执行之后,`flag`变量包含值`False`,我们就说这个字符串包含数字。否则，我们会说字符串不包含数字。

您可以在下面的代码中观察到这一点。

```py
myStr = "I am 23 but it feels like I am 15."
flag = False
for character in myStr:
    flag = character.isdigit()
    if flag:
        break
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

## 使用 map()和 any()函数检查 Python 字符串是否包含数字

在前面的章节中，我们使用了各种方法来检查 python 字符串是否包含数字。为此，我们在每个示例中使用 For 循环遍历了输入字符串。然而，我们也可以在不显式迭代字符串的情况下检查字符串是否包含数字。为此，我们将使用`map()`函数。

`map()`函数将一个函数作为它的第一个输入参数，比如说`func`，将一个可迭代对象比如说`iter`作为它的第二个输入参数。执行时，它执行第一个输入参数中给出的函数`func`，并将`iter`的元素作为`func`的输入参数。执行后，它返回一个 map 对象，该对象包含当它以`iter`的元素作为输入参数执行时由`func`返回的值。

您可以使用`list()`构造函数将地图对象转换成列表。

例如，看看下面的源代码。

```py
from math import sqrt

numbers = [1, 4, 9, 16, 25]
print("The numbers are:", numbers)
square_roots = list(map(sqrt, numbers))
print("The square roots are:", square_roots)
```

输出:

```py
The numbers are: [1, 4, 9, 16, 25]
The square roots are: [1.0, 2.0, 3.0, 4.0, 5.0]
```

在上面的代码中，我们将`sqrt()`函数作为第一个输入参数，将一个包含正数的列表作为`map()`函数的第二个参数。执行后，我们获得了一个包含输入列表元素平方根的列表。

为了使用`map()`函数检查 python 字符串是否包含数字，我们将创建一个函数`myFunc`。函数`myFunc`应该将一个字符作为它的输入参数。执行后，如果字符是数字，它应该返回`True`。否则，它应该返回`False`。

在`myFunc`里面，我们可以用`isdigit()`的方法来检查输入的字符是否是数字。

我们将把`myFunc()`函数作为第一个参数，把输入字符串作为第二个输入参数传递给`map()` 方法。执行后，`map()`函数将返回一个地图对象。

当我们将 map 对象转换成一个列表时，我们会得到一个布尔值列表`True`和`False`。列表中值的总数将等于输入字符串中的字符数。此外，每个布尔值对应于字符串中相同位置的一个字符。换句话说，列表的第一个元素对应于输入字符串中的第一个字符，列表的第二个元素对应于输入字符串中的第二个字符，依此类推。

如果字符串中有任何数字字符，输出列表中对应的元素将是`True`。因此，如果从 map 对象获得的列表在其至少一个元素中包含值`True`，我们将得出输入字符串包含数字的结论。

为了确定输出列表是否包含值`True`作为其元素，我们将使用`any()`函数。

`any()`函数将列表作为其输入参数，如果至少有一个值是`True`，则返回`True`。否则返回`False`。

因此，如果`any()`函数在执行后返回`True`，我们将说输入字符串包含一个数字。否则，我们将断定输入字符串不包含数字字符。

您可以在下面的代码中观察到这一点。

```py
def myFunc(character):
    return character.isdigit()

myStr = "I am 23 but it feels like I am 15."
flag = any(list(map(myFunc, myStr)))
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

这种方法也适用于较大的字符串，因为我们不需要逐个字符地遍历输入字符串。

除了定义 myFunc，还可以将一个 [lambda 函数](https://www.pythonforbeginners.com/basics/lambda-function-in-python)传递给 map()函数。这将使你的代码更加简洁。

```py
 myStr = "I am 23 but it feels like I am 15."
flag = any(list(map(lambda character: character.isdigit(), myStr)))
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

除了使用`isdigit()`方法，还可以使用 `isnumeric()`方法以及`map()`函数和`any()` 函数来检查 python 字符串是否包含数字，如下例所示。

```py
def myFunc(character):
    return character.isnumeric()

myStr = "I am 23 but it feels like I am 15."
flag = any(list(map(myFunc, myStr)))
print("The string is:")
print(myStr)
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

前面讨论的所有方法都使用标准的字符串方法和字符串值。然而，python 也为我们提供了正则表达式模块‘re’来处理文本数据和字符串。

现在让我们讨论如何使用正则表达式检查 python 字符串是否包含数字。

## 使用正则表达式检查 Python 字符串是否包含数字

[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)用于在 python 中操作文本数据。我们还可以使用正则表达式来检查 python 字符串是否包含数字。为此，我们可以使用`match()`方法和`search()`方法。让我们逐一讨论这两种方法。

### 使用 match()方法验证 Python 字符串是否包含数字

`match()`方法用于查找特定模式在字符串中第一次出现的位置。它将正则表达式模式作为第一个参数，将输入字符串作为第二个参数。如果存在与正则表达式模式匹配的字符串字符序列，它将返回一个 match 对象。否则，它返回 None。

要使用`match()`方法检查字符串是否包含数字，我们将遵循以下步骤。

*   首先，我们将定义数字的正则表达式模式。因为数字是一个或多个数字字符的序列，所以数字的正则表达式模式是`“\d+”`。
*   定义模式后，我们将把模式和输入字符串传递给`match()`方法。
*   如果`match()`方法返回`None`，我们就说这个字符串不包含任何数字。否则，我们会说字符串包含数字。

您可以在下面的示例中观察到这一点。

```py
import re

myStr = "23 but it feels like I am 15."
match_object = re.match(r"\d+", myStr)
print("The string is:")
print(myStr)
flag = False
if match_object:
    flag = True
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
23 but it feels like I am 15.
String contains numbers?: True
```

只有当数字出现在字符串的开头时, `match()` 方法才有效。在其他情况下，它无法检查给定的字符串是否包含数字。您可以在下面的示例中观察到这一点。

```py
import re

myStr = "I am 23 but it feels like I am 15."
match_object = re.match(r"\d+", myStr)
print("The string is:")
print(myStr)
flag = False
if match_object:
    flag = True
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: False
```

在上面的例子中，您可以看到字符串包含数字。然而，程序断定字符串中没有任何数字。为了避免在检查字符串中的数字时出错，我们可以使用`search()`方法来代替`match()`方法。

### 使用 search()方法检查 Python 字符串是否包含数字

`search()`方法用于查找字符串中特定模式的位置。它将正则表达式模式作为第一个参数，将输入字符串作为第二个参数。如果存在与正则表达式模式匹配的字符串字符序列，它将返回一个 match 对象，该对象包含该模式第一次出现的位置。否则返回`None`。

为了使用`search()`方法检查一个字符串是否包含一个数字，我们将遵循使用`match()`方法的所有步骤。唯一的区别是我们将使用`search()`方法而不是`match()`方法。您可以在下面的示例中观察到这一点。

```py
import re

myStr = "I am 23 but it feels like I am 15."
match_object = re.search(r"\d+", myStr)
print("The string is:")
print(myStr)
flag = False
if match_object:
    flag = True
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

`search()`方法在整个字符串中搜索。执行后，它返回一个匹配对象，该对象包含输入模式的第一个匹配项。因此，即使字符串中只有一个数字，`search()`方法也会找到它。然而，这并不适用于`match()`方法。

### 使用 findall()方法检查 Python 字符串是否包含数字

`findall()`方法在语义上类似于 `search()`方法。不同之处在于`findall()`方法返回模式的所有出现，而不是模式的第一次出现。当输入字符串中不存在模式时，`findall()`方法返回`None`。

您可以使用如下所示的`findall()`方法检查 python 字符串是否包含数字。

```py
import re

myStr = "I am 23 but it feels like I am 15."
match_object = re.findall(r"\d+", myStr)
print("The string is:")
print(myStr)
flag = False
if match_object:
    flag = True
print("String contains numbers?:", flag)
```

输出:

```py
The string is:
I am 23 but it feels like I am 15.
String contains numbers?: True
```

## 结论

在本文中，我们讨论了检查 python 字符串是否包含数字的不同方法。在所有这些方法中，使用正则表达式的方法是最有效的。如果我们讨论编写更多 pythonic 代码，您可以使用`map()`函数和`any()`函数来检查字符串是否包含数字。与字符串方法相比，正则表达式是更好的选择。因此，为了获得最佳的执行时间，您可以使用`re.search()`方法来检查一个字符串是否包含数字。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！