# 在 Python 中删除字符串中的引号

> 原文：<https://www.pythonforbeginners.com/basics/remove-quotes-from-a-string-in-python>

由于各种模块的可用性，Python 是自然语言处理和文本分析中最常用的编程语言之一。我们使用 python 中的字符串来分析文本数据。在 Python 中，单引号或双引号将每个字符串括起来。但是，输入字符串之间可能包含引号。本文讨论了在 Python 中从字符串中移除引号的不同方法。

## 在 Python 中使用 For 循环删除字符串中的引号

在 Python 中，我们使用 for 循环来迭代一个像字符串或列表这样的可迭代对象。要使用 for 循环从字符串中删除引号，我们将使用以下步骤。

*   首先，我们将创建一个名为`quotes`的列表来存储单引号和双引号字符。
*   然后，我们将创建一个名为`newStr`的空字符串来存储输出字符串。
*   现在，我们将使用 for 循环遍历输入字符串中的字符。
*   在迭代过程中，如果我们发现除了单引号或双引号以外的字符，我们将使用[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)操作符将这些字符附加到`newStr`中。为了检查一个字符是单引号还是双引号，我们将使用成员运算符。
*   如果我们在字符串中发现单引号或双引号字符，我们将跳过它。
*   在执行 for 循环之后，我们将在变量`newStr`中获得输出字符串。

您可以在下面的示例中观察整个过程。

```py
input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
quotes = ["'", '"']
newStr = ""
for character in input_string:
    if character not in quotes:
        newStr += character
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

不使用名为`quotes`的列表和成员操作符，我们可以直接比较字符串中出现单引号和双引号的字符。为此，我们将在 if 语句中使用等号运算符来比较字符。其余的过程与上面相同。

```py
input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
newStr = ""
for character in input_string:
    if character == "'" or character == '"':
        continue
    else:
        newStr += character
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

在这里，您可以看到我们没有使用列表和成员操作符。相反，我们使用等号运算符直接比较了 if 语句中的字符。

## 在 Python 中使用 filter()函数和 join()方法移除字符串中的引号

filter()函数用于从 iterable 对象中排除元素。filter()函数的语法如下。

```py
filter(input_function,iterable_object)
```

这里，

*   `iterable_object`是我们需要从中排除元素的 python 对象。
*   `input_function`是一个接受`iterable_object`的一个元素并返回`True`或`False`的函数。

执行后，`filter()`函数返回一个`filter`对象。一个`filter`对象是一个可迭代的对象，它包含了`input_function`返回`True`的`iterable_object`的所有元素。

`join()` 方法用于从给定的 iterable 对象创建一个字符串。当在分隔符字符串上调用时， `join()` 方法将 iterable 对象作为其输入参数。

执行后，它返回一个字符串，其中 iterable 对象的所有元素都由分隔符分隔。
要使用 `join()`方法和`filter()`函数在 Python 中删除字符串中的引号，我们将使用以下步骤。

*   首先，我们将创建一个函数 `isNotQuotes()`，它将一个字符作为它的输入参数。如果字符是单引号或双引号字符，它返回`False`。否则，它返回`True`。
*   之后，我们将使用 `filter()`函数从字符串中排除引号。为此，我们将把`isNotQuotes`函数作为第一个输入参数，把输入字符串作为第二个参数传递给过滤函数。在执行了`filter()`函数之后，我们将得到一个`filter`对象，它包含输入字符串中除引号之外的所有字符。
*   现在，我们将使用 `join()` 方法来获取输出字符串。为此，我们将使用空字符串作为分隔符。我们将调用分隔符上的 `join()`方法，将 filter 对象作为它的输入参数。

在执行了`join()`方法之后，我们将得到没有任何引号的输出字符串。您可以在下面的示例中观察到这一点。

```py
def isNotQuotes(character):
    if character == '"':
        return False
    if character == "'":
        return False
    return True

input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
filter_object=filter(isNotQuotes,input_string)
newStr = "".join(filter_object)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

## 使用列表理解删除 Python 中字符串的引号

[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)用于从现有的可迭代对象创建一个新列表。在 Python 中，我们可以使用 list comprehension 和 `join()`方法来删除字符串中的引号。为此，我们将使用以下步骤。

*   首先，我们将创建一个名为`quotes`的列表来存储单引号和双引号字符。
*   然后，我们将使用 list comprehension 从输入字符串中获取一个字符列表，不包括引号。
*   一旦我们获得了字符列表，我们将在一个空字符串上调用`join()`方法，并将字符列表作为输入参数传递给它。
*   在执行了 `join()`方法之后，我们将得到想要的字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
quotes = ["'", '"']
myList = [character for character in input_string if character not in quotes]
newStr = "".join(myList)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

## 在 Python 中使用 replace()方法移除字符串中的引号

`replace()`方法用于将字符串中的一个字符替换为另一个字符。在字符串上调用时，`replace()`方法将需要替换的元素作为第一个输入参数，新字符作为第二个参数。执行后，它返回修改后的字符串。
要使用 `replace()`方法在 Python 中删除字符串中的引号，我们将使用以下步骤。

*   首先，我们将对输入字符串调用`replace()` 方法。这里，我们将把单引号字符作为第一个输入参数，一个空字符串作为第二个输入参数传递给`replace()` 方法。`replace()` 方法将返回一个字符串`tempStr`作为输出。
*   同样，我们将调用`tempStr`上的`replace()`方法。这一次，我们将把双引号字符作为第一个输入参数，一个空字符串作为第二个输入参数传递给`replace()`方法。

在第二次执行 replace 方法后，我们将得到所需的输出字符串，其中没有引号字符。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
tempStr = input_string.replace("'", "")
newStr = tempStr.replace('"', "")
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

## 在 Python 中使用 re.sub()函数删除字符串中的引号

[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)为我们提供了各种字符串操作的函数。我们可以用 re。`sub()`Python 中移除字符串引号的函数。
`re.sub()`函数的语法如下。

```py
re.sub(old_character, new_character, input_string)
```

这里，

*   `input_string`是我们需要替换或删除的字符串。
*   `old_character`是需要从`input_string`中移除的字符。
*   `new_character`是插入到`input_string`中代替`old_character`的字符。

为了使用`re.sub()`函数在 python 中删除给定字符串的引号，我们将使用单引号和双引号字符作为`old_character`，使用空字符串作为`new_character`。

*   首先，我们将把单引号作为第一个输入参数，一个空字符串作为第二个输入参数，给定的字符串作为第三个输入参数传递给`re.sub()`函数。执行后，`sub()`函数将返回一个字符串。我们将把它命名为`tempStr`。
*   现在，我们将把双引号作为第一个输入参数，一个空字符串作为第二个输入参数，`tempStr`作为第三个输入参数传递给`re.sub()`函数。

执行后，`re.sub()`函数将返回没有引号的字符串。您可以在下面的代码中观察到这一点。

```py
import re

input_string = "Pythonf'orb''eginn'er's"
print("The input string is:", input_string)
tempStr = re.sub("'", "", input_string)
newStr = re.sub('"', "", tempStr)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Pythonf'orb''eginn'er's
The output string is: Pythonforbeginners
```

## 从 Python 字符串中移除第一个和最后一个引号字符

如果我们有一个只在开头和结尾有引号字符的字符串，如在`"'Aditya'"`或`'"Aditya"'`中，我们可以使用下面的方法从字符串中删除引号。

## 使用 ast 模块从字符串中删除第一个和最后一个引号字符

`ast`模块为我们提供了 `literal_eval()`函数来计算以字符串形式编写的表达式。`literal_eval()` 函数将一个字符串作为其输入参数，对其求值，然后返回输出。
当我们将字符串`'"Aditya"'`或`"'Aditya'"` 传递给`literal_eval()`函数时，它将输入字符串视为表达式，将`"Aditya"`或 `'Aditya'`视为相应的值。您可以在下面的示例中观察到这一点。

```py
import ast

input_string = "'Aditya'"
newStr=ast.literal_eval(input_string)
print("The input string is:", input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Aditya'
The output string is: Aditya
```

如果输入字符串在不同于第一个和最后一个位置的位置包含额外的引号，这种方法就不起作用。如果您试图使用`literal_eval()`函数从这样的输入字符串中删除引号，程序将运行到`SyntaxError`，如下例所示。

```py
import ast

input_string = "'Adity'a'"
print("The input string is:", input_string)
newStr=ast.literal_eval(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Adity'a'
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 5, in <module>
    newStr=ast.literal_eval(input_string)
  File "/usr/lib/python3.8/ast.py", line 59, in literal_eval
    node_or_string = parse(node_or_string, mode='eval')
  File "/usr/lib/python3.8/ast.py", line 47, in parse
    return compile(source, filename, mode, flags,
  File "<unknown>", line 1
    'Adity'a'
           ^
SyntaxError: invalid syntax
```

在这种方法中，输入字符串的第一个和最后一个位置都必须包含引号字符。

如果我们在第一个位置有一个引用字符，而不是在最后一个位置，程序将运行到`SyntaxError`。类似地，如果我们在最后一个位置有一个引用字符，而不是在第一个位置，程序将再次运行到一个`SyntaxError`。
你可以在下面的例子中观察到这一点。

```py
import ast

input_string = "'Aditya"
print("The input string is:", input_string)
newStr=ast.literal_eval(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Aditya
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 5, in <module>
    newStr=ast.literal_eval(input_string)
  File "/usr/lib/python3.8/ast.py", line 59, in literal_eval
    node_or_string = parse(node_or_string, mode='eval')
  File "/usr/lib/python3.8/ast.py", line 47, in parse
    return compile(source, filename, mode, flags,
  File "<unknown>", line 1
    'Aditya
          ^
SyntaxError: EOL while scanning string literal 
```

需要记住的另一个条件是，字符串开头和结尾的引号字符应该相同。如果字符串的开头是单引号，结尾是双引号，反之亦然，程序将再次运行到如下所示的`SyntaxError`。

```py
import ast

input_string = "'Aditya\""
print("The input string is:", input_string)
newStr=ast.literal_eval(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Aditya"
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 5, in <module>
    newStr=ast.literal_eval(input_string)
  File "/usr/lib/python3.8/ast.py", line 59, in literal_eval
    node_or_string = parse(node_or_string, mode='eval')
  File "/usr/lib/python3.8/ast.py", line 47, in parse
    return compile(source, filename, mode, flags,
  File "<unknown>", line 1
    'Aditya"
           ^
SyntaxError: EOL while scanning string literal
```

如果输入字符串的开头和结尾不包含引号字符，程序将会遇到一个`ValueError`异常。您可以在下面的示例中观察到这一点。

```py
import ast

input_string = "Aditya"
print("The input string is:", input_string)
newStr = ast.literal_eval(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Aditya
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 5, in <module>
    newStr = ast.literal_eval(input_string)
  File "/usr/lib/python3.8/ast.py", line 99, in literal_eval
    return _convert(node_or_string)
  File "/usr/lib/python3.8/ast.py", line 98, in _convert
    return _convert_signed_num(node)
  File "/usr/lib/python3.8/ast.py", line 75, in _convert_signed_num
    return _convert_num(node)
  File "/usr/lib/python3.8/ast.py", line 66, in _convert_num
    _raise_malformed_node(node)
  File "/usr/lib/python3.8/ast.py", line 63, in _raise_malformed_node
    raise ValueError(f'malformed node or string: {node!r}')
ValueError: malformed node or string: <_ast.Name object at 0x7ffbe7ec60d0>
```

因此，您应该记住，只有当字符串只在开头和结尾包含引号字符时，才可以使用`literal_eval()`函数。

## 使用 eval()函数删除字符串中的第一个和最后一个引号字符

`eval()`功能的工作方式类似于 `literal_eval()`功能。它还将一个字符串表达式作为输入参数，对该表达式求值，并返回结果值。
您可以使用如下所示的`eval()`函数删除字符串的第一个和最后一个引号。

```py
input_string = "'Aditya'"
print("The input string is:", input_string)
newStr = eval(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Aditya'
The output string is: Aditya
```

针对`string_eval()`功能提到的所有条件对于`eval()` 功能都是真实的。因此，你应该记住你不能对每个字符串使用`eval()`函数。

## 使用 json 模块从字符串中删除第一个和最后一个引号字符

双引号中的 [python 字符串](https://www.pythonforbeginners.com/basics/strings-quotes)是有效的 json 对象。因此，我们可以使用 [json 模块](https://www.pythonforbeginners.com/json/parsingjson)从输入字符串的第一个和最后一个位置移除引号。
json 模块中的 `loads()` 函数将一个 json 对象作为其输入参数，并返回一个对应于 JSON 对象的 python 字符串。
因为我们的字符串在第一个和最后一个位置包含额外的引号。它将被视为有效的 json 对象。因此，我们可以将它传递给`loads()`函数，以获得如下所示的输出字符串。

```py
import json
input_string = '"Aditya"'
print("The input string is:", input_string)
newStr = json.loads(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: "Aditya"
The output string is: Aditya
```

这里，您应该记住，带单引号的字符串不是有效的 json 字符串。因此，如果您试图使用 json 模块从给定的字符串中删除单引号，程序将会出错，如下例所示。

```py
import json
input_string = "'Aditya'"
print("The input string is:", input_string)
newStr = json.loads(input_string)
print("The output string is:", newStr)
```

输出:

```py
The input string is: 'Aditya'
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 4, in <module>
    newStr = json.loads(input_string)
  File "/usr/lib/python3.8/json/__init__.py", line 357, in loads
    return _default_decoder.decode(s)
  File "/usr/lib/python3.8/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/usr/lib/python3.8/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

在除了额外的双引号之外的所有其他情况下，如果您试图从字符串中删除引号，使用 json 模块的方法将会出错。因此，您应该记住，您只能在一种情况下使用这种方法。

## 使用 strip()函数删除字符串中的第一个和最后一个引号字符

使用 ast 模块、json 模块或`eval()`函数来删除字符串开头和结尾的引号有很多限制。使用这些方法时，您的程序很可能会遇到异常。除了上述方法，我们可以使用`strip()` 方法从字符串的开头和结尾删除引号。
`strip()`方法在字符串上调用时，将一个字符作为其输入参数。执行后，它从字符串的开头和结尾删除该字符的所有匹配项，并返回一个新字符串。
要删除字符串开头和结尾的引号，我们将使用下面的方法。

*   首先，我们将声明一个字符串`temp1`，并将其初始化为输入字符串。
*   现在，我们将使用 while 循环来删除引号字符。在 while 循环中，我们将使用以下步骤。
*   首先，我们将声明一个名为`temp2`的临时字符串，并将其初始化为`temp1`。
*   现在，我们将调用`temp1`上的`strip()`方法。这里，我们将把一个单引号字符作为输入参数传递给`strip()`方法。我们将在`temp3`中存储`strip()` 方法的返回值。
*   同样，我们将调用`temp3`上的`strip()`方法。这一次，我们将把双引号作为输入参数传递给`strip()` 方法。我们将输出存储在`temp4`中。
*   现在，我们将检查`temp4`是否等于`temp2`。如果是，则所有的引号都已从字符串中删除，因为在当前迭代中字符串没有发生变化。因此，我们将使用 [break 语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)退出 while 循环。
*   如果`temp2`不等于`temp4`，字符串的开头和结尾仍然包含引号。因此，我们需要另一个迭代。为此，我们将把`temp4`分配给`temp1`。

在执行 while 循环之后，我们将获得所需的字符串，并从其开始和结束处删除引号。您可以在下面的代码中观察到这一点。

```py
input_string = "'Pythonforbeginners'"
print("The input string is:", input_string)
temp1 = input_string
while True:
    temp2 = temp1
    tem3 = temp2.strip("'")
    temp4 = tem3.strip('"')
    if temp4 == temp2:
        newStr = temp2
        print("The output string is:", newStr)
        break
    else:
        temp1 = temp4
```

输出:

```py
The input string is: 'Pythonforbeginners'
The output string is: Pythonforbeginners
```

在`literal_eval()`方法和 `eval()`函数失败的情况下，这种方法可以成功地删除引号。因此，您可以自由使用这种方法。例如，看看下面的例子。

```py
input_string = "'''''''Pythonforbeginners'\""
print("The input string is:", input_string)
temp1 = input_string
while True:
    temp2 = temp1
    tem3 = temp2.strip("'")
    temp4 = tem3.strip('"')
    if temp4 == temp2:
        newStr = temp2
        print("The output string is:", newStr)
        break
    else:
        temp1 = temp4
```

输出:

```py
The input string is: '''''''Pythonforbeginners'"
The output string is: Pythonforbeginners
```

在上面的例子中，您可以观察到我们使用了一个输入字符串，它的左边包含七个单引号，右边包含一个带双引号的单引号。即使在这种不对称的情况下，程序也能正确工作，不会遇到任何错误。

## 结论

在本文中，我们讨论了在 Python 中从字符串中移除引号的各种方法。在所有这些方法中，我建议您使用`replace()`方法和 `re.sub()` 函数。使用这些函数的方法是最有效的。
我希望你喜欢阅读这篇文章。想要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中[字典理解的文章。你可能也会喜欢这篇关于机器学习](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)中[回归的文章。](https://codinginfinite.com/regression-in-machine-learning-with-examples/)