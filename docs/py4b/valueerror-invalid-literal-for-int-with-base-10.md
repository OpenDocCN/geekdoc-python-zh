# ValueError:基数为 10 的 int()的文本无效

> 原文：<https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10>

Python 值错误:基数为 10 的 int()的无效文字是一个异常，当我们尝试使用 int()方法将字符串文字转换为整数，并且字符串文字包含除数字以外的字符时，可能会发生这种情况。在这篇文章中，我们将试图理解这种异常背后的原因，并将研究在我们的程序中避免它的不同方法。

## Python 中的“ValueError:以 10 为基数的 int()的无效文字”是什么？

ValueError 是 python 中的一个异常，当将类型正确但值不正确的参数传递给方法或函数时，会出现该异常。消息的第一部分，即**“value error”告诉我们，由于不正确的值作为参数传递给了 int()函数，因此出现了异常。**消息的第二部分**“基数为 10 的 int()的无效文字”告诉我们，我们试图将输入转换为整数，但输入中有十进制数字系统中的数字以外的字符。**

## int()函数的工作原理

python 中的 int()函数将一个字符串或一个数字作为第一个参数，并使用一个可选的参数基来表示数字格式。基数有一个默认值 10，用于十进制数，但是我们可以为基数传递一个不同的值，例如 2 表示二进制数，16 表示十六进制数。在本文中，我们将只使用带有第一个参数的 int()函数，base 的默认值将始终为零。这可以从下面的例子中看出。

我们可以将浮点数转换成整数，如下例所示。当我们使用 int()函数将一个浮点数转换成整数时，输出中的数字会去掉小数点后的数字。

```py
 print("Input Floating point number is")
myInput= 11.1
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input Floating point number is
11.1
Output Integer is:
11
```

我们可以将由数字组成的字符串转换为整数，如下例所示。这里的输入只包含数字，因此它将被直接转换成整数。

```py
print("Input String is:")
myInput= "123"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
123
Output Integer is:
123
```

上面两个例子中显示的两种输入类型是 int()函数能够正常工作的唯一输入类型。对于其他类型的输入，当它们作为参数传递给 int()函数时，将生成 ValueError，并显示消息“对于以 10 为基数的 int()无效”。现在，我们将看看可以在 int()函数中为其生成 ValueError 的各种类型的输入。

## 什么时候出现“ValueError:以 10 为基数的 int()的无效文字”？

如上所述，当带有不适当值的输入被传递给 int()函数时，可能会出现基数为 10 的“value error:invalid literal for int()”。这可能发生在下列情况下。

1.Python 值错误:当 int()方法的输入是字母数字而不是数字时，基数为 10 的 int()的文字无效，因此输入无法转换为整数。这可以用下面的例子来理解。

在本例中，我们将一个包含字母数字字符的字符串传递给 int()函数，由于这个原因，会出现 ValueError，并在输出中显示消息“value error:invalid literal for int()with base 10”。

```py
print("Input String is:")
myInput= "123a"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
 Input String is:
123a
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-9-36c8868f7082>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: '123a' 
```

2.Python 值错误:当 int()函数的输入包含空格字符时，基数为 10 的 int()的文字无效，因此输入无法转换为整数。这可以用下面的例子来理解。

在本例中，我们将一个包含空格的字符串传递给 int()函数，由于这个原因，会出现 ValueError，并在输出中显示消息“value error:invalid literal for int()with base 10”。

```py
print("Input String is:")
myInput= "12 3"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
12 3
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-10-d60c59d37000>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: '12 3' 
```

3.当 int()函数的输入包含任何标点符号(如句号“.”)时，就会出现 Python `ValueError: invalid literal for int() with base 10`或者逗号“，”。因此，输入不能转换成整数。这可以用下面的例子来理解。

在这个例子中，我们传递一个包含句点字符的字符串导致 ValueError 发生的 int()函数，并在输出中显示消息“ValueError:对于以 10 为基数的 int()无效的文本”。

```py
print("Input String is:")
myInput= "12.3"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
12.3
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-11-9146055d9086>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: '12.3'
```

## 如何避免“ValueError:以 10 为基数的 int()的无效文字”？

我们可以避免 value error:invalid literal for int()with base 10 exception 使用先发制人的措施来检查传递给 int()函数的输入是否只包含数字。我们可以使用几种方法来检查传递给 int()的输入是否只包含数字，如下所示。

1.我们可以使用正则表达式来检查传递给 int()函数的输入是否只包含数字。如果输入包含数字以外的字符，我们可以提示用户输入不能转换为整数。否则，我们可以正常进行。

在下面给出的 python 代码中，我们定义了一个正则表达式“[^\d]”，它匹配十进制中除数字以外的所有字符。re.search()方法搜索模式，如果找到了模式，则返回一个 match 对象。否则 re.search()方法返回 None。

每当 re.search()返回 None 时，可以实现输入没有除数字之外的字符，因此输入可以被转换成如下的整数。

```py
import re
print("Input String is:")
myInput= "123"
print(myInput)
matched=re.search("[^\d]",myInput)
if matched==None:
    myInt=int(myInput)
    print("Output Integer is:")
    print(myInt)
else:
    print("Input Cannot be converted into Integer.")
```

输出:

```py
Input String is:
123
Output Integer is:
123
```

如果输入包含除数字以外的任何字符，re.search()将包含一个 match 对象，因此输出将显示一条消息，说明输入不能转换为整数。

```py
import re
print("Input String is:")
myInput= "123a"
print(myInput)
matched=re.search("[^\d]",myInput)
if matched==None:
    myInt=int(myInput)
    print("Output Integer is:")
    print(myInt)
else:
    print("Input Cannot be converted into Integer.") 
```

输出:

```py
Input String is:
123a
Input Cannot be converted into Integer.
```

2.我们还可以使用 isdigit()方法来检查输入是否仅由数字组成。isdigit()方法接受一个字符串作为输入，如果作为参数传递给它的输入字符串只包含十进制的数字，则返回 True。否则，它返回 False。在检查输入字符串是否仅由数字组成后，我们可以将输入转换为整数。

在这个例子中，我们使用了 isdigit()方法来检查给定的输入字符串是否只包含数字。由于输入字符串“123”仅由数字组成，isdigit()函数将返回 True，并且使用 int()函数将输入转换为整数，如输出所示。

```py
print("Input String is:")
myInput= "123"
print(myInput)
if myInput.isdigit():
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
else:
    print("Input cannot be converted into integer.") 
```

输出:

```py
Input String is:
123
Output Integer is:
123 
```

如果输入字符串包含除数字以外的任何其他字符，isdigit()函数将返回 False。因此，输入字符串不会被转换成整数。

在本例中，给定的输入是“123a ”,它包含一个字母表，因此 isdigit()函数将返回 False，并且在输出中将显示一条消息，说明输入不能转换为整数，如下所示。

```py
print("Input String is:")
myInput= "123a"
print(myInput)
if myInput.isdigit():
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
else:
    print("Input cannot be converted into integer.") 
```

输出:

```py
Input String is:
123a
Input cannot be converted into integer.
```

3.输入字符串可能包含一个浮点数并带有一个句点字符。在数字之间。要使用 int()函数将这样的输入转换为整数，首先我们将检查输入字符串是否包含浮点数，即在数字之间只有一个句点字符，或者没有使用正则表达式。如果是，我们将首先把输入转换成一个可以传递给 int()函数的浮点数，然后我们将显示输出。否则，它将被通知输入不能被转换成整数。

在这个例子中，“^\d+\.\d$ "表示以一个或多个数字开始的模式，有一个句点符号。以一个或多个数字结尾，这是浮点数的模式。因此，如果输入字符串是浮点数，re.search()方法将不会返回 None，输入将使用 float()函数转换为浮点数，然后转换为整数，如下所示。

```py
import re
print("Input String is:")
myInput= "1234.5"
print(myInput)
matched=re.search("^\d+\.\d+$",myInput)
if matched!=None:
    myFloat=float(myInput)
    myInt=int(myFloat)
    print("Output Integer is:")
    print(myInt)
else:
    print("Input is not a floating point literal.") 
```

输出:

```py
Input String is:
1234.5
Output Integer is:
1234
```

如果输入不是浮点文字，re.search()方法将返回一个 None 对象，并在输出中显示输入不是浮点文字的消息，如下所示。

```py
import re
print("Input String is:")
myInput= "1234a"
print(myInput)
matched=re.search("^\d+\.\d$",myInput)
if matched!=None:
    myFloat=float(myInput)
    myInt=int(myFloat)
    print("Output Integer is:")
    print(myInt)
else:
    print("Input is not a floating point literal.") 
```

输出:

```py
Input String is:
1234a
Input is not a floating point literal. 
```

对于使用正则表达式的两种方法，我们可以在使用 re.match()对象编写命名模式之后，使用 groupdict()方法编写一个单独的程序。groupdict()将返回输入中已命名的已捕获组的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/),因此可用于识别可转换为整数的字符串。

4.我们还可以在 python 中使用异常处理，使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 在错误发生时处理 ValueError。在代码的 try 块中，我们通常会执行代码。每当 ValueError 发生时，它将在 try 块中引发，并由 except 块处理，并向用户显示一条正确的消息。

如果输入只包含数字并且格式正确，输出将如下所示。

```py
print("Input String is:")
myInput= "123"
print(myInput)
try:
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
except ValueError:
    print("Input cannot be converted into integer.") 
```

输出:

```py
Input String is:
123
Output Integer is:
123 
```

如果输入包含除数字以外的字符，如字母或标点符号，将从 int()函数中抛出 ValueError，该函数将被 except 块捕获，并向用户显示一条消息，说明输入不能转换为整数。

```py
print("Input String is:")
myInput= "123a"
print(myInput)
try:
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
except ValueError:
    print("Input cannot be converted into integer.")
```

输出:

```py
Input String is:
123a
Output Integer is:
Input cannot be converted into integer. 
```

## 结论

在本文中，我们看到了为什么“value error:invalid literal for int()with base 10”会出现在 python 中，并理解了其背后的原因和机制。我们还看到，通过首先检查 int()函数的输入是否仅由数字组成，或者不使用不同的方法(如正则表达式和内置函数),可以避免这种错误。