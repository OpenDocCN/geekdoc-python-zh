# Python 中的无效语法:语法错误的常见原因

> 原文：<https://realpython.com/invalid-syntax-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**识别无效 Python 语法**](/courses/identify-invalid-syntax/)

Python 以其简单的语法而闻名。然而，当你第一次学习 Python，或者当你对另一种编程语言有扎实的基础时，你可能会遇到一些 Python 不允许的事情。如果你曾经在试图运行你的 Python 代码时收到过 **`SyntaxError`** ，那么这个指南可以帮助你。在本教程中，您将看到 Python 中无效语法的常见例子，并学习如何解决这个问题。

**本教程结束时，你将能够:**

*   在 Python 中识别无效语法
*   **`SyntaxError`** 溯流而上
*   **解决**无效语法或完全阻止它

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 中的无效语法

当您运行 Python 代码时，解释器将首先解析它，将其转换为 Python 字节码，然后执行它。解释器将在程序执行的第一阶段发现 Python 中的任何无效语法，也称为**解析阶段**。如果解释器不能成功解析您的 Python 代码，那么这意味着您在代码中的某个地方使用了无效的语法。解释器会试图告诉你错误发生在哪里。

当你第一次学习 Python 时，获得一个`SyntaxError`可能会令人沮丧。Python 将试图帮助你确定无效语法在你代码中的位置，但是它提供的[回溯](https://realpython.com/python-traceback/)可能会有点混乱。有时候，它指向的代码完全没问题。

**注意:**如果你的代码在语法上**是正确的**，那么你可能会得到其他不是`SyntaxError`的异常。要了解更多关于 Python 的其他异常以及如何处理它们，请查看 [Python 异常:简介](https://realpython.com/python-exceptions/)。

在 Python 中不能像其他异常一样处理无效语法。即使您试图用无效语法将`try`和`except`块包装在代码周围，您仍然会看到解释器抛出一个`SyntaxError`。

[*Remove ads*](/account/join/)

## `SyntaxError`异常和追溯

当解释器在 Python 代码中遇到无效语法时，它会引发一个 **`SyntaxError`** 异常，并提供一些有用信息的回溯来帮助你[调试](https://realpython.com/python-debug-idle/)错误。以下是 Python 中包含无效语法的一些代码:

```py
 1# theofficefacts.py
 2ages = {
 3    'pam': 24,
 4    'jim': 24 5    'michael': 43
 6}
 7print(f'Michael is {ages["michael"]} years old.')
```

您可以在第 4 行的[字典](https://realpython.com/python-dicts/)中看到无效的语法。第二个条目`'jim'`缺少一个逗号。如果您试图按原样运行这段代码，那么您会得到以下回溯:

```py
$ python theofficefacts.py
File "theofficefacts.py", line 5
 'michael': 43
 ^
SyntaxError: invalid syntax
```

注意，回溯消息在第 5 行找到错误，而不是第 4 行。Python 解释器试图指出无效语法的位置。然而，它只能真正指向它第一次注意到问题的地方。当您得到一个`SyntaxError`回溯并且回溯所指向的代码看起来没问题时，那么您将想要开始回溯代码，直到您可以确定哪里出了问题。

在上面的例子中，根据逗号后面的内容，省略逗号没有问题。例如，第 5 行中的`'michael'`后面缺少一个逗号没有问题。但是一旦解释器遇到没有意义的东西，它只能给你指出它首先发现的它无法理解的东西。

**注意:**本教程假设你知道 Python 的**回溯**的基础知识。要了解更多关于 Python 回溯以及如何阅读它们，请查看[了解 Python 回溯](https://realpython.com/python-traceback/)和[充分利用 Python 回溯](https://realpython.com/courses/python-traceback/)。

有一些`SyntaxError`回溯的元素可以帮助您确定无效语法在代码中的位置:

*   **遇到无效语法的文件名**
*   **遇到问题的代码的行号**和复制行
*   **一个脱字符号(`^` )** 在复制代码下面的一行，显示代码中有问题的地方
*   **异常类型`SyntaxError`之后的错误消息**，它可以提供帮助您确定问题的信息

在上面的例子中，给出的文件名是`theofficefacts.py`，行号是 5，插入符号指向字典键`michael`的右引号。`SyntaxError`回溯可能不会指出真正的问题，但它会指出解释器无法理解语法的第一个地方。

您可能会看到 Python 引发的另外两个异常。这些等同于`SyntaxError`，但名称不同:

1.  `IndentationError`
2.  `TabError`

这些异常都继承自`SyntaxError`类，但是它们是缩进的特例。当代码的缩进级别不匹配时，会引发一个`IndentationError`。当您的代码在同一个文件中同时使用制表符和空格时，会引发一个`TabError`。在后面的小节中，您将仔细研究这些异常。

## 常见语法问题

当您第一次遇到一个`SyntaxError`时，了解为什么会出现问题以及您可以做些什么来修复 Python 代码中的无效语法是很有帮助的。在下面的小节中，您将看到一些更常见的引发`SyntaxError`的原因，以及如何修复它们。

### 误用赋值运算符(`=` )

在 Python 中有几种情况下你不能给对象赋值。一些例子是分配给文字和函数调用。在下面的代码块中，您可以看到几个尝试这样做的例子以及由此产生的`SyntaxError`回溯:

>>>

```py
>>> len('hello') = 5
  File "<stdin>", line 1
SyntaxError: can't assign to function call

>>> 'foo' = 1
  File "<stdin>", line 1
SyntaxError: can't assign to literal

>>> 1 = 'foo'
  File "<stdin>", line 1
SyntaxError: can't assign to literal
```

第一个例子试图将值`5`分配给`len()`调用。在这种情况下,`SyntaxError`的信息非常有用。它告诉你不能给函数调用赋值。

第二个和第三个例子试图将一个[字符串](https://realpython.com/python-strings/)和一个整数赋给文字。同样的规则也适用于其他文字值。回溯消息再次表明，当您试图为文本赋值时会出现问题。

**注意:**上面的例子缺少了重复的代码行和在回溯中指向问题的插入符号(`^`)。当你在 REPL 中试图从一个文件中执行这段代码时，你看到的异常和回溯会有所不同。如果这些代码在一个文件中，那么您将得到重复的代码行和指向问题的插入符号，就像您在本教程的其他例子中看到的那样。

很可能你的意图不是给一个文字或者一个函数调用赋值。例如，如果您不小心遗漏了额外的等号(`=`)，就会发生这种情况，这会将赋值转换为比较。如下图所示，比较是有效的:

>>>

```py
>>> len('hello') == 5
True
```

大多数时候，当 Python 告诉你你正在给不能赋值的东西赋值时，你可能首先要检查一下，确保这个语句不应该是一个[布尔表达式](https://realpython.com/python-boolean/)。当您试图给一个 [Python 关键字](https://realpython.com/python-keywords/)赋值时，也可能会遇到这个问题，这将在下一节中介绍。

[*Remove ads*](/account/join/)

### 拼错、遗漏或误用 Python 关键字

Python 关键字是一组**受保护的单词**，在 Python 中有特殊的含义。这些词不能在代码中用作标识符、[变量](https://realpython.com/python-variables/)或函数名。它们是语言的一部分，只能在 Python 允许的上下文中使用。

有三种常见的错误使用关键词的方式:

1.  拼错关键字
2.  **缺少**一个关键字
3.  **误用**关键字

如果你**在你的 Python 代码中拼错了**一个关键词，那么你会得到一个`SyntaxError`。例如，如果您拼错了关键字`for`，会发生什么情况:

>>>

```py
>>> fro i in range(10):
  File "<stdin>", line 1
    fro i in range(10):
        ^
SyntaxError: invalid syntax
```

信息显示为`SyntaxError: invalid syntax`，但这并没有多大帮助。回溯指向 Python 可以检测到出错的第一个地方。要修复此类错误，请确保所有 Python 关键字拼写正确。

关键词的另一个常见问题是当你**完全错过**它们时:

>>>

```py
>>> for i range(10):
  File "<stdin>", line 1
    for i range(10):
              ^
SyntaxError: invalid syntax
```

同样，异常消息并不那么有用，但是回溯确实试图为您指出正确的方向。如果您从插入符号向后移动，那么您可以看到 [`for`循环](https://realpython.com/python-for-loop/)语法中缺少了`in`关键字。

你也可以**误用**一个受保护的 Python 关键字。记住，关键字只允许在特定的情况下使用。如果使用不当，Python 代码中就会出现无效语法。一个常见的例子是在循环之外使用 [`continue`或`break`](https://realpython.com/python-for-loop/#the-break-and-continue-statements) 。在开发过程中，当您正在实现一些东西，并且碰巧将逻辑移到循环之外时，这很容易发生:

>>>

```py
>>> names = ['pam', 'jim', 'michael']
>>> if 'jim' in names:
...     print('jim found')
...     break ...
  File "<stdin>", line 3
SyntaxError: 'break' outside loop

>>> if 'jim' in names:
...     print('jim found')
...     continue ...
  File "<stdin>", line 3
SyntaxError: 'continue' not properly in loop
```

在这里，Python 很好地告诉了你到底哪里出了问题。消息`"'break' outside loop"`和`"'continue' not properly in loop"`帮助你弄清楚该做什么。如果这段代码在一个文件中，那么 Python 也会有一个插入符号指向被误用的关键字。

另一个例子是，如果你试图将一个 Python 关键字赋给一个变量，或者使用一个关键字来定义一个函数:

>>>

```py
>>> pass = True
  File "<stdin>", line 1
    pass = True
         ^
SyntaxError: invalid syntax
  >>> def pass():
  File "<stdin>", line 1
    def pass():
           ^
SyntaxError: invalid syntax
```

当你试图给 [`pass`](https://realpython.com/lessons/pass-statement/) 赋值时，或者当你试图定义一个名为`pass`的新函数时，你会得到一个`SyntaxError`并再次看到`"invalid syntax"`消息。

在 Python 代码中解决这种类型的无效语法可能有点困难，因为代码从外部看起来很好。如果您的代码看起来不错，但是您仍然得到了一个`SyntaxError`，那么您可以考虑根据您正在使用的 Python 版本的关键字列表来检查您想要使用的变量名或函数名。

受保护的关键字列表在 Python 的每个新版本中都有所变化。例如，在 Python 3.6 中，您可以使用`await`作为变量名或函数名，但是从 Python 3.7 开始，这个词已经被添加到关键字列表中。现在，如果你试图使用`await`作为变量或函数名，如果你的代码是 Python 3.7 或更高版本的，这将导致一个`SyntaxError`。

另一个例子是 [`print`](https://realpython.com/python-print/) ，这在 Python 2 和 Python 3 中有所不同:

| 版本 | `print`类型 | 接受一个值 |
| --- | --- | --- |
| [Python 2](https://realpython.com/python-print/#print-was-a-statement-in-python-2) | 关键字 | 不 |
| [Python 3](https://realpython.com/python-print/#print-is-a-function-in-python-3) | 内置函数 | 是 |

`print`在 Python 2 中是一个关键字，所以不能给它赋值。然而，在 Python 3 中，它是一个可以赋值的内置函数。

您可以运行以下代码来查看正在运行的任何 Python 版本中的关键字列表:

```py
import keyword
print(keyword.kwlist)
```

`keyword`也提供了有用的`keyword.iskeyword()`。如果你只是需要一种快速的方法来检查`pass`变量，那么你可以使用下面的一行程序:

>>>

```py
>>> import keyword; keyword.iskeyword('pass')
True
```

这段代码会很快告诉你，你试图使用的标识符是否是一个关键字。

[*Remove ads*](/account/join/)

### 缺少圆括号、方括号和引号

通常，Python 代码中无效语法的原因是右括号、中括号或引号丢失或不匹配。在很长的嵌套括号行或更长的多行代码块中，很难发现这些问题。借助 Python 的回溯功能，您可以发现不匹配或缺失的引号:

>>>

```py
>>> message = 'don't'
  File "<stdin>", line 1
    message = 'don't'
                   ^
SyntaxError: invalid syntax
```

在这里，回溯指向在结束单引号后有一个`t'`的无效代码。要解决这个问题，您可以进行两种更改之一:

1.  **用反斜杠(`'don\'t'`)转义**单引号
2.  **将整个字符串用双引号(`"don't"`)括起来**

另一个常见的错误是忘记关闭字符串。对于双引号和单引号字符串，情况和回溯是相同的:

>>>

```py
>>> message = "This is an unclosed string
  File "<stdin>", line 1
    message = "This is an unclosed string
                                        ^
SyntaxError: EOL while scanning string literal
```

这一次，回溯中的插入符号直接指向问题代码。`SyntaxError`消息`"EOL while scanning string literal"`更加具体，有助于确定问题。这意味着 Python 解释器在一个打开的字符串关闭之前到达了行尾(EOL)。要解决这个问题，请用与您用来开始字符串的引号相匹配的引号来结束字符串。在这种情况下，这将是一个双引号(`"`)。

在 Python 中， [f 字符串](https://realpython.com/python-f-strings/)内的语句中缺少引号也会导致无效语法:

```py
 1# theofficefacts.py
 2ages = {
 3    'pam': 24,
 4    'jim': 24,
 5    'michael': 43
 6}
 7print(f'Michael is {ages["michael]} years old.')
```

这里，对打印的 f 字符串中的`ages`字典的引用缺少了键引用的右双引号。产生的回溯如下:

```py
$ python theofficefacts.py
 File "theofficefacts.py", line 7
 print(f'Michael is {ages["michael]} years old.')
 ^
SyntaxError: f-string: unterminated string
```

Python 会识别问题，并告诉您它存在于 f 字符串中。消息`"unterminated string"`也指出了问题所在。在这种情况下，插入符号仅指向 f 字符串的开头。

这可能不像脱字符号指向 f 字符串的问题区域那样有用，但是它确实缩小了您需要查看的范围。在 f-string 里面有一个未结束的字符串。你只需要找到在哪里。要解决此问题，请确保所有内部 f 字符串引号和括号都存在。

对于缺少圆括号和方括号的情况也是如此。例如，如果你从一个[列表](https://realpython.com/python-lists-tuples/)中漏掉了右方括号，那么 Python 会发现并指出来。然而，这也有一些变化。第一个是把右括号从列表中去掉:

```py
# missing.py
def foo():
 return [1, 2, 3 
print(foo())
```

当你运行这段代码时，你会被告知对 [`print()`](https://realpython.com/python-print/) 的调用有问题:

```py
$ python missing.py
 File "missing.py", line 5
 print(foo())
 ^
SyntaxError: invalid syntax
```

这里发生的是 Python 认为列表包含三个元素:`1`、`2`和`3 print(foo())`。Python 使用[空格](https://realpython.com/lessons/whitespace-expressions-and-statements/)对事物进行逻辑分组，因为没有逗号或括号将`3`和`print(foo())`分开，Python 将它们聚集在一起作为列表的第三个元素。

另一种变化是在列表中的最后一个元素后添加一个尾随逗号，同时仍保留右方括号:

```py
# missing.py
def foo():
 return [1, 2, 3, 
print(foo())
```

现在你得到了一个不同的追溯:

```py
$ python missing.py
 File "missing.py", line 6

 ^
SyntaxError: unexpected EOF while parsing
```

在前面的例子中，`3`和`print(foo())`被合并为一个元素，但是在这里您可以看到一个逗号将两者分开。现在，对`print(foo())`的调用被添加为列表的第四个元素，Python 到达了文件的末尾，没有右括号。回溯告诉您 Python 到达了文件的末尾(EOF ),但是它期望的是别的东西。

在这个例子中，Python 需要一个右括号(`]`)，但是重复的行和插入符号没有多大帮助。Python 很难识别缺失的圆括号和方括号。有时候，你唯一能做的就是从插入符号开始向后移动，直到你能识别出什么是丢失的或错误的。

[*Remove ads*](/account/join/)

### 弄错字典语法

你之前看到过如果你去掉字典元素中的逗号，你会得到一个`SyntaxError`。Python 字典的另一种无效语法是使用等号(`=`)来分隔键和值，而不是冒号:

>>>

```py
>>> ages = {'pam'=24}
  File "<stdin>", line 1
    ages = {'pam'=24}
                 ^
SyntaxError: invalid syntax
```

同样，这个错误消息也不是很有帮助。然而，重复的行和插入符号非常有用！他们正指向问题人物。

如果您将 Python 语法与其他编程语言的语法混淆，这种类型的问题是常见的。如果您将定义字典的行为与`dict()`调用混淆，您也会看到这一点。要解决这个问题，您可以用冒号替换等号。您也可以切换到使用`dict()`:

>>>

```py
>>> ages = dict(pam=24)
>>> ages
{'pam': 24}
```

如果语法更有用，您可以使用`dict()`来定义字典。

### 使用错误的缩进

`SyntaxError`有两个子类专门处理缩进问题:

1.  `IndentationError`
2.  `TabError`

当其他编程语言使用花括号来表示代码块时，Python 使用[空格](https://realpython.com/lessons/whitespace-expressions-and-statements/)。这意味着 Python 希望代码中的空白行为是可预测的。如果代码块中有一行的空格数错误，它将引发 **`IndentationError`** :

```py
 1# indentation.py
 2def foo():
 3    for i in range(10):
 4        print(i)
 5  print('done') 6
 7foo()
```

这可能很难看到，但是第 5 行只缩进了 2 个空格。它应该与`for`循环语句一致，超出 4 个空格。幸运的是，Python 可以很容易地发现这一点，并会很快告诉您问题是什么。

不过，这里也有一点含糊不清。`print('done')`线是打算在`for`循环的后的*还是*循环块`for`内的*？当您运行上述代码时，您会看到以下错误:*

```py
$ python indentation.py
 File "indentation.py", line 5
 print('done')
 ^
IndentationError: unindent does not match any outer indentation level
```

尽管回溯看起来很像`SyntaxError`回溯，但它实际上是一个`IndentationError`。错误消息也很有帮助。它告诉您该行的缩进级别与任何其他缩进级别都不匹配。换句话说，`print('done')`缩进了 2 个空格，但是 Python 找不到任何其他代码行匹配这个缩进级别。您可以通过确保代码符合预期的缩进级别来快速解决这个问题。

另一种类型的`SyntaxError`是 **`TabError`** ，每当有一行包含用于缩进的制表符或空格，而文件的其余部分包含另一行时，就会看到这种情况。这可能会隐藏起来，直到 Python 为您指出来！

如果您的制表符大小与每个缩进级别中的空格数一样宽，那么它可能看起来像所有的行都在同一级别。但是，如果一行使用空格缩进，另一行使用制表符缩进，那么 Python 会指出这是一个问题:

```py
 1# indentation.py
 2def foo():
 3    for i in range(10):
 4        print(i)
 5    print('done') 6
 7foo()
```

这里，第 5 行缩进了一个制表符，而不是 4 个空格。根据您的系统设置，这个代码块可能看起来非常好，也可能看起来完全错误。

然而，Python 会立即注意到这个问题。但是，在运行代码以查看 Python 会告诉您什么是错误的之前，看看不同制表符宽度设置下的代码示例可能会对您有所帮助:

```py
$ tabs 4 # Sets the shell tab width to 4 spaces
$ cat -n indentation.py
 1   # indentation.py
 2   def foo():
 3       for i in range(10)
 4           print(i)
 5       print('done') 6 
 7   foo()

$ tabs 8 # Sets the shell tab width to 8 spaces (standard)
$ cat -n indentation.py
 1   # indentation.py
 2   def foo():
 3       for i in range(10)
 4           print(i)
 5           print('done') 6 
 7   foo()

$ tabs 3 # Sets the shell tab width to 3 spaces
$ cat -n indentation.py
 1   # indentation.py
 2   def foo():
 3       for i in range(10)
 4           print(i)
 5      print('done') 6 
 7   foo()
```

请注意上面三个示例之间的显示差异。大多数代码为每个缩进级别使用 4 个空格，但是第 5 行在所有三个示例中都使用了一个制表符。标签的宽度根据**标签宽度**的设置而变化:

*   **如果标签宽度是 4** ，那么`print`语句将看起来像是在`for`循环之外。控制台将在循环结束时打印`'done'`。
*   **如果标签宽度为 8** ，这是许多系统的标准，那么`print`语句看起来就像是在`for`循环中。控制台会在每个数字后打印出`'done'`。
*   **如果标签宽度为 3** ，那么`print`语句看起来不合适。在这种情况下，第 5 行与任何缩进级别都不匹配。

当您运行代码时，您将得到以下错误和追溯:

```py
$ python indentation.py
 File "indentation.py", line 5
 print('done')
 ^
TabError: inconsistent use of tabs and spaces in indentation
```

注意`TabError`而不是通常的`SyntaxError`。Python 指出了问题所在，并给出了有用的错误消息。它清楚地告诉你，在同一个文件中混合使用了制表符和空格来缩进。

解决这个问题的方法是让同一个 Python 代码文件中的所有行都使用制表符或空格，但不能两者都用。对于上面的代码块，修复方法是移除制表符并用 4 个空格替换，这将在`for`循环完成后打印`'done'`。

[*Remove ads*](/account/join/)

### 定义和调用函数

在定义或调用函数时，您可能会在 Python 中遇到无效的语法。例如，如果在函数定义的末尾使用分号而不是冒号，您会看到一个`SyntaxError`:

>>>

```py
>>> def fun();
  File "<stdin>", line 1
    def fun();
             ^
SyntaxError: invalid syntax
```

这里的回溯非常有用，插入符号直接指向问题字符。您可以通过去掉冒号的分号来清除 Python 中的这种无效语法。

此外，函数定义和函数调用中的**关键字参数**需要有正确的顺序。关键字参数*总是*在位置参数之后。不使用该顺序将导致`SyntaxError`:

>>>

```py
>>> def fun(a, b):
...     print(a, b)
...
>>> fun(a=1, 2)
  File "<stdin>", line 1
SyntaxError: positional argument follows keyword argument
```

这里，错误消息再次非常有助于告诉您该行到底出了什么问题。

### 更改 Python 版本

有时，在一个版本的 Python 中运行良好的代码在新版本中会崩溃。这是由于语言句法的官方变化。最著名的例子是`print`语句，它从 Python 2 中的关键字变成了 Python 3 中的内置函数:

>>>

```py
>>> # Valid Python 2 syntax that fails in Python 3
>>> print 'hello'
  File "<stdin>", line 1
    print 'hello'
                ^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print('hello')?
```

这是与`SyntaxError`一起提供的错误消息闪耀的例子之一！它不仅告诉您在`print`调用中丢失了括号，而且还提供了正确的代码来帮助您修复语句。

您可能遇到的另一个问题是，当您在阅读或学习新版本 Python 中有效的语法时，在您正在编写的版本中却无效。这方面的一个例子是 [f-string](https://realpython.com/python-f-strings/) 语法，这在 Python 之前的版本中是不存在的:

>>>

```py
>>> # Any version of python before 3.6 including 2.7
>>> w ='world'
>>> print(f'hello, {w}')
  File "<stdin>", line 1
    print(f'hello, {w}')
                      ^
SyntaxError: invalid syntax
```

在 Python 之前的版本中，解释器不知道任何关于 f 字符串的语法，只会提供一个通用的`"invalid syntax"`消息。在这种情况下，问题是代码*看起来*非常好，但是它是用旧版本的 Python 运行的。如果有疑问，请仔细检查您运行的 Python 版本！

Python 语法在继续发展，在 [Python 3.8](https://realpython.com/python38-new-features/) 中引入了一些很酷的新特性:

*   [海象算子(赋值表达式)](https://docs.python.org/3.8/whatsnew/3.8.html#assignment-expressions)
*   [调试用 F-string 语法](https://docs.python.org/3.8/whatsnew/3.8.html#f-strings-support-for-self-documenting-expressions-and-debugging)
*   [仅位置参数](https://docs.python.org/3.8/whatsnew/3.8.html#positional-only-parameters)

如果您想尝试这些新特性，那么您需要确保您正在 Python 3.8 环境中工作。否则，你会得到一个`SyntaxError`。

Python 3.8 还提供了新的 **`SyntaxWarning`** 。在语法有效但看起来可疑的情况下，您会看到这个警告。这种情况的一个例子是，在一个列表中，两个元组之间缺少了一个逗号。这在 Python 之前的版本中是有效的语法，但是代码会引发一个`TypeError`，因为元组是不可调用的:

>>>

```py
>>> [(1,2)(2,3)]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object is not callable
```

这个`TypeError`意味着你不能像调用函数一样调用元组，这是 Python 解释器认为你在做的事情。

在 Python 3.8 中，这段代码仍然会引发`TypeError`，但是现在您还会看到一个`SyntaxWarning`来指示您可以如何修复这个问题:

>>>

```py
>>> [(1,2)(2,3)]
<stdin>:1: SyntaxWarning: 'tuple' object is not callable; perhaps you missed a comma?
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
TypeError: 'tuple' object is not callable
```

新`SyntaxWarning`附带的有用信息甚至提供了一个提示(`"perhaps you missed a comma?"`)，为您指出正确的方向！

[*Remove ads*](/account/join/)

## 结论

在本教程中，您已经看到了`SyntaxError`回溯给了您什么信息。您还看到了 Python 中无效语法的许多常见示例，以及这些问题的解决方案。这不仅会加速你的工作流程，还会让你成为一个更有帮助的代码评审者！

当你写代码时，试着使用理解 Python 语法并提供反馈的 IDE。如果您将本教程中的许多无效 Python 代码示例放到一个好的 IDE 中，那么它们应该会在您开始执行代码之前突出显示问题行。

在学习 Python 的时候获得一个 **`SyntaxError`** 可能会令人沮丧，但是现在你知道如何理解回溯消息，以及在 Python 中你可能会遇到什么形式的无效语法。下一次你得到一个`SyntaxError`，你将会更好的装备来快速解决问题！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**识别无效 Python 语法**](/courses/identify-invalid-syntax/)********