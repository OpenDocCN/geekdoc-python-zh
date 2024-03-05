# Python 3 的 f-Strings:改进的字符串格式化语法(指南)

> 原文：<https://realpython.com/python-f-strings/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 3 的 f-Strings:改进的字符串格式化语法**](/courses/python-3-f-strings-improved-string-formatting-syntax/)

从 Python 3.6 开始，f 字符串是格式化字符串的一种很好的新方法。它们不仅比其他格式更易读、更简洁、更不易出错，而且速度也更快！

到本文结束时，您将了解如何以及为什么今天开始使用 f 弦。

但首先，这是在 f 弦出现之前的生活，那时你必须在雪地里走着上山去上学。

***参加测验:****通过我们的交互式“Python f-Strings”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/f-strings/)

## Python 中的“老派”字符串格式

在 Python 3.6 之前，有两种将 Python 表达式嵌入字符串文字进行格式化的主要方式:%-formatting 和`str.format()`。您将看到如何使用它们以及它们的局限性。

[*Remove ads*](/account/join/)

### 选项# 1:%-格式化

这是 Python 格式的 OG，从一开始就存在于语言中。你可以在 [Python 文档](https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting)中阅读更多内容。请记住，文档中不建议使用%格式，文档中包含以下说明:

> 这里描述的格式化操作展示了导致许多常见错误的各种怪癖(比如不能正确显示元组和字典)。
> 
> 使用新的格式化字符串或`str.format()`接口有助于避免这些错误。这些替代方案还提供了更强大、更灵活、更可扩展的文本格式化方法。"([来源](https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting))

#### 如何使用%格式

String 对象有一个使用`%`操作符的内置操作，可以用来格式化字符串。实际情况是这样的:

>>>

```py
>>> name = "Eric"
>>> "Hello, %s." % name
'Hello, Eric.'
```

为了插入多个变量，必须使用这些变量的元组。你可以这样做:

>>>

```py
>>> name = "Eric"
>>> age = 74
>>> "Hello, %s. You are %s." % (name, age)
'Hello Eric. You are 74.'
```

#### 为什么%格式不好

您刚才看到的代码示例足够易读。然而，一旦你开始使用几个参数和更长的字符串，你的代码将很快变得不容易阅读。事情已经开始变得有些混乱了:

>>>

```py
>>> first_name = "Eric"
>>> last_name = "Idle"
>>> age = 74
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> "Hello, %s  %s. You are %s. You are a %s. You were a member of %s." % (first_name, last_name, age, profession, affiliation)
'Hello, Eric Idle. You are 74\. You are a comedian. You were a member of Monty Python.'
```

不幸的是，这种格式不是很好，因为它冗长并且会导致错误，比如不能正确显示元组或字典。幸运的是，前方有更光明的日子。

### 选项 2: str.format()

Python 2.6 中引入了这种完成工作的新方法。你可以查看最新的 Python 字符串格式技术指南以获得更多信息。

#### 如何使用 str.format()

`str.format()`是对%格式的改进。它使用普通的函数调用语法，并且通过被转换为字符串的对象上的`__format__()`方法是[可扩展的。](https://www.python.org/dev/peps/pep-3101/#controlling-formatting-on-a-per-type-basis)

使用`str.format()`，替换字段由花括号标记:

>>>

```py
>>> "Hello, {}. You are {}.".format(name, age)
'Hello, Eric. You are 74.'
```

您可以通过引用变量的索引来以任意顺序引用变量:

>>>

```py
>>> "Hello, {1}. You are {0}.".format(age, name)
'Hello, Eric. You are 74.'
```

但是如果您插入变量名，您将获得额外的好处，能够传递对象，然后在大括号之间引用参数和方法:

>>>

```py
>>> person = {'name': 'Eric', 'age': 74}
>>> "Hello, {name}. You are {age}.".format(name=person['name'], age=person['age'])
'Hello, Eric. You are 74.'
```

你也可以使用`**`来用字典做这个巧妙的把戏:

>>>

```py
>>> person = {'name': 'Eric', 'age': 74}
>>> "Hello, {name}. You are {age}.".format(**person)
'Hello, Eric. You are 74.'
```

与% formatting 相比，这绝对是一个升级，但也不完全是美好的事情。

#### 为什么 str.format()不好

使用`str.format()`的代码比使用%格式的代码更容易阅读，但是当您处理多个参数和更长的字符串时,`str.format()`仍然非常冗长。看看这个:

>>>

```py
>>> first_name = "Eric"
>>> last_name = "Idle"
>>> age = 74
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> print(("Hello, {first_name}  {last_name}. You are {age}. " + 
>>>        "You are a {profession}. You were a member of {affiliation}.") \
>>>        .format(first_name=first_name, last_name=last_name, age=age, \
>>>                profession=profession, affiliation=affiliation))
'Hello, Eric Idle. You are 74\. You are a comedian. You were a member of Monty Python.'
```

如果你在字典中有你想要传递给`.format()`的变量，那么你可以用`.format(**some_dict)`解包它，并通过字符串中的键引用值，但是必须有一个更好的方法来做到这一点。

[*Remove ads*](/account/join/)

## f-Strings:在 Python 中格式化字符串的一种新的改进方法

好消息是 f 弦在这里拯救世界。他们切！他们掷骰子！他们做薯条！好吧，它们什么都不做，但是它们确实使格式化变得更容易。他们加入了 Python 3.6 的聚会。你可以在 2015 年 8 月 Eric V. Smith 写的《PEP 498 中读到所有相关内容。

也称为“格式化字符串”，f-string 是在开头有一个`f`和花括号的字符串，花括号包含将被其值替换的表达式。表达式在运行时被求值，然后使用`__format__`协议格式化。一如既往，当你想了解更多时， [Python 文档](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)是你的好朋友。

这里有一些 f 弦可以让你的生活更轻松的方法。

### 简单语法

语法类似于您在`str.format()`中使用的语法，但不太详细。看看这是多么容易阅读:

>>>

```py
>>> name = "Eric"
>>> age = 74
>>> f"Hello, {name}. You are {age}."
'Hello, Eric. You are 74.'
```

使用大写字母`F`也是有效的:

>>>

```py
>>> F"Hello, {name}. You are {age}."
'Hello, Eric. You are 74.'
```

你喜欢 f 弦了吗？我希望，在这篇文章结束时，你会回答 [`>>> F"Yes!"`](https://twitter.com/dbader_org/status/992847368440561664) 。

### 任意表达式

因为 f 字符串是在运行时计算的，所以可以在其中放入任何和所有有效的 Python 表达式。这允许你做一些漂亮的事情。

你可以做一些非常简单的事情，就像这样:

>>>

```py
>>> f"{2 * 37}"
'74'
```

但是你也可以调用函数。这里有一个例子:

>>>

```py
>>> def to_lowercase(input):
...     return input.lower()

>>> name = "Eric Idle"
>>> f"{to_lowercase(name)} is funny."
'eric idle is funny.'
```

您也可以选择直接调用方法:

>>>

```py
>>> f"{name.lower()} is funny."
'eric idle is funny.'
```

您甚至可以使用从带有 f 字符串的类中创建的对象。假设您有以下类:

```py
class Comedian:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

    def __str__(self):
        return f"{self.first_name}  {self.last_name} is {self.age}."

    def __repr__(self):
        return f"{self.first_name}  {self.last_name} is {self.age}. Surprise!"
```

你可以这样做:

>>>

```py
>>> new_comedian = Comedian("Eric", "Idle", "74")
>>> f"{new_comedian}"
'Eric Idle is 74.'
```

[`__str__()`和`__repr__()`方法](https://realpython.com/operator-function-overloading/)处理如何将对象表示为字符串，所以你需要确保在你的类定义中至少包含其中一个方法。如果非要选一个，就选`__repr__()`，因为它可以代替`__str__()`。

由`__str__()`返回的字符串是对象的非正式字符串表示，应该是可读的。`__repr__()`返回的字符串是官方表示，应该是明确的。调用`str()`和`repr()`比直接使用`__str__()`和`__repr__()`要好。

默认情况下，f 字符串将使用`__str__()`，但是如果您包含转换标志`!r`，您可以确保它们使用`__repr__()`:

>>>

```py
>>> f"{new_comedian}"
'Eric Idle is 74.'
>>> f"{new_comedian!r}"
'Eric Idle is 74\. Surprise!'
```

如果你想阅读一些导致 f 字符串支持完整 Python 表达式的对话，你可以在这里[阅读。](https://mail.python.org/pipermail/python-ideas/2015-July/034726.html)

[*Remove ads*](/account/join/)

### 多行 f 字符串

可以有多行字符串:

>>>

```py
>>> name = "Eric"
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> message = (
...     f"Hi {name}. "
...     f"You are a {profession}. "
...     f"You were in {affiliation}."
... )
>>> message
'Hi Eric. You are a comedian. You were in Monty Python.'
```

但是请记住，您需要在多行字符串的每一行前面放置一个`f`。以下代码不起作用:

>>>

```py
>>> message = (
...     f"Hi {name}. "
...     "You are a {profession}. "
...     "You were in {affiliation}."
... )
>>> message
'Hi Eric. You are a {profession}. You were in {affiliation}.'
```

如果你不在每一行前面加一个`f`，那么你就只有普通的、旧的、普通的琴弦，而没有闪亮的、新的、花哨的 f 弦。

如果您想将字符串分布在多行中，您还可以选择用`\`来转义 return:

>>>

```py
>>> message = f"Hi {name}. " \
...           f"You are a {profession}. " \
...           f"You were in {affiliation}."
...
>>> message
'Hi Eric. You are a comedian. You were in Monty Python.'
```

但是如果你使用`"""`就会发生这种情况:

>>>

```py
>>> message = f"""
... Hi {name}. 
... You are a {profession}. 
... You were in {affiliation}.
... """
...
>>> message
'\n    Hi Eric.\n    You are a comedian.\n    You were in Monty Python.\n'
```

阅读 [PEP 8](https://pep8.org/) 中的缩进指南。

### 速度

f 弦中的`f`也可以代表“快”。

f 字符串比% formatting 和`str.format()`都要快。正如您已经看到的，f 字符串是在运行时计算的表达式，而不是常量值。以下是这些文件的摘录:

> F 字符串提供了一种使用最小语法将表达式嵌入字符串文字的方法。应该注意，f 字符串实际上是一个在运行时计算的表达式，而不是一个常数值。在 Python 源代码中，f-string 是一个文字字符串，前缀为`f`，包含大括号内的表达式。表达式将被替换为它们的值。([来源](https://www.python.org/dev/peps/pep-0498/#abstract))

在运行时，花括号内的表达式在其自身的范围内进行计算，然后与 f 字符串的字符串文字部分放在一起。然后返回结果字符串。这就够了。

这里有一个速度对比:

>>>

```py
>>> import timeit
>>> timeit.timeit("""name = "Eric"
... age = 74
... '%s is %s.' % (name, age)""", number = 10000)
0.003324444866599663
```

>>>

```py
>>> timeit.timeit("""name = "Eric"
... age = 74
... '{} is {}.'.format(name, age)""", number = 10000)
0.004242089427570761
```

>>>

```py
>>> timeit.timeit("""name = "Eric"
... age = 74
... f'{name} is {age}.'""", number = 10000)
0.0024820892040722242
```

如你所见，f 弦出现在顶部。

然而，情况并非总是如此。当他们第一次实现时，他们有一些[速度问题](https://stackoverflow.com/questions/37365311/why-are-literal-formatted-strings-so-slow-in-python-3-6-alpha-now-fixed-in-3-6)，需要比`str.format()`更快。引入了一个特殊的 [`BUILD_STRING`操作码](https://bugs.python.org/issue27078)。

[*Remove ads*](/account/join/)

## Python f-Strings:讨厌的细节

既然你已经了解了 f 弦为什么如此伟大，我相信你一定想走出去开始使用它们。当你冒险进入这个勇敢的新世界时，有一些细节要记住。

### 引号

您可以在表达式中使用各种类型的引号。只要确保在 f 字符串的外部没有使用与表达式中相同类型的引号。

这段代码将起作用:

>>>

```py
>>> f"{'Eric Idle'}"
'Eric Idle'
```

此代码也将工作:

>>>

```py
>>> f'{"Eric Idle"}'
'Eric Idle'
```

您也可以使用三重引号:

>>>

```py
>>> f"""Eric Idle"""
'Eric Idle'
```

>>>

```py
>>> f'''Eric Idle'''
'Eric Idle'
```

如果您发现您需要在字符串的内部和外部使用相同类型的引号，那么您可以使用`\`进行转义:

>>>

```py
>>> f"The \"comedian\" is {name}, aged {age}."
'The "comedian" is Eric Idle, aged 74.'
```

### 字典

说到引号，你在查字典的时候要小心。如果您打算对字典的键使用单引号，那么请记住确保对包含这些键的 f 字符串使用双引号。

这将起作用:

>>>

```py
>>> comedian = {'name': 'Eric Idle', 'age': 74}
>>> f"The comedian is {comedian['name']}, aged {comedian['age']}."
The comedian is Eric Idle, aged 74.
```

但是这将会是一场混乱，因为有一个语法错误:

>>>

```py
>>> comedian = {'name': 'Eric Idle', 'age': 74}
>>> f'The comedian is {comedian['name']}, aged {comedian['age']}.'
  File "<stdin>", line 1
    f'The comedian is {comedian['name']}, aged {comedian['age']}.'
                                    ^
SyntaxError: invalid syntax
```

如果在字典键周围使用与 f 字符串外部相同类型的引号，那么第一个字典键开头的引号将被解释为字符串的结尾。

[*Remove ads*](/account/join/)

### 大括号

为了让大括号出现在字符串中，您必须使用双大括号:

>>>

```py
>>> f"{{70 + 4}}"
'{70 + 4}'
```

请注意，使用三大括号将导致字符串中只有一个大括号:

>>>

```py
>>> f"{{{70 + 4}}}"
'{74}'
```

但是，如果使用三个以上的大括号，可以显示更多的大括号:

>>>

```py
>>> f"{{{{70 + 4}}}}"
'{{70 + 4}}'
```

### 反斜杠

正如您前面看到的，您可以在 f 字符串的字符串部分使用反斜杠转义。但是，不能在 f 字符串的表达式部分使用反斜杠进行转义:

>>>

```py
>>> f"{\"Eric Idle\"}"
  File "<stdin>", line 1
    f"{\"Eric Idle\"}"
                      ^
SyntaxError: f-string expression part cannot include a backslash
```

您可以通过预先计算表达式并在 f 字符串中使用结果来解决这个问题:

>>>

```py
>>> name = "Eric Idle"
>>> f"{name}"
'Eric Idle'
```

### 行内注释

表达式不应包含使用`#`符号的注释。您将得到一个语法错误:

>>>

```py
>>> f"Eric is {2 * 37 #Oh my!}."
  File "<stdin>", line 1
    f"Eric is {2 * 37 #Oh my!}."
                                ^
SyntaxError: f-string expression part cannot include '#'
```

## 开始格式化吧！

您仍然可以使用旧的格式化字符串的方法，但是有了 f 字符串，您现在有了一种更简洁、可读性更好、更方便的方法，既更快又不容易出错。通过使用 f 字符串来简化您的生活是开始使用 Python 3.6 的一个很好的理由，如果您还没有做出改变的话。(如果你还在用 Python 2，别忘了 [2020](https://pythonclock.org/) 马上就来了！)

根据 Python 的[禅，当你需要决定如何做某事时，那么“应该有一个——最好只有一个——显而易见的方法去做。”虽然 f-strings 不是格式化字符串的唯一可能的方法，但是它们很有可能成为完成这项工作的一种显而易见的方法。](https://www.python.org/dev/peps/pep-0020/)

## 延伸阅读

如果你想阅读关于字符串插值的详细讨论，看看 [PEP 502](https://www.python.org/dev/peps/pep-0502/) 。此外， [PEP 536 草案](https://www.python.org/dev/peps/pep-0536/)对 f 弦的未来有更多的想法。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

要获得更多关于字符串的乐趣，请查看以下文章:

*   Dan Bader 的 Python 字符串格式化最佳实践
*   Colin OKeefe 的 Python Web 抓取实用介绍

快乐的蟒蛇！

***参加测验:****通过我们的交互式“Python f-Strings”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/f-strings/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 3 的 f-Strings:改进的字符串格式化语法**](/courses/python-3-f-strings-improved-string-formatting-syntax/)*********