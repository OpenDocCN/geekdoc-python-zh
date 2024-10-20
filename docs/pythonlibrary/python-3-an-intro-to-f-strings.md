# python 3:f 弦介绍

> 原文：<https://www.blog.pythonlibrary.org/2018/03/13/python-3-an-intro-to-f-strings/>

Python 3.6 增加了另一种进行字符串插值的方法，称为“f-strings”或格式化字符串文字( [PEP 498](https://www.python.org/dev/peps/pep-0498/) )。f 字符串背后的想法是使字符串插值更简单。要创建一个 f 字符串，你只需要在字符串前面加上字母“f”。字符串本身可以用与使用 **str.format()** 相同的方式进行格式化。换句话说，字符串中可以有用花括号括起来的替换字段。这里有一个简单的例子:

```py

>>> name = 'Mike'
>>> f"Hello {name}"
'Hello Mike'

```

Python 文档中有一个有趣的例子，演示了如何嵌套替换字段。不过，我做了一点修改，让它更简单:

```py

>>> total = 45.758
>>> width = 14
>>> precision = 4
>>> f"Your total is {total:{width}.{precision}}"
'Your total is          45.76'

```

这里我们创建了三个变量，第一个是浮点数，另外两个是整数。然后我们创建我们的 f-string，告诉它我们想把 **total** 变量放到我们的字符串中。但是你会注意到，在字符串的替换字段中，我们嵌套了**宽度**和**精度**变量来格式化总数本身。在这种情况下，我们告诉 Python，我们希望 total 字段的宽度是 14 个字符，float 的精度是 4，所以结果是 45.76，您会注意到它是向上舍入的。

f 字符串还支持日期格式:

```py

>>> import datetime
>>> today = datetime.datetime.today()
>>> f"{today:%B %d, %Y}"
'March 13, 2018'

```

我个人很喜欢 PEP 498 中给出的例子，它实际上展示了如何使用日期格式从日期中提取星期几:

```py

>>> from datetime import datetime
>>> date = datetime(1992, 7, 4)
>>> f'{date} was on a {date:%A}'
'1992-07-04 00:00:00 was on a Saturday'

```

您也可以在 f 弦中重复使用相同的变量:

```py

>>> spam = 'SPAM'
>>> f"Lovely {spam}! Wonderful {spam}!"
'Lovely SPAM! Wonderful SPAM!'

```

文档确实指出，在嵌套引号时，必须小心 f 字符串。比如，你显然不能做这样的事情:

```py

>>> value = 123
>>> f"Your value is "{value}""

```

这是一个语法错误，就像使用常规字符串一样。您也不能在格式字符串中直接使用反斜杠:

```py

>>> f"newline: {ord('\n')}"
Traceback (most recent call last):
  Python Shell, prompt 29, line 1
Syntax Error: f-string expression part cannot include a backslash: , line 1, pos 0 
```

文档指出，作为一种变通方法，您可以将反斜杠放入变量中:

```py

>>> newline = ord('\n')
>>> f"newline: {newline}"
'newline: 10'

```

在这个例子中，我们将换行符转换成它的序数值。文档中提到的最后一点是，不能将 f 字符串用作 docstring。根据 Nick Coghlan 的说法，这是因为需要在编译时知道文档字符串，但是 f 字符串直到运行时才被解析。

### 包扎

此时，您应该有足够的信息开始在自己的代码中使用 f 字符串。这是对 Python 语言的一个有趣的补充，虽然不是绝对必要的，但我可以看到它在某些方面使字符串插值更简单。开心快乐编码！

### 进一步阅读

https://docs.python.org/3/reference/lexical_analysis.html#f-strings