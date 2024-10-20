# 关于 Python 正则表达式的更多信息

> 原文：<https://www.pythoncentral.io/more-about-python-regular-expressions/>

在本系列的第一部分中，我们看了正则表达式的基本语法和一些简单的例子。在这一部分中，我们将了解一些更高级的语法和 Python 必须提供的一些其他特性。

## 正则表达式捕获组

到目前为止，我们已经使用正则表达式在一个字符串中进行了搜索，并使用返回的`MatchObject`来提取匹配的整个子字符串。现在我们来看看如何从匹配的子字符串中提取部分。

**这个正则表达式:**

```py

\d{2}-\d{2}-\d{4}

```

**将匹配以下格式的日期:**

*   两位数的日期。
*   一个连字符。
*   两位数的月份。
*   一个连字符。
*   四位数的年份。

**例如:**

```py

>>> s = 'Today is 31-05-2012'

>>> mo = re.search(r'\d{2}-\d{2}-\d{4}', s)

>>> print(mo.group())

31-05-2012

```

**我们可以*通过将这个正则表达式的各个部分放在括号中来捕获*:**

```py

(\d{2})-(\d{2})-(\d{4})

```

如果 Python 匹配这个正则表达式，我们就可以分别检索每个*捕获的组*。

```py

>>> mo = re.search(r'(\d{2})-(\d{2})-(\d{4})', s)

>>> # Note: The entire matched string is still available

>>> print(mo.group())

31-05-2012

>>> # The first captured group is the date

>>> print(mo.group(1))

31

>>> # And this is its start/end position in the string

>>> print('%s %s' % (mo.start(1), mo.end(1)))

9 11

>>> # The second captured group is the month

>>> print(mo.group(2))

05

>>> # The third captured group is the year

>>> print(mo.group(3))

2012

```

当您开始编写更复杂的正则表达式时，使用有意义的名称而不是数字来引用它们会很有用。语法是`(...)`，其中...是要捕获的正则表达式，name 是要为组指定的名称。

```py

>>> s = "Joe's ID: abc123"

>>> # A normal captured group

>>> mo = re.search(r'ID: (.+)', s)

>>> print(mo.group(1))

abc123

>>> # A named captured group

>>> mo = re.search(r'ID: (?P<id>.+)', s)

>>> print(mo.group('id'))

abc123

```

### **使用正则表达式重用捕获的组**

我们还可以获取捕获的组，稍后在正则表达式中重用它们！`(?P=name)`表示*匹配之前在命名组*中匹配的任何内容。**比如:**

```py

>>> s = 'abc 123 def 456 def 789'

>>> mo = re.search(r'(?P<foo>def) \d+', s)

>>> print(mo.group())

def 456

>>> print(mo.group('foo'))

def

>>> # Capture 'def' in a group

>>> mo = re.search(r'(?P<foo>def) \d+ (?P=foo)', s)

>>> print(mo.group())

def 456 def

>>> mo.group('foo')

def

```

### **Python 正则表达式断言**

有时我们想匹配的东西*只有*后面有其他东西，这意味着 Python 在搜索字符串时需要提前查看。这被称为*前瞻断言*，语法为`(?=...)`，其中...是需要跟随的内容的正则表达式。

在下面的例子中，正则表达式`ham(?= and eggs)`意味着*匹配‘火腿’，但前提是它后面跟有‘和鸡蛋’*。

```py

>>> s = 'John likes ham and eggs.'

>>> mo = re.search(r'ham(?= and eggs)', s)

>>> print(mo.group())

ham

```

注意匹配的子串只有*火腿*，没有*火腿鸡蛋*。*和鸡蛋*部分只是对*火腿*部分进行匹配的要求。让我们看看如果不满足这个要求会发生什么。

```py

>>> s = 'John likes ham and mushrooms.'

>>> mo = re.search(r'ham(?= and eggs)', s)

>>> print(mo)

None

```

```py

>>> s = 'John likes ham, eggs and mushrooms.'

>>> mo = re.search(r'ham(?= and eggs)', s)

>>> print(mo)

None

```

可惜 Python 只做简单的字符匹配，只会匹配字符串*火腿*，只要后面跟*和鸡蛋*。人工智能和语义分析是另外一篇文章。🙂

我们还可以做*否定前瞻断言*，也就是说，一个元素只有在*而不是*后跟其他东西时才匹配。

```py

>>> s = 'My name is John Doe.'

>>> # Syntax is (?!...)

>>> mo = re.search( r'John(?! Doe)', s)

>>> print(mo)

None

```

```py

>>> s = 'My name is John Jones.'

>>> mo = re.search(r'John(?! Doe)', s)

>>> print(mo.group())

John

```