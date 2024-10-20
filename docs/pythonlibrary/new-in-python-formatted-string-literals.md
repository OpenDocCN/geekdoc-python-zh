# Python 中的新特性:格式化字符串文字

> 原文：<https://www.blog.pythonlibrary.org/2017/02/08/new-in-python-formatted-string-literals/>

Python 3.6 增加了另一种进行字符串替换的方法，他们称之为“格式化字符串文字”。你可以在 [PEP 498](https://www.python.org/dev/peps/pep-0498) 中读到关于这个概念的所有内容。这里我有点不高兴，因为 Python 的禅宗说*应该有一种——最好只有一种——显而易见的方法来做这件事*。现在 Python 有三种方式。在谈论最新的弦乐演奏方式之前，让我们回顾一下过去。

* * *

### 旧字符串替换

Python 刚开始的时候，他们按照 C++的方式使用 **%s、%i** 等进行字符串替换。这里有几个例子:

```py

>>> The %s fox jumps the %s' % ('quick', 'crevice')
'The quick fox jumps the crevice'
>>> foo = 'The total of your purchase is %.2f' % 10
>>> foo
'The total of your purchase is 10.00'

```

上面的第二个示例演示了如何将一个数字格式化为精度设置为两位小数的浮点数。这种字符串替换方法也支持关键字参数:

```py

>>> 'Hi, my name is %(name)s' % {'name': 'Mike'}
Out[21]: 'Hi, my name is Mike'

```

语法有点奇怪，我总是要查找它才能正确工作。

虽然这些字符串替换方法仍然受支持，但人们发明了一种新的方法，这种方法应该更清晰、功能更强。让我们看看这是什么样子:

```py

>>> bar = 'You need to pay {}'.format(10.00)
>>> bar
'You need to pay 10.0'
>>> swede = 'The Swedish chef is know for saying {0}, {1}, {2}'.format('bork', 'cork', 'spork')
>>> swede
'The Swedish chef is know for saying bork, cork, spork'

```

我认为这是一个非常聪明的新添加。不过，还有一个额外的增强，那就是您实际上可以使用关键字参数来指定字符串替换中的内容:

```py

>>> swede = 'The Swedish chef is know for saying {something}, {something}, {something}'
>>> swede.format(something='bork')
'The Swedish chef is know for saying bork, bork, bork'
>>> test = 'This is a {word} of your {something}'.format(word='Test', something='reflexes')
>>> test
'This is a Test of your reflexes'

```

这很酷，实际上也很有用。你会看到一些程序员会争论哪种方法更好。我看到一些人甚至声称，如果你做大量的字符串替换，原来的方法实际上比新的方法更快。不管怎样，这让你对旧的做事方式有了一个简要的了解。让我们看看有什么新的！

* * *

### 使用格式化字符串文字

从 Python 3.6 开始，我们得到格式化的字符串或 f 字符串。格式化字符串的语法与我们之前看到的稍有不同:

```py

>>> name = 'Mike'
>>> f'My name is {name}'
'My name is Mike'

```

让我们把它分解一下。我们要做的第一件事是定义一个要插入字符串的变量。接下来我们想告诉 Python 我们想创建一个格式化的字符串文字。为此，我们在字符串前面加上字母“f”。这意味着字符串将被格式化。最后一部分与上一节的最后一个例子非常相似，我们只需要将变量名插入到字符串中，并用一对花括号括起来。然后 Python 变了一些魔术，我们打印出了一个新的字符串。这实际上非常类似于一些 Python 模板语言，比如 mako。

f-string 也支持某些类型的转换，比如 str() via !!s '和 repr() via '！r '这里有一个更新的例子:

```py

>>> f'My name is {name!r}'
Out[11]: "My name is 'Mike'"

```

您会注意到输出中的变化非常微妙，因为添加的只是插入变量周围的一些单引号。让我们来看看更复杂一点的东西，即浮点数！

```py

>>> import decimal
>>> gas_total = decimal.Decimal('20.345')
>>> width = 10
>>> precision = 4
>>> f'Your gas total is: {gas_total:{width}.{precision}}'
'Your gas total is:      20.34'

```

这里，我们导入 Python 的十进制模块，并创建一个表示气体总量的实例。然后我们设置字符串的宽度为 10 个字符，精度为 4。最后，我们告诉 f 字符串为我们格式化它。正如您所看到的，插入的文本在前端有一些填充，使其宽度为 10 个字符，精度基本上设置为 4，这截断了 5，而不是向上舍入。

* * *

### 包扎

新的格式化字符串文字或 f-string 并没有给格式化字符串增加任何新的东西。然而[声称](https://www.python.org/dev/peps/pep-0498/#id24)比以前的方法更灵活，更不容易出错。我强烈推荐阅读文档和 PEP 498 来帮助您了解这个新特性，这样您就可以确定这是否是您将来进行字符串替换的方式。

* * *

### 相关阅读

*   [Python 3.6 的新特性](https://docs.python.org/3.6/whatsnew/3.6.html#pep-498-formatted-string-literals)
*   PEP 498 - [文字字符串插值](https://www.python.org/dev/peps/pep-0498/)
*   Python 中的新特性:[变量注释的语法](https://www.blog.pythonlibrary.org/2017/01/12/new-in-python-syntax-for-variable-annotations/)
*   Python 中的新特性:[数字文字中的下划线](https://www.blog.pythonlibrary.org/2017/01/11/new-in-python-underscores-in-numeric-literals/)