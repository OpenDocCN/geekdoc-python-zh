# Python 报价模块:如何生成随机报价？

> 原文：<https://www.askpython.com/python-modules/quote-module>

大家好，今天我们来学习如何使用 python 中的 **quote** 模块获取不同作者的报价。所以让我们开始吧！

## 生成随机关键字

为了从各种背景中获取报价，我们每次都会生成一个随机关键字，该程序将围绕该关键字返回特定作者的报价。

为了获得任意的英语单词，我们使用了`random_word`模块。random_word 模块可用于生成单个随机单词或随机单词列表。

如果导入模块时出现错误，您可以使用`pip`命令安装模块。我们先来看下面几行代码。

```py
from random_word import RandomWords
r = RandomWords()
w = r.get_random_word()
print(w)

```

这里我们从模块中导入了一个名为`RandomWords`的函数，并创建了一个相同的对象，这将有助于提取单词。

后来，我们对创建的对象应用了`get_random_word`函数来创建一个随机单词，并将其存储到一个变量中。

该代码从英语词典中随机生成一个单词。

## 使用 Python 中的报价模块获得随机报价

现在我们有了一个随机的关键字，下一步是使用`quote`库为该关键字生成一个报价。

如果导入库时出现错误，确保事先使用`pip`命令安装报价库。

让我们看看下面的代码。

```py
from quote import quote
res = quote('family',limit=1)
print(res)

```

为了生成随机报价，我们将使用报价模块中的`quote`函数。quote 函数需要一个关键字来搜索报价。

我们还设置了限制值来限制生成的报价数量。但是在打印输出时，我们会得到这样的结果:

```py
[{'author': 'J.R.R. Tolkien', 'book': 'The Fellowship of the Ring', 'quote': "I don't know half of you half as well as I should like; and I like less than half of you half as well as you deserve."}]

```

其背后的原因是 quote 函数返回一个字典列表，其中每个[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)包含关于特定报价的信息。

因此，我们将从字典中提取报价值。为此，我们将使用下面几行代码。

```py
for i in range(len(res)):
    print(res[i]['quote'])

```

我们在这里做的是遍历列表，对于每个字典值，我们将只打印`quote`键旁边的值。

现在我们得到如下输出:

```py
I don't know half of you half as well as I should like; and I like less than half of you half as well as you deserve.

```

## 用一个随机的词得到一个随机的报价

现在我们学习了如何使用不同的模块生成关键字和报价，让我们将它们结合起来，根据特定的关键字生成报价。

相同的代码如下所示。

```py
from random_word import RandomWords
from quote import quote

r = RandomWords()
w = r.get_random_word()
print("Keyword Generated: ",w)

res = quote(w, limit=1)
for i in range(len(res)):
    print("\nQuote Generated: ",res[i]['quote'])

```

结果如下:

```py
Keyword Generated:  fenman

Quote Generated:  The fenman gazed at Wimsey with a slow pity for his bird-witted feebleness of mind.

```

## 结论

今天我们学习了使用 Python 编程语言生成随机关键字和关键字周围的引号。

您也可以通过在多个关键字上生成多个引号来进行尝试！编码快乐！

感谢您的阅读！