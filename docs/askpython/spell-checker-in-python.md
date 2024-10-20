# Python 中的拼写检查器

> 原文：<https://www.askpython.com/python/examples/spell-checker-in-python>

Python 中的拼写检查器是一种检查文本中拼写错误的软件功能。拼写检查功能通常嵌入在软件或服务中，如文字处理器、电子邮件客户端、电子词典或搜索引擎。

* * *

## 用 Python 构建拼写检查器

让我们开始构建我们的拼写检查工具吧！

### 1.导入模块

我们将用两个不同的模块来构建我们的拼写检查工具:

*   拼写检查模块
*   textblob 模块

让我们从一个一个地安装和导入开始。

为了在 python 中构建拼写检查器，我们需要导入**拼写检查器模块**。如果您没有该模块，您可以使用 [pip 软件包管理器](https://www.askpython.com/python-modules/python-pip)安装该模块。

```py
C:\Users\Admin>pip install spellchecker

```

你也可以用同样的方法安装 **textblob 模块**

```py
C:\Users\Admin>pip install textblob

```

* * *

### 2.使用 textblob 模块进行拼写检查

**TextBlob** 在 python 编程语言中是一个用于处理文本数据的 python 库。它提供了一个简单的 API，用于处理常见的自然语言处理任务，如词性标注、名词短语提取、情感分析、分类、翻译等。

**correct()函数:**纠正输入文本最直接的方法是使用`correct()`方法。

```py
from textblob import TextBlob
#Type in the incorrect spelling
a = "eies"
print("original text: "+str(a))
b = TextBlob(a)
#Obtain corrected spelling as an output
print("corrected text: "+str(b.correct()))

```

**输出:**

```py
original text: eies
corrected text: eyes

```

* * *

### 3.使用拼写检查器模块进行拼写检查

让我们看看拼写检查模块如何纠正句子错误！

```py
#import spellchecker library
from spellchecker import SpellChecker

#create a variable spell and instance as spellchecker()
spell=SpellChecker()
'''Create a while loop under this loop you need to create a variable called a word and make this variable that takes the real-time inputs from the user.'''

while True:
    w=input('Enter any word of your choice:')
    w=w.lower()
'''if the word that presents in the spellchecker dictionary, It
will print “you spelled correctly" Else you need to find the best spelling for that word'''
    if w in spell:
        print("'{}' is spelled correctly!".format(w))
    else:
        correctwords=spell.correction(w)
        print("The best suggestion for '{}' is '{}'".format(w,correctwords))

```

```py
Enter any word of your choice:gogle
The best suggestion for 'gogle' is 'google'

```

拼写检查器实例将在该程序中被多次调用。它能容纳大量的单词。如果你输入任何拼写错误的单词，如果它不在拼写检查词典中，它会纠正下来。这就是你对这个图书馆的了解。

## 结论

简单来说，这是关于如何使用 Python 编程语言来构建你自己的拼写检查器，它易于编码，易于学习和理解，只需要很少的几行代码。