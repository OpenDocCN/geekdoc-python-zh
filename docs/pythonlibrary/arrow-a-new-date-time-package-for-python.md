# 箭头 Python 的一个新的日期/时间包

> 原文：<https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/>

[arrow](https://github.com/crsmithdev/arrow) 项目试图将 Python 的时间和日期时间模块封装到一个 API 中。它还声称要填补这些模块中的功能空白，如时间跨度、ISO-8601 和人性化。你可以把 arrow 看作 Python 的 datetime 和 time 模块的替代物，就像可以用 [requests](http://docs.python-requests.org/en/latest/) 项目代替 Python 的 urllib 一样。在撰写本文时，Arrow 支持 Python 2.6、2.7 和 3.3。

### 安装箭头

要开始使用 arrow，只需点击安装即可:

```py

pip install arrow

```

* * *

### 使用箭头

arrow 包使用起来非常简单。让我们看几个常见的例子:

```py

>>> import arrow
>>> arrow.now()
 >>> now = arrow.now()
>>> now.ctime()
'Fri Jul 25 15:41:30 2014'
>>> pacific = now.to("US/Pacific")
>>> pacific
 >>> pacific.timestamp
1406320954 
```

这里我们得到了当天的日期和时间。然后，我们将它存储在一个变量中，并将时区更改为太平洋标准时间。我们还可以获得时间戳值，即从 epoch 开始的秒数。让我们再看几个例子:

```py

>>> day = arrow.get("2014-07-13")
>>> day.format("MM-DD-YYYY")
'07-13-2014'
>>> day.humanize()
u'12 days ago'

```

我们在这里选择一个日期，然后重新格式化它的显示方式。如果您调用 arrow 的**人源化()**方法，它会告诉您是多少天前的事了。如果你通过【T2 现在】()方法得到现在的时间，然后叫做人性化，你会得到不同的信息。

### 包扎

我最喜欢这个包的地方是它很好地包装了 Python 的日期和时间模块。能够通过一个公共界面访问他们的每一个功能是很好的。作者已经使日期和时间操作更容易使用。我认为值得你花时间试一试。开心快乐编码！