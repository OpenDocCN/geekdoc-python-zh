# 另一个 Python 日期时间的替代品:Pendulum

> 原文：<https://www.blog.pythonlibrary.org/2016/07/13/yet-another-python-datetime-replacement-pendulum/>

我偶然发现了另一个新的库，据称它比 Python 的 datetime 模块更好。它叫做[钟摆](http://pendulum.eustace.io/)。根据它的文档，钟摆受到 PHP 的[碳](http://carbon.nesbot.com/)的严重影响。

这些库总是很有趣，尽管我不确定是什么让这个库比 Arrow 或 Delorean 更好。在 Reddit 上有一些评论，钟摆的创造者说他发现 Arrow 在某些情况下表现奇怪。

无论如何，本文不是要比较这些相互竞争的库。只是为了看看钟摆本身是如何工作的。这些不同库的主要目的之一是创建一个具有更“Pythonic 式”API 和时区意识的 datetime 模块。

让我们安装它，以便我们可以检查它:

```py

pip install pendulum

```

如果您不喜欢将它直接安装到您的主 Python 安装中，那么您可以随意将其安装到 virtualenv 中。无论您将它安装在哪里，您都会看到它安装了一些自己的依赖项，如 tzlocal、python-translate、codegen 和 polib。

现在已经安装好了，让我们打开 Python 的解释器，试一试:

```py

>>> import pendulum
>>> pendulum.now('Japan')
 >>> pendulum.now() 
```

这里我们获取我所在位置的当前时间，然后我们询问日本的当前时间。你会注意到钟摆自动检测你的时区。如您所料，您也可以从日期创建类似日期时间的对象:

```py

>>> from pendulum import Pendulum
>>> Pendulum.create_from_date(2016, 7, 8, 'US/Eastern')
 >>> Pendulum.create_from_date(2016, 7, 8, 'US/Central')
 >>> Pendulum.create_from_date(2016, 7, 8, 'US/Mountain') 
```

这个例子演示了如何通过指定不同的时区来获得各种类似 datetime 的对象。然而，我喜欢钟摆的一点是它如何根据时间段计算日期:

```py

>>> today = pendulum.now()
>>> today
 >>> tomorrow = pendulum.now().add(days=1)
>>> tomorrow
 >>> last_week = pendulum.now().subtract(weeks=1)
>>> last_week 
```

我发现它加减时间段的方式非常直观。钟摆还有一个有趣的间隔概念:

```py

>>> interval = pendulum.interval(days=365)
>>> interval.weeks
52
>>> interval.years
1
>>> interval.days
365
>>> interval.for_humans
 >>> interval.for_humans()
'52 weeks 1 day' 
```

这里我们创建了一个 365 天的时间间隔。然后，我们可以询问它关于这段时间的各种信息，例如它包含多少天、多少周和多少年。您可以做的另一件有趣的事情是创建一个日期时间，然后向它询问与创建日期相关的其他日期:

```py

>>> dt = pendulum.create(2012, 1, 31, 12, 0, 0)
>>> dt.start_of('decade')
 >>> dt.start_of('century')
 >>> dt
 >>> dt.end_of('month')
 >>> dt 
```

在早期版本的 Pendulum 中，这些调用实际上会改变 dt 对象的位置。这将意味着我的 dt 对象将不是 2012-1-31，而是更改为十年的开始，然后是世纪，最后在月末结束，也就是 2001-01-30。在撰写本文时，此问题已得到解决，以与您在上面看到的内容相匹配。

### 包扎

钟摆是一个非常有趣的图书馆。在过去的一周左右的时间里，已经有了很多更新，因为作者正在根据用户的反馈修改 API。我发现我在写这篇文章时试图使用的函数在我准备发表时已经改变了。我确实认为这个库值得一看，但是当至少有另外两个有价值的竞争者已经存在一段时间时，我们将会看到它是否存在。一定要检查文档，因为它变化很大，这篇文章可能会过时。

### 相关项目

*   [箭头](http://crsmithdev.com/arrow/)
*   [德罗林](https://pypi.python.org/pypi/Delorean)
*   [时刻](https://github.com/zachwill/moment)

### 相关文章

*   python "[与 Delorean 共度时光](https://www.blog.pythonlibrary.org/2014/09/03/python-taking-time-with-delorean/)
*   arrow "[Python 的新日期/时间包](https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/)