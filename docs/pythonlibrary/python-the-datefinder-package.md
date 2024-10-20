# python-datefinder 包

> 原文：<https://www.blog.pythonlibrary.org/2016/02/04/python-the-datefinder-package/>

本周早些时候，我偶然发现了另一个有趣的包，叫做 [datefinder](https://github.com/akoumjian/datefinder) 。这个包背后的思想是，它可以接受任何包含日期的字符串，并将它们转换成 Python datetime 对象的列表。我在以前的工作中会喜欢这个包，在那里我做了大量的文本文件和数据库查询解析，因为很多时候找到日期并将其转换成我可以轻松使用的格式是非常麻烦的。

无论如何，要安装这个方便的软件包，你需要做的就是:

```py

pip install datefinder

```

我应该注意到，当我运行这个程序时，它最后还安装了以下软件包:

*   pyyaml-3.11 型自动步枪
*   日期解析器-0.3.2
*   jdatetime-1.7.2
*   python-dateutil-2.4.2
*   pytz-2015.7
*   regex-2016.1.10
*   六-10 . 0
*   umalqurra-0.2

因为所有这些额外的东西，你可能想先把这个包安装到一个 virtualenv 中。让我们来看看一些代码。这是我尝试的一个快速演示:

```py

>>> import datefinder
>>> data = '''Your appointment is on July 14th, 2016\. Your bill is due 05/05/2016'''
>>> matches = datefinder.find_dates(data)
>>> for match in matches:
...     print(match)
... 
2016-07-14 00:00:00
2016-05-05 00:00:00

```

如您所见，它与这两种常见的日期格式配合得非常好。我过去不得不支持的另一种格式是相当典型的 [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) 日期格式。让我们看看 datefinder 如何处理这个问题。

```py

>>> data = 'Your report is due: 2016-02-04T20:16:26+00:00'
>>> matches = datefinder.find_dates(x)
>>> for i in matches: 
...     print(i)
... 
2016-02-04 00:00:00
2016-02-04 20:16:26

```

有趣的是，这种特殊版本的 ISO 8601 格式会导致 datefinder 返回两个匹配项。第一个只是日期，而第二个既有日期又有时间。无论如何，希望你会发现这个包对你的项目有用。玩得开心！