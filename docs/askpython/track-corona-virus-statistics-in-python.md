# 用 Python 追踪冠状病毒统计数据的简单方法

> 原文：<https://www.askpython.com/python/examples/track-corona-virus-statistics-in-python>

在本教程中，我们将使用 COVID19Py 库来跟踪 Python 中的 Corona 病毒统计数据。

大家好！我们正在经历这个困难时期，尽我们所能帮助我们周围的人是有意义的。作为程序员，您可以帮助传播关于 COVID 病毒的信息，帮助人们找到获得疫苗的地方，等等。

这是一个用 Python 编写的预建的 Corona 病毒统计跟踪器。你所要做的就是安装它，执行某些功能，你就可以访问来自世界各地的信息。所以让我们开始吧。

* * *

## *设置 COVID19Py*

Python 安装非常简单。只需在命令提示符下键入 [pip 命令](https://www.askpython.com/python-modules/python-pip)。

```py
pip install COVID19Py

```

在. py 文件中键入以下命令以导入此包:

```py
import COVID19Py

```

这个包只有一行预处理，使用起来非常简单。

```py
covid19 = COVID19Py.COVID19()

```

* * *

## *追踪 Python 中的 COVID19 信息*

现在我们有了一个 package 对象，我们可以开始使用它的方法了。

使用 getLatest()函数收集关于受影响者、已康复者以及全球死亡人数的最新统计数据。它给你一个字典列表。

```py
L= covid19.getLatest()
print(L)

```

```py
{'confirmed': 277161199, 'deaths': 5377197, 'recovered': 0}

```

然后使用 getLocations()函数过滤收集的大量数据。

```py
LS = covid19.getLocations()
print(LS[0])

```

```py
{'id': 0, 'country': 'Afghanistan', 'country_code': 'AF', 'country_population': 37172386, 'province': '', 'last_updated': '2021-12-23T08:34:35.628637Z', 'coordinates': {'latitude': '33.93911', 'longitude': '67.709953'}, 'latest': {'confirmed': 157841, 'deaths': 7341, 'recovered': 0}}

```

需要国家代码来查看某个国家的数据。因此，这里有一个简单的方法来获取捆绑包中包含的所有国家代码。

```py
C= {}
for i in locations:
    C[i.get('country')] = i.get('country_code')

```

只需输入这段代码就可以获得印度的统计数据。

```py
code = C.get('India')
india = covid19.getLocationByCountryCode("IN")
for i in india:
    print(i.get("latest"))

```

```py
{'confirmed': 34765976, 'deaths': 478759, 'recovered': 0}

```

* * *

## 结论

既然你已经掌握了所有的工具，找到你想要的所有数据，这样你就可以比较和对比其他国家的统计数据。

您还可以使用 Google Trends API 来查看 Corona 病毒在整个互联网中的趋势。

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 统计模块——要知道的 7 个函数！](https://www.askpython.com/python-modules/statistics-module)
2.  [Python 中如何计算汇总统计？](https://www.askpython.com/python/examples/calculate-summary-statistics)
3.  [数据分析与数据科学](https://www.askpython.com/python/data-analytics-vs-data-science)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *