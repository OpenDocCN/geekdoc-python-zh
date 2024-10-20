# 通过 Plotly 和 Python 在线绘制数据

> 原文：<https://www.blog.pythonlibrary.org/2014/10/27/plotting-data-online-via-plotly-and-python/>

我在工作中不怎么绘图，但我最近听说了一个名为[plottly](https://plot.ly/)的网站，它为任何人的数据提供绘图服务。他们甚至有一个 Python 的 plotly 包(还有其他的)！因此，在这篇文章中，我们将学习如何与他们的包情节。让我们来做一些有趣的图表吧！

* * *

### 入门指南

您将需要 plotly 包来阅读本文。您可以使用 pip 获取软件包并安装它:

```py

pip install plotly

```

现在你已经安装好了，你需要去 Plotly 网站创建一个免费账户。一旦完成，您将获得一个 API 密钥。为了使事情变得非常简单，您可以使用您的用户名和 API 密匙来创建一个凭证文件。下面是如何做到这一点:

```py

import plotly.tools as tls

tls.set_credentials_file(
        username="your_username", 
        api_key="your_api_key")

# to get your credentials
credentials = tls.get_credentials_file()

```

如果您不想保存凭据，也可以通过执行以下操作来登录他们的服务:

```py

import plotly.plotly as py
py.sign_in('your_username','your_api_key')

```

出于本文的目的，我假设您已经创建了凭证文件。我发现这使得与他们的服务交互变得更容易使用。

* * *

### 创建图表

Plotly 似乎默认为散点图，所以我们就从这里开始。我决定从一个人口普查网站获取一些数据。您可以下载美国任何一个州的人口数据以及其他数据。在本例中，我下载了一个 CSV 文件，其中包含爱荷华州每个县的人口。让我们来看看:

```py

import csv
import plotly.plotly as py

#----------------------------------------------------------------------
def plot_counties(csv_path):
    """
    http://census.ire.org/data/bulkdata.html
    """
    counties = {}
    county = []
    pop = []

    counter = 0
    with open(csv_path) as csv_handler:
        reader = csv.reader(csv_handler)
        for row in reader:
            if counter  == 0:
                counter += 1
                continue
            county.append(row[8])
            pop.append(row[9])

    trace = dict(x=county, y=pop)
    data = [trace]
    py.plot(data, filename='ia_county_populations')

if __name__ == '__main__':
    csv_path = 'ia_county_pop.csv'
    plot_counties(csv_path)

```

如果您运行这段代码，您应该会看到如下所示的图形:

[https://plot.ly/~driscollis/0.embed?width=640&height=480](https://plot.ly/~driscollis/0.embed?width=640&height=480)

你也可以点击查看图表[。无论如何，正如您在上面的代码中看到的，我所做的只是读取 CSV 文件并提取出县名和人口。然后我将这些数据放入两个不同的 Python 列表中。最后，我为这些列表创建了一个字典，然后将这个字典包装在一个列表中。所以你最终得到一个包含字典的列表，而字典包含两个列表！为了制作散点图，我将数据传递给了 plotly 的 **plot** 方法。](https://plot.ly/~driscollis/0)

* * *

### 转换为条形图

现在让我们看看能否将散点图转换成条形图。首先，我们将摆弄一下绘图数据。以下是通过 Python 解释器完成的:

```py

>>> scatter = py.get_figure('driscollis', '0')
>>> print scatter.to_string()
Figure(
    data=Data([
        Scatter(
            x=[u'Adair County', u'Adams County', u'Allamakee County', u'..', ],
            y=[u'7682', u'4029', u'14330', u'12887', u'6119', u'26076', '..'  ]
        )
    ])
)

```

这显示了我们如何使用用户名和图的唯一编号来获取数字。然后我们打印出数据结构。您会注意到它没有打印出整个数据结构。现在，让我们进行条形图的实际转换:

```py

from plotly.graph_objs import Data, Figure, Layout

scatter_data = scatter.get_data()
trace_bar = Bar(scatter_data[0])
data = Data([trace_bar])
layout = Layout(title="IA County Populations")
fig = Figure(data=data, layout=layout)
py.plot(fig, filename='bar_ia_county_pop')

```

这将在以下 URL 创建一个条形图:[https://plot.ly/~driscollis/1](https://plot.ly/~driscollis/1)。这是图表的图像:

[https://plot.ly/~driscollis/1.embed?width=640&height=480](https://plot.ly/~driscollis/1.embed?width=640&height=480)

这段代码与我们最初使用的代码略有不同。在这种情况下，我们显式地创建了一个**条**对象，并将散点图的数据传递给它。然后我们将这些数据放入一个**数据**对象中。接下来，我们创建了一个**布局**对象，并给我们的图表加了一个标题。然后，我们使用数据和布局对象创建了一个**图形**对象。最后我们绘制了条形图。

* * *

### 将图形保存到磁盘

Plotly 还允许您将图形保存到硬盘上。您可以将其保存为以下格式:png、svg、jpeg 和 pdf。假设您手头还有上一个示例中的 Figure 对象，您可以执行以下操作:

```py

py.image.save_as(fig, filename='graph.png')

```

如果您想使用其他格式保存，那么只需在文件名中使用该格式的扩展名。

* * *

### 包扎

至此，您应该能够很好地使用 plotly 包了。还有许多其他可用的图形类型，所以请务必通读 Plotly 的文档。它们还支持流式图形。据我了解，Plotly 允许你免费创建 10 个图表。在那之后，你要么删除一些图片，要么支付月费。

* * *

### 附加阅读

*   Plotly Python [文档](https://plot.ly/python/)
*   Plotly [用户指南](https://plot.ly/python/user-guide/)