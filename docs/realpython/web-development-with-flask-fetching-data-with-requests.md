# 全栈开发——获取数据，用 D3 可视化，用 Dokku 部署

> 原文：<https://realpython.com/web-development-with-flask-fetching-data-with-requests/>

在本教程中，我们将构建一个 web 应用程序来从 [NASDAQ-100](http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx) 中获取数据，并用 D3 将其可视化为气泡图。然后最重要的是，我们将通过 Dokku 在数字海洋上部署它。

> **注意:**气泡图非常适合在一个小空间内可视化数百个值。然而，它们更难阅读，因为很难区分相似大小的圆。如果您只处理几个值，条形图可能是更好的选择，因为它更容易阅读。

本教程使用的主要工具:Python v2.7.8、Flask v0.10.1、 [Requests](https://realpython.com/python-requests/) v2.4.1、D3 v3.4.11、Dokku v0.2.3 和 Bower v1.3.9

首先从这个 [repo](https://github.com/realpython/flask-stock-visualizer) 中找到并下载文件 *_app_boilerplate.zip* 。这个文件包含一个烧瓶样本。下载完成后，解压文件和文件夹，激活一个 virtualenv，然后[用 Pip](https://realpython.com/what-is-pip/) 安装依赖项:

```py
pip install -r requirements.txt
```

然后测试以确保它能工作:启动服务器，打开浏览器，并导航到`http://localhost:5000/`。你应该看到“你好，世界！”盯着你。

## 获取数据

在 *app.py* 文件中创建新的路线和查看功能:

```py
@app.route("/data")
def data():
    return jsonify(get_data())
```

更新导入:

```py
from flask import Flask, render_template, jsonify
from stock_scraper import get_data
```

因此，当调用该路由时，它将返回值从名为`get_data()`的函数转换为 JSON，然后返回它。这个函数驻留在一个名为 *stock_scraper.py* 的文件中，这让你大吃一惊！-从纳斯达克 100 指数获取数据。

[*Remove ads*](/account/join/)

### 剧本

将 *stock_scraper.py* 添加到主目录中。

**轮到你了**:按照以下步骤，自己创建脚本:

1.  从[http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?下载 CSV 渲染=下载](http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download)。
2.  从 CSV 中获取相关数据:股票名称、股票代码、当前价格、净变化、百分比变化、成交量和价值。
3.  将解析后的数据转换为 Python 字典。
4.  归还字典。

怎么样了？需要帮助吗？让我们来看一个可能的解决方案:

```py
import csv
import requests

URL = "http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download"

def get_data():
    r = requests.get(URL)
    data = r.text
    RESULTS = {'children': []}
    for line in csv.DictReader(data.splitlines(), skipinitialspace=True):
        RESULTS['children'].append({
            'name': line['Name'],
            'symbol': line['Symbol'],
            'symbol': line['Symbol'],
            'price': line['lastsale'],
            'net_change': line['netchange'],
            'percent_change': line['pctchange'],
            'volume': line['share_volume'],
            'value': line['Nasdaq100_points']
        })
    return RESULTS
```

发生了什么事？

1.  这里，我们通过 GET 请求获取 URL，然后将响应对象`r`转换为 [unicode](https://realpython.com/python-encodings-guide/) 。
2.  然后我们使用`CSV`库将逗号分隔的文本转换成`DictReader()`类的实例，它将数据映射到字典而不是列表。
3.  最后，在遍历数据、创建字典列表(其中每个字典代表不同的股票)之后，我们返回`RESULTS` dict。

> 注意:你也可以使用字典理解来创建个人字典。这是一个更有效的方法，但是你牺牲了可读性。你的电话。

[测试](https://realpython.com/python-testing/)的时间:启动服务器，然后导航到`http://localhost:5000/data`。如果一切顺利，您应该看到一个包含相关股票数据的对象。

有了手头的数据，我们现在可以在前端可视化它。

## 可视化

除了 [HTML 和 CSS](https://realpython.com/html-css-python/) ，我们还将使用 [Bootstrap](http://getbootstrap.com/) 、JavaScript/jQuery 和 [D3](http://d3js.org/) 来驱动我们的前端。我们还将使用客户端依赖管理工具 [Bower](http://bower.io/) 来下载和管理这些库。

**轮到你了**:按照[安装说明](https://github.com/bower/bower#install)在你的机器上设置 Bower。*提示:在安装 Bower* 之前，您需要安装 [Node.js](http://nodejs.org/download/) 。

准备好了吗？

## 鲍尔

需要两个文件来启动 bower-*[bower . JSON](http://bower.io/docs/creating-packages/#bowerjson)*和*。[bower RC](http://bower.io/docs/config/)T7】。*

后一个文件用于配置 Bower。将其添加到主目录:

```py
{ "directory":  "static/bower_components" }
```

这只是指定我们希望依赖项安装在应用程序的*静态*目录中的 *bower_components* 目录(约定)中。

同时，第一个文件 *bower.json* 存储了 bower 清单——这意味着它包含关于 Bower 组件以及应用程序本身的元数据。该文件可通过`bower init`命令交互创建。现在就做。接受所有的默认值。

现在，我们可以安装依赖项了。

```py
$ bower install bootstrap#3.2.0 jquery#2.1.1 d3#3.4.11 --save
```

标志将包添加到 *bower.json* dependencies 数组中。看看这个。此外，确保 *bower.json* 中的依赖版本与我们指定的版本相匹配——即`bootstrap#3.20`。

安装完依赖项后，让我们在应用程序中访问它们。

[*Remove ads*](/account/join/)

### 更新*index.html*T2】*

```py
<!DOCTYPE html>
<html>
  <head>
    <title>Flask Stock Visualizer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href={{ url_for('static', filename='./bower_components/bootstrap/dist/css/bootstrap.min.css') }} rel="stylesheet" media="screen">
    <link href={{ url_for('static', filename='main.css') }} rel="stylesheet" media="screen">
  </head>
  <body>
    <div class="container">
    </div>
    <script src={{ url_for('static', filename='./bower_components/jquery/dist/jquery.min.js') }}></script>
    <script src={{ url_for('static', filename='./bower_components/bootstrap/dist/js/bootstrap.min.js') }}></script>
    <script src={{ url_for('static', filename='./bower_components/d3/d3.min.js') }}></script>
    <script src={{ url_for('static', filename='main.js') }}></script>
  </body>
</html>
```

## D3

有了这么多的数据可视化框架，为什么还要有 [D3](http://d3js.org/) ？D3 是相当低的级别，所以它可以让你构建你想要的框架类型。一旦将数据附加到 DOM，就可以使用 CSS3、HTML5 和 SVG 的组合来创建实际的可视化。然后你可以通过 D3 内置的数据驱动的[转换](https://github.com/mbostock/d3/wiki/Transitions)来增加交互性。

平心而论，这个图书馆并不适合所有人。因为你有很大的自由来构建你想要的东西，所以学习曲线相当高。如果你正在寻找一个快速的开始，看看 [Python-NVD3](https://github.com/areski/python-nvd3) ，它是 D3 的一个包装器，用来使使用 D3 变得非常非常容易。尽管我们在本教程中没有使用它，因为 Python-NVD3 不支持气泡图。

**轮到你了**:浏览 D3 [介绍教程](http://d3js.org/#introduction)。

现在让我们编码。

### 设置

将以下代码添加到 *main.js* 中:

```py
// Custom JavaScript $(function()  { console.log('jquery is working!'); createGraph(); }); function  createGraph()  { // Code goes here }
```

在这里，在初始页面加载之后，我们记录“jquery 正在工作！”然后启动一个名为`createGraph()`的函数。测试一下。启动服务器，然后导航到`http://localhost:5000/`，打开 [JavaScript](https://realpython.com/python-vs-javascript/) 控制台，刷新页面。您应该看到“jquery 正在工作！”如果一切顺利就发短信。

将以下标记添加到*index.html*文件中，在具有`container`的`id`的`<div>`标记内(在第 10 行之后)，以保存 D3 气泡图:

```py
<div id="chart"></div>
```

### 主配置

将以下代码添加到 *main.js* 中的`createGraph()`函数中:

```py
var  width  =  960;  // chart width var  height  =  700;  // chart height var  format  =  d3.format(",d");  // convert value to integer var  color  =  d3.scale.category20();  // create ordinal scale with 20 colors var  sizeOfRadius  =  d3.scale.pow().domain([-100,100]).range([-50,50]);  // https://github.com/mbostock/d3/wiki/Quantitative-Scales#pow
```

请务必查阅代码注释和官方 D3 [文档](https://github.com/mbostock/d3/wiki/API-Reference)。你不明白的地方就去查。*程序员必须自力更生！*

### 气泡配置

```py
var  bubble  =  d3.layout.pack() .sort(null)  // disable sorting, use DOM tree traversal .size([width,  height])  // chart layout size .padding(1)  // padding between circles .radius(function(d)  {  return  20  +  (sizeOfRadius(d)  *  30);  });  // radius for each circle
```

同样，将上述代码添加到`createGraph()`函数中，并检查[文档](https://github.com/mbostock/d3/wiki/Pack-Layout)中的任何问题。

[*Remove ads*](/account/join/)

### SVG Config

接下来，将下面的代码添加到`createGraph()`中，它选择了带有`chart`的`id`的元素，然后附加上圆圈和一些属性:

```py
var  svg  =  d3.select("#chart").append("svg") .attr("width",  width) .attr("height",  height) .attr("class",  "bubble");
```

继续使用`createGraph()`函数，我们现在需要获取数据，这可以用 D3 异步完成。

### 请求数据

```py
// REQUEST THE DATA d3.json("/data",  function(error,  quotes)  { var  node  =  svg.selectAll('.node') .data(bubble.nodes(quotes) .filter(function(d)  {  return  !d.children;  })) .enter().append('g') .attr('class',  'node') .attr('transform',  function(d)  {  return  'translate('  +  d.x  +  ','  +  d.y  +  ')'}); node.append('circle') .attr('r',  function(d)  {  return  d.r;  }) .style('fill',  function(d)  {  return  color(d.symbol);  }); node.append('text') .attr("dy",  ".3em") .style('text-anchor',  'middle') .text(function(d)  {  return  d.symbol;  }); });
```

因此，我们点击前面设置的`/data`端点来返回数据。这段代码的剩余部分只是将气泡和文本添加到 DOM 中。这是标准的样板代码，为我们的数据稍作修改。

### 工具提示

因为我们在图表上的空间有限，仍然在`createGraph()`函数中，所以让我们添加一些工具提示来显示每只特定股票的附加信息。

```py
// tooltip config var  tooltip  =  d3.select("body") .append("div") .style("position",  "absolute") .style("z-index",  "10") .style("visibility",  "hidden") .style("color",  "white") .style("padding",  "8px") .style("background-color",  "rgba(0, 0, 0, 0.75)") .style("border-radius",  "6px") .style("font",  "12px sans-serif") .text("tooltip");
```

这些只是与工具提示相关的 CSS 样式。我们仍然需要添加实际数据。更新我们将圆圈附加到 DOM 的代码:

```py
node.append("circle") .attr("r",  function(d)  {  return  d.r;  }) .style('fill',  function(d)  {  return  color(d.symbol);  }) .on("mouseover",  function(d)  { tooltip.text(d.name  +  ": $"  +  d.price); tooltip.style("visibility",  "visible"); }) .on("mousemove",  function()  { return  tooltip.style("top",  (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px"); }) .on("mouseout",  function(){return  tooltip.style("visibility",  "hidden");});
```

对此进行测试，导航至`http://localhost:5000/`。现在，当你用吸尘器清扫一圈时，你会看到一些底层元数据——公司名称和股票价格。

**轮到你了**:添加更多元数据。你认为还有哪些相关的数据？想想我们在这里展示的——价格的相对变化。您也许可以计算以前的价格并显示:

1.  当前价格
2.  相对变化
3.  先前价格

### 重构

股票如果我们只是想用修改后的市值加权指数——纳斯达克-100 点[列](http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx)——大于 0.1 来想象股票会怎么样？

向`get_data()`函数添加一个条件:

```py
def get_data():
    r = requests.get(URL)
    data = r.text
    RESULTS = {'children': []}
    for line in csv.DictReader(data.splitlines(), skipinitialspace=True):
        if float(line['Nasdaq100_points']) > .01:
            RESULTS['children'].append({
                'name': line['Name'],
                'symbol': line['Symbol'],
                'symbol': line['Symbol'],
                'price': line['lastsale'],
                'net_change': line['netchange'],
                'percent_change': line['pctchange'],
                'volume': line['share_volume'],
                'value': line['Nasdaq100_points']
            })
    return RESULTS
```

现在，让我们在 *main.js* 的气泡配置部分增加每个气泡的半径；相应地修改代码:

```py
// Radius for each circle .radius(function(d)  {  return  20  +  (sizeOfRadius(d)  *  60);  });
```

[*Remove ads*](/account/join/)

### CSS

最后，让我们给 *main.css* 添加一些基本样式:

```py
body  { padding-top:  20px; font:  12px  sans-serif; font-weight:  bold; }
```

好看吗？准备部署了吗？

## 正在部署

Dokku 是一个开源的、类似 Heroku 的平台即服务(PaaS)，由 Docker 提供支持。设置完成后，你可以用 Git 把你的应用程序推送到上面。

我们使用数字海洋作为我们的主机。让我们开始吧。

### 设置数字海洋

如果你还没有帐户，请注册一个帐户。然后按照这个[指南](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2)添加一个公钥。

创建新的 Droplet -指定名称、大小和位置。对于图像，单击“应用程序”选项卡并选择 Dokku 应用程序。确保选择您的 SSH 密钥。

创建完成后，通过在浏览器中输入新创建的 Droplet 的 IP 来完成设置，这将带您进入 Dokku 设置屏幕。确认公钥是正确的，然后单击“完成设置”。

现在 VPS 可以接受推送了。

### 部署配置

1.  用下面的代码创建一个 proc file:`web: gunicorn app:app`。(此文件包含启动 web 进程必须运行的命令。)
2.  安装 gunicorn: `pip install gunicorn`
3.  更新*需求. txt* 文件:`pip freeze > requirements.txt`
4.  初始化一个新的本地 Git repo: `git init`
5.  添加遥控器:`git remote add dokku dokku@192.241.208.61:app_name`(一定要添加自己的 IP 地址。)

### 更新 *app.py* :

```py
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

所以，我们首先尝试从 app 的环境中抓取端口，如果找不到，默认为端口 5000。

确保也更新导入:

```py
import os
```

### 部署！

提交您的更改，然后按:`git push dokku master`。如果一切顺利，您应该在终端中看到应用程序的 URL:

```py
=====> Application deployed:
 http://192.241.208.61:49155
```

测试一下。导航到[http://192.241.208.61:49155](http://192.241.208.61:49155)(同样，确保添加您自己的 IP 地址和正确的端口。)你应该看看你的直播 app！(预览见本文顶部的图片。)

[*Remove ads*](/account/join/)

## 接下来的步骤

想更进一步吗？向应用程序添加以下功能:

1.  错误处理
2.  单元测试
3.  集成测试
4.  持续集成/交付

> 这些功能(以及更多！)将包含在 2014 年 10 月初推出的下一期[真正的 Python](https://realpython.com) 课程中！

如有疑问，请在下方评论。

干杯！*****