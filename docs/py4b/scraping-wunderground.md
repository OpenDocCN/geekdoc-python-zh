# 刮痧地下

> 原文：<https://www.pythonforbeginners.com/scraping/scraping-wunderground>

## 概观

使用 API 既有趣又有教育意义。

谷歌、Reddit 和 Twitter 等许多公司向公众发布了它的 API，这样开发者就可以开发基于其服务的产品。

与 API 一起工作让你学会了引擎盖下的螺母和螺栓。

在这篇文章中，我们将研究地下天气 API。

## 地下天气

我们将建立一个应用程序，将连接到' [Wunderground](https://www.wunderground.com/ "wunderground") '和检索。
天气预报等。

Wunderground 提供全球各地的本地和长期天气预报、天气报告、
地图&热带天气状况。

## 应用程序接口

API 是一种协议，旨在被软件组件用作相互通信的接口。API 是一组编程指令和
标准，用于访问基于 web 的软件应用程序(如上所述)。

借助 API，应用程序可以在没有任何用户知识或
干预的情况下相互对话。

## 入门指南

当我们想要使用一个 API 时，我们需要做的第一件事是查看
公司是否提供任何 API 文档。既然我们想为
神童世界写一份申请，我们就去神童世界[网站](https://www.wunderground.com/ "wunderground")

在页面底部，您应该会看到“面向开发人员的天气 API”。

## API 文档

大多数 API 特性都需要一个 API 密匙，所以在开始使用天气 API 之前，让我们先注册一个
密匙。

在文档中，我们还可以看到 API 请求是通过 HTTP
发出的，数据特性返回 JSON 或 XML。

要阅读完整的 API 文档，请参见此[链接](https://www.wunderground.com/weather/api/d/docs?MR=1 "api_docs_wu")。

在获得密钥之前，我们需要先创建一个免费帐户。

## API 密钥

下一步是注册 API 密匙。只需填写您的姓名、电子邮件地址、
项目名称和网站，您就可以开始了。

互联网上的许多服务(如 Twitter、脸书..)要求你
有一个“API 密匙”。

应用编程接口密钥(API key)是由调用 API 的
计算机程序传入的代码，用于向网站标识调用程序、其开发者、
或其用户。

API 密钥用于跟踪和控制 API 的使用方式，例如
防止恶意使用或滥用 API。

API 密钥通常既作为唯一的标识符，又作为用于
认证的秘密令牌，并且通常具有一组对与之相关联的 API
的访问权限。

## 美国城市的现状

Wunderground 在他们的 API 文档中为我们提供了一个例子。

美国城市的现状

```py
http://api.wunderground.com/api/0def10027afaebb7/conditions/q/CA/San_Francisco.json 
```

如果您点击“显示响应”按钮，或者将该 URL 复制并粘贴到您的
浏览器中，您应该会看到类似这样的内容:

```py
{
	"response": {
		"version": "0.1"
		,"termsofService": "http://www.wunderground.com/weather/api/d/terms.html"
		,"features": {
		"conditions": 1
		}
	}
		,	"current_observation": {
		"image": {
		"url":"http://icons-ak.wxug.com/graphics/wu2/logo_130x80.png",
		"title":"Weather Underground",
		"link":"http://www.wunderground.com"
		},
		"display_location": {
		"full":"San Francisco, CA",
		"city":"San Francisco",
		"state":"CA",
		"state_name":"California",
		"country":"US",
		"country_iso3166":"US",
		"zip":"94101",
		"magic":"1",
		"wmo":"99999",
		"latitude":"37.77500916",
		"longitude":"-122.41825867",
		"elevation":"47.00000000"
		},
		..... 
```

## 锡达拉皮兹的现状

在“代码示例”[页面](https://www.wunderground.com/weather/api/d/docs?d=resources/code-samples&MR=1 "code_samples")上，我们可以看到检索锡达拉皮兹
当前温度的完整 Python 代码。

复制并粘贴到你最喜欢的编辑器，并保存为任何你喜欢的。

请注意，您必须用自己的 API 密钥替换“0def10027afaebb7”。

```py
import urllib2
import json
f = urllib2.urlopen('http://api.wunderground.com/api/0def10027afaebb7/geolookup/conditions/q/IA/Cedar_Rapids.json')
json_string = f.read()

parsed_json = json.loads(json_string)

location = parsed_json['location']['city']

temp_f = parsed_json['current_observation']['temp_f']

print "Current temperature in %s is: %s" % (location, temp_f)

f.close() 
```

要在终端中运行程序:

```py
python get_current_temp.py 
```

您的程序将返回锡达拉皮兹的当前温度:

锡达拉皮兹现在的温度是 68.9 度

## 下一步是什么？

既然我们已经查看并测试了 Wunderground 提供的示例，让我们自己创建一个程序。

地下天气为我们提供了一大堆
可以利用的[数据特征](https://www.wunderground.com/weather/api/d/docs?d=data/index&MR=1 "datafeatures")。

重要的是，你要通读那里的信息，了解如何使用不同的功能。

#### 标准请求 URL 格式

*"大多数 API 特性都可以使用以下格式访问。*

*注意，几个特性可以组合成一个请求。"*

http://api.wunderground.com/api/0def10027afaebb7/features/settings/q/query.format

**其中:**

**0def10027afaebb7:** 您的 API 密钥

**特性:**以下一个或多个数据特性

**设置(可选):**示例:lang:FR/pws:0

**查询:**您想要获取天气信息的位置

**格式:** json，或 xml

我想做的是检索巴黎的天气预报。

预测功能返回未来 3 天的天气摘要。

这包括高温和低温，字符串文本预报和条件。

## 巴黎天气预报

要检索巴黎的天气预报，我首先要找到法国的国家代码，我可以在这里找到:

[各国天气](https://www.wunderground.com/weather-by-country.asp "weather_by_country")

下一步是在 API 文档中寻找“特性:预测”。

我们需要的字符串可以在这里找到:

[http://www.wunderground.com/weather/api/d/docs?d =数据/预测](https://www.wunderground.com/weather/api/d/docs?d=data/forecast "forecast_api")

通过阅读文档，我们应该能够构建一个 URL。

### 进行 API 调用

我们现在有了需要的 URL，可以开始我们的程序了。

现在是时候对 Weather Underground 进行 API 调用了。

注意:在这个程序中，我们将使用“[请求](https://www.pythonforbeginners.com/requests/requests-in-python "requests")”模块，而不是像我们在上面的例子中那样使用 urllib2 模块。

使用“请求”模块进行 API 调用非常容易。

```py
r = requests.get("http://api.wunderground.com/api/your_api_key/forecast/q/France/
Paris.json") 
```

现在，我们有一个名为“r”的响应对象。我们可以从这个物体中获得我们需要的所有信息。

### 创建我们的应用程序

打开您选择的编辑器，在第一行，导入请求模块。

注意，requests 模块带有一个内置的 JSON 解码器，我们可以对 JSON 数据使用
。这也意味着，我们不必导入 JSON
模块(就像我们在前面的例子中使用 urllib2 模块时所做的那样)

```py
import requests 
```

为了开始提取我们需要的信息，我们首先要看到“r”对象返回给我们什么键。

下面的代码将返回键，并应返回[u'response '，u'forecast']

```py
import requests

r = requests.get("http://api.wunderground.com/api/your_api_key/forecast/q/France/
Paris.json")

data = r.json()

print data.keys() 
```

### 获取我们想要的数据

将 URL(从上面)复制并粘贴到 JSON 编辑器中。

我使用[http://jsoneditoronline.org/](http://jsoneditoronline.org/ "jsoneditoronline")，但是任何 JSON 编辑器都应该做这项工作。

这将显示所有数据的更简单的概览。

[http://API . wunderground . com/API/your _ API _ key/forecast/q/France/Paris . JSON](https://api.wunderground.com/api/your_api_key/forecast/q/France/Paris.json "json_url")

请注意，通过输入以下命令，可以通过终端获得相同的信息:

```py
r = requests.get("http://api.wunderground.com/api/your_api_key/forecast/q/France/
Paris.json")
print r.text 
```

检查给我们的输出后，我们可以看到我们感兴趣的数据在“forecast”键中。回到我们的程序，从那个键中打印出
数据。

```py
import requests

r = requests.get("http://api.wunderground.com/api/your_api_key/forecast/q/France/
Paris.json")

data = r.json()

print data['forecast'] 
```

结果存储在变量“数据”中。

为了访问我们的 JSON 数据，我们简单地使用括号符号，就像这样:
data['key']。

让我们通过添加“简单预测”来浏览更多的数据

```py
import requests

r = requests.get("http://api.wunderground.com/api/your_api_key/forecast/q/France/
Paris.json")

data = r.json()

print data['forecast']['simpleforecast'] 
```

我们的产出仍然有点多，但是坚持住，我们就快到了。

我们程序的最后一步是添加['forecastday']，而不是打印出每一个条目，我们将使用 for 循环来遍历字典。

我们可以像这样访问任何我们想要的东西，只要查找你感兴趣的数据。

在这个程序中，我想得到巴黎的天气预报。

让我们看看代码是什么样子的。

```py
import requests

r = requests.get("http://api.wunderground.com/api/0def10027afaebb7/forecast/q/France/Paris.json")
data = r.json()

for day in data['forecast']['simpleforecast']['forecastday']:
    print day['date']['weekday'] + ":"
    print "Conditions: ", day['conditions']
    print "High: ", day['high']['celsius'] + "C", "Low: ", day['low']['celsius'] + "C", '
' 
```

运行程序。

$ python get_temp_paris.py

```py
Monday:
Conditions:  Partly Cloudy
High:  23C Low:  10C

Tuesday:
Conditions:  Partly Cloudy
High:  23C Low:  10C

Wednesday:
Conditions:  Partly Cloudy
High:  24C Low:  14C

Thursday:
Conditions:  Mostly Cloudy
High:  26C Low:  15C 
```

预测功能只是众多功能之一。剩下的我就交给你去探索了。

一旦你理解了一个 API 和它在 JSON 中的输出，你就明白了大多数 API 是如何工作的。

#### 更多阅读

[Python API 综合列表](https://www.pythonforbeginners.com/api/list-of-python-apis "pythonapi")
[地下天气](https://www.wunderground.com/ "wunderground_api")