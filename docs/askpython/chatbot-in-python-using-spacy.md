# 使用 spaCy NLP 库在 Python 中构建聊天机器人

> 原文：<https://www.askpython.com/python/examples/chatbot-in-python-using-spacy>

读者你好！欢迎来到本教程，在这里我们将用 python 构建一个天气机器人，它将用自然语言与用户互动。没有任何进一步的到期让我们开始吧。

***必读:[NLP 简介](https://www.askpython.com/python/examples/introduction-to-nlp)***

## 什么是聊天机器人？

你们一定都访问过一个网站，在那里有一条信息说“嗨！我能帮你什么”然后我们点击它，开始和它聊天。你有没有想过谁和我们互动？嗯，它是智能软件，与我们互动，并回应我们的查询。

让我们再举一个现实生活中的例子，比如苹果公司的 Siri、亚马逊公司的 Alexa、谷歌助手等等。每当我们说“Alexa，在 Spotify 上播放我的音乐播放列表”，你的音乐播放列表就会开始播放。这些智能助理使用人工智能和机器学习，并接受过用户提供的各种输入的训练。

聊天机器人可以执行各种任务，如预订火车票，提供特定主题的信息，寻找你附近的餐馆等。聊天机器人就是为用户完成这些任务而创建的，让他们从自己搜索这些信息中解脱出来。

在本教程中，您将使用 **[spacy NLP 库](https://spacy.io/)** 创建一个聊天机器人，它可以告诉用户城市的当前天气，并且能够用自然语言与用户交谈。这个聊天机器人将使用 OpenWeather API 告诉用户世界上任何一个城市的当前天气。

***推荐阅读*** : [自然语言处理 Top 5 Python 库](https://www.askpython.com/python/top-python-libraries-for-natural-language-processing)

## 用 Python 创建聊天机器人的先决条件

*   最新版本的 Python 可以从[https://www.python.org/downloads/](https://www.python.org/downloads/)下载
*   在本教程中，我们将为 OpenWeather 使用一个 API 键。要获取 API 密钥，请访问 [OpenWeather](https://home.openweathermap.org/) 并创建一个帐户。请确认您的电子邮件地址。注册成功后，请访问 API 密钥部分，查看为您的帐户生成的 API 密钥。这个 API 密钥是一个字母数字字符序列。

满足上述要求后，我们就可以进入下一步了。

## 安装库

在本教程中，我们将需要两个库`**[spacy](https://www.askpython.com/python/examples/pos-tagging-in-nlp-using-spacy)**`和`**[requests](https://www.askpython.com/python-modules/requests-in-python)**`。空间库将帮助你的聊天机器人理解用户的句子，请求库将允许聊天机器人发出 HTTP 请求。

安装`spacy`:

```py
pip install -U spacy

```

接下来，我们将下载 spacy 的英语语言模型:

```py
python -m spacy download en_core_web_md

```

如果出现以下错误，则需要安装`wheel`:

```py
Output
ERROR: Failed building wheel for en-core-web-md

```

安装车轮:

```py
pip install -U wheel

```

然后，再次下载英语语言模型。

要确认 spacy 安装正确，请在终端中执行以下命令打开 Python 解释器:

```py
python

```

现在，导入空间并加载英语语言模型:

```py
>>> import spacy
>>> nlp = spacy.load("en_core_web_md")

```

如果这两条语句正确执行，则 spacy 安装成功。您可以关闭 python 解释器:

```py
>>> exit()

```

`**requests**`库预装了 Python。如果在导入请求模块时收到错误消息，则需要安装库:

```py
pip install requests

```

## 创建聊天机器人

好了，安装了上面的库，我们可以开始编码了。

### 步骤 1–创建天气函数

在这里，我们将创建一个函数，机器人将使用它来获取一个城市的当前天气。

打开您最喜欢的 IDE，并将以下代码添加到 python 文件中:

```py
import requests

api_key = "your_api_key"

def get_weather(city_name):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)

    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]

    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None

```

让我们来理解代码！

首先，我们导入 **`requests`** 库，这样我们就可以发出 HTTP 请求并使用它们。在下一行中，您必须用为您的帐户生成的 API 密钥替换`**your_api_key**`。

接下来，我们定义一个函数`**get_weather**()`，它将城市的名称作为参数。在函数内部，我们为 OpenWeather API 构造了 URL。我们将通过这个 URL 发出 get 请求。URL 以 JSON 格式返回城市的天气信息。之后，我们使用`requests.get()`函数向 API 端点发出 GET 请求，并将结果存储在响应变量中。之后，使用`response.json()`将 GET 请求的结果转换成 Python 字典。我们这样做是为了方便访问。

接下来，我们将天气条件提取成一个 [**天气变量**](https://www.askpython.com/python/examples/gui-weather-app-in-python) 。

接下来，我们处理一些条件。在 **`if`** 块中，我们确保 API 响应的状态代码为 200(这意味着我们成功获取了天气信息)并返回天气描述。

如果请求有问题，错误代码会打印到控制台，并且不会返回任何内容。

以上就是关于 get_weather()函数的全部内容。现在，让我们用一些输入来测试这个函数。将代码粘贴到您的 IDE 中，并用为您的帐户生成的 api 密钥替换 **your_api_key** 。

### 代码片段

```py
import requests

def get_weather(city_name):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)

    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]

    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None

weather = get_weather("Patna")
print(weather)

```

### 输出

```py
mist

```

多神奇啊！我们有一个函数可以获取世界上任何一个城市的天气情况。

### 步骤 2–创建聊天机器人功能

在这里，我们将创建一个功能正常的聊天机器人，它使用`get_weather`()函数获取一个城市的天气状况，使用 **spacy NLP 库**用自然语言与用户进行交互。将以下代码片段添加到前面的代码中。您不需要为此创建新文件。

首先，我们将导入空间库并加载英语语言模型:

```py
import spacy

nlp = spacy.load("en_core_web_md")

```

之后，`get_weather()`添加以下代码:

```py
weather = nlp("Weather Conditions in a city")
def chatbot(statement):
  statement = nlp(statement)

```

在上面的代码片段中，变量`weather`和`statement`被标记化，这对空间计算用户输入`statement`和`weather`之间的语义相似度是必要的。聊天机器人函数将`statement`作为一个参数，它将与存储在变量天气中的句子进行比较。

接下来，我们将使用 spaCy 库的 similarity()函数。similarity()方法计算两个语句的语义相似度，并给出一个介于 0 和 1 之间的值，其中数字越大表示相似度越大。该功能用于使聊天机器人变得智能，以便它可以将用户给出的句子与基本句子进行比较，并给出所需的输出。当我们测试聊天机器人时，情况会变得更加清楚🙂

但是，我们必须为相似性设置一个最小值，以使聊天机器人决定用户希望通过输入语句了解城市的温度。所以，我们把最小值设为 0.75。您可以根据您的项目需求更改该值。

到目前为止，我们的代码是这样的:

```py
import spacy

nlp = spacy.load("en_core_web_md")

weather = nlp("Weather Conditions in a city")

def chatbot(statement):
  statement = nlp(statement)
  min_similarity = 0.75

```

现在是本教程的最后也是最有趣的部分。我们将用户输入与存储在变量`weather`中的基本句子进行比较，我们还将从用户给出的句子中提取城市名称。

**添加以下代码:**

```py
if weather.similarity(statement) >= min_similarity:

    for ent in statement.ents:
      if ent.label_ == "GPE": # GeoPolitical Entity
        city = ent.text

        city_weather = get_weather(city)
        if city_weather is not None:
          return "In " + city +", the current weather is: " + city_weather
        else:
          return "Something went wrong."
      else:
        return "You need to tell me a city to check."

else:
    return "Sorry I don't understand that. Please rephrase your statement."

```

为了提取命名实体，我们使用 spaCy 的[命名实体识别](https://spacy.io/usage/linguistic-features#named-entities)特性。为了提取城市的名称，使用了一个循环来遍历 spaCy 从用户输入中提取的所有实体，并检查实体标签是否为“GPE”(地理政治实体)。如果是，那么我们将实体的名称存储在变量`city`中。一旦提取了城市名称，就调用 get_weather()函数，城市作为参数传递，返回值存储在变量`city_weather`中。

现在，如果 get_weather()函数成功地获取了天气，那么它将被传递给用户，否则如果发生了错误，将向用户显示一条消息。

至此，您终于使用 spaCy 库创建了一个聊天机器人，它可以理解用户用自然语言输入的内容，并给出想要的结果。

## 完整的聊天机器人程序代码

```py
import spacy
import requests

nlp = spacy.load("en_core_web_md")

api_key = "019947b686adde825c5c6104b3e13d7e"

def get_weather(city_name):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)

    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]

    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None

weather = nlp("Weather Conditions in a city")
def chatbot(statement):
  statement = nlp(statement)
  min_similarity = 0.75

  if weather.similarity(statement) >= min_similarity:

    for ent in statement.ents:
      if ent.label_ == "GPE": # GeoPolitical Entity
        city = ent.text

        city_weather = get_weather(city)
        if city_weather is not None:
          return "In " + city +", the current weather is: " + city_weather
        else:
          return "Something went wrong."
      else:
        return "You need to tell me a city to check."

  else:
    return "Sorry I don't understand that. Please rephrase your statement."

print("Hi! I am Windy a weather bot.........")
statement = input("How can I help you ?\n")

response = chatbot(statement)
print(response)

```

我们已经要求聊天机器人提供比哈尔邦的天气情况，让我们看看我们得到了什么输出:

```py
Hi! I am Windy a weather bot.........
How can I help you ?
How is the weather in Bihar
In Bihar, the current weather is: broken clouds

```

## 摘要

看到我们的聊天机器人给我们提供天气情况真的很有趣。请注意，我用自然语言询问聊天机器人，聊天机器人能够理解并计算输出。

最后，您已经创建了一个聊天机器人，并且您可以向它添加许多功能。

## 参考

[空间](https://spacy.io/)