# Lyricize:一个使用马尔可夫链创建歌词的 Flask 应用程序

> 原文：<https://realpython.com/lyricize-a-flask-app-to-create-lyrics-using-markov-chains/>

新程序员[是](http://www.reddit.com/r/learnpython/comments/xjlsh/i_just_finished_codeacademys_python_course/) [总](http://www.reddit.com/r/learnpython/comments/zc2c2/done_learning_python_what_next/) [寻找](http://www.reddit.com/r/learnpython/comments/ul1b8/any_good_projects_for_beginners/) [新](http://www.reddit.com/r/learnpython/comments/1a9yie/worked_with_python_for_almost_a_year_now_how_to/) [项目](http://www.reddit.com/r/learnpython/comments/1iqv3c/please_help_me_prepare_a_roadmap_as_to_what_i/)——他们也应该如此！做你自己的兼职项目不仅是获得实践经验的最好方式，而且如果你想从爱好转向职业，那么兼职项目是开始建立工作组合的好方法。

## 从创意到 MVP

在这篇文章中，我们将完成启动一个(最低限度的)MVP 的过程，从最初的概念到一个可共享的原型。最后，你将创建自己版本的 [Lyricize](http://lyricize.herokuapp.com) ，这是一个小应用程序，它使用艺术家或乐队的歌词，根据概率生成“新的”听起来相似的歌词。我们不会呈现典型的“如何复制所有这些代码”教程，而是一步一步地展示思考过程和创作过程中的*实际上*是什么。

> 请注意，这不一定是关于建立下一个杀手级创业公司；我们只是在寻找一个项目，可以 1)一个有趣的学习机会，2)与他人分享。

在我们开始之前，先看一下示例应用程序，看看你将创建什么。本质上，您可以使用马尔可夫链根据特定艺术家的歌词生成新的歌词。例如，尝试搜索“鲍勃·迪伦”，将行数改为三行。很酷，对吧？我刚刚执行了相同的搜索，结果是:

**那边站着你的承诺所有的船**
**我准备好了峡谷**
**我是一个钩子**

好深。不管怎样，让我们开始吧…

* * *

[*Remove ads*](/account/join/)

## 找到感兴趣的话题

所以，第一步:找一个你有兴趣了解更多的话题。下面这款应用的灵感来自于一份旧的[大学作业](http://www.cs.princeton.edu/courses/archive/fall13/cos126/assignments/markov.html)(不可否认，这不是最常见的灵感来源)，它使用马尔可夫链来生成给定大量样本文本的“真实”文本。马尔可夫模型突然出现在[各种场景](http://en.wikipedia.org/wiki/Markov_chain#Applications)。(我们将很快深入什么是马尔可夫模型。)我发现基于概率的文本生成的想法特别有趣；具体来说，我想知道如果您使用歌曲歌词作为样本文本来生成“新”歌词会发生什么…

到网上！快速的网络搜索显示了一些基于马尔可夫的歌词生成器网站，但是没有一个和我想的一样。此外，埋头研究别人完成的代码并不是学习马尔可夫发生器实际工作方式的有效方法；我们自己造吧。

那么……马尔可夫发生器是如何工作的？基本上，马尔可夫链是根据某些模式出现的频率从一些文本中生成的。例如，考虑以下字符串作为我们的示例文本:

```py
bobby
```

我们将从本文中构建最简单的马尔可夫模型，这是一个*阶为 0* 的马尔可夫模型，作为预测任何特定字母出现的可能性的一种方式。这是一个简单明了的频率表:

```py
b: 3/5
o: 1/5
y: 1/5
```

然而，这是一个相当糟糕的语言模型；除了字母出现的频率，我们还想看看给定前一个字母的情况下，某个字母*出现的频率。因为我们依赖于前面的一个字母，这是一个 1 阶马尔可夫模型:*

```py
given "b":
  "b" is next: 1/3
  "o" is next: 1/3
  "y" is next: 1/3
given "o":
  "b" is next: 1
given "y":
  [terminates]: 1
```

从这里，你可以想象更高阶的马尔可夫模型；阶 2 模型将从测量在双字母串“bo”之后出现的每个字母的频率开始，等等。通过增加阶数，我们得到一个看起来更像真实语言的模型；例如，一个 5 阶马尔可夫模型，如果给出了大量样本输入，包括单词“python ”,很可能会在字符串“python”后面加上一个“n ”,而一个低得多的阶模型可能会提出一些创造性的单词。

## 开始开发

我们如何着手建立一个马尔可夫模型的粗略近似？本质上，我们上面用高阶模型概述的结构是字典的字典。您可以想象一个包含各种单词片段(即“bo”)作为关键字的`model`字典。然后，这些片段中的每一个将依次指向一个字典，这些内部字典将各个下一个字母(“y”)作为关键字，将它们各自的频率作为值。

让我们从制作一个`generateModel()`方法开始，该方法接收一些样本文本和一个马尔可夫模型顺序，然后返回这个字典的字典:

```py
def generateModel(text, order):
    model = {}
    for i in range(0, len(text) - order):
        fragment = text[i:i+order]
        next_letter = text[i+order]
        if fragment not in model:
            model[fragment] = {}
        if next_letter not in model[fragment]:
            model[fragment][next_letter] = 1
        else:
            model[fragment][next_letter] += 1
    return model
```

我们遍历所有可用的文本，直到最后一个可用的完整片段+下一个字母，以便不超出字符串的结尾，将我们的`fragment`字典添加到`model`中，每个`fragment`保存一个总`next_letter`频率的字典。

将该函数复制到 Python shell 中，并进行试验:

>>>

```py
>>> generateModel("bobby", 1)
{'b': {'y': 1, 'b': 1, 'o': 1}, 'o': {'b': 1}}
```

那就行了！我们有频率的计数，而不是相对概率，但我们可以利用它；没有理由我们需要标准化每个字典来增加 100%的概率。

现在让我们在一个`getNextCharacter()`方法中使用这个`model`,该方法将在给定一个模型和一个片段的情况下，根据给定的模型概率决定一个合适的下一个字母:

```py
from random import choice
def getNextCharacter(model, fragment):
    letters = []
    for letter in model[fragment].keys():
        for times in range(0, model[fragment][letter]):
            letters.append(letter)
    return choice(letters)
```

这不是最有效的设置，但它构建起来很简单，并且目前有效。我们简单地建立了一个字母列表，给出它们在片段后出现的总频率，并从列表中随机选择。

剩下的就是在第三种方法中使用这两种方法，这种方法实际上会生成指定长度的文本。要做到这一点，我们需要在添加新字符时跟踪我们正在构建的当前文本片段:

```py
def generateText(text, order, length):
    model = generateModel(text, order)

    currentFragment = text[0:order]
    output = ""
    for i in range(0, length-order):
        newCharacter = getNextCharacter(model, currentFragment)
        output += newCharacter
        currentFragment = currentFragment[1:] + newCharacter
    print output
```

让我们把它变成一个完整的可运行脚本，它以马尔可夫顺序和输出文本长度作为参数:

```py
from random import choice
import sys

def generateModel(text, order):
    model = {}
    for i in range(0, len(text) - order):
        fragment = text[i:i+order]
        next_letter = text[i+order]
        if fragment not in model:
            model[fragment] = {}
        if next_letter not in model[fragment]:
            model[fragment][next_letter] = 1
        else:
            model[fragment][next_letter] += 1
    return model

def getNextCharacter(model, fragment):
    letters = []
    for letter in model[fragment].keys():
        for times in range(0, model[fragment][letter]):
            letters.append(letter)
    return choice(letters)

def generateText(text, order, length):
    model = generateModel(text, order)
    currentFragment = text[0:order]
    output = ""
    for i in range(0, length-order):
        newCharacter = getNextCharacter(model, currentFragment)
        output += newCharacter
        currentFragment = currentFragment[1:] + newCharacter
    print output

text = "some sample text"
if __name__ == "__main__":
    generateText(text, int(sys.argv[1]), int(sys.argv[2]))
```

现在，我们将通过非常科学的方法生成示例文本，即根据一些复制粘贴的 Alanis Morisette 歌词，将字符串直接放入代码中。

[*Remove ads*](/account/join/)

## 测试

保存脚本并尝试一下:

```py
$ python markov.py 2 100
I wounts
You ho's humortel whime
 mateend I wass
How by Lover
```

```py
$ python markov.py 4 100
stress you to cosmic tears
All they've cracked you (honestly) at the filler in to like raise
$ python markov.py 6 100
tress you place the wheel from me
Please be philosophical
Please be tapped into my house
```

那真是太珍贵了。最后两次试听是她歌词的体面代表(虽然 order 2 的第一个样本看起来更像 bjrk)。这些结果对于一个快速的代码草图来说是足够令人鼓舞的，所以让我们把这个东西变成一个真正的项目。

## 下一次迭代

第一个障碍:我们如何自动获取大量歌词？一种选择是有选择地从歌词网站抓取内容，但这听起来像是为可能的低质量结果付出了很多努力，加上考虑到大多数歌词聚合器的可疑性和音乐行业的严酷性，潜在的法律灰色地带。相反，让我们看看是否有开放的 API。前往通过[programmableweb.com](http://www.programmableweb.com/search/lyrics)搜索，我们实际上发现 *14* 不同的歌词 API 列表。不过，这些列表并不总是最新的，所以让我们通过最近列出的来搜索。

[LYRICSnMUSIC](http://www.lyricsnmusic.com/api) 提供免费的、 [RESTful API](https://realpython.com/api-integration-in-python/) 使用 [JSON](https://realpython.com/python-json/) 返回多达 150 个字符的歌词。这对于我们的用例来说听起来很完美，特别是考虑到大多数歌曲的重复；只要一个样本就够了，没有必要收集完整的歌词。去拿一个新的[键](http://www.lyricsnmusic.com/api_keys/new)，这样你就可以访问他们的 API 了。

在我们永久确定这个来源之前，让我们先尝试一下他们的 API。根据他们的文档，我们可以提出一个示例请求，如下所示:

```py
http://api.lyricsnmusic.com/songs?api_key=[YOUR_API_KEY_HERE]&artist=coldplay
```

它在浏览器中返回的 JSON 结果有点难以阅读；通过它们在一个[格式器](http://jsonformatter.curiousconcept.com/)中进行更好的查看。看起来我们成功地取回了基于 Coldplay 歌曲的字典列表:

```py
[ { "title":"Don't Panic", "url":"http://www.lyricsnmusic.com/coldplay/don-t-panic-lyrics/4294612", "snippet":"Bones sinking like stones \r\nAll that we've fought for \r\nHomes, places we've grown \r\nAll of us are done for \r\n\r\nWe live in a beautiful world \r\nYeah we ...", "context":null, "viewable":true, "instrumental":false, "artist":{ "name":"Coldplay", "url":"http://www.lyricsnmusic.com/coldplay" } }, { "title":"Shiver", "url":"http://www.lyricsnmusic.com/coldplay/shiver-lyrics/4294613", "snippet":"So I look in your direction\r\nBut you pay me no attention, do you\r\nI know you don't listen to me\r\n'Cause you say you see straight through me, don't you...", "context":null, "viewable":true, "instrumental":false, "artist":{ "name":"Coldplay", "url":"http://www.lyricsnmusic.com/coldplay" } }, ... ]
```

没有办法限制响应，但是我们只对提供的每个“片段”感兴趣，这对这个项目来说看起来很好。

我们对马尔可夫发生器的初步实验是有教育意义的，但是我们当前的模型并不是最适合生成歌词的任务。一方面，我们可能应该使用单个单词作为我们的标记，而不是一个字符一个字符地看待事物；试图模仿语言本身很有趣，但为了生成假歌词，我们将希望坚持真正的英语。这听起来有点棘手，但是，我们已经走了很长的路来理解马尔可夫链是如何运作的，这是该练习的最初目标。在这一点上，我们到达了一个十字路口:为了更多的学习，重新发明隐喻的轮子(这可能是伟大的编码实践)，或者看看其他人已经创造了什么。

我选择了懒惰的方式，回去搜索内部网络。GitHub 上的一个善良的灵魂已经[实现了](https://github.com/MaxWagner/PyMarkovChain)一个基本的基于单词的马尔可夫链，甚至上传到了 [PyPI](https://pypi.python.org/pypi/PyMarkovChain/) 。快速浏览一下[代码](https://github.com/ketaro/markov-cranberries/blob/master/markov.py)，似乎这个模型只有 0 阶。这对于我们自己构建可能已经足够快了，而高阶模型可能需要更多的工作。就目前来说，我们还是用别人的预包装车轮吧；如果我们使用完整的单词，至少 0 阶模型不会听起来像 bjrk。

因为我们想轻松地与朋友和家人分享我们的创作，所以把它变成一个 web 应用程序是有意义的。现在，选择一个网络框架。就我个人而言，我是迄今为止对 Django 最熟悉的，但这在这里似乎有点过头了；毕竟，我们甚至不需要自己的数据库。让我们试试烧瓶。

## 添加烧瓶

按照惯例，[启动一个虚拟环境](https://realpython.com/python-virtual-environments-a-primer/) -如果你还没有的话！如果这不是一个熟悉的过程，看看我们的[以前的帖子](https://realpython.com/python-web-applications-with-flask-part-i/#toc_5)来学习如何设置。

```py
$ mkdir lyricize
$ cd lyricize
$ virtualenv --no-site-packages venv
$ source venv/bin/activate
```

还是像往常一样，安装必要的需求，并将它们放在 requirements.txt 文件中:

```py
$ pip install PyMarkovChain flask requests
$ pip freeze > requirements.txt
```

我们还添加了[请求](http://docs.python-requests.org/en/latest/)库，这样我们就可以向歌词 API 发出 web 请求。

现在，制作应用程序。为了简单起见，让我们将它分成两个页面:主页面将向用户呈现一个基本表单，用于选择艺术家姓名和要生成的多行歌词，而第二个“歌词”页面将呈现结果。让我们从一个名为 **app.py** 的准系统 Flask 应用程序开始，它使用了一个**index.html**模板:

```py
from flask import Flask, render_template

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
```

到目前为止，这个应用程序所能做的就是加载 index.html 模板的内容。让我们把它做成一个基本形式:

```py
<html>
 <body>
  <form action="#" method="post" class="lyrics">
    Artist or band name: <input name="artist" type="text" /><br />
    Number of lines:
    <select name="lines">
      {% for n in range(1,11) %}
        <option value="{{n}}">{{n}}</option>
      {% endfor %}
    </select>
    <br /><br />
    <input class="button" type="submit" value="Lyricize">
  </form>
 </body>
</html>
```

将这个 index.html 保存在一个名为 *templates* 的单独文件夹中，以便 Flask 可以找到它。这里我们使用 Flask 的 Jinja2 模板来创建一个“选择”下拉列表，它基于一个覆盖数字 1 到 10 的循环。在我们添加任何其他内容之前，启动此页面以确保我们设置正确:

```py
$ python app.py
* Running on http://127.0.0.1:5000/
```

您现在应该能够在浏览器中访问 http://127.0.0.1:5000/

现在让我们决定我们想要在结果页面上显示什么，以便我们知道我们需要传递给它什么:

```py
<html>
 <body>
  <div align="center" style="padding-top:20px;">
   <h2>
   {% for line in result %}
     {{ line }}<br />
   {% endfor %}
   </h2>
   <h3>{{ artist }}</h3>
   <br />
   <form action="{{ url_for('index') }}">
    <input type="submit" value="Do it again!" />
   </form>
  </div>
 </body>
</html>
```

这里我们循环遍历一个**结果**数组，逐行显示每一行。在下面，我们显示被选中的**艺术家**，并链接回主页。在你的/templates 目录下，将此文件另存为【lyrics.html*。

我们还需要更新 index.html 的表单操作，以指向这个结果页面:

```py
<form action="{{ url_for('lyrics') }}" method="post" class="lyrics">
```

现在为生成的歌词页面写一条路线:

```py
@app.route('/lyrics', methods=['POST'])
def lyrics():
    artist = request.form['artist']
    lines = int(request.form['lines'])

    if not artist:
        return redirect(url_for('index'))

    return render_template('lyrics.html', result=['hello', 'world'], artist=artist)
```

这个页面从表单中获取 POST 请求，解析出提供的**艺术家**和**行数**——我们还没有生成任何歌词，只是给模板一个虚拟的结果列表。我们还需要添加我们所依赖的必要的 Flask 功能- `url_for`和`redirect`:

```py
from flask import Flask, render_template, url_for, redirect
```

测试一下，确保没有任何损坏:

```py
$ python app.py
```

太好了，现在是项目的实质内容。在歌词()中，让我们根据传入的艺术家参数从 LYRICSnMUSIC 获得一个响应:

```py
# Get a response of sample lyrics from the provided artist
uri = "http://api.lyricsnmusic.com/songs"
params = {
    'api_key': API_KEY,
    'artist': artist,
}
response = requests.get(uri, params=params)
lyric_list = response.json()
```

使用请求，我们获取一个特定的 URL，其中包含一个参数字典:提供的艺术家姓名和我们的 API 键。这个私有 API 密钥应该*而不是*出现在您的代码中；毕竟，您会希望与他人共享这些代码。相反，让我们创建一个单独的文件来保存这个值作为一个变量:

```py
$ echo "API_KEY=[youractualapikeygoeshere]" > .env
```

我们已经创建了一个特殊的“环境”文件，如果我们只需在应用程序顶部添加以下内容，Flask 就可以读取该文件:

```py
import os
API_KEY = os.environ.get('API_KEY')
```

最后，让我们添加马尔可夫链功能。既然我们正在使用别人的包，这最终变得相当琐碎。首先，在顶部添加导入:

```py
from pymarkovchain import MarkovChain
```

然后，在我们从 API 收到一个**歌词**响应后，我们简单地创建一个 MarkovChain，加载歌词数据，并生成一个句子列表:

```py
mc = MarkovChain()
mc.generateDatabase(lyrics)

result = []
for line in range(0, lines):
    result.append(mc.generateString())
```

总的来说，app.py 现在应该是这样的:

```py
from flask import Flask, url_for, redirect, request, render_template
import requests
from pymarkovchain import MarkovChain
import os

API_KEY = os.environ.get('API_KEY')

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/lyrics', methods=['POST'])
def lyrics():
    artist = request.form['artist']
    lines = int(request.form['lines'])

    if not artist:
        return redirect(url_for('index'))

    # Get a response of sample lyrics from the artist
    uri = "http://api.lyricsnmusic.com/songs"
    params = {
        'api_key': API_KEY,
        'artist': artist,
    }
    response = requests.get(uri, params=params)
    lyric_list = response.json()

    # Parse results into a long string of lyrics
    lyrics = ''
    for lyric_dict in lyric_list:
        lyrics += lyric_dict['snippet'].replace('...', '') + ' '

    # Generate a Markov model
    mc = MarkovChain()
    mc.generateDatabase(lyrics)

    # Add lines of lyrics
    result = []
    for line in range(0, lines):
        result.append(mc.generateString())

    return render_template('lyrics.html', result=result, artist=artist)

if __name__ == '__main__':
    app.run()
```

试试吧！一切都应该在本地工作。现在，与世界分享它…

[*Remove ads*](/account/join/)

## 部署到 Heroku

让我们在 Heroku 上托管，因为(对于这些最低要求)我们可以免费这么做。为此，我们需要对代码做一些小的调整。首先，添加一个 [Procfile](https://devcenter.heroku.com/articles/procfile) ，它将告诉 Heroku 如何为应用程序提供服务:

```py
$ echo "web: python app.py" > Procfile
```

接下来，因为 Heroku 指定了一个运行应用程序的随机端口，所以您需要在顶部传递一个端口号:

```py
PORT = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
app.config.from_object(__name__)
```

当应用程序运行时，确保通过这个端口

```py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
```

我们还必须指定“0.0.0.0”的主机，因为 Flask 默认情况下在本地计算机上秘密运行，而我们希望该应用程序在 Heroku 上公开可用的 IP 上运行。

最后，从代码中删除`app.debug=True`,这样如果出错，用户就不会看到完整的 stacktrace 错误。

初始化一个 git 存储库(如果你还没有的话)，创建一个新的 Heroku 应用程序，并把你的代码放进去！

```py
$ git init
$ git add .
$ git commit -m "First commit"
$ heroku create
$ git push heroku master
```

请参见 Heroku 文档以了解此部署流程的更详细的概要。确保在 Heroku 上添加 API_KEY 变量:

```py
$ heroku config:set API_KEY=[youractualapikeygoeshere]
```

我们都准备好了！是时候与世界分享你的创作了——或者继续黑下去:)

## 结论和后续步骤

如果你喜欢这个内容，你可能会对我们目前学习网页开发的课程感兴趣，或者对我们最新的 T2 Kickstarter 感兴趣，它涵盖了更先进的技术。或者——在这里玩一下应用程序。

**可能的后续步骤:**

*   这个 HTML 看起来像是 90 年代初写的；使用 [Bootstrap](https://github.com/mbr/flask-bootstrap) 或者只是一些基本的 CSS 样式
*   在忘记代码的作用之前，给代码添加一些注释！(这是留给读者的一个练习:o)
*   将歌词路径中的代码抽象为单独的方法(即，一个从歌词 API 返回响应的方法和另一个用于生成马尔可夫模型的独立方法)；随着代码规模和复杂性的增长，这将使代码更容易维护和测试
*   创建一个能够使用更高阶的马尔可夫发生器
*   使用 [Flask-WTF](https://flask-wtf.readthedocs.org/en/latest/) 改进表格和表格验证
*   说到这里:让它更安全！现在，有人可能会发送不合理的 POST 请求，将他们自己的代码注入页面，或者用许多快速重复的请求来拒绝站点；添加一些可靠的输入验证和基本速率限制
*   添加更好的错误处理；如果一个 API 调用时间太长或者由于某种原因失败了怎么办？
*   将结果放入一个文本到语音转换引擎，学习使用另一个马尔可夫模型改变音调模式，并设置为一个节拍；很快你就会在排行榜上名列前茅！****