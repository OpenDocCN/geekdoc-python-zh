# 用 Bottle、SQLAlchemy 和 Twitter API 构建一个简单的 Web 应用

> 原文：<https://realpython.com/building-a-simple-web-app-with-bottle-sqlalchemy-twitter-api/>

去年 10 月，我们向 PyBites 的观众发出挑战，要求他们制作一个网络应用程序，以便更好地浏览[每日 Python 提示](https://twitter.com/python_tip)提要。在这篇文章中，我将分享我一路构建和学习到的东西。

在本文中，您将了解到:

1.  如何克隆项目回购，设置 app？
2.  如何通过 Tweepy 模块使用 Twitter API 加载推文。
3.  如何使用 SQLAlchemy 来存储和管理数据(提示和标签)。
4.  如何使用类似 Flask 的微型 web 框架 Bottle 构建一个简单的 web 应用程序。
5.  如何使用 pytest 框架添加测试？
6.  更好的 Code Hub 指南如何带来更易维护的代码。

如果你想继续下去，详细阅读代码(并可能有所贡献)，我建议你分叉[repo](https://github.com/pybites/pytip)。让我们开始吧。

## 项目设置

首先，*名称空间是一个非常棒的想法*，所以让我们在一个[虚拟环境中工作](https://realpython.com/python-virtual-environments-a-primer/)。使用 Anaconda，我像这样创建它:

```py
$ virtualenv -p <path-to-python-to-use> ~/virtualenvs/pytip
```

在 Postgres 中创建生产和测试数据库:

```py
$  psql psql  (9.6.5,  server  9.6.2) Type  "help"  for  help. #  create  database  pytip; CREATE  DATABASE #  create  database  pytip_test; CREATE  DATABASE
```

我们需要证书来连接到数据库和 Twitter [API](https://realpython.com/api-integration-in-python/) ( [首先创建一个新的应用](https://apps.twitter.com/))。根据最佳实践，配置应该存储在环境中，而不是代码中。将以下 env 变量放在*~/virtualenvs/py tip/bin/activate*的末尾，该脚本处理虚拟环境的激活/停用，确保为您的环境更新这些变量:

```py
export DATABASE_URL='postgres://postgres:password@localhost:5432/pytip'
# twitter
export CONSUMER_KEY='xyz'
export CONSUMER_SECRET='xyz'
export ACCESS_TOKEN='xyz'
export ACCESS_SECRET='xyz'
# if deploying it set this to 'heroku'
export APP_LOCATION=local
```

在同一个脚本的 deactivate 函数中，我取消了它们的设置，因此在停用(离开)虚拟环境时，我们将事情排除在 shell 范围之外:

```py
unset DATABASE_URL
unset CONSUMER_KEY
unset CONSUMER_SECRET
unset ACCESS_TOKEN
unset ACCESS_SECRET
unset APP_LOCATION
```

现在是激活虚拟环境的好时机:

```py
$ source ~/virtualenvs/pytip/bin/activate
```

克隆存储库，在启用虚拟环境的情况下，安装要求:

```py
$ git clone https://github.com/pybites/pytip && cd pytip
$ pip install -r requirements.txt
```

接下来，我们导入包含以下内容的推文集合:

```py
$ python tasks/import_tweets.py
```

然后，验证表是否已创建，tweets 是否已添加:

```py
$  psql \c  pytip pytip=#  \dt List  of  relations Schema  |  Name  |  Type  |  Owner --------+----------+-------+----------
  public  |  hashtags  |  table  |  postgres public  |  tips  |  table  |  postgres (2  rows) pytip=#  select  count(*)  from  tips; count -------
  222 (1  row) pytip=#  select  count(*)  from  hashtags; count -------
  27 (1  row) pytip=#  \q
```

现在让我们[运行测试](https://realpython.com/python-testing/):

```py
$ pytest
========================== test session starts ==========================
platform darwin -- Python 3.6.2, pytest-3.2.3, py-1.4.34, pluggy-0.4.0
rootdir: realpython/pytip, inifile:
collected 5 items

tests/test_tasks.py .
tests/test_tips.py ....

========================== 5 passed in 0.61 seconds ==========================
```

最后，运行瓶子应用程序:

```py
$ python app.py
```

浏览到 http://localhost:8080 ,瞧:你应该会看到按受欢迎程度降序排列的提示。点击左边的标签链接，或者使用搜索框，你可以很容易地过滤它们。在这里我们看到了*熊猫*的提示举例:

[![PyBites: Daily Python tips](img/f696ac1c0bbd9925ac89b215a47ff42e.png)](https://files.realpython.com/media/pytip-pandas.9487f06c45d6.png)

> 我用 [MUI](https://www.muicss.com/) 做的设计——一个轻量级 CSS 框架，遵循 Google 的材料设计准则。

[*Remove ads*](/account/join/)

## 实施细节

### DB 和 SQLAlchemy

我使用 [SQLAlchemy](https://realpython.com/python-sqlite-sqlalchemy/) 与数据库接口，以避免不得不编写大量(冗余的) [SQL](https://realpython.com/python-sql-libraries/) 。

在 *[tips/models.py](https://github.com/pybites/pytip/blob/master/tips/models.py)* 中，我们定义了我们的模型- `Hashtag`和`Tip`-SQLAlchemy 将映射到 DB 表:

```py
from sqlalchemy import Column, Sequence, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Hashtag(Base):
    __tablename__ = 'hashtags'
    id = Column(Integer, Sequence('id_seq'), primary_key=True)
    name = Column(String(20))
    count = Column(Integer)

    def __repr__(self):
        return "<Hashtag('%s', '%d')>" % (self.name, self.count)

class Tip(Base):
    __tablename__ = 'tips'
    id = Column(Integer, Sequence('id_seq'), primary_key=True)
    tweetid = Column(String(22))
    text = Column(String(300))
    created = Column(DateTime)
    likes = Column(Integer)
    retweets = Column(Integer)

    def __repr__(self):
        return "<Tip('%d', '%s')>" % (self.id, self.text)
```

在 *[tips/db.py](https://github.com/pybites/pytip/blob/master/tips/db.py)* 中，我们导入了这些模型，现在很容易使用 db，例如与`Hashtag`模型接口:

```py
def get_hashtags():
    return session.query(Hashtag).order_by(Hashtag.name.asc()).all()
```

并且:

```py
def add_hashtags(hashtags_cnt):
    for tag, count in hashtags_cnt.items():
        session.add(Hashtag(name=tag, count=count))
    session.commit()
```

### 查询 Twitter API

我们需要从 Twitter 上检索数据。为此，我创建了*[tasks/import _ tweets . py](https://github.com/pybites/pytip/blob/master/tasks/import_tweets.py)*。我将它打包在*任务*下，因为它应该在每日 cronjob 中运行，以寻找新的提示并更新现有推文的统计数据(点赞和转发的数量)。为了简单起见，我每天都重新创建表格。如果我们开始依赖与其他表的 FK 关系，我们肯定应该选择 update 语句而不是 delete+add。

我们在项目设置中使用了这个脚本。让我们更详细地看看它是做什么的。

首先，我们创建一个 API 会话对象，并将其传递给 [tweepy。光标](http://docs.tweepy.org/en/v3.5.0/cursor_tutorial.html)。API 的这个特性非常好:它处理分页，遍历时间轴。对于小费的数量——在我写这篇文章的时候是 222 英镑——这真的很快。`exclude_replies=True`和`include_rts=False`参数很方便，因为我们只想要每日 Python Tip 自己的推文(而不是转发推文)。

从提示中提取标签只需要很少的代码。

首先，我为标签定义了一个正则表达式:

```py
TAG = re.compile(r'#([a-z0-9]{3,})')
```

然后，我用`findall`获取所有标签。

我把它们交给了[收藏。计数器](https://docs.python.org/3.7/library/collections.html),它返回一个类似 dict 的对象，标签作为键，并作为值计数，按值降序排列(最常见)。我排除了太常见的 python 标记，它会扭曲结果。

```py
def get_hashtag_counter(tips):
    blob = ' '.join(t.text.lower() for t in tips)
    cnt = Counter(TAG.findall(blob))

    if EXCLUDE_PYTHON_HASHTAG:
        cnt.pop('python', None)

    return cnt
```

最后，*[tasks/import _ tweets . py](https://github.com/pybites/pytip/blob/master/tasks/import_tweets.py)*中的`import_*`函数实际导入 tweets 和 hashtags，调用 *tips* 目录/包的`add_*` DB 方法。

[*Remove ads*](/account/join/)

### 用瓶子做一个简单的 web 应用

有了这些准备工作，制作一个 web 应用程序变得异常简单(如果你以前使用过 Flask 的话，这并不奇怪)。

首先认识[瓶](https://bottlepy.org/docs/dev/):

*Bottle 是一个快速、简单、轻量级的 [WSGI](http://www.wsgi.org/) 微型 web 框架，用于 [Python](https://www.python.org) 。它作为一个单独的文件模块发布，除了 [Python 标准库](https://docs.python.org/3/library/)之外，没有任何依赖关系。*

很好。最终的 web 应用程序由< 30 LOC and can be found in [app.py](https://github.com/pybites/pytip/blob/master/app.py) 组成。

对于这个简单的应用程序，只需要一个带有可选标记参数的方法。类似于 Flask，路由是由 decorators 处理的。如果用标签调用，它过滤标签上的提示，否则显示所有提示。视图装饰器定义要使用的模板。像 Flask(和 Django)一样，我们返回一个用于模板的字典。

```py
@route('/')
@route('/<tag>')
@view('index')
def index(tag=None):
    tag = tag or request.query.get('tag') or None
    tags = get_hashtags()
    tips = get_tips(tag)

    return {'search_tag': tag or '',
            'tags': tags,
            'tips': tips}
```

根据[文档](https://bottlepy.org/docs/dev/tutorial.html#static-files)，要处理静态文件，您需要在导入之后的顶部添加以下代码片段:

```py
@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static')
```

最后，我们希望确保我们只在本地主机上以[调试模式](https://bottlepy.org/docs/dev/tutorial.html#tutorial-debugging)运行，因此我们在项目设置中定义了`APP_LOCATION` env 变量:

```py
if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True, reloader=True)
```

### 瓶子模板

Bottle 自带了一个快速、强大、易学的内置模板引擎，名为 [SimpleTemplate](https://bottlepy.org/docs/dev/stpl.html) 。

在[视图](https://github.com/pybites/pytip/tree/master/views)子目录中，我定义了一个 *header.tpl* 、 *index.tpl* 和 *footer.tpl* 。对于标签云，我使用了一些简单的内联 CSS 来增加标签大小，参见 *[header.tpl](https://github.com/pybites/pytip/blob/master/views/header.tpl)* :

```py
% for tag in tags:
  <a style="font-size: {{ tag.count/10 + 1 }}em;" href="/{{ tag.name }}">#{{ tag.name }}</a>&nbsp;&nbsp;
% end
```

在 *[index.tpl](https://github.com/pybites/pytip/blob/master/views/index.tpl)* 中我们循环了一下提示:

```py
% for tip in tips:
  <div class='tip'>
    <pre>{{ !tip.text }}</pre>
    <div class="mui--text-dark-secondary"><strong>{{ tip.likes }}</strong> Likes / <strong>{{ tip.retweets }}</strong> RTs / {{ tip.created }} / <a href="https://twitter.com/python_tip/status/{{ tip.tweetid }}" target="_blank">Share</a></div>
  </div>
% end
```

如果你熟悉 Flask 和 Jinja2，这应该看起来很熟悉。嵌入 Python 甚至更容易，输入更少— `(% ...` vs `{% ... %}`。

所有的 css、图片(和 JS，如果我们会用到的话)都放在 [static](https://github.com/pybites/pytip/tree/master/static) 子文件夹中。

这就是用 Bottle 制作一个基本 web 应用程序的全部内容。一旦正确定义了数据层，事情就简单多了。

[*Remove ads*](/account/join/)

### 用 pytest 添加测试

现在，让我们通过添加[一些测试](https://github.com/pybites/pytip/tree/master/tests)来使这个项目更健壮一点。测试 DB 需要更深入地研究 [pytest](https://realpython.com/pytest-python-testing/) 框架，但是我最终使用了 [pytest.fixture](https://docs.pytest.org/en/latest/fixture.html) decorator 来建立和拆除一个包含一些测试 tweets 的数据库。

我没有调用 Twitter API，而是使用了 *[tweets.json](https://github.com/pybites/pytip/blob/master/tests/tweets.json)* 中提供的一些静态数据。并且，我没有使用 live DB，而是在 *[tips/db.py](https://github.com/pybites/pytip/blob/master/tips/db.py)* 中，检查 pytest 是否是调用者(`sys.argv[0]`)。如果是这样，我使用测试数据库。我可能会对此进行重构，因为 [Bottle 支持使用配置文件](https://bottlepy.org/docs/dev/configuration.html)。

hashtag 部分更容易测试(`test_get_hashtag_counter`)，因为我可以在多行字符串中添加一些 hash tag。不需要固定装置。

### 代码质量很重要——更好的代码中枢

[更好的代码中枢](https://bettercodehub.com/)指导你写，嗯，更好的代码。在编写测试之前，项目得分为 7:

[![BCH screenshot #1](img/e7affa5c14511188f9ea8a7d284049fc.png)](https://files.realpython.com/media/bch1.f5045e9b9392.png)

不错，但我们可以做得更好:

1.  我通过使代码更加模块化，将 DB 逻辑从 app.py (web 应用程序)中取出，放在 tips 文件夹/包中(重构 [1](https://github.com/pybites/pytip/commit/3e9909284899fdf01bb15b01dea7db7e6ff0e7e8) 和 [2](https://github.com/pybites/pytip/commit/c09bb7a69deb992ed379599d36c23f49527ce693) )，将它提升到了 9

2.  然后，随着测试的进行，该项目获得了 10:

[![BCH screenshot #2](img/65b1fba5347b5751edec9b3ee096f425.png)](https://files.realpython.com/media/bch2.5b64d7596c90.png)

## 结论和学习

我们的[代码挑战#40](https://pybit.es/codechallenge40_review.html) 提供了一些好的实践:

1.  我建立了一个有用的应用程序，可以扩展(我想添加一个 API)。
2.  我用了一些很酷的值得探索的模块: [Tweepy](https://realpython.com/twitter-bot-python-tweepy/) ，SQLAlchemy，和 Bottle。
3.  我学习了更多的 pytest，因为我需要夹具来测试与数据库的交互。
4.  最重要的是，必须使代码可测试，应用程序变得更加模块化，这使得它更容易维护。更好的 Code Hub 在这个过程中帮助很大。
5.  我使用我们的[分步指南](https://pybit.es/deploy-flask-heroku.html)将应用程序部署到 [Heroku](https://pytip.herokuapp.com/) 。

### 我们挑战你

学习和提高你的编码技能[的最好方法是练习](https://pybit.es/learn-by-doing.html)。在 PyBites，我们通过组织 Python 代码挑战巩固了这一概念。查看我们的[成长收藏](https://pybit.es/pages/challenges.html)、[叉回购](https://github.com/pybites/challenges)，获取编码！

通过对你的作品提出[拉取请求](https://github.com/pybites/challenges/blob/master/INSTALL.md)，让我们知道你是否创造了一些很酷的东西。我们已经看到人们在这些挑战中竭尽全力，我们也是如此。

编码快乐！***