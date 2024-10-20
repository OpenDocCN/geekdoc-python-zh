# 如何用 Python 连接 Twitter

> 原文：<https://www.blog.pythonlibrary.org/2014/09/26/how-to-connect-to-twitter-with-python/>

有几个第三方包包装了 Twitter 的 API。我们将关注 tweepy 和 T2 的 twitter。tweepy [文档](http://tweepy.readthedocs.org/en/v2.3.0/)比 twitter 的更广泛一些，但是我觉得 twitter 包有更多具体的例子。让我们花一些时间来看看如何使用这些软件包！

* * *

### 入门指南

要开始使用 Twitter 的 API，您需要创建一个 Twitter 应用程序。为此，你必须去他们的开发者网站创建一个新的应用程序。创建应用程序后，您将需要获得您的 API 密钥(或者生成一些)。您还需要生成您的访问令牌。

因为 Python 中没有包含这两个包，所以您也需要安装它们。要安装 tweepy，只需执行以下操作:

```py

pip install tweepy

```

要安装 twitter，你可以做同样的事情:

```py

pip install twitter

```

现在你应该准备好出发了！

* * *

### 发布状态更新

你应该能够用这些包做的一个基本事情是在你的 Twitter 账户上发布一个更新。让我们看看这两个软件包在这方面是如何工作的。我们从十二层开始。

```py

import tweepy

auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(token, token_secret)
client = tweepy.API(auth)
client.update_status("#Python Rocks!")

```

嗯，这是相当直截了当的。我们必须用我们的密钥创建一个 OAuth 处理程序，然后设置访问令牌。最后，我们创建了一个表示 Twitter API 的对象，并更新了状态。这个方法对我很有效。现在让我们看看我们是否能让 twitter 包工作。

```py

import twitter

auth=twitter.OAuth(token, token_secret, key, secret)
client = twitter.Twitter(auth=auth)
client.statuses.update(status="#Python Rocks!")

```

这段代码也非常简单。事实上，我认为 twitter 包的 OAuth 实现比 tweepy 的更干净。

注意:我在使用 twitter 包时有时会得到以下错误:**错误的认证数据，代码 215** 。我不完全确定为什么当你查找这个错误时，它应该是因为你使用 Twitter 的旧 API 而引起的。如果是这样的话，那么它就永远不会起作用。

接下来，我们将看看如何获得我们的时间线。

* * *

### 获取时间线

在这两个包中获得你自己的 Twitter 时间表真的很容易。让我们来看看 tweepy 的实现:

```py

import tweepy

auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(token, token_secret)
client = tweepy.API(auth)

timeline = client.home_timeline()

for item in timeline:
    text = "%s says '%s'" % (item.user.screen_name, item.text)
    print text

```

所以在这里我们得到认证，然后我们调用 **home_timeline()** 方法。这将返回一个对象的 iterable，我们可以循环遍历这些对象并从中提取各种数据。在这种情况下，我们只提取屏幕名称和 Tweet 的文本。让我们看看 twitter 包是如何做到这一点的:

```py

import twitter

auth=twitter.OAuth(token, token_secret, key, secret)
client = twitter.Twitter(auth=auth)

timeline = client.statuses.home_timeline()
for item in timeline:
    text = "%s says '%s'" % (item["user"]["screen_name"],
                             item["text"])
    print text

```

twitter 包非常相似。主要区别在于它返回一个字典列表。

如果你想得到别人的时间线。在十二岁时，你会这样做:

```py

import tweepy

auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(token, token_secret)
client = tweepy.API(auth)

user = client.get_user(screen_name='pydanny')
timeline = user.timeline()

```

twitter 包有点不同:

```py

import twitter

auth=twitter.OAuth(token, token_secret, key, secret)
client = twitter.Twitter(auth=auth)

user_timeline = client.statuses.user_timeline(screen_name='pydanny')

```

在这种情况下，我认为 twitter 包更干净一些，尽管有人可能会说 tweepy 的实现更直观。

* * *

### 让你的朋友和追随者

几乎每个人在 Tritter 上都有朋友(他们追随的人)和追随者。在本节中，我们将了解如何访问这些项目。twitter 包并没有一个很好的例子来寻找你的 Twitter 朋友和追随者，所以在这一节我们将只关注 tweepy。

```py

import tweepy

auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(token, token_secret)
client = tweepy.API(auth)

friends = client.friends()

for friend in friends:
    print friend.name

```

如果您运行上面的代码，您会注意到它打印出来的最大朋友数是 20。如果你想打印出你所有的朋友，那么你需要使用光标。使用光标有两种方式。您可以使用它来返回页面或特定数量的项目。在我的例子中，我关注了 32 个人，所以我选择了 items 的工作方式:

```py

for friend in tweepy.Cursor(client.friends).items(200):
    print friend.name

```

这段代码将迭代多达 200 项。如果你有很多朋友或者你想迭代别人的朋友，但是不知道他们有多少，那么使用 pages 方法更有意义。让我们来看看这可能是如何工作的:

```py

for page in tweepy.Cursor(client.friends).pages():
    for friend in page:
        print friend.name

```

这很简单。获取您的关注者列表完全相同:

```py

followers = client.followers()
for follower in followers:
    print follower.name

```

这也将只返回 20 个项目。我有很多追随者，所以如果我想获得他们的列表，我必须使用上面提到的光标方法之一。

* * *

### 包扎

这些包提供了比本文所涵盖的更多的功能。我特别推荐看 tweepy，因为使用 Python 的自省工具比 twitter 包更直观、更容易理解。如果你还没有一堆这样的应用程序，你可以很容易地使用 tweepy，并围绕它创建一个用户界面，让你与你的朋友保持同步。另一方面，对于初学者来说，这仍然是一个很好的程序。

* * *

### 附加阅读

*   tweepy - [github](https://github.com/tweepy/tweepy) 或[读取文档](http://tweepy.readthedocs.org/en/v2.3.0/)
*   推特上的 [PyPI](https://pypi.python.org/pypi/twitter)
*   其他 Python Twitter [客户端](https://dev.twitter.com/overview/api/twitter-libraries)