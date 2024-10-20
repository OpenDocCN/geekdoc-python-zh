# 通过 Python 和 Tweepy 使用 Twitter

> 原文：<https://www.blog.pythonlibrary.org/2019/08/06/using-twitter-with-python-and-tweepy/>

Twitter 是一个流行的社交网络，人们用它来相互交流。Python 有几个可以用来与 Twitter 交互的包。这些软件包对于创建 Twitter 机器人或下载大量数据进行离线分析非常有用。其中一个比较流行的 Python Twitter 包叫做 [Tweepy](https://www.tweepy.org/) 。在本文中，您将学习如何使用 Tweepy 和 Twitter。

Tweepy 让您可以访问 Twitter 的 API，它公开了以下内容(还有更多！):

*   小鸟叫声
*   转发
*   喜欢
*   直接消息
*   追随者

这允许你用 Tweepy 做很多事情。让我们来看看如何开始吧！

* * *

### 入门指南

您需要做的第一件事是创建一个 Twitter 帐户，并获得访问 Twitter 所需的凭证。为此，你需要在这里申请一个开发者账户。

创建之后，您可以获得或生成以下内容:

*   消费者 API 密钥
*   消费者 API 秘密
*   访问令牌
*   访问机密

如果您丢失了这些项目，您可以返回您的开发者帐户，重新生成新的项目。也可以撤销旧的。

注意:默认情况下，您收到的访问令牌是只读的。如果你想用 Tweepy 发送推文，那么你需要确保将**应用类型**设置为“读写”。

* * *

### 安装 Tweepy

接下来，您需要安装 Tweepy 包。这是通过使用 pip 实现的:

```py
pip install tweepy

```

现在您已经安装了 Tweepy，您可以开始使用它了！

* * *

### 使用 Tweepy

你可以使用 Tweepy 以编程的方式在 Twitter 上做几乎任何事情。例如，您可以使用 Tweepy 来获取和发送推文。您可以使用它来访问有关用户的信息。你可以转发、关注/取消关注、发帖等等。

让我们看一个获取用户家庭时间表的例子:

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

tweets = api.home_timeline()
for tweet in tweets:
    print('{real_name} (@{name}) said {tweet}\n\n'.format(
        real_name=tweet.author.name, name=tweet.author.screen_name,
        tweet=tweet.text))

```

这里的前几行代码是您在 Twitter 开发者个人资料页面上找到的凭证。实际上不建议在您的代码中硬编码这些值，但是为了简单起见，我在这里这样做了。如果您要创建共享的代码，您会希望让代码的用户将他们的凭证导出到他们的环境中，并使用 Python 的`os.getenv()`或通过使用`argparse`的命令行参数。

接下来，您通过创建一个`OAuthHandler()`登录 Twitter

对象并使用恰当命名的`set_access_token()`设置访问令牌

功能。然后你可以创建一个`API()`

允许您访问 Twitter 的实例。

在这种情况下，您调用`home_timeline()`

它会返回您家庭时间轴中的前二十条推文。这些推文来自你的朋友或关注者，也可能是 Twitter 决定在你的时间表中推广的随机推文。在这里，您可以打印出作者的姓名、Twitter 句柄以及他们的推文文本。

让我们来看看您是如何使用`api`获取自己的信息的

您之前创建的对象:

```py
>>> me = api.me()
>>> me.screen_name
'driscollis'
>>> me.name
'Mike Driscoll'
>>> me.description
('Author of books, blogger @mousevspython and Python enthusiast. Also part of '
 'the tutorial team @realpython')

```

您可以使用`api`

来获得关于你自己的信息。上面的代码演示了如何获取您的屏幕名称、实际名称以及您在 Twitter 上设置的描述。你可以得到比这更多的东西。比如你可以获得你的关注者，时间线等。

* * *

### 获取推文

使用 Tweepy 获取推文也很容易。如果你知道别人的用户名，你可以获得自己或别人的推文。

让我们从获取您的推文开始:

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
my_tweets = api.user_timeline()
for t in my_tweets:
    print(t.text)

```

在这里，您像在上一节中一样连接到 Twitter。然后你叫`user_timeline()`

以获得对象列表，然后对其进行迭代。在这种情况下，您最终只能打印出 tweet 的文本。

让我们尝试获取一个特定的用户。在这种情况下，我们将使用一个相当受欢迎的程序员，凯利沃恩:

```py
>>> user = api.get_user('kvlly')
>>> user.screen_name
'kvlly'
>>> user.name
'Kelly Vaughn ðŸž'
>>> for t in tweets:
        print(t.text)

```

她发了很多微博，所以我不会在这里复制她的微博。然而正如你所看到的，获得一个用户是很容易的。你需要做的就是通过`get_user()`

一个有效的 Twitter 用户名，然后你就可以访问该用户的任何公开信息。

* * *

### 发送推文

看推文很好玩，但是发呢？Tweepy 也可以为您完成这项任务。

让我们来看看如何实现:

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

api.update_status("This is your tweet message")

```

您应该关注的主要代码是最后一行:`update_status()`

。这里你传入一个字符串，它是 tweet 消息本身。只要没有错误，您应该会在您的 Twitter 时间线中看到这条推文。

现在我们来学习如何发送附有照片的推文:

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

api.update_with_media('/path/to/an/image.jpg',
                      "This is your tweet message")

```

在这种情况下，你需要使用`update_with_media()`

方法，它接受一个到您想要上传的图片的路径和一个字符串，也就是 tweet 消息。

请注意，虽然 **update_with_media()** 易于使用，但也不推荐使用。

所以让我们更新这个例子，改为使用 **media_upload** ！

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

media_list = []
response = api.media_upload('/home/mdriscoll/Downloads/sebst.png')
media_list.append(response.media_id_string)
api.update_status("This is your tweet message", media_ids=media_list)

```

在这里，您将媒体上传到 Twitter，并从返回的响应中获取它的 id 字符串。然后将该字符串添加到 Python 列表中。最后，您使用 **update_status()** 来发布一些东西，但是您也将 **media_ids** 参数设置为您的图片列表。

* * *

### 列出关注者

我要讲的最后一个话题是如何在 Twitter 上列出你的关注者。

让我们来看看:

```py
import tweepy

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

followers = api.followers()
for follower in followers:
    print(follower.screen_name)

```

此代码将列出您帐户的最新 20 个关注者。你只需要打电话给`followers()`

去拿名单。如果你需要得到的不仅仅是最新的 20 个，你可以使用`count`

参数来指定需要多少个结果。

* * *

### 包扎

使用 Tweepy，您可以做更多事情。例如，你可以获得赞，发送和阅读直接消息，上传媒体，等等。这是一个非常好和易于使用的软件包。您还可以使用 Tweepy 实时读写 Twitter，这允许您创建一个 Twitter 机器人。如果你还没有尝试过 Tweepy，你绝对应该试一试。很好玩啊！

* * *

### 相关阅读

*   Tweepy 的[文档](http://docs.tweepy.org/en/3.7.0/index.html)
*   如何用 Tweepy - [真正的 Python](https://realpython.com/twitter-bot-python-tweepy/) 用 Python 制作一个 Twitter 机器人