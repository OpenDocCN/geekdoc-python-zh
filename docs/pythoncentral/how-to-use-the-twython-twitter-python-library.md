# 如何使用 Twython (Twitter) Python 库

> 原文：<https://www.pythoncentral.io/how-to-use-the-twython-twitter-python-library/>

Python 是一种非常通用的语言，具有非常多样化的库生态系统。这就是为什么他们的座右铭是“包括电池”-男孩，他们是对的！使用 Python，你可以与包括脸书、Twitter 和 LinkedIn 在内的各种社交网络联系和互动。

今天，我想向您展示如何使用 Twython 和 Python 在 Twitter 上执行一些基本任务。

Twython 是一个非常健壮、成熟的 Twitter 库。已经维持了 2 年多。它由其核心团队积极维护，并定期从社区收到补丁。这不是一个 alpha 状态库；它经过测试，并在许多商业应用中使用。

你可以在这里找到 Twython 的 Github 页面:[https://github.com/ryanmcgrath/twython](https://github.com/ryanmcgrath/twython "Twython")。您将在库页面上看到库的源代码和贡献者列表。他们还提供了一个简单的。py 文件下载给你使用。

## 安装 Python 的 Twython

*简单安装*将为您获取最新版本，并将其安装在 Python 能够访问的地方。

如果你不熟悉*简易安装*，这是一个与`setuptools`捆绑在一起的简易 Python 模块(`easy_install`)，可以让你自动下载、构建、安装和管理 Python 包。官方 Python 3.x 和 Python 2.x 发行版中包含了简易安装。

安装 *Easy Install* 后，只需运行下面的命令，让它下载并打包最新版本的 Twython:

```py

easy_install twython

```

## 创建 Twython 对象

让我们从导入库并创建一个 Twython 对象开始。我们将在所有与 Twitter 的交互中使用这个对象。

```py

from twython import Twython

twitter = Twython()

```

## 获取 Twitter 用户的时间表

在我们的第一个演示中，我们将执行开发人员希望从 Twitter 获得的最常见的功能之一:*获取用户的时间表*。作为一个例子，我将搜索我自己的时间线。方法如下:

```py

from twython import Twython

twitter = Twython()

# First, let's grab a user's timeline. Use the

# 'screen_name' parameter with a Twitter user name.

user_timeline = twitter.getUserTimeline(screen_name="pythoncentral")

```

## 在 Twython 中打印用户推文

现在我们有了用户的时间线，我们可以使用 Twython 来显示单个 Tweets。因为`user_timeline`变量只是一个集合，你可以跳过或获取你需要的 Tweets 数量。

```py

# And print the tweets for that user.

for tweet in user_timeline:

print(tweet['text'])

```

## 运行 Twython 脚本

保存您的脚本(在我的例子中，它被命名为`twythonExample.py`)，并使用 Python 命令运行它:

```py

python twythonExample.py

```

您应该会在控制台上看到一些推文。如果没有，请仔细检查 Twython 库是否正确安装在 Python 安装的 Libs 文件夹中。

代码本身是不言自明的；每条推文中都有丰富的数据。在本例中，我们将进入并打印“text”元素。

使用 Twython，让 Twitter 交互更简单！

## 更多 Twython 的东西来尝试

### 获取 Gravatar 图像

现在让我们尝试获取用户的 Gravatar 图像。

```py

from twython import Twython

twitter = Twython()

```

抓取用户的 Gravatar 图标非常简单。你也可以决定你想要多大尺寸的 gravatar。

```py

print(twitter.getProfileImageUrl('pythoncentral', size='bigger'))

print(twitter.getProfileImageUrl('pythoncentral'))

```

如您所见，我们可以请求特定大小的 gravatar，或者如果您只是想要默认大小，则根本不需要请求大小。

### 用 Twython 搜索 Twitter

来点推特搜索怎么样？Twython 让这个问题变得无关紧要。

```py

from twython import Twython

twitter = Twython()

# Some basic search functionality.

print(twitter.search(q='pythoncentral'))

```

您应该会在控制台上看到一组 Tweets 结果，其中包含您编写的查询。最重要的是，这个过程真的很快，在搜索过程中开销最小。

### 获取每日和每周的 Twitter 趋势

最后，Twython 提供了一种获取每日和每周趋势的方法。两者都很容易调用:

```py

from twython import Twython

twitter = Twython()

# Displaying the Daily Trends.

trends = twitter.getDailyTrends()

print(trends)

# Displaying the Weekly Trends.

weeklyTrends = twitter.getWeeklyTrends()

printweeklyTrends

```

Twython 是一个非常强大的库，有一个很棒的贡献者团队已经为它工作了很长时间。如果您在 Python 中有任何 Twitter 需求，我会强烈推荐这个库。