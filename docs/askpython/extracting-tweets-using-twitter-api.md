# 使用 API 和 Python 从 Twitter 中提取 tweets

> 原文：<https://www.askpython.com/python/examples/extracting-tweets-using-twitter-api>

读者们好，在这篇文章中，我将向你们介绍 Twitter API，即 Tweepy，它用于使用 Python 检索推文。我希望你会喜欢阅读这篇文章。

## 使用 Python 从 Twitter 中提取 Tweets 的要求

让我们先来看看我们需要做些什么。

### 1.Twitter 开发者账户

为了访问 Tweepy API，创建一个开发者帐户是很重要的，这个帐户必须得到 twitter 的批准。因此，请确保您提供了正确的细节和使用 Tweepy 的适当理由。

以下是创建开发者账户的方法。

*   访问位于 dev.twitter.com 的 twitter 开发者网站。
*   点击右上角的“登录”按钮，在开发者网站上创建一个帐户。

![Twitter Developer](img/afd0b3795c18ae1084eaf3263bc382a6.png)

Twitter Developer site

*   登录后，点击导航栏上的开发者链接。
*   点击你的账户，从出现的下拉菜单中选择“应用程序”。

![Image 7](img/e4cc7e2a737781034d868c0973aa6dc8.png)

Drop-down

*   点击“创建应用程序”按钮，并填写您的应用程序的详细信息。
*   为应用程序创建访问令牌。将此访问令牌复制到一个文件中，并确保其安全。
*   完成后，记下您的 OAuth 设置，包括——消费者密钥、消费者密码、OAuth 访问令牌、OAuth 访问令牌密码。

### 2.电子表格阅读器软件

你需要一个可以阅读电子表格的软件，比如微软 Excel 或者 LibreOffice Reader。

## 从 Twitter 中提取推文的代码

在这个编码示例中，我们将使用 Tweepy 从 twitter.com 提取数据。

### 1.导入所需的库并设置 OAuth 令牌

因此，首先，导入必要的库，如 tweepy 和 pandas，并声明 OAuth 令牌，该令牌是在 twitter 开发人员仪表板上创建应用程序时获得的。

```py
from tweepy import *

import pandas as pd
import csv
import re 
import string
import preprocessor as p

consumer_key = <enter your consumer key>
consumer_secret = <enter key>
access_key= <enter key>
access_secret = <enter key>

```

### 2.使用 Tweepy 的 OAuthhandler 授权

现在我们已经定义了密钥，我们将继续用 tweepy 的 OAuthHandler 授权我们自己。我们将如下所示传递密钥。

```py
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

```

我们现在将这些授权细节传递给 tweepy，如下所示。

```py
api = tweepy.API(auth,wait_on_rate_limit=True)

```

### 3.从 Twitter 中提取特定的推文

您可以通过名称`search_words`定义一个变量，并指定您想要检索 tweets 的单词。

Tweepy 检查所有 tweets 中的特定关键字并检索内容。这可以是标签、@提及，甚至是普通的单词。

有时，甚至转发都是摘录，为了避免这种情况，我们过滤了转发。

```py
search_words = "#"      #enter your words
new_search = search_words + " -filter:retweets"

```

现在，对于 Tweepy 光标中的每条 Tweepy，我们搜索单词并传递它，如下所示。然后，我们将内容写入一个 csv 文件，如 utf-8 编码后所示。

### 4.拉推元数据

在下面的代码片段中，我希望只检索 tweet 的创建时间、文本、用户名和位置。

```py
for tweet in tweepy.Cursor(api.search,q=new_search,count=100,
                           lang="en",
                           since_id=0).items():
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),tweet.user.screen_name.encode('utf-8'), tweet.user.location.encode('utf-8')])

```

我们现在将在追加模式下打开一个 [csv 文件](https://www.askpython.com/python-modules/python-csv-module)，并将 twitter 的内容写入该文件。

```py
csvFile = open('file-name', 'a')
csvWriter = csv.writer(csvFile)

```

### 5.使用 Python 和 Tweepy 从 Twitter 中提取 Tweets 的完整代码

整个代码如下所示。您可以执行这个命令，在 python 文件所在的工作目录中找到一个包含所有数据的 csv 文件。

```py
from tweepy import *

import pandas as pd
import csv
import re 
import string
import preprocessor as p

consumer_key = <enter your consumer key>
consumer_secret = <enter key>
access_key= <enter key>
access_secret = <enter key>

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

csvFile = open('file-name', 'a')
csvWriter = csv.writer(csvFile)

search_words = "#"      # enter your words
new_search = search_words + " -filter:retweets"

for tweet in tweepy.Cursor(api.search,q=new_search,count=100,
                           lang="en",
                           since_id=0).items():
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),tweet.user.screen_name.encode('utf-8'), tweet.user.location.encode('utf-8')])

```

上述代码的输出是一个 csv 文件，如下所示:

![Image 22](img/dead58dec4bdfb7a974fb9dd1915cc76.png)

Output CSV file

请注意，输出会因搜索关键字而异。

## 结论

因此，我们已经到了本文的结尾，并尝试从 Tweepy 中检索一些信息。希望你喜欢这样做！请在下面的评论区告诉我们您的反馈。