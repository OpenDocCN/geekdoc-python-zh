# 使用 Python 和 Flask 开始使用 Slack API

> 原文：<https://realpython.com/getting-started-with-the-slack-api-using-python-and-flask/>

光滑的托管聊天应用程序 [Slack](https://slack.com/) 今年风靡一时。该工具的采用并不是空洞的宣传——它对于同开发人员交流和学习非常有用。例如，软件开发者社区如 [DC Python](https://pythoncommunity.herokuapp.com/) 、[达拉斯沃斯开发者](http://dfwdevs.org/)和[丹佛开发者](http://denverdevs.org/)建立了他们自己的 Slack 频道。

然而，如果 Slack 只是一个美化了的 AOL 即时通讯工具，它就不会那么有用。这是使用 Slack web 应用程序编程接口(API)检索和发送消息的的[编程访问，这是真正发挥威力的地方。](https://realpython.com/api-integration-in-python/)

在这篇文章中，我们将看到如何通过 API 和官方的 SlackClient Python 助手库来使用 Slack。我们将获取一个 API 访问令牌，并编写一些 Python 代码来通过 API 列出、检索和发送数据。让我们现在开始挖吧！

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

*这是[马特·马凯](http://www.mattmakai.com/)的客座博文，他是 [Twilio](https://www.twilio.com/) 的开发者传道者，也是[全栈 Python](http://www.fullstackpython.com/) 的作者。*

## 我们需要的工具

在这篇博文中，将使用几个工具来运行代码，包括:

*   一个[免费的 Slack 账户，在这个账户上你有 API 访问权限](https://slack.com/)或者注册一个 [Slack 开发者聚会团队](http://dev4slack.xoxco.com/)
*   Python 2 或 3
*   Slack 团队构建的官方 Python [slackclient](https://github.com/slackhq/python-slackclient) 代码库
*   松弛 API [测试令牌](https://api.slack.com/docs/oauth-test-tokens)
*   [烧瓶网状微结构](http://flask.pocoo.org/)；如果你对它不熟悉，可以查看一下[真正的 Python 课程](https://realpython.com/courses/)、 [Flask by Example 系列](https://realpython.com/flask-by-example-part-1-project-setup/)或 [Full Stack Python 的 Flask 页面](https://www.fullstackpython.com/flask.html)

打开 [Slack API 文档](https://api.slack.com/web)进行参考也很方便。你可以在这篇文章中写代码或者[用完成的项目](https://github.com/makaimc/slack-api-python-examples)克隆配套的 GitHub 库。

现在我们知道了需要使用什么工具，让我们开始创建一个新的 [virtualenv](https://realpython.com/python-virtual-environments-a-primer/) 来将我们的应用程序依赖项与您正在处理的其他 Python 项目隔离开来:

```py
$ mkdir slackapi
$ virtualenv venv
```

激活 virtualenv:

```py
$ source venv/bin/activate
```

根据您的 virtualenv 和 shell 的设置，您的提示符现在应该是这样的:

```py
(venv)$
```

让 shell 保持开放，因为我们通过 Slack 构建的官方 slackclient API helper 库建立了 Slack 访问。

> 社区还创建了其他精彩的 Python 助手库。为了简单起见，我们将只安装和使用 slackclient，但是一旦我们完成了这里的工作，你也可以尝试像 [slacker](https://github.com/os/slacker) 、 [slack](https://github.com/kn/slack) 和 [pyslack](https://github.com/loisaidasam/pyslack) 这样的库。

用 [pip](https://realpython.com/what-is-pip/) 将 slackclient 助手库安装到您的 virtualenv 中:

```py
$ pip install slackclient==1.0.0
```

现在我们已经安装了 helper 库，我们需要为我们的团队和帐户获取一个 Slack 访问令牌。

[*Remove ads*](/account/join/)

## Slack Web API

前往 Slack Web API 的[登录页面:](https://api.slack.com/web)

[![Slack: sign in](img/48573a64c37701c77058dd0fc5b28029.png)](https://files.realpython.com/media/slack-sign-in.900deaeb59ca.png)

登录后，您可以向下滚动 web API 页面，在这里您会看到一个生成测试令牌的按钮:

[![Slack: generate test token page](img/c571b93f0406c057fa13e905ebe5a3bf.png)](https://files.realpython.com/media/slack-generate-test-token.ff3aaeb706b8.png)

为您拥有管理权限的松散团队生成测试令牌。在这篇博文中，这个令牌将很好地满足我们的开发目的，但是您也可以[创建一个 OAuth 流](https://api.slack.com/docs/oauth-test-tokens)，其他用户可以在其中通过他们自己的帐户生成令牌进行身份验证。

我们一会儿就需要那个测试令牌，所以把它放在手边。让我们切换到我们的 Python 环境设置，这样我们就可以尝试 API 了。在 virtualenv 仍然有效的情况下，启动 Python REPL:

```py
(venv)$ python
Python 3.5.0 (v3.5.0:374f501f4567, Sep 12 2015, 11:00:19)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

让我们用一个测试调用来测试我们的 API 令牌；在 REPL 提示符下键入以下代码:

>>>

```py
>>> from slackclient import SlackClient
>>> slack_client = SlackClient('your test token here')
>>> slack_client.api_call("api.test")
```

如果使用令牌的 API 测试成功，REPL 应该返回类似下面的字典:

```py
{u'args': {u'token': u'xoxp-361113305843-7621238052-8691112296227-d0d4824abe'}, u'ok': True}
```

如果您返回`{u'ok': False, u'error': u'invalid_auth'}`,那么仔细检查您是否正确地将 Slack 令牌复制到了在 REPL 上输入的第二行。

用 REPL 中的另一行代码再输入一个快速测试来验证我们的身份:

>>>

```py
>>> slack_client.api_call("auth.test")
```

您应该会看到与本词典类似的另一本词典:

```py
{u'user_id': u'U0S77S29J', u'url': u'https://fullstackguides.slack.com/', u'team_id': u'T0S8V1ZQA', u'user': u'matt', u'team': u'Full Stack Guides, u'ok': True}
```

厉害！我们被授权通过我们的帐户开始使用 Slack API。现在有趣的事情开始了，我们可以开始以编程的方式获取松弛数据和处理消息！

## 松弛 API 基础知识

用快捷的 `Ctrl` + `D` 或`exit()`命令退出 REPL。回到命令行，将松弛令牌作为环境变量导出:

```py
(venv)$ export SLACK_TOKEN='your slack token pasted here'
```

我们将使用`os`模块在 Python 脚本中捕获环境变量，而不是将其硬编码到源代码中。

进入你最喜欢的文本编辑器，比如 [Vim](https://realpython.com/vim-and-python-a-match-made-in-heaven/) 、 [Emacs](https://realpython.com/emacs-the-best-python-editor/) 或 [Sublime Text](https://realpython.com/setting-up-sublime-text-3-for-full-stack-python-development/) ，这样我们就可以删减一些新的 Python 代码。创建一个名为 *app.py* 的新文件，并开始用以下导入内容填充它:

```py
import os
from slackclient import SlackClient
```

同样，`os`模块将用于提取我们刚刚导出的`SLACK_TOKEN`环境变量。SlackClient 导入应该看起来很熟悉，因为它是我们之前在 REPL 上写的同一行:

```py
SLACK_TOKEN = os.environ.get('SLACK_TOKEN')

slack_client = SlackClient(SLACK_TOKEN)
```

在上面两行中，我们截取了`SLACK_TOKEN`环境变量值并实例化了 SlackClient 助手库。接下来，让我们创建一个通过 API 调用列出频道的函数。Slack 在一个带有两个键的字典中返回结果:`ok`和`channels`。`ok`允许我们知道 API 调用是否成功，如果它的值是`True`，那么`channels`包含了我们在通道列表中需要的数据。

```py
def list_channels():
    channels_call = slack_client.api_call("channels.list")
    if channels_call.get('ok'):
        return channels_call['channels']
    return None
```

最后，让我们添加一个方便的[主函数](https://realpython.com/python-main-function/)，当我们在命令行上用`python app.py`调用 Python 文件时，它将允许我们打印所有的通道:

```py
if __name__ == '__main__':
    channels = list_channels()
    if channels:
        print("Channels: ")
        for c in channels:
            print(c['name'] + " (" + c['id'] + ")")
    else:
        print("Unable to authenticate.")
```

这就是我们目前需要的所有代码。是时候尝试一下了。用`python app.py`从命令行执行脚本。您将看到类似以下通道列表的输出:

```py
Channels:
general (C0S82S5RS)
python (C0S8HABL3)
random (C0S8F4432)
```

我们在频道名称旁边的括号中打印出的频道 ID 是什么？Slack 的 API 需要一个惟一的通道引用，所以我们使用 ID 而不是名称作为标识符，而不是人类可读的通道名称。

我们可以编写一些代码，使用 [channel.info API 方法](https://api.slack.com/methods/channels.info)来获取基于 ID 的特定通道的数据。

在 main 中添加一个新函数和几个新行，以输出来自每个通道的最新消息，这仅在更详细的`channel.info` API 调用中可用。

更新代码:

```py
import os
from slackclient import SlackClient

SLACK_TOKEN = os.environ.get('SLACK_TOKEN', None)

slack_client = SlackClient(SLACK_TOKEN)

def list_channels():
    channels_call = slack_client.api_call("channels.list")
    if channels_call['ok']:
        return channels_call['channels']
    return None

def channel_info(channel_id):
    channel_info = slack_client.api_call("channels.info", channel=channel_id)
    if channel_info:
        return channel_info['channel']
    return None

if __name__ == '__main__':
    channels = list_channels()
    if channels:
        print("Channels: ")
        for c in channels:
            print(c['name'] + " (" + c['id'] + ")")
            detailed_info = channel_info(c['id'])
            if detailed_info:
                print(detailed_info['latest']['text'])
    else:
        print("Unable to authenticate.")
```

> 注意，这段代码极大地增加了脚本执行的 API 调用，从 1 个增加到 N+1 个，其中 N 是 Slack 返回的通道数。

通过执行`python app.py`再次运行新脚本:

```py
Channels:
general (C0S82S5RS)
yada yada yada.
python (C0S8HABL3)
This is posted to #python and comes from a bot named webhookbot.
random (C0S8F4432)
<@U0SAEJ99T|samb> has joined the channel
```

不错！现在我们既有了频道列表，也有了获取每个频道及其 ID 的详细信息的方法。接下来，让我们通过发送和接收消息，在我们的一个通道中与其他用户进行交互。

[*Remove ads*](/account/join/)

## 发送消息

现在我们可以更深入地了解 Slack API，因为我们知道我们的 API 调用正在工作，并且有了通道 ID。我们给#总频道发个消息吧。

在`channel_info`下增加一个名为`send_message`的新功能:

```py
def send_message(channel_id, message):
    slack_client.api_call(
        "chat.postMessage",
        channel=channel_id,
        text=message,
        username='pythonbot',
        icon_emoji=':robot_face:'
    )
```

接收一个频道的 ID，然后从我们的“Python bot”向该频道发布一条消息。另外，修改`main`函数，这样当我们运行这个文件时，`main`将调用我们新的`send_message`函数:

```py
if __name__ == '__main__':
    channels = list_channels()
    if channels:
        print("Channels: ")
        for channel in channels:
            print(channel['name'] + " (" + channel['id'] + ")")
            detailed_info = channel_info(channel['id'])
            if detailed_info:
                print('Latest text from ' + channel['name'] + ":")
                print(detailed_info['latest']['text'])
            if channel['name'] == 'general':
                send_message(channel['id'], "Hello " +
                             channel['name'] + "! It worked!")
        print('-----')
    else:
        print("Unable to authenticate.")
```

保存更改并运行`python app.py`。为你的懈怠团队打开#通用通道。您应该看到您的 Python bot 向频道发布了一条新消息:

[![Slack API: send message](img/6c21c3a0269bea496f01530a528f49df.png)](https://files.realpython.com/media/slack-send-message.40d559dd2ed9.png)

厉害！所以我们可以发送消息，但是如果我们想知道#general 频道的用户在说什么呢？

## 接收消息

我们可以设置一个传出的 webhook，它将通过 HTTP POST 请求提醒我们的 Python 应用程序。这部分比发送消息稍微复杂一些，因为我们需要接收一个或多个 POST 请求。

首先，我们需要一个简单的 web 服务器，它可以处理来自 Slack webhook 的入站 POST 请求。使用以下代码创建一个名为 *receive.py* 的新文件:

```py
import os
from flask import Flask, request, Response

app = Flask(__name__)

SLACK_WEBHOOK_SECRET = os.environ.get('SLACK_WEBHOOK_SECRET')

@app.route('/slack', methods=['POST'])
def inbound():
    if request.form.get('token') == SLACK_WEBHOOK_SECRET:
        channel = request.form.get('channel_name')
        username = request.form.get('user_name')
        text = request.form.get('text')
        inbound_message = username + " in " + channel + " says: " + text
        print(inbound_message)
    return Response(), 200

@app.route('/', methods=['GET'])
def test():
    return Response('It works!')

if __name__ == "__main__":
    app.run(debug=True)
```

在上面的 Python 文件中，我们:

1.  进口烧瓶
2.  实例化新的 Flask 应用程序上下文
3.  引入`SLACK_WEBHOOK_SECRET`环境变量，稍后我们将从 Slack 控制台中获得该变量
4.  建立一个可以接收 Slack 的 HTTP POST 请求的路由，只要发送给我们的 webhook 密钥与我们的环境变量中的密钥相匹配，就会将输出打印到命令行
5.  为响应 GET 请求的测试目的创建另一个路由
6.  将我们的 Flask 应用程序设置为在我们用 Python 运行这个脚本时运行

安装 Flask ( `pip install flask`)，然后用`python receive.py`命令启动 Flask 应用程序，我们将看到一些调试输出，表明开发服务器正在运行。

*   运行于 http://127.0.0.1:5000/(按 `Ctrl` + `C` 退出)
*   使用 stat 重新启动
*   调试器处于活动状态！
*   调试器 pin 码:144-609-426

我们已经准备好接收我们的 POST 请求 webhook，只是大多数开发环境不公开 localhost 之外的路由。我们需要一个本地主机隧道，当我们开发代码时，它将为我们提供一个外部可访问的域名。我通常使用 [ngrok](https://ngrok.com/) ，因为它是[简单、免费和令人敬畏的](https://www.twilio.com/blog/2015/09/6-awesome-reasons-to-use-ngrok-when-testing-webhooks.html)。还有其他选项如[本地隧道](https://localtunnel.me/)和[前向](https://forwardhq.com/)。

在新的终端窗口中下载并运行 ngrok(或另一个本地主机隧道工具)后，您将获得一个子域，该子域将发送到该子域的请求转发到您的本地主机服务器。下面是 ngrok 在控制台中用`./ngrok http 5000`命令启动时的样子:

[![Terminal session running ngrok](img/e5b7461823a8e3689b550059074f5d8e.png)](https://files.realpython.com/media/ngrok.1b717f68fa09.png)

记下您的转发 URL，在本例中是`https://6940e7da.ngrok.io`，因为 Slack 需要它来设置我们的出站 webhook。然后，通过打开您的 web 浏览器并转到转发 URL，测试我们的 ngrok 转发 URL 是否正确连接到我们的 Flask 应用程序。我们应该看到“成功了！”消息。

现在我们可以在我们的 Slack 配置中使用 ngrok 转发 URL。进入 [Slack Outgoing Webhooks 页面](https://api.slack.com/outgoing-webhooks)，点击“outgoing webhook integration”链接，如下图所示:

[![Slack: outgoing webhooks settings page](img/d0e6e50ff7c26f3728a48c5857cc3317.png)](https://files.realpython.com/media/slack-outgoing-webhooks.2cc1ffcbafdd.png)

向下滚动到集成设置部分。选择“#general”作为收听频道。将您的 ngrok 转发 URL 加上“/slack”复制到 URL 文本框中:

[![Slack outgoing webhooks settings page](img/03b18f1ae87fa17ed9b6c33732b7b312.png)](https://files.realpython.com/media/slack-outgoing-webhooks-settings.95c74c1d8027.png)

复制生成的令牌。向下滚动并按下“保存设置”按钮。

暂停一下你的 Flask 服务器。正如我们前面对 Slack 令牌所做的那样，使用`export`命令将传出的 webhook 令牌作为环境变量公开:

```py
(venv)$ export SLACK_WEBHOOK_SECRET='generated outgoing webhook token here'
```

然后重启 Flask 服务器，这样它就可以获取生成的`SLACK_WEBHOOK_SECRET`。最后，是测试接收消息的时候了！

去你的 Slack #通用频道。您应该会看到外发的 webhook 集成已经添加到通道中:

[![Slack webhook added](img/c043d21221a03c41a8a6d6ba516c1243.png)](https://files.realpython.com/media/slack-webhook-added.23083d66b742.png)

在 Slack 中，输入类似“testing”的消息，然后按回车键。回到运行 Flask 应用程序的命令行。您应该看到 POST 请求输出的消息:

```py
matt in general says: testing
127.0.0.1 - - [21/May/2016 12:39:56] "POST /slack HTTP/1.1" 200 -
```

现在，我们已经有了从一个或多个通道接收消息的方法，并且可以添加我们想要的任何 Python 代码来处理输入。这是构建 bot 或将消息发送到另一个服务进行处理的一个很好的挂钩。

[*Remove ads*](/account/join/)

## 包装完毕

呜哇！全部完成！实际上，你可以用 Slack API 做更多的事情。既然你已经有了基本的东西，这里还有几个想法可以尝试:

1.  结合 [Twilio API，通过短信](https://www.twilio.com/blog/2016/05/build-sms-slack-bot-python.html)与 Slack 频道进行通信
2.  尝试不同的 Slack 客户端或完全放弃助手库，并使用[请求库来实现重试逻辑](https://realpython.com/caching-external-api-requests/)
3.  [编写并定制一个完全松弛的机器人](https://medium.com/@julianmartinez/how-to-write-a-slack-bot-with-python-code-examples-4ed354407b98#.gdc0pkni0)

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

* * *

目前就这些。

如果您有任何问题，请在下面留言或通过以下方式联系我:

*   推特: [@mattmakai](https://twitter.com/mattmakai) 和 [@fullstackpython](https://twitter.com/fullstackpython)
*   GitHub: [makaimc](https://github.com/makaimc)
*   Twitch(使用 Python 和 Swift 进行实时编码): [mattmakai](https://www.twitch.tv/mattmakai)***