# Python 101:如何用 Twilio 发送短信/彩信

> 原文：<https://www.blog.pythonlibrary.org/2014/09/23/python-101-how-to-send-smsmms-with-twilio/>

我一直听到一些关于一种叫做 [Twilio](http://www.twilio.com) 的新型网络服务的传言，这种服务可以让你发送短信和彩信等。他们的 REST API 也有一个方便的 Python 包装器。如果你注册了 Twilio，他们会给你一个试用账户，甚至不需要你提供信用卡，我很感激。您将收到一个 Twilio 号码，可用于发送您的信息。因为您使用的是试用帐户，所以在实际发送消息之前，您必须授权您想要发送消息的任何电话号码。让我们花点时间了解一下这是如何工作的！

* * *

### 入门指南

首先，你需要注册 Twilio，还需要安装 Python twilio 包装器。要安装后者，只需执行以下操作:

```py

pip install twilio

```

一旦安装完成并且你注册了，我们就可以继续了。

* * *

### 使用 Twilio 发送短信

通过 Twilio 发送短信非常简单。您需要查看您的 twilio 帐户，以获得 sid 和认证令牌以及您的 Twilio 号码。一旦你有了这三条关键信息，你就可以发送短信了。让我们看看如何:

```py

from twilio.rest import TwilioRestClient

#----------------------------------------------------------------------
def send_sms(msg, to):
    """"""
    sid = "text-random"
    auth_token = "youAreAuthed"
    twilio_number = "123-456-7890"

    client = TwilioRestClient(sid, auth_token)

    message = client.messages.create(body=msg,
                                     from_=twilio_number,
                                     to=to,
                                     )

if __name__ == "__main__":
    msg = "Hello from Python!"
    to = "111-111-1111"
    send_sms(msg, to)

```

当您运行上面的代码时，您将在手机上收到一条消息，内容如下:**从您的 Twilio 试用帐户发送——来自 Python 的 Hello！**。如你所见，这真的很简单。您所需要做的就是创建一个 **TwilioRestClient** 的实例，然后创建一条消息。剩下的交给提里奥。发送彩信几乎同样简单。让我们在下一个例子中看看！

* * *

### 发送彩信

大多数情况下，您实际上并不想将您的 sid、身份验证令牌或 Twilio 电话号码放在代码本身中。相反，它通常存储在数据库或配置文件中。因此，在本例中，我们将把这些信息放入一个配置文件，并使用 [ConfigObj](https://pypi.python.org/pypi/configobj) 提取它们。下面是我的配置文件的内容:

```py

[twilio]
sid = "random-text"
auth_token = "youAreAuthed!"
twilio_number = "123-456-7890"

```

现在，让我们编写一些代码来提取这些信息并发送一条彩信:

```py

import configobj
from twilio.rest import TwilioRestClient

#----------------------------------------------------------------------
def send_mms(msg, to, img):
    """"""
    cfg = configobj.ConfigObj("/path/to/config.ini")
    sid = cfg["twilio"]["sid"]
    auth_token = cfg["twilio"]["auth_token"]
    twilio_number = cfg["twilio"]["twilio_number"]

    client = TwilioRestClient(sid, auth_token)

    message = client.messages.create(body=msg,
                                     from_=twilio_number,
                                     to=to,
                                     MediaUrl=img
                                     )

if __name__ == "__main__":
    msg = "Hello from Python!"
    to = "111-111-1111"
    img = "http://www.website.com/example.jpg"
    send_mms(msg, to=to, img=img)

```

这段代码生成与上一段代码相同的消息，但它也发送一幅图像。您会注意到，Trilio 要求您使用 HTTP 或 HTTPS URL 将照片附加到您的彩信中。否则，这也很容易做到！

* * *

### 包扎

在现实世界中你能用它做什么？我敢肯定，企业会想用它来发送优惠券或提供新产品代码。这听起来也像是一个乐队或政治家用来向他们的粉丝、拥护者等传递信息的东西。我看到一篇文章，有人用它自动给自己发送体育比分。我发现这项服务很容易使用。我不知道他们的定价是否有竞争力，但试用帐户肯定足以进行测试，当然值得一试。

* * *

### 附加阅读

*   Twilio Python [文档](http://www.twilio.com/docs/python/install#more-documentation)
*   使用 Python 来[保存分数](http://impythonist.wordpress.com/2014/09/07/how-i-satisfied-a-request-from-my-friend-with-python/)
*   星期五的乐趣:[创建你自己的简单得令人讨厌的消息应用程序，就像你一样](http://readwrite.com/2014/07/11/one-click-messaging-app)