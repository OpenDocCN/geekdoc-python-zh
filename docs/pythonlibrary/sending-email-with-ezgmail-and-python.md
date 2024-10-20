# 使用 EZGmail 和 Python 发送电子邮件

> 原文：<https://www.blog.pythonlibrary.org/2019/06/04/sending-email-with-ezgmail-and-python/>

你有没有想过用 Python 编程语言用 GMail 发邮件？2018 年，[用 Python](https://automatetheboringstuff.com/) 自动化枯燥的东西的畅销书作者 Al Sweigart 创建了一个名为 [EZGmail](https://pypi.org/project/EZGmail/) 的包。你也可以使用谷歌自己的绑定来做这类事情，但是这比使用 EZGmail 要复杂得多。

在本文中，我们将快速了解如何使用这个包。

* * *

### 安装

您的第一步是使用 pip 安装 EZGmail。方法如下:

```py

pip install ezgmail

```

然后转到[https://developers.google.com/gmail/api/quickstart/python](https://developers.google.com/gmail/api/quickstart/python)，点击**启用 Gmail API** 按钮。这将允许你下载一个 *credentials.json* 文件，并给你一个客户端 ID 和客户端密码。您可以在 Google 的 Python API 客户端使用后一种凭证，如果需要，您可以在这里管理这些凭证[。](https://console.developers.google.com/apis/credentials)

现在，将凭证文件复制到您计划编写代码的位置。然后，您需要在您的终端中运行 Python，运行位置与您下载的凭证文件的位置相同。

下一步是运行`ezgmail.init(). This will open up a web browser to Gmail where it will ask you to allow access to your application. If you grant access, EZGmail will download a tokens file so that it doesn't need to have you reauthorize it every time you use it.`

要验证一切都正常工作，您可以运行以下代码:

```py

>>> ezgmail.EMAIL_ADDRESS
'your_email_address@gmail.com'

```

这应该会打印出您的 Gmail 帐户名称。

* * *

### 使用 EZGmail 发送电子邮件

您可以使用 EZGmail 发送和阅读电子邮件。

让我们看一个发送电子邮件的简单例子:

```py

>>> email = 'joe@schmo.com'
>>> subject = 'Test from EZGmail'
>>> text = 'This is only a test'
>>> ezgmail.send(email, subject, text)

```

这将向您指定的帐户发送一封带有主题和文本的电子邮件。您还可以传入要发送的附件列表。最后，EZGmail 支持抄送和密件抄送，尽管如果你使用它们或电子邮件字段本身发送到多个地址，参数只接受字符串。这意味着电子邮件地址需要在字符串中用逗号分隔，而不是电子邮件地址列表。

* * *

### 阅读 Gmail

你也可以用 EZGmail 阅读邮件。最好的两种方法是使用`recent() and `unread() methods.``

以下是一些例子:

```py

>>> recent = ezgmail.recent()
>>> unread = ezgmail.unread()

```

这些方法中的每一个都返回一个列表`GmailThread objects. A `GmailThread has the following attributes:``

*   信息
*   发报机
*   接受者
*   科目
*   身体
*   时间戳

您可以遍历这些列表，并根据需要提取任意或所有这些项目。

还有一个方便的功能，您可以使用它来打印您的电子邮件摘要:

```py

>>> ezgmail.summary(unread)

```

当您运行这段代码时，EZGmail 可能需要一段时间来下载和解析电子邮件。但是请注意，默认情况下，它最多只能下载 25 封电子邮件的数据。如果需要，您可以将最大值更改为一个更高的数字，但是要注意有一个[数据配额限制](https://developers.google.com/gmail/api/v1/reference/quota)。

* * *

### 包扎

EZGmail 包非常简洁。本教程没有涉及到它，但是你也可以使用 EZGmail 通过`search() function. Be sure to give the package a try or at least study the source code. Happy coding!`来搜索你的邮件

* * *

### 相关阅读

*   如何用 Python 发送电子邮件