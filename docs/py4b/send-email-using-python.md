# 使用 Python 发送电子邮件

> 原文：<https://www.pythonforbeginners.com/api/send-email-using-python>

电子邮件是我们生活中最重要的部分之一。在本文中，我们将讨论如何使用 python 发送电子邮件。

## 如何使用 Python 发送电子邮件？

在本文中，我们将使用简单邮件传输协议(SMTP)通过 python 发送电子邮件。为此，我们将使用 smtplib 模块。此外，我们将使用 gmail.com 电子邮件 id 发送电子邮件。我们需要指定邮件平台，因为不同的邮件平台使用不同的端口号来发送电子邮件。因此，我们需要知道邮件服务提供商使用 python 发送电子邮件所使用的端口号。

建议阅读:[用 Python 创建聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 使用 Python 中的 smtplib 模块发送电子邮件的步骤

为了使用 smtplib 模块发送邮件，我们将遵循以下步骤。

*   首先，我们将使用 import 语句导入 smtplib 模块。
*   之后，我们将使用 SMTP()方法创建一个会话。SMTP()方法将邮件服务器位置作为第一个输入参数，将端口号作为第二个输入参数。这里，我们将传递“smtp.gmail.com”作为邮件服务器位置，传递 587 作为端口号。执行后，SMTP()方法创建一个 SMTP 会话。
*   我们将在 SMTP 会话中使用传输层安全性(TLS)。为此，我们将对 SMTP()方法返回的会话对象调用 starttls()方法。
*   启动 TLS 后，我们将登录我们的 Gmail 帐户。为此，我们将使用 login()方法。在会话对象上调用 login()方法时，它接受用户名作为第一个输入参数，接受密码作为第二个输入参数。如果用户名和密码正确，您将登录 Gmail。否则，程序将会出错。
*   登录后，我们将使用 sendmail()方法发送电子邮件。在会话对象上调用 sendmail()方法时，它将发送者的电子邮件地址作为第一个输入参数，接收者的电子邮件地址作为第二个输入参数，要发送的消息作为第三个输入参数。执行后，邮件会发送到收件人的电子邮件地址。
*   发送电子邮件后，我们将通过调用 session 对象上的 quit()方法来终止 SMTP 会话。

下面是使用 python 发送电子邮件的代码。

```py
import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
email_id="[[email protected]](/cdn-cgi/l/email-protection)" #your email
password= "*********" password of gmail account
s.login("email_id", "password")
message= "Email Body"
s.sendmail("email_id", "receiver_email_id", message)
s.quit()
```

## 结论

在本文中，我们讨论了如何使用 python 发送电子邮件。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解的文章。你可能也会喜欢这篇关于 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 中的[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)