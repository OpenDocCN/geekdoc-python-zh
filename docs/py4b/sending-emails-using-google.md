# 使用谷歌发送电子邮件

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/sending-emails-using-google>

## 概观

系统管理员和开发人员的一项常见任务是在出现错误时使用脚本发送电子邮件。

## 为什么使用 Gmail？

使用谷歌的 SMTP 服务器可以免费使用，而且转发电子邮件的效果非常好。请注意，谷歌有一个发送限制:“如果你向超过 500 个收件人发送消息，或者如果你发送了大量无法送达的消息，谷歌将暂时禁用你的帐户。”只要你同意，你就可以走了。

## 我从哪里开始？

使用 SMTP(简单邮件传输协议)服务器通过 Python 的 smtplib 发送邮件。由于我们将使用谷歌的 SMTP 服务器来发送我们的电子邮件，我们将需要收集信息，如服务器，端口，认证。谷歌搜索很容易找到这些信息。

#### 谷歌的标准配置说明

## 入门指南

首先打开您最喜欢的文本编辑器，并在脚本顶部导入 smtplib 模块。

```py
import smtplib
```

已经在顶部，我们将创建一些 SMTP 头。

```py
fromaddr = '[[email protected]](/cdn-cgi/l/email-protection)'
toaddrs  = '[[email protected]](/cdn-cgi/l/email-protection)'
msg = 'Enter you message here’
```

完成后，创建一个 SMTP 对象，用于连接服务器。

```py
server = smtplib.SMTP("smtp.gmail.com:587”)
```

接下来，我们将使用 Gmail 所需的 starttls()函数。

```py
server.starttls()
```

接下来，登录到服务器:

```py
server.login(username,password)
```

然后，我们将发送电子邮件:

```py
server.sendmail(fromaddr, toaddrs, msg)
```

#### 最终方案

你可以在下面看到完整的程序，现在你应该能理解它是干什么的了。

```py
import smtplib
# Specifying the from and to addresses

fromaddr = '[[email protected]](/cdn-cgi/l/email-protection)'
toaddrs  = '[[email protected]](/cdn-cgi/l/email-protection)'

# Writing the message (this message will appear in the email)

msg = 'Enter you message here'

# Gmail Login

username = 'username'
password = 'password'

# Sending the mail  

server = smtplib.SMTP('smtp.gmail.com:587')
server.starttls()
server.login(username,password)
server.sendmail(fromaddr, toaddrs, msg)
server.quit()
```

##### 更多阅读

[使用 Python 发送电子邮件](https://www.pythonforbeginners.com/code-snippets-source-code/using-python-to-send-email "python_send_emails")