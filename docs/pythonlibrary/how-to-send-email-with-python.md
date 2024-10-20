# 如何用 Python 发送电子邮件

> 原文：<https://www.blog.pythonlibrary.org/2010/05/14/how-to-send-email-with-python/>

在我工作的地方，我们运行许多用 Python 编写的登录脚本。当其中一个脚本出现错误时，我们想知道。因此，我们编写了一个简单的 Python 脚本，通过电子邮件将错误发送给我们。从那时起，我就需要想办法用一些更高级的脚本来发送附件。如果你是这个博客的长期读者，那么你可能还记得 [wxPyMail](https://www.blog.pythonlibrary.org/2008/08/16/wxpymail-creating-an-application-to-send-emails/) ，这是一个简单的 wxPython 程序，可以发送电子邮件。在本文中，您将发现如何仅使用 Python 的标准库来发送电子邮件。我们将重点讨论 smtplib 和电子邮件模块。

## 使用 smtplib 发送电子邮件

用 smtplib 发邮件超级简单。你想看看有多简单吗？你当然知道！让我们来看看:

```py

import smtplib
import string

SUBJECT = "Test email from Python"
TO = "mike@mydomain.com"
FROM = "python@mydomain.com"
text = "blah blah blah"
BODY = string.join((
        "From: %s" % FROM,
        "To: %s" % TO,
        "Subject: %s" % SUBJECT ,
        "",
        text
        ), "\r\n")
server = smtplib.SMTP(HOST)
server.sendmail(FROM, [TO], BODY)
server.quit()

```

请注意，电子邮件的实际连接和发送只有两行代码。剩下的代码只是设置要发送的消息。在工作中，我们将所有这些都包装在一个可调用的函数中，并向它传递一些信息，比如错误是什么以及将错误发送给谁。如果需要登录，请在创建服务器变量后添加一行，执行以下操作:server.login(username，password)

## 发送带有附件的电子邮件

现在让我们看看如何发送带有附件的电子邮件。对于这个脚本，我们还将使用*电子邮件*模块。这里有一个简单的例子，基于我最近写的一些代码:

```py

import os
import smtplib

from email import Encoders
from email.MIMEBase import MIMEBase
from email.MIMEMultipart import MIMEMultipart
from email.Utils import formatdate

filePath = r'\\some\path\to\a\file'

def sendEmail(TO = "mike@mydomain.com",
              FROM="support@mydomain.com"):
    HOST = "mail.mydomain.com"

    msg = MIMEMultipart()
    msg["From"] = FROM
    msg["To"] = TO
    msg["Subject"] = "You've got mail!"
    msg['Date']    = formatdate(localtime=True)

    # attach a file
    part = MIMEBase('application', "octet-stream")
    part.set_payload( open(filePath,"rb").read() )
    Encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(filePath))
    msg.attach(part)

    server = smtplib.SMTP(HOST)
    # server.login(username, password)  # optional

    try:
        failed = server.sendmail(FROM, TO, msg.as_string())
        server.close()
    except Exception, e:
        errorMsg = "Unable to send email. Error: %s" % str(e)

if __name__ == "__main__":
    sendEmail()

```

这里发生了相当多的事情，所以让我们来看看新的内容。首先，我们从*电子邮件*模块导入所有我们需要的零碎信息。然后我们创建一个发送电子邮件的函数。接下来，我们创建一个 *MIMEMultipart* 对象。这个方便的东西可以保存我们的电子邮件。它使用一个类似 dict 的界面来添加字段，如收件人、发件人、主题等。您会注意到我们还有一个日期字段。这只是抓取你的电脑的当前日期，并将其转换为适当的 MIME 电子邮件格式。

我们最感兴趣的是如何附加文件。这里我们创建了一个 *MIMEBase* 对象，并将其有效负载设置为我们想要附加的文件。注意，我们需要告诉它将文件作为二进制文件读取，即使文件是纯文本的。接下来，我们用 64 进制对数据流进行编码。最后两步是添加一个头，然后将 MIMEBase 对象附加到我们的 MIMEMultipart 对象。如果您有多个文件要附加，那么您可能希望将这一部分放入某种循环中，并对这些文件进行循环。事实上，我在前面提到的 wxPyMail 示例中就是这么做的。

无论如何，一旦你完成了所有这些，你就可以做和上面的 smtplib 例子中一样的事情了。唯一的区别是我们更改了下面的行:

```py

server.sendmail(FROM, TO, msg.as_string())

```

请注意 msg.as_string。我们需要将对象转换成字符串来完成这项工作。我们还将 sendmail 函数放在一个 try/except 语句中，以防发生不好的事情，我们的电子邮件无法发送。如果我们愿意，我们可以将 try/except 封装在一个 *while* 循环中，这样如果失败，我们可以重试发送 X 次电子邮件。

## 包扎

我们已经介绍了如何发送一封简单的电子邮件以及如何发送一封带有附件的电子邮件。电子邮件模块内置了更多的功能，这里没有介绍，所以请务必阅读文档。祝编码愉快！

## 附加阅读

*   [smtplib 模块](http://docs.python.org/library/smtplib.html)
*   Python 的[邮件模块](http://docs.python.org/library/email.html)