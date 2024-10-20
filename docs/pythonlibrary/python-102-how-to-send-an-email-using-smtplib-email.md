# Python 102:如何使用 smtplib + email 发送电子邮件

> 原文：<https://www.blog.pythonlibrary.org/2013/06/26/python-102-how-to-send-an-email-using-smtplib-email/>

几年前我就这个话题写过一篇[文章](https://www.blog.pythonlibrary.org/2010/05/14/how-to-send-email-with-python/)，但是我觉得现在是我重温的时候了。为什么？嗯，最近我在一个发送电子邮件的程序上做了很多工作，我一直在看我以前的文章，觉得我第一次写它的时候错过了一些东西。因此，在本文中，我们将了解以下内容:

*   电子邮件的基本知识——有点像原始文章的翻版
*   如何使用“收件人”、“抄送”和“密件抄送”行发送电子邮件
*   如何一次发送到多个地址
*   如何使用电子邮件模块添加附件和正文

我们开始吧！

### 如何使用 Python 和 smtplib 发送电子邮件

我们将从原始文章中稍加修改的代码版本开始。我注意到我忘记了在原来的中设置主机变量，所以这个例子会更完整一点:

```py

import smtplib
import string

HOST = "mySMTP.server.com"
SUBJECT = "Test email from Python"
TO = "mike@someAddress.org"
FROM = "python@mydomain.com"
text = "Python rules them all!"
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

您会注意到这段代码中没有用户名或密码。如果您的服务器需要身份验证，那么您需要添加以下代码:

```py

server.login(username, password)

```

这应该在您创建了**服务器**对象之后添加。通常，您会希望将这段代码放入一个函数中，并使用其中的一些参数来调用它。您甚至可能希望将这些信息放入配置文件中。让我们接下来做那件事。

```py

#----------------------------------------------------------------------
def send_email(host, subject, to_addr, from_addr, body_text):
    """
    Send an email
    """
    BODY = string.join((
            "From: %s" % from_addr,
            "To: %s" % to_addr,
            "Subject: %s" % subject ,
            "",
            body_text
            ), "\r\n")
    server = smtplib.SMTP(host)
    server.sendmail(from_addr, [to_addr], BODY)
    server.quit()

if __name__ == "__main__":
    host = "mySMTP.server.com"
    subject = "Test email from Python"
    to_addr = "mike@someAddress.org"
    from_addr = "python@mydomain.com"
    body_text = "Python rules them all!"
    send_email(host, subject, to_addr, from_addr, body_text)

```

现在，您可以通过查看函数本身来了解实际代码有多小。那是 13 行！如果我们不把正文中的每一项都放在自己的行上，我们可以把它变得更短，但是可读性会差一些。现在我们将添加一个配置文件来保存服务器信息和 from 地址。为什么？在我的工作中，我们可能会使用不同的电子邮件服务器来发送电子邮件，或者如果电子邮件服务器升级了，名称改变了，那么我们只需要改变配置文件而不是代码。如果我们的公司被另一家公司收购并合并，同样的事情也适用于发件人地址。我们将使用 configObj 包而不是 Python 的 ConfigParser，因为我发现 configObj 更简单。如果你还没有 Python 包索引 (PyPI)的话，你应该下载一份。

让我们看一下配置文件:

```py

[smtp]
server = some.server.com
from_addr = python@mydomain.com

```

这是一个非常简单的配置文件。在其中，我们有一个标记为 **smtp** 的部分，其中有两个项目:server 和 from_addr。我们将使用 configObj 读取该文件，并将其转换为 Python 字典。下面是代码的更新版本:

```py

import os
import smtplib
import string
import sys

from configobj import ConfigObj

#----------------------------------------------------------------------
def send_email(subject, to_addr, body_text):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config.ini")

    if os.path.exists(config_path):
        cfg = ConfigObj(config_path)
        cfg_dict = cfg.dict()
    else:
        print "Config not found! Exiting!"
        sys.exit(1)

    host = cfg_dict["smtp"]["server"]
    from_addr = cfg_dict["smtp"]["from_addr"]

    BODY = string.join((
            "From: %s" % from_addr,
            "To: %s" % to_addr,
            "Subject: %s" % subject ,
            "",
            body_text
            ), "\r\n")
    server = smtplib.SMTP(host)
    server.sendmail(from_addr, [to_addr], BODY)
    server.quit()

if __name__ == "__main__":
    subject = "Test email from Python"
    to_addr = "mike@someAddress.org"
    body_text = "Python rules them all!"
    send_email(subject, to_addr, body_text)

```

我们在这段代码中添加了一个小检查。我们想首先获取脚本本身所在的路径，这就是 **base_path** 所代表的。接下来，我们将路径和文件名结合起来，得到配置文件的完全限定路径。然后，我们检查该文件是否存在。如果存在，我们创建一个字典，如果不存在，我们打印一条消息并退出脚本。为了安全起见，我们应该在 ConfigObj 调用周围添加一个异常处理程序，尽管该文件可能存在，但可能已损坏，或者我们可能没有权限打开它，这将引发一个异常。这将是一个你可以自己尝试的小项目。不管怎样，假设一切顺利，我们拿到了字典。现在，我们可以使用普通的字典语法提取主机和 from_addr 信息。

现在我们准备学习如何同时发送多封电子邮件！

### 如何一次发送多封邮件

如果你在网上搜索这个话题，你可能会遇到这个 StackOverflow [问题](http://stackoverflow.com/questions/6941811/send-email-to-multiple-recipients-from-txt-file-with-python-smtplib)，在这里我们可以学习如何通过 smtplib 模块发送多封电子邮件。让我们稍微修改一下上一个例子，这样我们就可以发送多封电子邮件了！

```py

import os
import smtplib
import string
import sys

from configobj import ConfigObj

#----------------------------------------------------------------------
def send_email(subject, body_text, emails):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config.ini")

    if os.path.exists(config_path):
        cfg = ConfigObj(config_path)
        cfg_dict = cfg.dict()
    else:
        print "Config not found! Exiting!"
        sys.exit(1)

    host = cfg_dict["smtp"]["server"]
    from_addr = cfg_dict["smtp"]["from_addr"]

    BODY = string.join((
            "From: %s" % from_addr,
            "To: %s" % ', '.join(emails),
            "Subject: %s" % subject ,
            "",
            body_text
            ), "\r\n")
    server = smtplib.SMTP(host)
    server.sendmail(from_addr, emails, BODY)
    server.quit()

if __name__ == "__main__":
    emails = ["mike@example.org", "someone@gmail.com"]
    subject = "Test email from Python"
    body_text = "Python rules them all!"
    send_email(subject, body_text, emails)

```

您会注意到，在这个例子中，我们删除了 **to_addr** 参数，并添加了一个 **emails** 参数，这是一个电子邮件地址列表。为此，我们需要在正文的 **To:** 部分创建一个逗号分隔的字符串，并将电子邮件列表传递给 sendmail 方法。因此，我们做下面的事情来创建一个简单的逗号分隔的字符串:**'，'。加入(邮件)**。很简单，是吧？

现在我们只需要弄清楚如何使用“抄送”和“密件抄送”字段发送邮件。让我们创建一个支持该功能的新版本的代码！

```py

import os
import smtplib
import string
import sys

from configobj import ConfigObj

#----------------------------------------------------------------------
def send_email(subject, body_text, to_emails, cc_emails, bcc_emails):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config.ini")

    if os.path.exists(config_path):
        cfg = ConfigObj(config_path)
        cfg_dict = cfg.dict()
    else:
        print "Config not found! Exiting!"
        sys.exit(1)

    host = cfg_dict["smtp"]["server"]
    from_addr = cfg_dict["smtp"]["from_addr"]

    BODY = string.join((
            "From: %s" % from_addr,
            "To: %s" % ', '.join(to_emails),
            "CC: %s" % ', '.join(cc_emails),
            "BCC: %s" % ', '.join(bcc_emails),
            "Subject: %s" % subject ,
            "",
            body_text
            ), "\r\n")
    emails = to_emails + cc_emails + bcc_emails

    server = smtplib.SMTP(host)
    server.sendmail(from_addr, emails, BODY)
    server.quit()

if __name__ == "__main__":
    emails = ["mike@somewhere.org"]
    cc_emails = ["someone@gmail.com"]
    bcc_emails = ["schmuck@newtel.net"]

    subject = "Test email from Python"
    body_text = "Python rules them all!"
    send_email(subject, body_text, emails, cc_emails, bcc_emails)

```

在这段代码中，我们传入 3 个列表，每个列表都有一个电子邮件地址。我们像以前一样创建 CC 和 BCC 字段，但是我们还需要将 3 个列表合并成一个，这样我们就可以将合并后的列表传递给 sendmail()方法。在 [StackOverflow](http://stackoverflow.com/questions/771907/python-how-to-store-a-draft-email-with-bcc-recipients-to-exchange-server-via-im) 上有传言说，一些电子邮件客户端可能会以奇怪的方式处理密件抄送字段，从而允许收件人通过电子邮件标题查看密件抄送列表。我无法确认这种行为，但我知道 Gmail 成功地从邮件标题中删除了密件抄送信息。我还没有发现哪个客户不知道，但是如果你知道，请在评论中告诉我们。

现在我们准备好使用 Python 的电子邮件模块了！

### 使用 Python 发送电子邮件附件

现在，我们将学习上一节的内容，并将其与 Python 电子邮件模块结合起来。电子邮件模块使得添加附件变得极其容易。代码如下:

```py

import os
import smtplib
import string
import sys

from configobj import ConfigObj
from email import Encoders
from email.mime.text import MIMEText
from email.MIMEBase import MIMEBase
from email.MIMEMultipart import MIMEMultipart
from email.Utils import formatdate

#----------------------------------------------------------------------
def send_email_with_attachment(subject, body_text, to_emails,
                               cc_emails, bcc_emails, file_to_attach):
    """
    Send an email with an attachment
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config.ini")
    header = 'Content-Disposition', 'attachment; filename="%s"' % file_to_attach

    # get the config
    if os.path.exists(config_path):
        cfg = ConfigObj(config_path)
        cfg_dict = cfg.dict()
    else:
        print "Config not found! Exiting!"
        sys.exit(1)

    # extract server and from_addr from config
    host = cfg_dict["smtp"]["server"]
    from_addr = cfg_dict["smtp"]["from_addr"]

    # create the message
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    if body_text:
        msg.attach( MIMEText(body_text) )

    msg["To"] = ', '.join(to_emails)
    msg["cc"] = ', '.join(cc_emails)

    attachment = MIMEBase('application', "octet-stream")
    try:
        with open(file_to_attach, "rb") as fh:
            data = fh.read()
        attachment.set_payload( data )
        Encoders.encode_base64(attachment)
        attachment.add_header(*header)
        msg.attach(attachment)
    except IOError:
        msg = "Error opening attachment file %s" % file_to_attach
        print msg
        sys.exit(1)

    emails = to_emails + cc_emails

    server = smtplib.SMTP(host)
    server.sendmail(from_addr, emails, msg.as_string())
    server.quit()

if __name__ == "__main__":
    emails = ["mike@somewhere.org", "nedry@jp.net"]
    cc_emails = ["someone@gmail.com"]
    bcc_emails = ["anonymous@circe.org"]

    subject = "Test email with attachment from Python"
    body_text = "This email contains an attachment!"
    path = "/path/to/some/file"
    send_email_with_attachment(subject, body_text, emails, 
                               cc_emails, bcc_emails, path)

```

在这里，我们重命名了我们的函数，并添加了一个新的参数， **file_to_attach** 。我们还需要添加一个头并创建一个 **MIMEMultipart** 对象。在我们添加附件之前，可以随时创建标题。我们向 MIMEMultipart 对象(msg)添加元素，就像我们向字典添加键一样。您会注意到，我们必须使用 email 模块的 formatdate 方法来插入正确格式化的日期。为了添加消息体，我们需要创建一个 **MIMEText** 的实例。如果你注意的话，你会发现我们没有添加密件抄送信息，但是你可以通过遵循上面代码中的约定很容易地做到这一点。接下来我们添加附件。我们将它包装在一个异常处理程序中，并使用带有语句的**来提取文件，并将其放在我们的 **MIMEBase** 对象中。最后，我们将它添加到 msg 变量中，然后发送出去。注意，我们必须在 sendmail 方法中将 msg 转换成一个字符串。**

### 包扎

现在你知道如何用 Python 发送电子邮件了。对于那些喜欢迷你项目的人来说，您应该返回并在代码的 **server.sendmail** 部分添加额外的错误处理，以防在这个过程中发生一些奇怪的事情，比如 SMTPAuthenticationError 或 SMTPConnectError。我们还可以在附加文件时加强错误处理，以捕捉其他错误。最后，我们可能希望获得这些不同的电子邮件列表，并创建一个已删除重复的规范化列表。如果我们从一个文件中读取电子邮件地址列表，这一点尤其重要。

还要注意，我们的 from 地址是假的。我们可以使用 Python 和其他编程语言来欺骗电子邮件，但这是非常糟糕的礼仪，而且可能是非法的，这取决于你住在哪里。你已经被警告了！明智地使用你的知识，享受 Python 带来的乐趣和收益！

**注意:以上代码中的所有电子邮件地址都是假的。这段代码是在 Windows 7 上使用 Python 2.6.6 测试的。**

### 附加阅读

*   smtplib 上的 Python [文档](http://docs.python.org/2/library/smtplib.html)
*   关于[电子邮件模块](http://docs.python.org/2/library/email)的 Python 文档
*   从向多个收件人发送电子邮件。带 Python smtplib 的 txt 文件
*   [python:如何用 to、CC、BCC 发送邮件？](http://stackoverflow.com/questions/1546367/python-how-to-send-mail-with-to-cc-and-bcc)
*   [如何用 Python 发送邮件](https://www.blog.pythonlibrary.org/2010/05/14/how-to-send-email-with-python/)

### 下载源代码

*   [python_email_102.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/06/python_email_102.zip)