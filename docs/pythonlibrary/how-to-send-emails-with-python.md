# 如何用 Python 发送电子邮件

> 原文：<https://www.blog.pythonlibrary.org/2021/09/21/how-to-send-emails-with-python/>

Python 提供了几个非常好的模块，可以用来制作电子邮件。它们是**电子邮件**和 **smtplib** 模块。在这两个模块中，您将花一些时间学习如何实际使用这些模块，而不是复习各种方法。

具体来说，您将涉及以下内容:

*   电子邮件的基础
*   如何一次发送到多个地址
*   如何使用“收件人”、“抄送”和“密件抄送”行发送电子邮件
*   如何使用电子邮件模块添加附件和正文

我们开始吧！

## 电子邮件基础-如何用 smtplib 发送电子邮件

**smtplib** 模块使用起来非常直观。你将写一个简单的例子，展示如何发送电子邮件。

打开您最喜欢的 Python IDE 或文本编辑器，创建一个新的 Python 文件。将以下代码添加到该文件并保存:

```py
import smtplib

HOST = "mySMTP.server.com"
SUBJECT = "Test email from Python"
TO = "mike@someAddress.org"
FROM = "python@mydomain.com"
text = "Python 3.4 rules them all!"

BODY = "\r\n".join((
"From: %s" % FROM,
"To: %s" % TO,
"Subject: %s" % SUBJECT ,
"",
text
))

server = smtplib.SMTP(HOST)
server.sendmail(FROM, [TO], BODY)
server.quit()
```

这里您只导入了 **smtplib** 模块。该代码的三分之二用于设置电子邮件。大多数变量都是显而易见的，所以您将只关注奇怪的一个，即 BODY。

在这里，您使用字符串的 **join()** 方法将前面的所有变量组合成一个字符串，其中每一行都以回车符("/r ")加新行("/n ")结束。如果你把正文打印出来，它会是这样的:

```py
'From: python@mydomain.com\r\nTo: mike@mydomain.com\r\nSubject: Test email from Python\r\n\r\nblah blah blah'
```

之后，建立一个到主机的服务器连接，然后调用 smtplib 模块的 sendmail 方法发送电子邮件。然后断开与服务器的连接。您会注意到这段代码中没有用户名或密码。如果您的服务器需要身份验证，那么您需要添加以下代码:

```py
server.login(username, password)
```

这应该在创建服务器对象后立即添加。通常，您会希望将这段代码放入一个函数中，并使用其中的一些参数来调用它。您甚至可能希望将这些信息放入配置文件中。

让我们将这段代码放入一个函数中。

```py
import smtplib

def send_email(host, subject, to_addr, from_addr, body_text):
    """
    Send an email
    """
    BODY = "\r\n".join((
            "From: %s" % from_addr,
            "To: %s" % to_addr,
            "Subject: %s" % subject ,
            "",
            body_text
            ))
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

现在，您可以通过查看函数本身来了解实际代码有多小。那是 13 行！如果你不把正文中的每一项都放在自己的行上，你可以把它变得更短，但是它没有可读性。现在，您将添加一个配置文件来保存服务器信息和 from 地址。

你为什么要这么做？许多组织使用不同的电子邮件服务器来发送电子邮件，或者如果电子邮件服务器升级并且名称改变，那么你只需要改变配置文件而不是代码。如果你的公司被另一家公司收购并合并，同样的事情也适用于发件人地址。

让我们看看配置文件(保存为 **email.ini** ):

```py
[smtp]
server = some.server.com
from_addr = python@mydomain.com
```

这是一个非常简单的配置文件。在其中，您有一个标记为 **smtp** 的部分，其中有两个项目:服务器和 **from_addr** 。您将使用 **ConfigParser** 来读取这个文件，并将它转换成一个 Python 字典。下面是代码的更新版本(保存为 **smtp_config.py** )

```py
import os
import smtplib
import sys

from configparser import ConfigParser

def send_email(subject, to_addr, body_text):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "email.ini")

    if os.path.exists(config_path):
        cfg = ConfigParser()
        cfg.read(config_path)
    else:
        print("Config not found! Exiting!")
        sys.exit(1)

    host = cfg.get("smtp", "server")
    from_addr = cfg.get("smtp", "from_addr")

    BODY = "\r\n".join((
        "From: %s" % from_addr,
        "To: %s" % to_addr,
        "Subject: %s" % subject ,
        "",
        body_text
    ))
    server = smtplib.SMTP(host)
    server.sendmail(from_addr, [to_addr], BODY)
    server.quit()

if __name__ == "__main__":
    subject = "Test email from Python"
    to_addr = "mike@someAddress.org"
    body_text = "Python rules them all!"
    send_email(subject, to_addr, body_text)
```

您在这段代码中添加了一个小检查。你想首先获取脚本本身所在的路径，这就是 **base_path** 所代表的。接下来，将路径和文件名结合起来，得到配置文件的完全限定路径。然后检查该文件是否存在。

如果存在，您创建一个 **ConfigParser** ，如果不存在，您打印一条消息并退出脚本。为了安全起见，您应该在 **ConfigParser.read()** 调用周围添加一个异常处理程序，尽管该文件可能存在，但可能已损坏，或者您可能没有权限打开它，这将引发一个异常。

这将是一个你可以自己尝试的小项目。无论如何，假设一切顺利，并且成功创建了 **ConfigParser** 对象。现在，您可以使用常用的 **ConfigParser** 语法从 _addr 信息中提取主机和**。**

现在你已经准备好学习如何同时发送多封电子邮件了！

## 一次发送多封电子邮件

能够一次发送多封电子邮件是一个很好的功能。

继续修改你的最后一个例子，这样你就可以发送多封电子邮件了！

```py
import os
import smtplib
import sys

from configparser import ConfigParser

def send_email(subject, body_text, emails):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "email.ini")

    if os.path.exists(config_path):
        cfg = ConfigParser()
        cfg.read(config_path)
    else:
        print("Config not found! Exiting!")
        sys.exit(1)

    host = cfg.get("smtp", "server")
    from_addr = cfg.get("smtp", "from_addr")

    BODY = "\r\n".join((
            "From: %s" % from_addr,
            "To: %s" % ', '.join(emails),
            "Subject: %s" % subject ,
            "",
            body_text
            ))
    server = smtplib.SMTP(host)
    server.sendmail(from_addr, emails, BODY)
    server.quit()

if __name__ == "__main__":
    emails = ["mike@someAddress.org", "someone@gmail.com"]
    subject = "Test email from Python"
    body_text = "Python rules them all!"
    send_email(subject, body_text, emails)
```

您会注意到，在这个例子中，您删除了 **to_addr** 参数，并添加了一个 emails 参数，这是一个电子邮件地址列表。为此，您需要在**正文**的 To:部分创建一个逗号分隔的字符串，并将电子邮件列表传递给 **sendmail** 方法。因此，您执行以下操作来创建一个简单的逗号分隔的字符串:**'，'。加入(邮件)**。很简单，是吧？

## 使用“收件人”、“抄送”和“密件抄送”行发送电子邮件

现在你只需要弄清楚如何使用“抄送”和“密件抄送”字段发送邮件。

让我们创建一个支持该功能的新版本的代码！

```py
import os
import smtplib
import sys

from configparser import ConfigParser

def send_email(subject, body_text, to_emails, cc_emails, bcc_emails):
    """
    Send an email
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "email.ini")

    if os.path.exists(config_path):
        cfg = ConfigParser()
        cfg.read(config_path)
    else:
        print("Config not found! Exiting!")
        sys.exit(1)

    host = cfg.get("smtp", "server")
    from_addr = cfg.get("smtp", "from_addr")

    BODY = "\r\n".join((
            "From: %s" % from_addr,
            "To: %s" % ', '.join(to_emails),
            "CC: %s" % ', '.join(cc_emails),
            "BCC: %s" % ', '.join(bcc_emails),
            "Subject: %s" % subject ,
            "",
            body_text
            ))
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

在这段代码中，你传入 3 个列表，每个列表有一个电子邮件地址。您创建的**抄送**和**密件抄送**字段与之前完全相同，但是您还需要将 3 个列表合并成一个，这样您就可以将合并后的列表传递给 **sendmail()** 方法。

在 StackOverflow 这样的论坛上有一些传言说，一些电子邮件客户端可能会以奇怪的方式处理密件抄送字段，从而允许收件人通过电子邮件标题看到密件抄送列表。我无法确认这种行为，但我知道 Gmail 成功地从邮件标题中删除了密件抄送信息。

现在您已经准备好使用 Python 的电子邮件模块了！

## 使用电子邮件模块添加附件/正文

现在，您将利用从上一节中学到的知识，将它与 Python 电子邮件模块结合起来，以便发送附件。

电子邮件模块使得添加附件变得极其容易。代码如下:

```py
import os
import smtplib
import sys

from configparser import ConfigParser
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

def send_email_with_attachment(subject, body_text, to_emails,
                               cc_emails, bcc_emails, file_to_attach):
    """
    Send an email with an attachment
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "email.ini")
    header = 'Content-Disposition', 'attachment; filename="%s"' % file_to_attach

    # get the config
    if os.path.exists(config_path):
        cfg = ConfigParser()
        cfg.read(config_path)
    else:
        print("Config not found! Exiting!")
        sys.exit(1)

    # extract server and from_addr from config
    host = cfg.get("smtp", "server")
    from_addr = cfg.get("smtp", "from_addr")

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
        encoders.encode_base64(attachment)
        attachment.add_header(*header)
        msg.attach(attachment)
    except IOError:
        msg = "Error opening attachment file %s" % file_to_attach
        print(msg)
        sys.exit(1)

    emails = to_emails + cc_emails

    server = smtplib.SMTP(host)
    server.sendmail(from_addr, emails, msg.as_string())
    server.quit()

if __name__ == "__main__":
    emails = ["mike@someAddress.org", "nedry@jp.net"]
    cc_emails = ["someone@gmail.com"]
    bcc_emails = ["anonymous@circe.org"]

    subject = "Test email with attachment from Python"
    body_text = "This email contains an attachment!"
    path = "/path/to/some/file"
    send_email_with_attachment(subject, body_text, emails, 
                               cc_emails, bcc_emails, path)
```

在这里，您重命名了您的函数并添加了一个新参数， **file_to_attach** 。您还需要添加一个头并创建一个 **MIMEMultipart** 对象。在添加附件之前，可以随时创建标题。

向 **MIMEMultipart** 对象( **msg** )添加元素，就像向字典中添加键一样。您会注意到，您必须使用 **email** 模块的 **formatdate** 方法来插入正确格式化的日期。

要添加消息体，您需要创建一个 **MIMEText** 的实例。如果您注意的话，您会发现您没有添加密件抄送信息，但是您可以通过遵循上面代码中的约定很容易地这样做。

接下来，添加附件。您将它包装在一个异常处理程序中，并使用带有语句的**来提取文件，并将其放入您的 **MIMEBase** 对象中。最后，你把它添加到 msg 变量中，然后发送出去。注意，您必须在 **sendmail** ()方法中将 msg 转换成一个字符串。**

## 包扎

现在你知道如何用 Python 发送电子邮件了。对于那些喜欢小型项目的人来说，您应该回去在服务器周围添加额外的错误处理。 **sendmail** 部分代码，以防在这个过程中发生一些奇怪的事情。

一个例子是 **SMTPAuthenticationError** 或 **SMTPConnectError** 。您还可以在附加文件的过程中加强错误处理，以捕捉其他错误。最后，您可能希望获得这些不同的电子邮件列表，并创建一个已删除重复项的规范化列表。如果您正在从文件中读取电子邮件地址列表，这一点尤其重要。

另外，请注意，您的发件人地址是假的。你可以使用 Python 和其他编程语言来欺骗电子邮件，但这是非常不礼貌的，而且可能是非法的，这取决于你住在哪里。你已经被警告了！

明智地使用你的知识，享受 Python 带来的乐趣和收益！

## 相关阅读

想学习更多 Python 基础知识？然后查看以下教程:

*   Python 101: [使用 JSON 的介绍](https://www.blog.pythonlibrary.org/2020/09/15/python-101-an-intro-to-working-with-json/)

*   python 101-[创建多个流程](https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/)

*   python 101-[用 pdb 调试你的代码](https://www.blog.pythonlibrary.org/2020/07/07/python-101-debugging-your-code-with-pdb/)