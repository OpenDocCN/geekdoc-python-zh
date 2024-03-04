# 使用 Python 发送电子邮件

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/using-python-to-send-email>

Python 在标准库中包含了几个模块，用于处理电子邮件和电子邮件服务器。

## smtplib 概述

smtplib 模块定义了一个 SMTP 客户端会话对象，该对象可用于向任何带有 SMTP 或 ESMTP 侦听器守护程序的互联网计算机发送邮件。

SMTP 代表简单邮件传输协议。smtplib 模块对于与邮件服务器通信以发送邮件非常有用。

使用 SMTP 服务器通过 Python 的 smtplib 发送邮件。

实际使用情况取决于电子邮件的复杂性和电子邮件服务器的设置，此处的说明基于通过 Gmail 发送电子邮件。

## smtplib 用法

这个例子取自 wikibooks.org 的这篇[帖子](https://en.wikibooks.org/wiki/Python_Programming/Email "python_programming")

```py
"""The first step is to create an SMTP object, each object is used for connection 
with one server."""

import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)

#Next, log in to the server
server.login("youremailusername", "password")

#Send the mail
msg = "
Hello!" # The /n separates the message from the headers
server.sendmail("[email protected]", "[email protected]", msg)

```

要包含 From、To 和 Subject 标题，我们应该使用 email 包，因为 smtplib 根本不修改内容或标题。

## 电子邮件包概述

Python 的 email 包包含许多用于编写和解析电子邮件消息的类和函数。

## 电子邮件包使用

我们从只导入我们需要的类开始，这也使我们不必在以后使用完整的模块名。

```py
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

```

然后，我们编写一些基本的消息头:

```py
fromaddr = "[email protected]"
toaddr = "[email protected]"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Python email"

```

接下来，我们将电子邮件正文附加到 MIME 消息中:

```py
body = "Python test mail"
msg.attach(MIMEText(body, 'plain'))

```

为了发送邮件，我们必须将对象转换为字符串，然后使用与上面相同的程序通过 SMTP 服务器发送..

```py
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.ehlo()
server.login("youremailusername", "password")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)

```

## 验证电子邮件地址

SMTP 协议包含一个命令，用于询问服务器地址是否有效。通常 VRFY 是禁用的，以防止垃圾邮件发送者找到合法的电子邮件地址，但如果它被启用，您可以向服务器询问地址，并收到一个状态码，表明有效性以及用户的全名。

这个例子基于这个[帖子](http://www.doughellmann.com/PyMOTW/smtplib/ "pymotw_smptlib")

```py
import smtplib

server = smtplib.SMTP('mail')
server.set_debuglevel(True)  # show communication with the server
try:
    dhellmann_result = server.verify('dhellmann')
    notthere_result = server.verify('notthere')
finally:
    server.quit()

print 'dhellmann:', dhellmann_result
print 'notthere :', notthere_result

```

## 使用 Gmail 发送邮件

这个例子取自[http://rosettacode.org/wiki/Send_an_email#Python](https://rosettacode.org/wiki/Send_an_email#Python "rosettacode.org")

```py
import smtplib

def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s
' % from_addr
    header += 'To: %s
' % ','.join(to_addr_list)
    header += 'Cc: %s
' % ','.join(cc_addr_list)
    header += 'Subject: %s

' % subject
    message = header + message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()

```

上述脚本的用法示例

```py
 sendemail(from_addr    = '[[email protected]](/cdn-cgi/l/email-protection)', 
          to_addr_list = ['[[email protected]](/cdn-cgi/l/email-protection)'],
          cc_addr_list = ['[[email protected]](/cdn-cgi/l/email-protection)'], 
          subject      = 'Howdy', 
          message      = 'Howdy from a python function', 
          login        = 'pythonuser', 
          password     = 'XXXXX') 
```

##### 收到的示例电子邮件

```py
 sendemail(from_addr    = '[[email protected]](/cdn-cgi/l/email-protection)', 
          to_addr_list = ['[[email protected]](/cdn-cgi/l/email-protection)'],
          cc_addr_list = ['[[email protected]](/cdn-cgi/l/email-protection)'], 
          subject      = 'Howdy', 
          message      = 'Howdy from a python function', 
          login        = 'pythonuser', 
          password     = 'XXXXX') 
```

##### 来源

```py
 [Python on Wikibooks.org](https://en.wikibooks.org/wiki/Python_Programming/Email "wikibooks_python")
[Rosettacode.org](https://rosettacode.org/wiki/Send_an_email#Python "rosettacode_python_email")
[Docs.python.org](https://docs.python.org/2/library/smtplib.html "python.org_smtplib")
[http://docs.python.org/2/library/email.mime.html](https://docs.python.org/2/library/email.mime.html "docs.python.org") 
```