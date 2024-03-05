# 使用 Python 发送电子邮件

> 原文：<https://realpython.com/python-send-email/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 发邮件**](/courses/sending-emails-python/)

你可能会发现这个教程，因为你想使用 Python 发送电子邮件。也许您希望从您的代码中接收电子邮件提醒，在用户创建帐户时向他们发送确认电子邮件，或者向您组织的成员发送电子邮件以提醒他们缴纳会费。手动发送电子邮件是一项耗时且容易出错的任务，但使用 Python 很容易实现自动化。

在本教程中，你将学习如何:

*   使用`SMTP_SSL()`和`.starttls()`建立一个**安全连接**

*   使用 Python 内置的`smtplib`库发送**基本邮件**

*   使用`email`包发送包含 **HTML 内容**和**附件**的电子邮件

*   使用包含联系人数据的 CSV 文件发送多封**个性化电子邮件**

*   使用 **Yagmail** 软件包，只需几行代码就可以通过您的 gmail 帐户发送电子邮件

在本教程的最后，你会发现一些事务性的电子邮件服务，当你想发送大量的电子邮件时，它们会很有用。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 开始使用

[Python](https://realpython.com/installing-python/) 自带内置 [`smtplib`](https://docs.python.org/3/library/smtplib.html) 模块，使用简单邮件传输协议(SMTP)发送邮件。`smtplib`对 SMTP 使用 [RFC 821](https://tools.ietf.org/html/rfc821) 协议。本教程中的示例将使用 Gmail SMTP 服务器发送电子邮件，但同样的原则也适用于其他电子邮件服务。尽管大多数电子邮件提供商使用与本教程中相同的连接端口，但您可以快速运行[谷歌搜索](https://www.google.co.uk/search?&q=gmail+smtp+server+and+port)来确认您的连接端口。

要开始学习本教程，[设置一个 Gmail 开发账户](https://realpython.com/python-send-email/#option-1-setting-up-a-gmail-account-for-development)，或者[设置一个 SMTP 调试服务器](https://realpython.com/python-send-email/#option-2-setting-up-a-local-smtp-server)，它会丢弃你发送的电子邮件并打印到命令提示符下。下面为您展示了这两种选择。本地 SMTP 调试服务器可用于修复电子邮件功能的任何问题，并确保您的电子邮件功能在发送任何电子邮件之前没有错误。

[*Remove ads*](/account/join/)

### 选项 1:为开发设立一个 Gmail 账户

如果你决定使用 Gmail 帐户发送邮件，我强烈建议你为代码开发设置一个一次性帐户。这是因为你必须调整你的 Gmail 帐户的安全设置，以允许从你的 Python 代码访问，也因为你可能会意外暴露你的登录信息。此外，我发现我的测试账户的收件箱很快就被测试邮件塞满了，这足以成为我建立一个新的 Gmail 账户进行开发的理由。

Gmail 的一个很好的特性是，你可以使用`+`符号给你的电子邮件地址添加任何修饰语，就在`@`符号之前。例如，发往`my+person1@gmail.com`和`my+person2@gmail.com`的邮件都会到达`my@gmail.com`。在测试电子邮件功能时，您可以使用它来模拟指向同一个收件箱的多个地址。

要设置用于测试代码的 Gmail 地址，请执行以下操作:

*   [创建一个新的 Google 帐户](https://accounts.google.com/signup)。
*   将 [*转到*](https://myaccount.google.com/lesssecureapps) 上允许不太安全的应用到*。请注意，这使得其他人更容易访问您的帐户。*

如果你不想降低你的 Gmail 帐户的安全设置，查看一下谷歌的[文档](https://developers.google.com/gmail/api/quickstart/python)关于如何使用 OAuth2 授权框架获得你的 Python 脚本的访问凭证。

### 选项 2:设置本地 SMTP 服务器

您可以使用 Python 预安装的`smtpd`模块，通过运行本地 SMTP 调试服务器来测试电子邮件功能。它不是将电子邮件发送到指定的地址，而是丢弃它们并将它们的内容打印到控制台。运行本地调试服务器意味着没有必要处理消息加密或使用凭证登录到电子邮件服务器。

您可以通过在命令提示符下键入以下命令来启动本地 SMTP 调试服务器:

```py
$ python -m smtpd -c DebuggingServer -n localhost:1025
```

在 Linux 上，使用前面带`sudo`的相同命令。

通过此服务器发送的任何电子邮件都将被丢弃，并在终端窗口中显示为每行一个 [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes-objects) 对象:

```py
---------- MESSAGE FOLLOWS ----------
b'X-Peer: ::1'
b''
b'From: my@address.com'
b'To: your@address.com'
b'Subject: a local test mail'
b''
b'Hello there, here is a test email'
------------ END MESSAGE ------------
```

在本教程的其余部分，我将假设您使用的是 Gmail 帐户，但如果您使用的是本地调试服务器，请确保使用`localhost`作为您的 SMTP 服务器，并使用端口 1025 而不是端口 465 或 587。除此之外，您不需要使用`login()`或使用 SSL/TLS 加密通信。

## 发送纯文本电子邮件

在我们开始发送带有 HTML 内容和附件的电子邮件之前，您将学习使用 Python 发送纯文本电子邮件。这些电子邮件你可以用简单的文本编辑器写出来。没有像文本格式或超链接这样的花哨东西。你过一会儿就会明白。

### 启动安全 SMTP 连接

当您通过 Python 发送电子邮件时，您应该确保您的 SMTP 连接是加密的，这样您的消息和登录凭证就不会被他人轻易访问。SSL(安全套接字层)和 TLS(传输层安全性)是可用于加密 SMTP 连接的两种协议。在使用本地调试服务器时，没有必要使用这两种方法。

有两种方法可以启动与电子邮件服务器的安全连接:

*   使用`SMTP_SSL()`启动一个从一开始就受到保护的 SMTP 连接。
*   启动一个不安全的 SMTP 连接，然后可以使用`.starttls()`进行加密。

在这两种情况下，Gmail 将使用 TLS 加密电子邮件，因为这是 SSL 的更安全的继任者。根据 Python 的[安全考虑](https://docs.python.org/3/library/ssl.html#ssl-security)，强烈建议您使用 [`ssl`](https://docs.python.org/3/library/ssl.html) 模块中的`create_default_context()`。这将加载系统的可信 CA 证书，启用主机名检查和证书验证，并尝试选择合理的安全协议和密码设置。

如果您想检查 Gmail 收件箱中电子邮件的加密情况，请进入*更多* → *显示原文*，查看列在*收到的*标题下的加密类型。

[`smtplib`](https://docs.python.org/3/library/smtplib.html) 是 Python 的内置模块，用于向任何装有 SMTP 或 ESMTP 监听守护程序的互联网机器发送电子邮件。

我将首先向您展示如何使用`SMTP_SSL()`,因为它实例化了一个从一开始就安全的连接，并且比`.starttls()`选项稍微简洁一些。请记住，Gmail 要求您在使用`SMTP_SSL()`时连接到 465 端口，在使用`.starttls()`时连接到 587 端口。

#### 选项 1:使用`SMTP_SSL()`

下面的代码示例创建了一个与 Gmail 的 SMTP 服务器的安全连接，使用`smtplib`的`SMTP_SSL()`启动一个 TLS 加密的连接。`ssl`的默认上下文验证主机名及其证书，并优化连接的安全性。确保填写您自己的电子邮件地址，而不是`my@gmail.com`:

```py
import smtplib, ssl

port = 465  # For SSL
password = input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login("my@gmail.com", password)
    # TODO: Send email here
```

使用`with smtplib.SMTP_SSL() as server:`确保连接在缩进代码块的末尾自动关闭。如果`port`为零，或者没有指定，`.SMTP_SSL()`将使用 SSL 上 SMTP 的标准端口(端口 465)。

将您的电子邮件密码存储在您的代码中是不安全的做法，尤其是当您打算与其他人共享它时。相反，使用`input()`让用户在运行脚本时输入密码，如上面的例子所示。如果您不想让您的密码在键入时显示在屏幕上，您可以导入 [`getpass`](https://docs.python.org/3/library/getpass.html) 模块，使用`.getpass()`代替盲输入您的密码。

#### 选项二:使用`.starttls()`

我们可以创建一个不安全的 SMTP 连接，并使用`.starttls()`对其进行加密，而不是使用`.SMTP_SSL()`来创建一个从一开始就是安全的连接。

为此，创建一个`smtplib.SMTP`的实例，它封装了一个 SMTP 连接并允许您访问它的方法。我建议在脚本开始时定义您的 SMTP 服务器和端口，以便于配置它们。

下面的代码片段使用了结构`server = SMTP()`，而不是我们在前面的例子中使用的格式`with SMTP() as server:`。为了确保你的代码在出错时不会崩溃，把你的主代码放在一个`try`块中，让一个`except`块把任何错误信息打印到`stdout`:

```py
import smtplib, ssl

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = "my@gmail.com"
password = input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

# Try to log in to server and send email
try:
    server = smtplib.SMTP(smtp_server,port)
    server.ehlo() # Can be omitted
    server.starttls(context=context) # Secure the connection
    server.ehlo() # Can be omitted
    server.login(sender_email, password)
    # TODO: Send email here
except Exception as e:
    # Print any error messages to stdout
    print(e)
finally:
    server.quit()
```

为了向服务器标识自己，应该在创建一个`.SMTP()`对象后调用`.helo()` (SMTP)或`.ehlo()` (ESMTP)，在`.starttls()`后再调用一次。如果需要，这个函数由`.starttls()`和`.sendmail()`隐式调用，所以除非你想检查服务器的 SMTP 服务扩展，否则没有必要显式使用`.helo()`或`.ehlo()`。

[*Remove ads*](/account/join/)

### 发送您的纯文本电子邮件

使用上述方法之一启动安全 SMTP 连接后，您可以使用`.sendmail()`发送您的电子邮件，这与 tin 上显示的差不多:

```py
server.sendmail(sender_email, receiver_email, message)
```

我建议在导入之后，在脚本的顶部定义电子邮件地址和消息内容，这样您可以很容易地更改它们:

```py
sender_email = "my@gmail.com"
receiver_email = "your@gmail.com"
message = """\
Subject: Hi there

This message is sent from Python."""

# Send email here
```

`message` [字符串](https://realpython.com/python-strings/)以`"Subject: Hi there"`开头，后跟两个换行符(`\n`)。这确保了`Hi there`显示为电子邮件的主题，并且换行后的文本将被视为消息正文。

下面的代码示例使用`SMTP_SSL()`发送一封纯文本电子邮件:

```py
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "my@gmail.com"  # Enter your address
receiver_email = "your@gmail.com"  # Enter receiver address
password = input("Type your password and press enter: ")
message = """\
Subject: Hi there

This message is sent from Python."""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
```

作为比较，这里有一个代码示例，它通过受`.starttls()`保护的 SMTP 连接发送一封纯文本电子邮件。如果需要，可以省略`server.ehlo()`行，因为它们被`.starttls()`和`.sendmail()`隐式调用:

```py
import smtplib, ssl

port = 587  # For starttls
smtp_server = "smtp.gmail.com"
sender_email = "my@gmail.com"
receiver_email = "your@gmail.com"
password = input("Type your password and press enter:")
message = """\
Subject: Hi there

This message is sent from Python."""

context = ssl.create_default_context()
with smtplib.SMTP(smtp_server, port) as server:
    server.ehlo()  # Can be omitted
    server.starttls(context=context)
    server.ehlo()  # Can be omitted
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
```

## 发送精美的电子邮件

Python 的内置`email`包允许你构建更多奇特的电子邮件，然后可以像你已经做的那样用`smtplib`传输。下面，你将学习如何使用`email`包发送带有 HTML 内容和附件的电子邮件。

### 包括 HTML 内容

如果你想格式化电子邮件中的文本(**粗体**、*斜体*等等)，或者如果你想添加任何图像、超链接或响应内容，那么 HTML 就非常方便了。当今最常见的电子邮件类型是 MIME(多用途互联网邮件扩展)多部分电子邮件，结合了 HTML 和纯文本。MIME 消息由 Python 的`email.mime`模块处理。有关详细描述，请查看文档。

由于并非所有的电子邮件客户端都默认显示 HTML 内容，而且出于安全原因，有些人选择只接收纯文本电子邮件，因此为 HTML 邮件添加纯文本替代内容非常重要。因为电子邮件客户端将首先呈现最后一个多部分附件，所以请确保在纯文本版本之后添加 HTML 消息。

在下面的例子中，我们的`MIMEText()`对象将包含我们的消息的 HTML 和纯文本版本，并且`MIMEMultipart("alternative")`实例将这些合并成一个具有两个可选呈现选项的消息:

```py
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "my@gmail.com"
receiver_email = "your@gmail.com"
password = input("Type your password and press enter:")

message = MIMEMultipart("alternative")
message["Subject"] = "multipart test"
message["From"] = sender_email
message["To"] = receiver_email

# Create the plain-text and HTML version of your message
text = """\
Hi,
How are you?
Real Python has many great tutorials:
www.realpython.com"""
html = """\
<html>
 <body>
 <p>Hi,<br>
 How are you?<br>
 <a href="http://www.realpython.com">Real Python</a> 
 has many great tutorials.
 </p>
 </body>
</html>
"""

# Turn these into plain/html MIMEText objects
part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)
message.attach(part2)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )
```

在这个例子中，首先将纯文本和 HTML 消息定义为字符串，然后将它们存储为`plain` / `html` `MIMEText`对象。然后，这些可以按此顺序添加到`MIMEMultipart("alternative")`消息中，并通过您与电子邮件服务器的安全连接发送出去。记得在纯文本选项后添加 HTML 消息，因为电子邮件客户端会尝试先呈现最后一个子部分。

### 使用`email`包添加附件

为了将二进制文件发送到设计用于处理文本数据的电子邮件服务器，需要在传输之前对它们进行编码。这通常使用 [`base64`](https://docs.python.org/3/library/base64.html) 来完成，它将二进制数据编码成可打印的 ASCII 字符。

下面的代码示例显示了如何发送附件为 [PDF 文件](https://realpython.com/creating-modifying-pdf/)的电子邮件:

```py
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

subject = "An email with attachment from Python"
body = "This is an email with attachment sent from Python"
sender_email = "my@gmail.com"
receiver_email = "your@gmail.com"
password = input("Type your password and press enter:")

# Create a multipart message and set headers
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message["Bcc"] = receiver_email  # Recommended for mass emails

# Add body to email
message.attach(MIMEText(body, "plain"))

filename = "document.pdf"  # In same directory as script

# Open PDF file in binary mode
with open(filename, "rb") as attachment:
    # Add file as application/octet-stream
    # Email client can usually download this automatically as attachment
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())

# Encode file in ASCII characters to send by email 
encoders.encode_base64(part)

# Add header as key/value pair to attachment part
part.add_header(
    "Content-Disposition",
    f"attachment; filename= {filename}",
)

# Add attachment to message and convert message to string
message.attach(part)
text = message.as_string()

# Log in to server using secure context and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, text)
```

`MIMEultipart()`消息接受 [RFC5233](https://tools.ietf.org/html/rfc5233.html) 样式的键/值对形式的参数，这些参数存储在一个字典中，并传递给 [`Message`](https://docs.python.org/3/library/email.compat32-message.html#module-email.message) 基类的 [`.add_header`方法](https://docs.python.org/2/library/email.message.html#email.message.Message.add_header)。

查看 Python 的`email.mime`模块的[文档](https://docs.python.org/3/library/email.mime.html)，了解更多关于使用 MIME 类的信息。

[*Remove ads*](/account/join/)

## 发送多封个性化邮件

假设您想给组织成员发送电子邮件，提醒他们缴纳会费。或者，您可能想给班上的学生发送个性化的电子邮件，告知他们最近作业的分数。在 Python 中，这些任务轻而易举。

### 用相关的个人信息制作一个 CSV 文件

发送多封个性化电子邮件的一个简单起点是[创建一个包含所有必需个人信息的 CSV(逗号分隔值)文件](https://realpython.com/python-csv/)。(确保不要在未经他人同意的情况下分享他人的隐私信息。)CSV 文件可以被视为一个简单的表格，其中第一行通常包含列标题。

下面是文件`contacts_file.csv`的内容，我将它保存在与我的 Python 代码相同的文件夹中。它包含一组虚构人物的姓名、地址和等级。我使用了`my+modifier@gmail.com`结构来确保所有的电子邮件都在我自己的收件箱里，在这个例子中是 my@gmail.com 的:

```py
name,email,grade
Ron Obvious,my+ovious@gmail.com,B+
Killer Rabbit of Caerbannog,my+rabbit@gmail.com,A
Brian Cohen,my+brian@gmail.com,C
```

创建 CSV 文件时，请确保用逗号分隔值，周围没有任何空格。

### 循环发送多封电子邮件

下面的代码示例向您展示了如何打开一个 CSV 文件并遍历其内容行(跳过标题行)。为了确保在您向所有联系人发送电子邮件之前代码能够正常工作，我为每个联系人打印了`Sending email to ...`，稍后我们可以用实际发送电子邮件的功能来替换它:

```py
import csv

with open("contacts_file.csv") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for name, email, grade in reader:
        print(f"Sending email to {name}")
        # Send email here
```

在上面的例子中，使用`with open(filename) as file:`确保你的文件在代码块的末尾关闭。`csv.reader()`使得逐行读取 CSV 文件并提取其值变得容易。`next(reader)`行跳过标题行，因此下面的行`for name, email, grade in reader:`在每个逗号处拆分后续行，并将结果值存储在当前联系人的字符串`name`、`email`和`grade`中。

如果您的 CSV 文件中的值在一侧或两侧包含空格，您可以使用`.strip()`方法删除它们。

### 个性化内容

您可以使用 [`str.format()`](https://realpython.com/python-string-formatting/) 来填充花括号占位符，从而在消息中添加个性化内容。比如`"hi {name}, you {result} your assignment".format(name="John", result="passed")`会给你`"hi John, you passed your assignment"`。

从 Python 3.6 开始，使用 [f-strings](https://realpython.com/python-f-strings/) 可以更优雅地完成字符串格式化，但是这需要在 f-string 本身之前定义占位符。为了在脚本的开头定义电子邮件消息，并在循环 CSV 文件时为每个联系人填充占位符，使用了较老的`.format()`方法。

考虑到这一点，您可以设置一个通用的邮件正文，其中的占位符可以根据个人情况进行定制。

### 代码示例

下面的代码示例允许您向多个联系人发送个性化电子邮件。它循环遍历每个联系人的带有`name,email,grade`的 CSV 文件，如上面的[示例所示。](https://realpython.com/python-send-email/#make-a-csv-file-with-relevant-personal-info)

一般消息在脚本的开头定义，对于 CSV 文件中的每个联系人，其`{name}`和`{grade}`占位符被填充，个性化电子邮件通过与 Gmail 服务器的安全连接发送出去，如您之前所见:

```py
import csv, smtplib, ssl

message = """Subject: Your grade

Hi {name}, your grade is {grade}"""
from_address = "my@gmail.com"
password = input("Type your password and press enter: ")

context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(from_address, password)
    with open("contacts_file.csv") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for name, email, grade in reader:
            server.sendmail(
                from_address,
                email,
                message.format(name=name,grade=grade),
            )
```

[*Remove ads*](/account/join/)

## 雅格邮件

有多个旨在使发送电子邮件更容易的库，如[信封](https://github.com/tomekwojcik/envelopes)、[侧翼](https://github.com/mailgun/flanker)和 [Yagmail](https://pypi.org/project/yagmail/) 。Yagmail 是专门为 gmail 设计的，它通过友好的 [API](https://realpython.com/python-api/) 极大地简化了发送电子邮件的过程，正如你在下面的代码示例中看到的:

```py
import yagmail

receiver = "your@gmail.com"
body = "Hello there from Yagmail"
filename = "document.pdf"

yag = yagmail.SMTP("my@gmail.com")
yag.send(
    to=receiver,
    subject="Yagmail test with attachment",
    contents=body, 
    attachments=filename,
)
```

这个代码示例使用`email`和`smtplib` 发送一封带有 [PDF](https://realpython.com/pdf-python/) 附件的电子邮件，这只是我们的[示例所需行数的一小部分。](https://realpython.com/python-send-email/#adding-attachments-using-the-email-package)

设置 Yagmail 时，你可以将 gmail 验证添加到操作系统的密钥环中，如文档中的[所述。如果不这样做，Yagmail 会在需要时提示您输入密码，并自动将其存储在 keyring 中。](https://yagmail.readthedocs.io/en/latest/api.html#authentication)

## 交易电子邮件服务

如果你打算发送大量的电子邮件，希望看到电子邮件的统计数据，并希望确保可靠的交付，它可能值得看看交易电子邮件服务。虽然以下所有服务都有发送大量电子邮件的付费计划，但它们也有免费计划，所以你可以尝试一下。其中一些免费计划无限期有效，可能足以满足您的电子邮件需求。

下面是一些主要交易电子邮件服务的免费计划的概述。点击提供商名称会将您带到他们网站的定价部分。

| 供应者 | 免费计划 |
| --- | --- |
| [发送网格](https://sendgrid.com/marketing/sendgrid-services-cro/#compare-plans) | 前 30 天 40，000 封电子邮件，然后每天 100 封 |
| [正在发送](https://www.sendinblue.com/pricing/) | 300 封电子邮件/天 |
| [气枪](https://www.mailgun.com/pricing-simple) | 前 10，000 封电子邮件免费 |
| [里程数](https://www.mailjet.com/pricing/) | 200 封电子邮件/天 |
| [亚马逊 SES](https://aws.amazon.com/free/?awsf.Free%20Tier%20Types=categories%23alwaysfree) | 每月 62，000 封电子邮件 |

你可以运行[谷歌搜索](https://www.google.co.uk/search?q=transactional+email+providers+comparison)来看看哪个提供商最适合你的需求，或者尝试几个免费计划来看看你最喜欢使用哪个 API。

## 发送网格代码示例

这里有一个使用 [Sendgrid](https://sendgrid.com/marketing/sendgrid-services-cro/#compare-plans) 发送电子邮件的代码示例，让您感受一下如何使用 Python 的事务性电子邮件服务:

```py
import os
import sendgrid
from sendgrid.helpers.mail import Content, Email, Mail

sg = sendgrid.SendGridAPIClient(
    apikey=os.environ.get("SENDGRID_API_KEY")
)
from_email = Email("my@gmail.com")
to_email = Email("your@gmail.com")
subject = "A test email from Sendgrid"
content = Content(
    "text/plain", "Here's a test email sent through Python"
)
mail = Mail(from_email, subject, to_email, content)
response = sg.client.mail.send.post(request_body=mail.get())

# The statements below can be included for debugging purposes
print(response.status_code)
print(response.body)
print(response.headers)
```

要运行此代码，您必须首先:

*   [注册一个(免费的)Sendgrid 账户](https://sendgrid.com/free/?source=sendgrid-python)
*   [请求 API 密钥](https://app.sendgrid.com/settings/api_keys)用于用户验证
*   通过在命令提示符下键入`setx SENDGRID_API_KEY "YOUR_API_KEY"`来添加您的 API 密钥(永久存储此 API 密钥)或键入`set SENDGRID_API_KEY YOUR_API_KEY`来仅存储当前客户端会话的 API 密钥

关于如何为 Mac 和 Windows 设置 Sendgrid 的更多信息可以在 [Github](https://github.com/sendgrid/sendgrid-python) 上的知识库自述文件中找到。

## 结论

您现在可以启动安全的 SMTP 连接，并向联系人列表中的人发送多封个性化电子邮件！

您已经学习了如何发送一封包含纯文本选项的 HTML 电子邮件，以及如何在电子邮件中附加文件。当你使用 gmail 账户时， [Yagmail](https://pypi.org/project/yagmail/) 包简化了所有这些任务。如果你打算发送大量的电子邮件，值得考虑交易电子邮件服务。

享受用 Python 发送电子邮件的乐趣，记住:[请勿发送垃圾邮件](https://www.youtube.com/watch?v=UO7HY7Nz398)！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 发邮件**](/courses/sending-emails-python/)******