# 使用 Python 发送电子邮件

> 原文：<https://www.pythoncentral.io/sending-email-with-python/>

Python 是一种众所周知的编程语言，在执行一个特殊的程序——解释器时，其源代码被部分转换成机器代码。许多人选择 Python 是因为它简单易学，可以跨多个平台工作。此外，python 软件通常可以免费下载，兼容各种类型的系统，加速开发。

和其他任何编程语言一样，Python 有自己的特点:

*   得益于基本语法，开发者可以轻松阅读和理解 Python 软件；
*   与其他编程语言相比，Python 可以使用更少的代码行来创建程序，从而帮助开发人员提高工作效率；
*   Python 标准库足够大，包含几乎任何任务的可重用代码；
*   Python 的另一个重要优势是可以很容易地与 C、C++、Java 等其他编程语言结合。

[![programming language](img/bae42660c59993ada581f152f3287b54.png)](https://www.pythoncentral.io/wp-content/uploads/2022/11/programming-language.jpg)

Python 有几个标准用例。它在编写服务器代码时很有用，因为它为复杂的服务器功能提供了许多由预写代码组成的库。数据科学家广泛使用 Python ML 库来运行机器学习模型，并构建分类器来高效地对数据 进行分类。最近，Python 也开始被积极地用于发送电子邮件。请阅读下面的内容，详细了解如何做到这一点。

## **如何使用 Python 发送 Gmail 邮件？**

如今，电子邮件营销是任何商业推广策略的重要组成部分。许多人已经看到了电子邮件通讯的有效性。在此了解[](https://stripo.email/blog/send-mass-emails-using-gmail/)如何使用 Gmail 发送批量邮件。使用 Python 发送 Gmail 邮件也有几种方式。最流行的是通过 SMTP 协议。

### **SMTP**

SMTP 是一种应用层协议。它用于在各种邮件服务器和外部服务(如移动设备上的邮件客户端)之间建立通信。SMTP 只是一种传递协议。因此，您无法使用它接收电子邮件。你可以发邮件。 [IMAP](https://www.techtarget.com/whatis/definition/IMAP-Internet-Message-Access-Protocol) 通常用于接收邮件。

包括 Gmail 在内的各种现代电子邮件服务不要求在内部邮件服务器上使用 SMTP。因此，该协议通常仅作为该服务通过 smtp.gmail.com 服务器的外部接口提供。邮件客户端主要在台式电脑或手机(雷鸟、Outlook 等)上使用。).

### **打开连接**

Python 已经有了一个允许你连接到 SMTP 服务器的库。它叫 smtplib，是 Python 自带的。这个库处理[协议](https://www.pythoncentral.io/python-generators-and-yield-keyword/)的各个部分，比如连接、认证、验证和发送电子邮件。

使用 smtplib 库时，可以提供到邮件服务器的连接。这样的连接是不安全的。它没有加密。默认情况下使用端口 25。因此，创建更可靠的连接是值得注意的。

### **确保安全连接**

[![secure connection](img/6281fb93fa47a91bc14a3c4a3fd5962e.png)](https://www.pythoncentral.io/wp-content/uploads/2022/11/secure-connection.jpg)

当通过 SSL/TLS 保护到 SMTP 协议的连接时，它通过 465 端口，通常被称为 SMTPS。不言而喻，为什么要提供这样的连接。你可能明白这个问题的重要性。

在库 smtplib 中有几种方法可以保护 SMTP 连接。最流行的方法包括建立不安全的连接，随后切换到[](https://www.cloudflare.com/learning/ssl/transport-layer-security-tls/)【TLS】。另一种选择是从一开始就创建一个安全的 SSL 连接。

### **创建电子邮件**

建立安全连接后，您可以直接创建电子邮件。事实上，电子邮件是由换行符连接的普通文本行。大多数电子邮件包含“发件人”、“收件人”、“主题”和“文本”等字段。每行包含几个具有特定数据的字段。没有二进制协议(根据“请求-响应”模式传输数据)，没有 XML，也没有 JSON。只有由字段分隔的行。要参数化字段，可以使用 Python 的字段格式。

### **Gmail 认证**

使用 SMTP 发送 Gmail 电子邮件之前，您可能需要进行身份验证。如果您使用 Gmail 邮件服务作为您的 ISP，您需要获得 Google 的许可才能通过 SMTP 协议进行连接，该协议被评为不太安全的方法。如果您使用其他邮件提供商的服务，则不需要执行任何附加步骤。

### **用 Python 发送电子邮件**

一旦你建立了 SMTP 连接并完成了谷歌认证，你就可以使用 Python 发送简单的邮件和带附件的邮件了。的使用。sendmail():是必需的。

## **结束语**

发送、检查和回复电子邮件是一项相当耗时的任务，尤其是当你为大量的人或客户做这件事时，你只需要更改收件人的详细信息。然而，你可以在这方面自动化许多事情，从长远来看，这将节省你很多时间。自动化发送电子邮件的一种方式是通过 Python。要成功地使用它，只需经历几个简单的步骤就足够了，本文已经对此进行了详细描述。

尽量考虑我们所有的建议，提供可靠的 SMTP 连接，用 Python 发邮件一定会成功！