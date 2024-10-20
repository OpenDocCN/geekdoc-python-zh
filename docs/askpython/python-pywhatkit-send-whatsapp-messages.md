# Python Pywhatkit–使用 Python 发送 WhatsApp 消息

> 原文：<https://www.askpython.com/python-modules/python-pywhatkit-send-whatsapp-messages>

这篇文章将向您介绍 python pywhatkit 库，以及如何使用它通过几行 python 代码自动发送 WhatsApp 消息。

## Python Pywhatkit

Pywhatkit 是一个流行的 python 库，可以自动向某人的 WhatsApp 手机号码发送消息。

它使用 WhatsApp 网络发送这些信息。

Pywhatkit 是用 python 3.4+编写的 WhatsApp Messenger 的 Python 包。它简单、优雅，而且是 100% python 式的。

## pywhatkit 库的特性

*   自动向 Whatsapp 上的联系人/联系人发送消息
*   自动向群组发送消息
*   播放 YouTube 视频或短片
*   也用于将文本(字符串)转换为手写
*   用 HTML 代码发送邮件

## 使用 Pywhatkit 通过 Python 发送 WhatsApp 消息

现在让我们进入使用 pywhatkit 的步骤，并使用 Python 发送我们的第一条 WhatsApp 消息。

### 1.安装库

因为在 Python3 中，pywhatkit 不是预装的，所以可以使用 [pip 命令](https://www.askpython.com/python-modules/python-pip)来安装它:

```py
pip install pywhatkit

```

### 2.发送 WhatsApp 消息:

使用 pywhatkit，Whatsapp 消息可以自动发送到 Whatsapp 上的任意号码。

**注意:你必须在浏览器中登录 WhatsApp，这意味着你需要在默认浏览器中设置你的 Whatsapp 网络账户。**

自动 WhatsApp 是使用 pywhatkit 库的 **sendmsg()** 方法发送的。它有几个特点，这些特点以例子的形式列出来，展示了如何向个人或群体发送消息或图像。

**语法** : pywhatkit.sendmsg("接收方手机号码"，"待发送消息"，小时，分钟)

方法的 ***参数*—**

*   接收人的手机号码:应该是字符串格式，必须包括国家代码，写在手机号码之前。
*   要发送的消息:字符串格式。
*   小时:该方法遵循 24 小时时间格式。
*   分钟:应该在 00-59 之间。

### 3.发送消息的代码

```py
import pywhatkit as pwk

# using Exception Handling to avoid unexpected errors
try:
     # sending message in Whatsapp in India so using Indian dial code (+91)
     pwk.sendwhatmsg("+91XXXXXX5980", "Hi, how are you?", 20, 34)

     print("Message Sent!") #Prints success message in console

     # error message
except: 
     print("Error in sending the message")

```

这个程序将在指定的时间(晚上 8:34)向接收者(传递的电话号码)发送一条消息，消息将是“嗨，你好吗？”

**注意**:默认情况下，该方法会在指定时间前 15 秒打开浏览器，以弥补在默认浏览器上加载 WhatsApp 网站的时间。

### 在 WhatsApp 中发送消息的更多功能

###### **发送消息后关闭标签页** (WhatsApp Web 标签页)。

```py
pwk.sendwhatmsg("+91XXXXXX5980", "Hi", 18, 15, True, 5)

```

这里我们将 5 秒作为关闭标签页的时间，true 也代表标签页是否需要关闭的布尔值，如果是 True，它将关闭，否则如果是 false，它将不会关闭

###### **向**组**发送图像**以及标题为 Hi

这里 Media/image.png 指的是要发送的图像

```py
pwk.sendwhats_image("Group_Name", "Media/image.png", "Hi")

pwk.sendwhats_image("Name", "Media/images.png")

```

###### **向群组发送消息**:

```py
pwk.sendwhatmsg_to_group("Group_Name", "Hey Guys! How's everybody?", 11, 0)

# it is similar to sending a message to a single person but here we are sending the message in a group

```

###### **即时群发消息**

我们用这个在一个组中即时发送消息，就好像我们写 0 小时，0 分钟，然后它会在 12:00 AM 发送消息

```py
pwk.sendwhatmsg_to_group_instantly("Group_Name", "Hey Guys Again!")

```

### 常见意外错误

您可能遇到的一些常见错误及其解决方案:

*   "语法错误:十进制整数文本中不允许前导零；对八进制整数使用 0o 前缀"

解决方法:用除 0 以外的任何数字开始一分钟的辩论。

*   "发出警告("互联网速度慢，提取信息可能需要更长时间")"
    "警告:互联网速度慢，提取信息可能需要更长时间"

解决方案:确保你有一个强大的互联网连接

## 结论

教程到此为止！希望您已经很好地了解了 pywhatkit 以及如何使用 Pywhatkit 库自动发送 WhatsApp 消息，并准备在您的代码中实现它。请继续关注更多关于 python 的教程。