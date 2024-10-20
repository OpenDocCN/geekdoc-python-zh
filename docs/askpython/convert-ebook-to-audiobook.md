# 使用 Python 将电子书转换为有声读物

> 原文：<https://www.askpython.com/python/examples/convert-ebook-to-audiobook>

读书是一个好习惯，但是听书让这个过程变得更加方便。有声读物取代了传统书籍，因为我们可以随时随地轻松地听它们。如果我们可以编写一个有声读物制作程序，将电子书 pdf 转换成有声读物并为我们阅读，这将有多大用处？

在本教程中，我们将使用 Python 构建一个有声读物制作工具，它可以为我们朗读一本书。

## 使用 Python 阅读电子书作为有声读物

让我们进入制作图书阅读器或者将 PDF 电子书转换成有声读物的 Python 脚本的过程吧！

### 1.安装所需的模块

首先，我们需要安装所需的模块，这将大大简化我们编写有声读物制作程序的工作。我们使用 [pip 包管理器](https://www.askpython.com/python-modules/python-pip)安装 pyttsx3 和 PyPDF3 模块。

pyttsx3 是 Python 中的文本到语音转换库，而 PyPDF3 是 Python 中读取和编辑 PDF 文件的库。

```py
pip install pyttsx3
pip install PyPDF3

```

### 2.导入 PDF 阅读器和 TTS 模块

在 python 文件中，我们从导入所需的模块开始。

```py
import PyPDF3
import pyttsx3

```

现在我们初始化 pyttsx3 引擎对象来读取。

```py
engine = pyttsx3.init()

```

### 3.打开并阅读 PDF

现在我们已经初始化了我们的语音引擎，我们需要打开 PDF 来阅读它的内容。我们将 pdf 的名称传递给 open 方法，如下所示:

如果 PDF 与 python 脚本不在同一个目录中，您需要传递名称和位置。

```py
book = open('sample.pdf', 'rb')

```

为了逐行阅读 pdf 内容，我们使用 PyPDF3 模块的 PdffileReader 方法，如下所示:

然后，我们使用 extractText 方法从 pdf 阅读器的对象中提取文本。

```py
pdfRead= PyPDF3.PdfFileReader(book)

#to start the reading from 1st page in the pdf
page = pdfRead.getPage(0)

#to extract text to read
text = page.extractText()

```

### 4.朗读 PDF

当我们打开时，阅读 pdf 内容，我们现在需要将这些数据输入到我们的 pyttsx3 库的语音引擎中

```py
#takes in message to read or text
engine.say(text)

engine.runAndWait()

```

在执行脚本时，代码开始读取传递的 PDF。最终代码如下所示:

```py
import PyPDF3
import pyttsx3

engine = pyttsx3.init()

book = open('sample.pdf', 'rb')
pdfRead= PyPDF3.PdfFileReader(book)

#to start the reading from 1st page in the pdf
page = pdfRead.getPage(0)

#to extract text to read
text = page.extractText()

#takes in message to read or text
engine.say(text)

engine.runAndWait()

```

### 5.改变演讲

pyttsx3 库为我们提供了各种类型的语音更改，例如:

改变**语速**的设置

```py
rate = engine.getProperty('rate')   # gets the current rate of speech
engine.setProperty('rate', 125)     # sets up new rate of speech (passed in as 125 to change to 1.25x or 150 to make it to 1.5x)

```

改变**语音**的设置

```py
voices = engine.getProperty('voices')       # gets the current voice type

#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. 0 for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

```

改变**音量的设置**

```py
volume = engine.getProperty('volume')   #gets the current volume (min=0 and max=1)

engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

```

我们还可以使用下面的代码**将完整的有声读物文件**——意思是完整的 pdf(书)以语音的形式保存在**音频文件** (type .mp3)中:

```py
engine.save_to_file('text, 'audiobook.mp3')

```

## 结论

这就是如何用 Python 编写有声读物制作程序的教程。我们希望你喜欢这个关于将 PDF 转换成有声读物的简短教程。继续玩这个脚本，让它更加直观和自动化！