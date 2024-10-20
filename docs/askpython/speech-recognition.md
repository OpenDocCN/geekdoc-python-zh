# Python 语音识别模块——完整介绍

> 原文：<https://www.askpython.com/python-modules/speech-recognition>

你好。今天，让我们学习如何使用 Python 编程语言中的`speech recognition`库将语音转换成文本。所以让我们开始吧！

## 语音识别简介

语音识别被定义为自动识别人类语音，并被认为是制作 Alexa 或 Siri 等应用程序时最重要的任务之一。

Python 附带了几个支持语音识别特性的库。我们将使用`speech recognition`库，因为它是最简单、最容易学习的。

## 导入语音识别模块

和往常一样，第一步是导入所需的库。在这种情况下，我们只需要导入`speech_recognition`库。

```py
import speech_recognition as SR

```

如果语句出错，您可能需要使用`pip`命令安装库。

## 用 Python 实现语音识别

为了将语音从音频转换成文本，我们需要来自`speech_recognition`模块的`Recognizer`类来创建一个对象，该对象包含进一步处理所需的所有函数。

### 1.正在加载音频

继续之前，我们需要下载一个音频文件。我用来开始的是艾玛·沃森的演讲，可以在这里找到。

我们下载了音频文件，并将其转换为`wav`格式，因为它最适合识别语音。但是要确保将它保存到 Python 文件所在的文件夹中。

为了加载音频，我们将使用`AudioFile`功能。该函数打开文件，读取其内容，并将所有信息存储在名为`source.`的音频文件实例中

我们将遍历源代码并做以下事情:

1.  每个音频都包含一些`noise`，可以使用`adjust_for_ambient_noise`功能移除。
2.  利用`record`方法读取音频文件，并将某些信息存储到一个变量中，供以后读取。

下面是加载音频的完整代码。

```py
import speech_recognition as SR
SR_obj = SR.Recognizer()

info = SR.AudioFile('speech.wav')
with info as source:
    SR_obj.adjust_for_ambient_noise(source)
    audio_data = SR_obj.record(source,duration=100)

```

这里我们还提到了一个称为`duration`的参数，因为对于较长的音频，识别语音需要更多的时间。所以威尔只会录下前 100 秒的音频。

### 2.从音频中读取数据

既然我们已经成功加载了音频，我们现在可以调用`recognize_google()`方法并识别音频中的任何语音。

该方法可能需要几秒钟，具体取决于您的互联网连接速度。在处理之后，该方法返回程序能够从第一个 100 秒中识别的最佳语音。

相同的代码如下所示。

```py
import speech_recognition as SR
SR_obj = SR.Recognizer()

info = SR.AudioFile('speech.wav')
with info as source:
    SR_obj.adjust_for_ambient_noise(source)
    audio_data = SR_obj.record(source,duration=100)
SR_obj.recognize_google(audio_data)

```

输出结果是音频中的一串句子，结果非常好。精确度可以通过使用更多的功能来提高，但是现在它只提供基本的功能。

```py
"I was appointed 6 months and I have realised for women's rights to often become synonymous with man heating if there is one thing I know it is that this has to stop someone is by definition is the belief that men and women should have equal rights and opportunities is the salary of the economic and social policy of the success of a long time ago when I was 8 I was confused sinkhole but I wanted to write the play Aise the width on preparing for the 14 isostasy sacralized elements of the media 15 my girlfriend Statue of Liberty sports team because they don't want to pay monthly 18 18 Mai Mela friends were unable to express their feelings I decided that I am business analyst at the seams and complicated to me some recent research has shown me feminism has become"

```

## 结论

恭喜你！今天，在本教程中，您学习了如何从音频中识别语音，并将其显示在屏幕上。

我还想提一下，语音识别是一个非常深奥和庞大的概念，我们在这里学到的知识仅仅触及了整个主题的表面。

感谢您的阅读！