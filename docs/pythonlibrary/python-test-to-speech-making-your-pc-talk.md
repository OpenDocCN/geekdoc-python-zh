# Python 文本到语音转换:让你的电脑说话

> 原文：<https://www.blog.pythonlibrary.org/2010/04/02/python-test-to-speech-making-your-pc-talk/>

在我被目前的工作录用后不久，我的老板发给我一个关于 Python 的脚本(我想是基于这篇文章)和一个叫做 [pyTTS](http://www.cs.unc.edu/Research/assist/doc/pytts/) 的文本到语音转换模块。这是在 Python 2.5 发布之后。无论如何，它基本上是 win32com 模块的一个很好的包装器，可以与微软语音 API (SAPI)通信。

我不会详细介绍 pyTTS，因为我上面链接的那篇文章已经介绍过了，但是我会给你一个快速的介绍。如果你去作者的[网站](http://mindtrove.info/clique/)你会发现他已经转移到屏幕阅读技术，并创建了一个名为 Clique 的程序。我不确定这是不是用 Python 写的。我在网站上寻找可以下载该软件的地方，最终找到了这个:【http://sourceforge.net/projects/uncassist/files/

据我所知，他只正式支持 Python 2.3-2.5。然而，由于 pyTTS 基本上只是包装了对 SAPI 的 win32com 调用，并且 PyWin32 模块支持 Python 2.x-3.x，所以我认为让 pyTTS 与新版本一起工作是相当容易的。

*注意:你将需要[微软 SAPI 5.1 可再发行版](http://www.cs.unc.edu/Research/assist/packages/SAPI5SpeechInstaller.msi)、[额外 MS voices](http://www.cs.unc.edu/Research/assist/packages/SAPI5VoiceInstaller.msi) 和 [PyWin32](http://sourceforge.net/projects/pywin32/files/) 。*

让我们快速看一下如何使用本模块:

```py

import pyTTS
tts = pyTTS.Create()
tts.SetVoiceByName('MSSam')
tts.Speak("Hello, fellow Python programmer")

```

你不必把声音设置成我认为的默认声音，但这样做很有趣。Speak 方法接受各种标志作为它的第二个参数。一个例子是 pyTTS.tts_async，它将说话置于异步模式。你也可以通过这样做来改变音量:tts。体积= 50。您可以选择 0-100%之间的任何值。

如果你看这篇[文章](http://mindtrove.info/articles/synthesizing-speech-with-pytts/)，它会教你如何让 pyTTS 念出你喂它的单词。

下面是您如何使用 PyWin32 完成上面的大部分示例:

```py

from win32com.client import constants
import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice", constants.SVSFlagsAsync)
speaker.Speak("Hello, fellow Python programmer")

```

在研究本文时，我注意到 pyTTS 背后的开发人员还开发了一个跨平台的文本到语音转换模块，名为 [pyttsx](http://pypi.python.org/pypi/pyttsx) 。我没有用过，但是我鼓励你试一试。其他值得一看的模块有 [pySpeech](http://code.google.com/p/pyspeech/) 和这个让 Python 识别语音的酷配方:【http://www.surguy.net/articles/speechrecognition.xml】T4

好吧，这更多的是对 Python 中酷的语音相关模块的调查，而不是对代码的调查。不过，我希望这将证明有助于你的努力。