# 如何用 Python 播放音乐？

> 原文：<https://www.askpython.com/python-modules/play-music-in-python>

你好，学习伙伴！今天我们将学习如何使用几行简单的代码在 Python 中播放音乐。

## 方法 playsound 模块

playsound 库是一个跨平台模块，可以播放音频文件。这没有任何依赖性，只需使用 pip 命令安装库，就可以开始了！

要播放音乐，我们只需使用`playsound`函数，并将音乐文件路径作为参数传递。该库适用于`mp3`和`wav`文件。

相同的代码如下所示:

```py
from playsound import playsound
playsound('Music1.mp3')

```

音乐在后台播放一次，然后程序准备好执行下一部分代码。

## 方法 pydub 库

pydub 库仅适用于。wav 文件格式。通过使用这个库，我们可以播放，分割，合并，编辑我们的。wav 音频文件。

为了让这个库工作，我们从`playdub.playback`模块导入了两个函数，即`AudioSegment`和`play`模块。

然后我们简单地载入歌曲。wav 格式并播放歌曲。相同的代码如下所示:

```py
from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_wav('Music1.wav')
play(song)

```

## 方法 3:使用零食声音套件

零食声音套件可用于播放几乎所有格式的音频文件，包括 WAV、AU、AIFF、MP3、CSL、SD、SMP 和 NIST/球体。

这个库需要 [GUI 模块`Tkinter`](https://www.askpython.com/python-modules/tkinter/tkinter-buttons) 来播放声音。所以我们需要在导入零食声音套件之前导入 tkinter 模块。

通过零食声音工具包播放音频文件包括创建一个 Tk 窗口并初始化它。然后调用`sound`函数和`read`函数来加载音乐。

最后，我们使用`play`功能来播放音乐。相同的代码如下所示:

```py
from Tkinter import *
import tkSnack

wind = Tk()
tkSnack.initializeSnack(wind)

snd = tkSnack.Sound()
snd.read('Music1.wav')
snd.play(blocking=1)

```

## 输出音乐

下面的音乐将是每种方法中播放的输出背景音乐。

Music Played

## 结论

今天，我们学习了使用简单的代码行和各种库在 python 中演奏音乐。厉害！

自己尝试代码，用 Python 播放美妙的音乐。感谢您的阅读！编码快乐！