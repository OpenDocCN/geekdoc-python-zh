# Python gtts 模块:如何在 Python 中将文本转换成语音？

> 原文：<https://www.askpython.com/python-modules/text-to-speech>

你好，学习伙伴！今天，我们将建立我们自己的文本到语音转换器！

我们开始吧！

## 项目介绍

**文本到语音**意味着将文本翻译成人类可以理解的语音。该应用程序对于那些在正确阅读句子和单词方面有困难的人来说非常有用。

在应用程序中，用户在输入框中输入一些文本，然后只需单击一下按钮，应用程序就会将文本转换成音频。

## 1.导入库

让我们从导入应用程序所需的库开始，如下所示:

1.  tkinter
2.  gTTS，
3.  播放声音

```py
from tkinter import *
from gtts import gTTS
from playsound import playsound

```

## 2.创建初始 Tkinter 窗口

首先，我们初始化了窗口，并为窗口添加了几何图形和其他配置，包括背景颜色和标题。

```py
window = Tk()
window.geometry("350x300") 
window.configure(bg='#FAD7A0')
window.title("TEXT TO SPEECH")

```

## 3.向窗口添加小部件

下一步包括在屏幕上添加标签、[输入框](https://www.askpython.com/python-modules/tkinter/tkinter-entry-widget)和[按钮](https://www.askpython.com/python-modules/tkinter/tkinter-buttons)。相同的代码如下所示。为了方便起见，突出显示了小部件声明。

对于这个应用程序，我们将使用三个按钮。一个是播放文本，第二个是重置应用程序，最后一个是退出应用程序。

```py
Label(window, text = "        TEXT TO SPEECH        ", font = "arial 20 bold", bg='black',fg="white").pack()

Msg = StringVar()

Label(window,text ="Enter Your Text Here: ", font = 'arial 20 bold', fg ='darkgreen').place(x=5,y=60)

entry_field = Entry(window, textvariable = Msg ,width ='30',font = 'arial 15 bold',bg="lightgreen")

entry_field.place(x=5,y=100)

Button(window, text = "PLAY TEXT", font = 'arial 15 bold' , width = '20',bg = 'orchid',fg="white").place(x=35,y=140)

Button(window, font = 'arial 15 bold',text = 'RESET APPLICATION', width = '20',bg = 'darkgreen',fg="white").place(x=35 , y = 190)

Button(window, font = 'arial 15 bold',text = 'EXIT APPLICATION', width = '20' , bg = 'red',fg="white").place(x=35 , y = 240)

```

## 4.为按钮创建将文本转换为语音的功能

我们将为这三个按钮定义三个功能，退出应用程序按钮非常简单，我们只需要销毁窗口。

下一个功能，即重置按钮，通过将其设置为空字符串来删除输入框的内容。需要最后一个函数来将文本转换成语音，这需要下面描述的几个函数。

1.  `get`:获取输入框中输入的文本，并存储在变量中
2.  `gTTS`:将传递给函数的消息转换成语音
3.  `save`:以 mp3 格式保存演讲
4.  `playsound`:播放上一步保存的语音

```py
def Text_to_speech():
    Message = entry_field.get()
    speech = gTTS(text = Message)
    speech.save('data.mp3')
    playsound('data.mp3')

def Exit():
    window.destroy()

def Reset():
    Msg.set("")

```

下一步是将`command`属性添加到按钮声明中，将函数连接到相应的按钮。

## 将文本转换为语音的最终代码

该项目的完整的最终代码如下所示。

```py
from tkinter import *
from gtts import gTTS
from playsound import playsound

def Text_to_speech():
    Message = entry_field.get()
    speech = gTTS(text = Message)
    speech.save('data.mp3')
    playsound('data.mp3')

def Exit():
    window.destroy()

def Reset():
    Msg.set("")

window = Tk()
window.geometry("350x300") 
window.configure(bg='#FAD7A0')
window.title("TEXT TO SPEECH")

Label(window, text = "        TEXT TO SPEECH        ", font = "arial 20 bold", bg='black',fg="white").pack()
Msg = StringVar()
Label(window,text ="Enter Your Text Here: ", font = 'arial 20 bold', fg ='darkgreen').place(x=5,y=60)

entry_field = Entry(window, textvariable = Msg ,width ='30',font = 'arial 15 bold',bg="lightgreen")
entry_field.place(x=5,y=100)

Button(window, text = "PLAY TEXT", font = 'arial 15 bold' , command = Text_to_speech ,width = '20',bg = 'orchid',fg="white").place(x=35,y=140)
Button(window, font = 'arial 15 bold',text = 'RESET APPLICATION', width = '20' , command = Reset,bg = 'darkgreen',fg="white").place(x=35 , y = 190)
Button(window, font = 'arial 15 bold',text = 'EXIT APPLICATION', width = '20' , command = Exit, bg = 'red',fg="white").place(x=35 , y = 240)

window.mainloop()

```

## 示例输出视频

下面的视频展示了该应用程序的工作原理。看看吧！

## 结论

恭喜你！我们已经成功地构建了文本到语音的 python tkinter 项目。希望你喜欢它！

感谢您的阅读！