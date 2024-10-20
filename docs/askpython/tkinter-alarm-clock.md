# Tkinter 闹钟——循序渐进指南

> 原文：<https://www.askpython.com/python-modules/tkinter/tkinter-alarm-clock>

你好。今天在本教程中，我们将开发一个基本的 Python Tkinter 闹钟。

难怪闹钟总是在我们睡觉、小睡时提醒我们，或者提醒我们我们总是不知道的工作。

***推荐阅读: [Python Tkinter GUI 计算器](https://www.askpython.com/python/examples/gui-calculator-using-tkinter)***

## 项目介绍

该项目利用了一些 python 库，即 datetime 和 Tkinter。

该项目利用当前的日期和时间以及一个功能，根据当前的日期和时间设置一个警报。

## 打造 Tkinter 闹钟

让我们不要再浪费时间了，现在就开始建设这个项目吧！

### 1.导入所需模块

在构建任何项目之前，第一步是导入项目所需的所有必要的库和模块。

```py
from tkinter import *
import datetime
import time
import winsound

```

让我们了解一下我们刚刚导入的每个模块:

1.  **Tkinter 模块**:帮助我们创建一个用户使用应用程序的窗口
2.  **[日期时间](https://www.askpython.com/python-modules/python-datetime-module)和[时间](https://www.askpython.com/python-modules/python-time-module)模块**:帮助我们处理日期和时间，并在需要时操纵它们。
3.  **winsound 模块**:有助于为我们的闹钟产生声音。

### 2.为警报创建功能

下一步是为闹钟创建函数。让我们先看看相同的代码。

```py
def Alarm(set_alarm_timer):
    while True:
        time.sleep(1)
        actual_time = datetime.datetime.now()
        cur_time = actual_time.strftime("%H:%M:%S")
        cur_date = actual_time.strftime("%d/%m/%Y")
        msg="Current Time: "+str(cur_time)
        print(msg)
        if cur_time == set_alarm_timer:
            winsound.PlaySound("Music.wav",winsound.SND_ASYNC)
            break

def get_alarm_time():
    alarm_set_time = f"{hour.get()}:{min.get()}:{sec.get()}"
    Alarm(alarm_set_time)

```

名为`Alarm`的函数处理应用程序的主要功能。该函数将用户在窗口输入框中设置的报警时间作为参数。

`sleep`函数停止程序的执行，直到得到用户输入的时间值。

然后，我们使用`datetime.now`函数获取当前日期和时间，并借助`strftime`函数将时间和日期存储到单独的变量中。

该程序检查当前时间是否与用户设置的报警时间相匹配。当条件为真时，使用`winsound`模块播放声音，否则计时器继续计时。

定义了一个新函数来从用户输入框中获取输入，并将其传递给前面的函数。

### 3.创建 Tkinter 窗口

最后一步是创建应用程序的主窗口，其中包含所有已定义的小部件和特性。相同的代码如下所示。

```py
window = Tk()
window.title("Alarm Clock")
window.geometry("400x160")
window.config(bg="#922B21")
window.resizable(width=False,height=False)

time_format=Label(window, text= "Remember to set time in 24 hour format!", fg="white",bg="#922B21",font=("Arial",15)).place(x=20,y=120)

addTime = Label(window,text = "Hour     Min     Sec",font=60,fg="white",bg="black").place(x = 210)
setYourAlarm = Label(window,text = "Set Time for Alarm: ",fg="white",bg="#922B21",relief = "solid",font=("Helevetica",15,"bold")).place(x=10, y=40)

hour = StringVar()
min = StringVar()
sec = StringVar()

hourTime= Entry(window,textvariable = hour,bg = "#48C9B0",width = 4,font=(20)).place(x=210,y=40)
minTime= Entry(window,textvariable = min,bg = "#48C9B0",width = 4,font=(20)).place(x=270,y=40)
secTime = Entry(window,textvariable = sec,bg = "#48C9B0",width = 4,font=(20)).place(x=330,y=40)

submit = Button(window,text = "Set Your Alarm",fg="Black",bg="#D4AC0D",width = 15,command = get_alarm_time,font=(20)).place(x =100,y=80)

window.mainloop()

```

## Tkinter 闹钟的完整代码

```py
from tkinter import *
import datetime
import time
import winsound

def Alarm(set_alarm_timer):
    while True:
        time.sleep(1)
        actual_time = datetime.datetime.now()
        cur_time = actual_time.strftime("%H:%M:%S")
        cur_date = actual_time.strftime("%d/%m/%Y")
        msg="Current Time: "+str(cur_time)
        print(msg)
        if cur_time == set_alarm_timer:
            winsound.PlaySound("Music.wav",winsound.SND_ASYNC)
            break

def get_alarm_time():
    alarm_set_time = f"{hour.get()}:{min.get()}:{sec.get()}"
    Alarm(alarm_set_time)

window = Tk()
window.title("Alarm Clock")
window.geometry("400x160")
window.config(bg="#922B21")
window.resizable(width=False,height=False)

time_format=Label(window, text= "Remember to set time in 24 hour format!", fg="white",bg="#922B21",font=("Arial",15)).place(x=20,y=120)
addTime = Label(window,text = "Hour     Min     Sec",font=60,fg="white",bg="black").place(x = 210)
setYourAlarm = Label(window,text = "Set Time for Alarm: ",fg="white",bg="#922B21",relief = "solid",font=("Helevetica",15,"bold")).place(x=10, y=40)

hour = StringVar()
min = StringVar()
sec = StringVar()

hourTime= Entry(window,textvariable = hour,bg = "#48C9B0",width = 4,font=(20)).place(x=210,y=40)
minTime= Entry(window,textvariable = min,bg = "#48C9B0",width = 4,font=(20)).place(x=270,y=40)
secTime = Entry(window,textvariable = sec,bg = "#48C9B0",width = 4,font=(20)).place(x=330,y=40)

submit = Button(window,text = "Set Your Alarm",fg="Black",bg="#D4AC0D",width = 15,command = get_alarm_time,font=(20)).place(x =100,y=80)

window.mainloop()

```

## 抽样输出

下面的视频展示了该应用程序的工作原理。您可以根据自己的喜好自定义窗口和变量。

## 结论

恭喜你！今天，我们成功地学习了如何使用 Python 的 Tkinter 模块制作闹钟。我们还学习了提取当前日期和时间，并在特定时刻播放声音。

希望你喜欢它！快乐学习！