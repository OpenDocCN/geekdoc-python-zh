# Python:如何判断 Windows 空闲了多长时间

> 原文：<https://www.blog.pythonlibrary.org/2010/05/05/python-how-to-tell-how-long-windows-has-been-idle/>

有一天，我收到了一个请求，要求创建一个脚本，可以告诉 Windows XP 机器空闲了多长时间，并在空闲了一定时间后提醒用户。我在谷歌上做了一点研究，找到了几个方法来完成这个壮举。我唯一能够使用的是 ctypes 示例，所以事不宜迟，让我们来看看吧！

以下 ctypes 相关代码摘自 [stackoverflow 论坛](http://stackoverflow.com/questions/911856/detecting-idle-time-in-python):

```py

from ctypes import Structure, windll, c_uint, sizeof, byref

# http://stackoverflow.com/questions/911856/detecting-idle-time-in-python
class LASTINPUTINFO(Structure):
    _fields_ = [
        ('cbSize', c_uint),
        ('dwTime', c_uint),
    ]

def get_idle_duration():
    lastInputInfo = LASTINPUTINFO()
    lastInputInfo.cbSize = sizeof(lastInputInfo)
    windll.user32.GetLastInputInfo(byref(lastInputInfo))
    millis = windll.kernel32.GetTickCount() - lastInputInfo.dwTime
    return millis / 1000.0

```

如果你不理解上面的代码，请在 ctypes 邮件列表上寻求帮助。我也不是完全理解。我明白了基本要点，但仅此而已。下面是我用来让上面的代码发挥作用的片段:

```py

while 1:
    GetLastInputInfo = int(get_idle_duration())
    print GetLastInputInfo
    if GetLastInputInfo == 480:
        # if GetLastInputInfo is 8 minutes, play a sound
        sound = r"c:\windows\media\notify.wav"
        winsound.PlaySound(sound, winsound.SND_FILENAME)
    if GetLastInputInfo == 560:
        # if GetLastInputInfo is 9 minutes, play a more annoying sound
        sound = r"c:\windows\media\ringout.wav"
        winsound.PlaySound(sound, winsound.SND_FILENAME)
        winsound.PlaySound(sound, winsound.SND_FILENAME)
        winsound.PlaySound(sound, winsound.SND_FILENAME)

    time.sleep(1)

```

在我的代码中，我检查机器是否空闲了 8 分钟和 9 分钟。根据空闲时间的长短，代码使用 Python winsound 模块播放特定的 wav 文件。在我们店里，有些机器闲置 10 分钟就会自动上锁。我们的用户不太喜欢这样，所以他们要求我们在机器即将锁定时发出声音警告他们。这就是这个脚本要完成的任务。希望你能更好地利用这些知识。