# 如何在 Python 中创建线程

> 原文：<https://www.pythoncentral.io/how-to-create-a-thread-in-python/>

## Python 线程简介

什么是线程？简单地说，试着把它们想象成在一个进程中同时运行几个程序。当您在程序中创建一个或多个线程时，它们会同时执行，彼此独立，最重要的是，它们可以毫无困难地共享信息。

这些特性使线程在网络编程等情况下变得轻量和方便，当您试图 ping(发送网络数据包或请求)数百个工作站，而您不想一个接一个地 ping 它们时！由于网络回复可能会延迟很长时间，如果不同时 ping 许多工作站，程序将会非常慢。本文将向您展示如何在 Python 中创建线程，以及如何在一般情况下使用它们。

## 在 Python 中创建线程

Python 中的线程很容易。你需要做的第一件事是使用下面的代码`import Thread`:

```py

from threading import Thread

```

要在 Python 中创建一个线程，你需要让你的类*像线程一样工作*。为此，您应该从`Thread`类中继承您的类:

```py

class MyThread(Thread):

    def __init__(self):

        pass

```

现在，我们的`MyThread`类是`Thread`类的子类。然后我们在类中定义一个`run`方法。当我们调用我们的`MyThread`类中任何对象的`start`方法时，这个函数将被执行。

完整类的代码如下所示。我们使用`sleep`函数让线程“休眠”(防止它在一段随机的时间内执行)。如果我们不这样做，代码会执行得如此之快，以至于我们无法注意到任何有价值的变化。

```py

class MyThread(Thread):

    def __init__(self, val):

        ''' Constructor. '''
线程。__init__(self) 
 self.val = val
def run(self): 
 for i in range(1，self . val):
print(' Value % d in thread % s ' %(I，self.getName()))
#在 1 ~ 3 秒之间随机休眠一段时间
seconds stosleep = randint(1，5)
print(' % s sleeping for % d seconds ... '% (self.getName()、secondsToSleep))
time . sleep(secondsToSleep)

```

为了创建线程，下一步是创建我们的线程支持类的一些对象(在这个例子中是两个)。我们调用每个对象的`start`方法——这反过来执行每个对象的`run`方法。

```py

# Run following code when the program starts

if __name__ == '__main__':

   # Declare objects of MyThread class

   myThreadOb1 = MyThread(4)

   myThreadOb1.setName('Thread 1')
mythtreadob 2 = mythtread(4)
mythtreadob 2.set name(' thread 2 ')”
#开始运行线程！
myth readob 1 . start()
myth readob 2 . start()
#等待线程完成...
myth readob 1 . join()
myth readob 2 . join()
打印('主终端...')

```

就是这样！注意，我们需要调用每个对象的`join`方法——否则，程序将在线程完成执行之前终止。

该程序的完整版本如下所示:

```py

from threading import Thread

from random import randint

import time
类 MyThread(线程):
def __init__(self，val): 
' ' '构造函数‘
线程。__init__(self) 
 self.val = val
def run(self): 
 for i in range(1，self . val):
print(' Value % d in thread % s ' %(I，self.getName()))
#在 1 ~ 3 秒之间随机休眠一段时间
seconds stosleep = randint(1，5)
print(' % s sleeping for % d seconds ... '% (self.getName()、secondsToSleep)
time . sleep(secondsToSleep)
#程序启动时运行以下代码
if _ _ name _ _ = = ' _ _ main _ _ ':
#声明 MyThread 类的对象
myth readob 1 = myth read(4)
myth readob 1 . setname(' Thread 1 ')
mythtreadob 2 = mythtread(4)
mythtreadob 2.set name(' thread 2 ')”
#开始运行线程！
myth readob 1 . start()
myth readob 2 . start()
#等待线程完成...
myth readob 1 . join()
myth readob 2 . join()
打印('主终端...')

```