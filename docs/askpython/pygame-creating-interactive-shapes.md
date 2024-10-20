# Pygame:创建交互式形状

> 原文：<https://www.askpython.com/python-modules/pygame-creating-interactive-shapes>

你好。今天我们将创建一个普通的 pygame 屏幕，但是为了增加趣味，我们将为它添加交互式的形状。听起来很有趣，对吗？

所以让我们开始吧！

## 步骤 1:创建一个基本的 Pygame 屏幕

我们的第一个任务是[创建一个 pygame 屏幕](https://www.askpython.com/python-modules/python-pygame)，首先导入必要的模块，包括`pygame`模块、`pygame.locals`模块。我们还为背景音乐添加了`mixer`模块。

### 1.创建窗口

我们首先初始化 pygame，并通过设置屏幕的高度和宽度来创建一个窗口对象。

我们还将添加一个运行循环来处理各种事件，比如按下`X`按钮时关闭窗口。

### 2.添加背景图像

我们要做的下一件事是[添加背景图像](https://www.askpython.com/python-modules/pygame-looping-background)，首先加载图像并缩放图像以填充整个窗口。

然后在运行循环中使用`blit`和`update`函数添加图像。

### 3.添加背景音乐

我们将使用调音台模块的功能将音乐添加到我们的程序中。

首先，我们从音乐文件文件夹中导入音乐。同样，我们使用`music.load`功能，然后使用`music.play`功能播放音乐。

我们还将使用`music.set_volume`功能设置音乐的音量。

**设计基本定制屏幕的完整代码如下:**

```py
import pygame
from pygame.locals import *
from pygame import mixer

pygame.init()
width = 500
height = 500
window = pygame.display.set_mode((width,height))
bg_img = pygame.image.load('Image Files/bg.png')
bg_img = pygame.transform.scale(bg_img,(width,height))

mixer.init()
mixer.music.load('Music File/Littleidea - wav music file.wav')
pygame.mixer.music.set_volume(0.05)
mixer.music.play()

runing = True
while runing:
    window.blit(bg_img,(0,0))
    for event in pygame.event.get():
        if event.type == QUIT:
            runing = False
    pygame.display.update()
pygame.quit()

```

## 步骤 2:在屏幕上添加一个正方形

为了画一个正方形，我们使用了`draw.rect`函数，它有三个参数:窗口对象名、矩形的颜色和矩形的尺寸(宽度和高度、x 和 y 坐标)。

我们将在运行循环之前定义块的宽度和高度。除此之外，我们还将声明块的颜色。

添加了所需代码行的代码如下所示。所做的更改会突出显示，供您参考。

```py
import pygame
from pygame.locals import *
from pygame import mixer
pygame.init()
width = 500
height = 500
window = pygame.display.set_mode((width,height))
bg_img = pygame.image.load('Image Files/bg.png')
bg_img = pygame.transform.scale(bg_img,(width,height))

x=y=50
color = "red"

mixer.init()
mixer.music.load('Music File/Littleidea - wav music file.wav')
pygame.mixer.music.set_volume(0.05)
mixer.music.play()
runing = True
while runing:
    window.blit(bg_img,(0,0))
    for event in pygame.event.get():
        if event.type == QUIT:
            runing = False

    pygame.draw.rect(window, color, pygame.Rect(x, y, 60, 60))

    pygame.display.update()
pygame.quit()

```

## 第三步:增加广场的互动性

现在用以下方法制作正方形:

1.  向上箭头键:将 y 坐标减少 2
2.  向下箭头键:将 y 坐标增加 2
3.  左箭头键:将 x 坐标减少 2
4.  右箭头键:将 x 坐标增加 2

但是在加入算术运算之前。我们将确保使用`key.get_pressed`函数捕获被按下的键，并将其存储在一个变量中。

然后，我们将检查变量，并根据捕获的键在坐标中应用必要的更改。

执行相同操作的代码行如下所示:

```py
key = pygame.key.get_pressed()
if key[pygame.K_UP]: 
    y -= 2
if key[pygame.K_DOWN]: 
    y += 2
if key[pygame.K_LEFT]: 
    x -= 2
if key[pygame.K_RIGHT]: 
    x += 2

```

## Pygame 中交互式形状的完整实现

下面的代码显示了最终完成的代码。希望你明白一切。

```py
import pygame
from pygame.locals import *
from pygame import mixer

pygame.init()

#window attributes
width = 500
height = 500
window = pygame.display.set_mode((width,height))
bg_img = pygame.image.load('Image Files/bg.png')
bg_img = pygame.transform.scale(bg_img,(width,height))

#square attributes
x=y=50
color = "red"

#music addition
mixer.init()
mixer.music.load('Music File/Littleidea - wav music file.wav')
pygame.mixer.music.set_volume(0.05)
mixer.music.play()

#the running loop
runing = True
while runing:

    #add background img
    window.blit(bg_img,(0,0))

    #handling events
    for event in pygame.event.get():
        #closing window function
        if event.type == QUIT:
            runing = False

    #add the square
    pygame.draw.rect(window, color, pygame.Rect(x, y, 60, 60))

    #moving square on pressing keys
    key = pygame.key.get_pressed()
    if key[pygame.K_UP]: 
        y -= 2
    if key[pygame.K_DOWN]: 
        y += 2
    if key[pygame.K_LEFT]: 
        x -= 2
    if key[pygame.K_RIGHT]: 
        x += 2

    #update display
    pygame.display.update()

#quit pygame
pygame.quit()

```

## 最终输出

下面的视频显示了上面代码的最终输出。你可以看到当箭头键被按下时，方块移动得多么完美！

## 结论

我希望这个基本的交互式形状教程能帮助你在 pygame 中学到一些新东西！

感谢您的阅读。