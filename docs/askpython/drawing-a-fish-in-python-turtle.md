# 蟒蛇龟——用蟒蛇皮画一条鱼

> 原文：<https://www.askpython.com/python/examples/drawing-a-fish-in-python-turtle>

嘿编码器！在本教程中，我将向你展示如何在 python turtle 模块的帮助下画一条鱼。如果你不知道什么是`turtle`模块，在查看教程[。](https://www.askpython.com/python-modules/python-turtle)

您需要导入 Python 附带的 turtle 库，不需要做任何额外的安装。

```py
import turtle

```

下一步包括创建一个画布来绘制鱼。我们可以根据需要给 canvas 变量命名。现在，我们将屏幕的名称命名为`fish_scr`。下面的代码为用户创建和显示屏幕。我们还添加了一些额外的属性，包括屏幕和笔的颜色。

```py
import turtle
fish_scr = turtle
fish_scr.color('black')
fish_scr.Screen().bgcolor("#85C1E9")

```

现在让我们创建一个为我们画鱼的函数。这个函数的名字是`Draw_Fish`，它将在屏幕上为我们画出这条鱼。`goto`函数将指针指向某个位置。`penup`和`pendown`功能控制何时绘制和何时不绘制。另外，`forward`和`backward`功能需要距离作为参数，另一方面，`left`和`right`功能需要转动角度作为参数。

```py
def Draw_Fish(i,j):
    fish_scr.penup()
    fish_scr.goto(i,j)
    fish_scr.speed(10)
    fish_scr.left(45)
    fish_scr.pendown()
    fish_scr.forward(100)
    fish_scr.right(135)
    fish_scr.forward(130)
    fish_scr.right(130)
    fish_scr.forward(90)
    fish_scr.left(90)
    fish_scr.right(90)
    fish_scr.circle(200,90)
    fish_scr.left(90)
    fish_scr.circle(200,90)
    fish_scr.penup()
    fish_scr.left(130)
    fish_scr.forward(200)
    fish_scr.pendown()
    fish_scr.circle(10,360)
    fish_scr.right(270)
    fish_scr.penup()
    fish_scr.forward(50)
    fish_scr.pendown()
    fish_scr.left(90)
    fish_scr.circle(100,45)
    fish_scr.penup()
    fish_scr.forward(300)
    fish_scr.left(135)
    fish_scr.pendown()
    fish_scr.right(180)

```

让我们用下面的代码在屏幕上画三条鱼。在我们画完鱼后，我们将使用`done`功能关闭应用程序屏幕。

```py
Draw_Fish(0,0)
Draw_Fish(150,150)
Draw_Fish(150,-150)
fish_scr.done()

```

## 完整代码

```py
import turtle
fish_scr = turtle
fish_scr.color('black')
fish_scr.Screen().bgcolor("#85C1E9")

def Draw_Fish(i,j):
    fish_scr.penup()
    fish_scr.goto(i,j)
    fish_scr.speed(10)
    fish_scr.left(45)
    fish_scr.pendown()
    fish_scr.forward(100)
    fish_scr.right(135)
    fish_scr.forward(130)
    fish_scr.right(130)
    fish_scr.forward(90)
    fish_scr.left(90)
    fish_scr.right(90)
    fish_scr.circle(200,90)
    fish_scr.left(90)
    fish_scr.circle(200,90)
    fish_scr.penup()
    fish_scr.left(130)
    fish_scr.forward(200)
    fish_scr.pendown()
    fish_scr.circle(10,360)
    fish_scr.right(270)
    fish_scr.penup()
    fish_scr.forward(50)
    fish_scr.pendown()
    fish_scr.left(90)
    fish_scr.circle(100,45)
    fish_scr.penup()
    fish_scr.forward(300)
    fish_scr.left(135)
    fish_scr.pendown()
    fish_scr.right(180)

Draw_Fish(0,0)
Draw_Fish(150,150)
Draw_Fish(150,-150)

fish_scr.done()

```

当我们执行上面的代码时，一个新的屏幕出现在系统屏幕上，鱼开始在应用程序的屏幕上绘制。同样如下图所示。

恭喜你！现在你知道如何使用 Python 中的 Turtle 模块在屏幕上画一条鱼了。感谢您的阅读！如果您喜欢本教程，我建议您也阅读以下教程:

*   [Python Pygame:简单介绍](https://www.askpython.com/python-modules/python-pygame)
*   [在 Python 中生成随机颜色的方法](https://www.askpython.com/python/examples/generate-random-colors)
*   [Python 中的简单游戏](https://www.askpython.com/python/examples/easy-games-in-python)

继续阅读，了解更多！编码快乐！😄