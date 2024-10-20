# Python 中的动画

> 原文：<https://www.askpython.com/python-modules/animation-in-python-celluloid>

使用 Python 中的动画，我们可以更有效地表达我们的数据。动画是一种方法，在这种方法中，数字被处理成移动的图像，由一系列图片产生的运动模拟就是动画。

在本文中，我们将使用赛璐珞库，它使得 [Python Matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 中的动画非常简单。

## 赛璐珞 Python 中的简单动画

对于初学者来说，matplotlib 动画教程可能会很复杂。赛璐珞使得使用 matplotlib 制作动画变得很容易。

使用赛璐珞，我们为我们的可视化“拍照”，以在每次迭代中创建一个帧。一旦所有的帧都被捕获，我们就可以用一个调用来创建一个动画。查看[自述文件](https://github.com/jwkvam/celluloid)了解更多详情。

你可以使用 [Python pip 命令](https://www.askpython.com/python-modules/python-pip)在 Python 中安装赛璐珞库

```py
pip install celluloid

```

## 使用赛璐珞制作动画的步骤

一旦你准备好了库，让我们开始制作动画。

### **1。从赛璐珞导入相机类**

首先，我们需要从赛璐珞模块导入 camera 类，并通过传递 Matplotlib figure 对象创建一个 Camera 对象。

```py
from celluloid import Camera
fig = plt.figure()
camera = Camera(fig)

```

### 2.在数据循环时创建快照

循环递增地在 Matplotlib 图形上绘制数据，并使用 camera 对象的`.snap( )`方法拍摄快照。

```py
#plotting data using loops and creating snapshot at each iteration
plt.plot(..)
camera.snap()

```

### 3.创建动画对象

创建完所有帧后，使用 camera 类的`.animate( )`方法。

```py
#Applying the animate method to create animations
animation = camera.animate()

#Saving the animation
animation.save('my_animation.mp4')

```

### Python 中动画的示例实现

现在让我们通过在 Python 中创建一个追踪正弦函数的动画来清楚地理解上面的步骤。

```py
#Importing required libraries
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
import ffmpeg

#Creating Data
x = np.linspace(0, 10, 100)

#defining a function to return sine of input values.
def fun(i):
    y = np.sin(i)
    return y

x_y = fun(x)

#Creating matplotlib figure and camera object
fig = plt.figure()
plt.xlim(0,10)
plt.ylim(-2,2)
camera = Camera(fig)

#Looping the data and capturing frame at each iteration
for i in x:
    plt.plot(x,x_y , color = 'green' , lw = 0.8)
    f = plt.scatter(i, fun(i) , color = 'red' , s = 200)
    plt.title('tracing a sin function')
    camera.snap()

#Creating the animation from captured frames
animation = camera.animate(interval = 200, repeat = True,
                           repeat_delay = 500)

```

<https://www.askpython.com/wp-content/uploads/2020/11/sine_wave.mp4>

在上面的代码中，我们定义了一个 fun()函数，它接受数值并返回输入值的正弦值。

当我们准备好相机对象时，我们遍历数据，每次迭代我们都传递跟踪器的新坐标(红色的点)并创建输出图像的快照。

在捕获所有帧后，我们应用带有以下输入参数的`.animate( )`方法:

*   `interval`–两帧之间的时间，单位为毫秒。
*   `repeat`–(*布尔*)指定我们是否要不断重复动画。
*   `repeat_delay`–如果 repeat 为真，我们使用它指定时间延迟来重复动画。

```py
#Saving the animation
animation.save('sine_wave.mp4')

```

**使用本库的一些限制:**

*   确保所有图的轴限制相同。
*   将艺术家传递给`legend`函数来分别绘制他们，因为图例将从先前的情节中累积。

## 结论

在本文中，我们发现了一个非常简单的使用赛璐珞库在 Python 中创建动画的方法。使用库来更好地学习它，并变得更有效率！快乐学习！！🙂