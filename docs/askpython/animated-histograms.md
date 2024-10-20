# Python 中的动画直方图——逐步实现

> 原文：<https://www.askpython.com/python/examples/animated-histograms>

嘿伙计们！今天，我们将使用 Python 编程语言进行编程，以获得动画直方图。

**Python 和 Matplotlib** 可用于创建静态 2D 图。但是 Matplotlib 有一个秘密的力量，可以用来创建**动态自动更新动画情节**。

我们开始吧！

* * *

## 1.导入模块

我们从**导入所有必要的模块/库**开始，包括`numpy`创建数据、`[matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib)`绘制直方图，最后`matplotlib.animation`绘制动画图。

我们还将导入 HTML 函数，以便将视频转换为 HTML 格式。

```py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
plt.style.use('seaborn')

```

* * *

## 2.创建数据集

为了**创建数据**，我们将需要 numpy 模块，首先修复一个随机状态，以便使用它。接下来，我们使用**行间距**函数初始化容器的数量。

接下来，我们将使用 **linspace** 函数创建随机的 1000 个数据点。最后一步是使用**直方图**功能将数据点转换成直方图数据点。

```py
np.random.seed(19680801)
HIST_BINS = np.linspace(-4, 4, 100)
data = np.random.randn(1000)
n, _ = np.histogram(data, HIST_BINS)

```

* * *

## 3.动画显示直方图

为了让直方图有动画效果，我们需要一个`animate`函数，它将生成一些随机数，并不断更新容器的高度。

```py
def prepare_animation(bar_container):

    def animate(frame_number):
        data = np.random.randn(1000)
        n, _ = np.histogram(data, HIST_BINS)

        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches

    return animate

```

* * *

## 3.显示动画直方图

在`hist()`函数的帮助下，可以得到一个`BarContainer`的实例(矩形实例的集合)。

然后我们将调用`prepare_animation`，在它下面定义了`animate`函数。

最后，我们将使用`to_html5_video`函数将情节转换成 **HTML** 格式。

```py
fig, ax = plt.subplots()
_, _, bar_container = ax.hist(data, HIST_BINS, lw=1,ec="red", fc="blue", alpha=0.5)
ax.set_ylim(top=55)
ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50,repeat=True, blit=True)
HTML(ani.to_html5_video())

```

* * *

## 在 Python 中显示动画直方图的完整实现

```py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
plt.style.use('seaborn')

np.random.seed(19680804)
HIST_BINS = np.linspace(-4, 4, 100)
data = np.random.randn(1000)
n, _ = np.histogram(data, HIST_BINS)

def prepare_animation(bar_container):

    def animate(frame_number):
        data = np.random.randn(1000)
        n, _ = np.histogram(data, HIST_BINS)

        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches

    return animate

fig, ax = plt.subplots()
_, _, bar_container = ax.hist(data, HIST_BINS, lw=1,ec="blue", fc="yellow", alpha=0.5)
ax.set_ylim(top=100)
ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50,repeat=True, blit=True)
HTML(ani.to_html5_video())

```

* * *

## 结论

我希望您在观看动画直方图时感到愉快！您可以尝试使用不同的数据、箱数，甚至改变直方图的速度。

编码快乐！😊

## 阅读更多

1.  [Python 情节:在 Python 中创建动画情节](https://www.askpython.com/python-modules/matplotlib/animated-plots)
2.  [3 个 Matplotlib 绘图技巧使绘图有效](https://www.askpython.com/python-modules/matplotlib/matplotlib-plotting-tips)
3.  [Python:绘制平滑曲线](https://www.askpython.com/python-modules/matplotlib/smooth-curves)

* * *