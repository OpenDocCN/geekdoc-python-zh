# Python 中的 auto clicker——2 种简单易行的方法

> 原文：<https://www.askpython.com/python/examples/autoclicker-in-python>

嗨，开发者们！！在本教程中，我们将看看 Python 中的**自动点击器。我们将首先学习它的含义以及如何用 Python 实现它。所以，事不宜迟，让我们直奔主题。**

**Auto clicker** 是一款 Python 软件，允许用户以很短的时间间隔连续点击鼠标。它由用户定义的密钥控制，可在所有环境下工作，包括 Windows、Mac 和 Linux。在 Python 中，我们将利用一个名为 PyAutoGUI 的包来完成这项工作。这将允许我们同时操作鼠标和监控键盘。

* * *

## 方法 1:使用 PyAutoGui

**PyAutoGUI** 使用(x，y)坐标，原点(0，0)在屏幕的左上角。当我们向右移动时，x 坐标增加，但是 y 坐标减少。

PyAutoGUI 目前**只在主显示器**上工作。对于第二台显示器的屏幕来说是不可信的。PyAutoGUI 执行的所有键盘操作都被传输到具有当前焦点的窗口。

### 代码实现

```py
import pyautogui
import time
def click(): 
    time.sleep(0.1)     
    pyautogui.click()
for i in range(20): 
    click()

```

* * *

## 方法 2:使用 Pynput

让我们尝试使用 Pynput 模块在 Python 中实现一个 autoclicker。

### 导入所需模块

```py
import time
import threading
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode

```

程序中导入了多个模块，包括导入按钮和控制器以控制鼠标动作，以及导入监听器和键码以跟踪键盘事件来处理自动点击动作的开始和停止。

### 声明重要变量

```py
delay = 0.001
button = Button.left
start_stop_key = KeyCode(char='s')
exit_key = KeyCode(char='e')

```

下一步是声明一些重要的变量，包括:

1.  **按钮变量**，设置为需要点击的鼠标按钮。
2.  **Begin_End**
3.  **退出 _ 键** **变量**关闭 autoclicker。

### 创建扩展线程的类

```py
class ClickMouse(threading.Thread):
    def __init__(self, delay, button):
        super(ClickMouse, self).__init__()
        self.delay = delay
        self.button = button
        self.running = False
        self.program_run = True

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_run = False

    def run(self):
        while self.program_run:
            while self.running:
                mouse.click(self.button)
                time.sleep(self.delay)
            time.sleep(0.1)

```

由于我们构建的线程，我们将能够管理鼠标点击。有两个选项:延时和按钮。此外，还有两个指示器指示程序是否正在执行。

#### 创建从外部处理线程的方法

*   **start_clicking():** 启动线程
*   **停止 _ 点击** **():** 停止线程
*   **exit():** 退出程序并复位

#### 创建一个将在线程启动时运行的方法

当线程启动时，这个方法将被调用。我们将循环迭代，直到 **run_prgm 的结果等于 True** 。循环内的循环迭代，直到游程的值为真。一旦进入两个循环，我们就按下设置按钮。

### 创建鼠标控制器的实例

```py
mouse = Controller()
thread = ClickMouse(delay, button)
thread.start()

```

### 创建设置键盘监听器的方法

```py
def on_press(key):
    if key == start_stop_key:
        if thread.running:
            thread.stop_clicking()
        else:
            thread.start_clicking()
    elif key == exit_key:
        thread.exit()
        listener.stop()

with Listener(on_press=on_press) as listener:
    listener.join()

```

如果您点击开始结束键，它将停止点击，如果标志设置为真。否则，它将开始。如果按下 exit 键，线程的 exit 方法被调用，监听器被终止。

* * *

## 结论

这是用 Python 开发自动点击器的两种截然不同的方法。它可以根据用户的需要进一步定制。

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [使用 Python 截图的 2 种方法](https://www.askpython.com/python/examples/capture-screenshots)
2.  [在 NumPy 中执行随机抽样的 4 种方式](https://www.askpython.com/python/random-sampling-in-numpy)
3.  [在 Python 中更容易调试的技巧](https://www.askpython.com/python/tricks-for-easier-debugging-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *