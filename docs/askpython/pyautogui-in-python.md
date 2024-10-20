# Python 中 PyAutoGUI 的自动化

> 原文：<https://www.askpython.com/python-modules/pyautogui-in-python>

让我们学习用 Python 中的 PyAutoGUI 实现任务自动化。每当我们遇到一个需要重复的任务时，我们都会想出具体的方法来避免它。这是人之常情。

在努力完成同一项任务的过程中，我们有了一个想法，我们可以创造一些自动运行的东西，只需要满足一系列条件就可以工作。

无论是需要电池和草地的割草机，还是一遍又一遍打印同一行的代码。

自动化已经成为我们人类生活中的一个重要部分，使用自动化可以让我们在工作过程中专注于其他任务。

然而，自动化需要工具来配合工作，这就是`pyautogui`模块发挥作用的地方。

`pyautogui`模块允许运行的脚本控制你的鼠标和键盘，像系统上的用户一样提供输入，允许系统上的应用程序之间的交互。

## 在 Python 中安装 PyAutoGUI

我们可以通过 [PIP 包管理器](https://www.askpython.com/python-modules/python-pip)在 Python 中安装 PyAutoGUI。您可以在任何使用 pip 的操作系统上使用相同的命令行进行安装。

```py
# Windows does not have any dependencies for installation
pip install pyautogui

# Mac has two dependencies for PyAutoGUI
pip3 install pyobjc-core
pip3 install pyobjc
pip3 install pyautogui

# Linux distributions require a single dependency installed
pip3 install python3-xlib
pip3 install pyautogui

```

一旦我们安装了依赖项(如果有的话)和模块，我们就可以开始了！

## 使用 Python PyAutoGUI

在使用 Python 中 PyAutoGUI 提供的所有强大功能之前，我们必须首先[在脚本中导入模块](https://www.askpython.com/python/python-import-statement)。

```py
# Importing the PyAutoGUI module
import pyautogui as pag

```

在本文中，我们将为`pyautogui`模块使用一个别名，我们在上面称之为 *pag* 。

### 1.PyAutoGUI 基本函数

在处理任何脚本之前，我们最好知道哪些组件执行什么样的任务。

也就是说，Python 中的`pyautogui`提供了多种处理输入的方法，

```py
# Gets the size of the primary monitor.
screenWidth, screenHeight = pag.size() 

# Gets the XY position of the mouse.
currentMouseX, currentMouseY = pag.position() 

# Move the mouse to XY coordinates.
pag.moveTo(100, 150)

# Allows the script to click with the mouse.
pag.click()

# Move the mouse to XY coordinates and click it.
pag.click(100, 200)

# Find where button.png appears on the screen and click it.
pag.click('button.png') 

# Double clicks the mouse.
pag.doubleClick()

# The writing functionality provided by PyAutoGUI imitates keyboard input
pag.write('Hello world!')

# Presses the Esc key.
pag.press('esc')

# The keyDown button causes the script to hold down on a specific key.
pag.keyDown('shift')

# You can pass a list of keys to press, which will be consecutively executed.
pag.press(['left', 'left', 'left', 'left'])

# Lets go of a certain key.
pag.keyUp('shift')

 # The hotkey() function allows for a selection of keys for hotkey usage.
pag.hotkey('ctrl', 'c')

# Make an alert box appear and pause the program until OK is clicked.
pag.alert('This is the message to display.')

```

同样需要注意的是，该模块还提供了在脚本中工作的关键字，这些关键字可以通过`pyautogui.KEY_NAMES`访问。

### 2.在 Python 中使用 PyAutoGUI 实现简单的自动化

我们可以创建一个简单的垃圾邮件自动化，使用一点 Python 和`pyautogui`模块在任何平台上连续发送消息。

让我们首先[导入](https://www.askpython.com/python/python-import-statement)几个模块来处理所需的功能。

```py
# Importing the pyautogui module
import pyautogui as pag

# Importing time to delay the input speed
import time

# Working with Tkinter allows us to use a GUI interface to select the file to read from
from tkinter import Tk
from tkinter.filedialog import askopenfilename

```

现在，这里是你如何做一个垃圾邮件机器人。

#### 2.1.提供一种输入方法。

我们可以通过手动键入消息来提供输入，但是，这甚至会使自动发送垃圾消息的目的落空。

因此，让我们用文件来解析一个文件，并将内容写入平台。我们将使用 [tkinter 模块](https://www.askpython.com/python/tkinter-gui-widgets)来选择要读取的文件。

```py
# The withdraw function hides the root window of Tkinter
Tk().withdraw()

# The askopenfilename() takes the file path from user selection.
filename = askopenfilename()

```

现在，我们通过`askopenfilename()`函数得到了文件的路径。该路径存储在`filename`变量中。

#### 2.2.创建一个延迟来调整垃圾邮件的速度。

我们还需要在每个消息之间创建一个延迟，以便平台能够一个接一个地接受消息，而不是由于平台输入滞后而由单个消息覆盖自身。

```py
# We take the input of the user and strip it such that we only receive a numeric input.
timeDelay = int(input("If you want a delay, enter the number of seconds for the delay : ").split()[0])

# In case the input time is designed to break the delay function, we can reset the timeDelay back to 1.
if timeDelay < 1:
    timeDelay = 1

# We need to place the cursor in the right place to begin writing to the platform.
time.sleep(5)

```

#### 2.3.垃圾邮件使用 PyAutoGUI！

我们现在可以使用`pyautogui`模块从文件中读取每个单词，并写入平台。

```py
f = open(filename, "r")
for word in f:
    time.sleep(timeDelay)
    pag.typewrite(word)
    pag.press("enter")

```

### 3.PyAutogui 在 Python 中的完整实现

我们现在完成了代码，您的最终代码应该是这样的，

```py
import pyautogui as pag
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename()
print(filename)

timeDelay = int(input("If you want a delay, enter the number of seconds for the delay : ").split()[0])

if timeDelay < 1:
    timeDelay = 1

time.sleep(5)

f = open(filename, "r")
for word in f:
    time.sleep(timeDelay)
    pag.typewrite(word)
    pag.press("enter")

```

## 结论

现在您已经完成了这篇文章，您知道 Python 中的`pyautogui`提供了什么，以及您可以用它来做什么。

虽然我们不一定推荐垃圾邮件，但修补是完全可以接受的😉

查看我们的其他文章，[使用熊猫模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)， [Numpy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)，以及[使用 Pygame](https://www.askpython.com/python/examples/pygame-graphical-hi-lo-game) 创建高低游戏。

## 参考

*   【PyAutoGUI 官方文档
*   [stack overflow to typeet()](https://stackoverflow.com/questions/51476348/unable-to-pass-variable-in-typewrite-function-in-python)