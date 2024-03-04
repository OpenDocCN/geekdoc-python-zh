# 如何在 Python 中检测按键

> 原文：<https://www.pythonforbeginners.com/basics/how-to-detect-keypress-in-python>

在创建运行图形用户界面的程序时，我们需要检测用户是否按了几次键。在本文中，我们将看到如何在 python 中检测按键。

## 使用 Python 中的键盘模块检测按键

为了在 python 中检测按键，我们可以使用键盘模块。它可以在 Windows 和 Linux 操作系统上运行，并且支持所有的热键。您可以使用 PIP 在机器上安装键盘模块，如下所示。

```py
pip install keyboard
```

为了检测按键，我们将使用键盘模块中定义的`is_pressed()`函数。`is_pressed()`接受一个字符作为输入，如果在键盘上按下具有相同字符的键，则返回`True`。因此，我们可以使用带有 while 循环的`is_pressed()` 函数来检测 python 中的按键，如下例所示。

```py
import keyboard
while True:
    if keyboard.is_pressed("a"):
        print("You pressed 'a'.")
        break 
```

输出:

```py
aYou pressed 'a'.
```

这里，我们已经执行了 while 循环，直到用户按下键“`a`”。按下其他键时，`is_pressed()`函数返回`False`，while 循环继续执行。一旦用户按下`“a”,`，if 块内的条件变为真，break 语句被执行。因此 while 循环终止。

代替 `is_pressed()`函数，我们可以使用`read_key()`函数来检测按键。`read_key()`函数返回用户按下的键。我们可以使用带有 while 循环的`read_key()`函数来检查用户是否按下了特定的键，如下所示。

```py
import keyboard
while True:
    if keyboard.read_key() == "a":
        print("You pressed 'a'.")
        break
```

输出:

```py
You pressed 'a'.
```

我们还可以使用键盘模块中定义的`wait()`函数来检测按键。`wait()`函数接受一个字符作为输入。在执行时，它会一直等待，直到用户按下作为输入参数传递给函数的键。一旦用户按下右键，该功能将停止执行。您可以在下面的示例中观察到这一点。

```py
import keyboard
keyboard.wait("a")
print("You pressed 'a'.") 
```

输出:

```py
You pressed 'a'.
```

## 结论

在本文中，我们讨论了使用键盘模块在 python 中检测按键的不同方法。要了解更多关于输入的内容，你可以阅读这篇关于用 python 从键盘获取用户输入的文章。您可能也会喜欢这篇关于 python 中的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。