# 用 3 种简单的方法退出 Python 程序！

> 原文：<https://www.askpython.com/python/examples/exit-a-python-program>

嘿，大家好。在本文中，我们将会看到一些被认为可以方便地执行这项任务的函数——**退出 Python 程序**。

* * *

## 技巧 1:使用 quit()函数

Python 函数提供的内置`quit() function`，可以用来退出 Python 程序。

**语法:**

```py
quit()

```

一旦系统遇到 quit()函数，它就完全终止程序的执行。

**举例:**

```py
for x in range(1,10):
    print(x*10)
    quit()

```

如上所示，在 for 循环的第一次迭代之后，解释器遇到 quit()函数并终止程序。

**输出:**

```py
10

```

* * *

## 技巧 2: Python sys.exit()函数

`Python sys module`包含一个内置函数，用于退出程序，从执行过程中出来——函数`sys.exit()`。

sys.exit()函数可以在任何时候使用，而不必担心代码损坏。

**语法:**

```py
sys.exit(argument)

```

让我们看看下面的例子来理解`sys.exit()`功能。

**举例:**

```py
import sys 

x = 50

if x != 100: 
	sys.exit("Values do not match")	 
else: 
	print("Validation of values completed!!") 

```

**输出:**

```py
Values do not match

```

* * *

## 技巧 3:使用 exit()函数

除了上面提到的技术，我们可以使用 Python 中内置的`exit() function`来退出程序的执行循环。

**语法:**

```py
exit()

```

**举例:**

```py
for x in range(1,10):
    print(x*10)
    exit()

```

exit()函数可以被认为是 quit()函数的替代函数，它使我们能够终止程序的执行。

**输出:**

```py
10

```

* * *

## 结论

到此，我们就结束了这个话题。`exit()`和`quit()`功能不能在操作和生产代码中使用。因为，这两个功能只有导入了站点模块才能实现。

因此，在上述方法中，最优选的方法是`sys.exit()`方法。

如果你遇到任何问题，欢迎在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   [如何退出 Python 程序？—堆栈溢出](https://stackoverflow.com/questions/19782075/how-to-stop-terminate-a-python-script-from-running/34029481#:~:text=To%20stop%20a%20running%20program,want%20to%20terminate%20the%20program.&text=Ctrl%20%2B%20Z%20should%20do%20it,caught%20in%20the%20python%20shell.)