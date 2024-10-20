# 什么是 Python compile()函数？

> 原文：<https://www.askpython.com/python/built-in-methods/what-is-python-compile-function>

嘿，伙计们！在本文中，我们将重点关注 **Python compile()函数**。

* * *

## 了解 Python compile()函数的工作原理

让我带大家回到系统/操作系统编程，从这里我们处理宏和函数的概念。宏基本上是一些预定义的代码块，在函数调用时用当前源代码执行。也就是说，在任何工作的程序代码中，整个功能块都是一次性执行的。

类似地，Python compile()函数帮助我们在函数中定义一段代码。此外，它还生成一个代码块对象，可用于在程序中的任何位置执行已定义的代码。

compile()函数中定义的输入源代码可以通过从该函数返回的 code 对象在任何程序中轻松执行。

因此，Python compile()函数有助于获得可重用性，并证明更加可靠。

* * *

## Python compile()函数的语法

`Python compile() function`接受源代码作为参数，并返回可随时执行的代码对象。

```py
compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)

```

现在让我们详细了解一下 Python compile()函数的参数表。

* * *

### compile()函数参数

*   `source`(必选):需要编译的源代码或字节串。
*   `filename`(必选):包含“源”的文件的名称。如果不存在，我们可以为源代码输入一个名称。
*   `mode`(必选):这是编译源代码的模式。这三种模式如下:

1.  **eval** :当源代码是要编译的单个表达式时，使用此模式。
2.  **single** :当源包含单个交互语句时使用。
3.  **exec** :当源代码包含要执行的语句块时，使用这种模式。

**可选参数:**

*   `flags`:默认值为 0。
*   `dont_inherit`:决定执行的流程。默认值为 False。
*   `Optimize`:默认值为-1。

* * *

## Python compile()函数示例

在下面的例子中，我们将一个变量“var = 10”作为源代码传递给了 compile()函数。此外，我们使用了'**单模**'来编译和执行传递给参数列表的源代码。

```py
var = 10
compile_obj = compile('var', 'Value', 'single')
exec(compile_obj)

```

使用 compile()函数，创建与传递的源代码相关联的代码对象。

然后，使用 Python exec()函数动态编译代码对象。

**输出:**

```py
10

```

现在，我们已经向 compile()函数传递了一个用于执行的表达式，因此这里使用了“ **eval mode** ”。

```py
action = 'print(100)'
compile_obj = compile(action, 'Display', 'eval')

exec(compile_obj)

```

**输出:**

```py
100

```

如下所示，我们将一串源代码传递给了 compile()函数，并使用' **exec mode** '通过使用 exec()函数创建的代码对象来执行代码块。

```py
action = 'x = 15\ny =10\nproduct=x*y\nprint("Product of x and y =",product)'
compile_obj = compile(action, 'Product', 'exec')

exec(compile_obj)

```

**输出:**

```py
Product of x and y = 150

```

* * *

## 结论

到此，我们就结束了这个话题。如果您遇到任何疑问，请随时在下面发表评论。

更多关于 Python 编程的帖子，请访问 [Python 教程 AskPython](https://www.askpython.com/) 。

* * *

## 参考

*   [Python compile()函数—文档](https://docs.python.org/3/library/functions.html#compile)
*   Python compile()函数— JournalDev