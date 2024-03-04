# 在 Python 中使用 While 循环

> 原文：<https://www.pythonforbeginners.com/loops/python-while-loop>

### 概观

在这篇文章中，我将讲述 Python 中的 While 循环。

如果你读过早先的帖子 [For 和 While 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)，你可能会认识到很多这种情况。

### 从 0 数到 9

这个小脚本会从 0 数到 9。

```py
i = 0
while i < 10:
    print i
    i = i + 1

```

### 它是做什么的？

在 while 和冒号之间，有一个值首先为真，但随后为假。

只要该语句为真，其余的代码就会运行。

将要运行的代码必须在缩进的块中。

i = i + 1 每运行一次，I 值就加 1。

注意不要形成一个永恒的循环，也就是循环一直持续到你按下 Ctrl+C。

```py
while True:
    print "Hello World"

```

这个循环意味着 while 循环将永远为真，并将永远打印 Hello World。