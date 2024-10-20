# 无换行符的 Python 打印

> 原文：<https://www.askpython.com/python/examples/python-print-without-newline>

有不同的方法可以让我们不用换行就能打印到控制台。让我们快速浏览一下这些方法。

* * *

## 1.使用打印()

我们可以使用`print()`函数来实现这一点，方法是适当地设置`end`(结束字符)关键字参数。

默认情况下，这是一个换行符(`\n`)。因此，我们必须改变这一点，以避免在结尾打印一个换行符。

这个选择有很多选项。我们可以使用空格来打印空格分隔的字符串。

```py
a = "Hello"
b = "from AskPython"

print(a, end=' ')
print(b)

```

这将打印字符串`a`和`b`，用一个空格分隔，而不是换行符。

**输出**

```py
Hello from AskPython

```

我们也可以用一个空字符串连续打印它们，没有任何间隙。

```py
a = "Hello"
b = "from AskPython"

print(a, end='')
print(b)

```

**输出**

```py
Hellofrom AskPython

```

* * *

## 2.打印不带换行符的列表元素

有时，当遍历一个列表时，我们可能需要在同一行打印它的所有元素。为此，我们可以再次使用与之前相同的逻辑，使用`end`关键字参数。

```py
a = ["Hello", "how", "are", "you?"]

for i in a:
    print(i, end=" ")

```

输出

```py
Hello how are you?

```

* * *

## 3.使用 sys 模块

我们也可以使用`sys` [模块](https://www.askpython.com/python-modules/python-modules)进行无换行符打印。

更具体地说，`sys.stdout.write()`函数使我们能够在不换行的情况下写入控制台。

```py
import sys
sys.stdout.write("Hello from AskPython.")
sys.stdout.write("This is printed on the same line too!")

```

**输出**

```py
Hello from AskPython.This is printed on the same line too!

```

* * *

## 4.创建我们自己的 C 风格 printf()函数

我们也可以在 Python 中创建我们的自定义`printf()`函数！是的，使用模块`functools`这是可能的，它允许我们通过`functools.partial()`从现有的函数中定义新的函数！

让我们对`print()`上的`end`关键字参数使用相同的逻辑，并用它来创建我们的`printf()`函数！

```py
import functools

# Create our printf function using
# print() invoked using the end=""
# keyword argument
printf = functools.partial(print, end="")

# Notice the semicolon too! This is very familiar for a
# lot of you!
printf("Hello!");
printf("This is also on the same line!");

```

**输出**

```py
Hello!This is also on the same line!

```

我们还可以将分号与此结合起来(Python 编译器不会抱怨),使我们的 C `printf()`函数恢复原样！

* * *

## 参考

*   JournalDev 关于不换行打印的文章
*   [StackOverflow 同一话题的问题](https://stackoverflow.com/questions/493386/how-to-print-without-a-newline-or-space)

* * *