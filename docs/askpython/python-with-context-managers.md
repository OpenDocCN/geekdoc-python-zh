# 上下文管理器——理解 Python with 关键字

> 原文：<https://www.askpython.com/python/python-with-context-managers>

Python `with`语句是一个非常有用的。这从 Python 2.5 开始就有了，现在几乎每个 Python 应用程序都在使用它！

这个说法到底有什么用，以至于大家都用？

最有用的东西(事实上唯一的东西！)它所做的就是打开并释放资源。

基本上，它处理打开和关闭程序特定部分可能需要使用的任何资源，并在之后自动关闭它。

现在，让我们用一些例子来更详细地看看这个陈述。

* * *

## 为什么我们需要上下文管理器？

考虑您处理文件的场景。在其他语言中，比如 C，我们必须像这样手动打开和关闭文件:

```py
# Open the file
file_obj = open('input.txt', 'r')

# File operations come here
...

# Manually close the file
file_obj.close()

```

现在,`with`语句会自动为您提取这些内容，这样就不需要每次都手动关闭文件了！

`with`语句有一个*上下文*(块)，在这个上下文中它起作用。这个这个作为声明的范围。

当程序从这个上下文中出来，`with`自动关闭你的文件！

因此，`with`通常被称为**上下文管理器**。

因此，可以像这样使用相同的文件处理过程，以及`with`语句:

```py
with open('input.txt', 'r') as file_obj:
    ...

```

注意这是多么的直观。Python `with`语句将*总是*在最后关闭文件，即使程序在上下文/块中异常终止。

这个安全特性使它成为所有 Python 程序员接受的(也是推荐的)选择！

* * *

## 使用 Python with 语句

现在，虽然有很多类已经实现了使用`with`的工具，我们还是想看看它是如何工作的，这样我们就可以自己写一个了！

*   首先，`with`语句在一个上下文对象中存储一个对象引用。

上下文对象是包含关于其状态的额外信息的对象，例如模块/范围等。这很有用，因为我们可以保存或恢复该对象的状态。

所以，保持对对象的引用是有意义的！

现在，让我们继续。一旦创建了上下文对象，它就调用对象上的`__enter__` dunder 方法。

*   `__enter__`语句是真正为对象打开资源的语句，比如文件/套接字。通常，如果需要，我们可以实现它来*保存*上下文对象状态。

现在，还记得`as`关键字吗？这实际上返回了上下文对象。因为我们需要由 [open()](https://www.askpython.com/python/built-in-methods/python-open-method) 返回的对象，所以我们使用`as`关键字来获取上下文对象。

使用`as`是可选的，特别是如果您在其他地方有对原始上下文对象的引用。

之后，我们进入嵌套语句块。

一旦嵌套块*或*结束，如果其中有异常，程序*总是*对上下文对象执行`__exit__`方法！

这就是我们之前谈到的安全第一的特性。所以不管发生什么，我们都会一直使用`__exit__`来释放资源，退出上下文。

最后，如果可行的话，`__exit__`可以被实现，以便*恢复*上下文对象状态，使得它回到它所属的任何状态。

好吧，那是一个相当长的解释。为了更清楚，让我们看一个为类创建我们自己的上下文管理器的例子。

* * *

## 为我们的类创建自己的上下文管理器

考虑下面的类，我们将有自己的上下文管理器来处理文件。

```py
class MyFileHandler():
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        # Originally, context object is None
        self.context_object = None

    # The context manager executes this first
    # Save the object state
    def __enter__(self):
        print("Entered the context!")
        self.context_object = open(self.filename, self.mode)
        return self.context_object

    # The context manager finally executes this before exiting
    # Information about any Exceptions encountered will go to
    # the arguments (type, value, traceback)
    def __exit__(self, type, value, traceback):
        print("Exiting the context....")
        print(f"Type: {type}, Value: {value}, Traceback: {traceback}")
        # Close the file
        self.context_manager.close()
        # Finally, restore the context object to it's old state (None)
        self.context_object = None

# We're simply reading the file using our context manager
with MyFileHandler('input.txt', 'r') as file_handle:
    for line in file_handle:
        print(line)

```

仔细观察类方法。我们的处理程序有`__init__`方法，它设置上下文对象和相关变量的初始状态。

现在，`__enter__` dunder 方法保存对象状态并打开文件。现在，我们在街区里面。

在块执行之后，上下文管理器最后执行`__exit__`，在那里上下文对象的原始状态被恢复，并且文件被关闭。

好的，现在让我们检查一下我们的输出。这个应该管用！

**输出**

```py
Entered the context!
Hello from AskPython

This is the second line

This is the last line!
Exiting the context....
Type: None, Value: None, Traceback: None

```

好的，看起来我们没有错误！我们刚刚为自定义类实现了自己的上下文管理器。

现在，有另一种创建上下文管理器的方法，它使用生成器。

然而，这有点不方便，一般不推荐，除非你确切地知道自己在做什么，因为你必须自己处理异常。

但是，为了完整起见，您可以在这里看一下使用这种方法[。一旦你熟悉了基于类的方法，我会推荐你阅读这篇文章。](https://preshing.com/20110920/the-python-with-statement-by-example/)

* * *

## 结论

在本文中，我们通过使用`with`语句学习了如何在 Python 中使用上下文管理器。

## 参考

*   一篇由[发表的关于 Python 上下文管理器的精彩文章](https://preshing.com/20110920/the-python-with-statement-by-example/)

* * *