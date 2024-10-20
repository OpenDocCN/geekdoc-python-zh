# Python 101 -赋值表达式

> 原文：<https://www.blog.pythonlibrary.org/2021/09/24/python-101-assignment-expressions-2/>

Python 在版本 **3.8** 中增加了赋值表达式。一般的想法是，赋值表达式允许你在表达式中给变量赋值。

这样做的语法是:

```py
NAME := expr

```

这个操作符被称为“walrus 操作符”，尽管它们的真名是“赋值表达式”。有趣的是，CPython 内部也将它们称为“命名表达式”。

你可以在 [PEP 572](https://www.python.org/dev/peps/pep-0572/) 中阅读所有关于赋值表达式的内容。让我们来看看如何使用赋值表达式！

## 使用赋值表达式

赋值表达式还是比较少见的。但是，您需要了解赋值表达式，因为您可能会不时地遇到它们。PEP 572 有一些很好的赋值表达式的例子。

```py
# Handle a matched regex
if (match := pattern.search(data)) is not None:
    ...

# A more explicit alternative to the 2-arg form of iter() invocation
while (value := read_next_item()) is not None:
    ...

# Share a subexpression between a comprehension filter clause and its output
filtered_data = [y for x in data if (y := f(x)) is not None]

```

在这 3 个例子中，您在表达式语句本身中创建了一个变量。第一个示例通过将 regex 模式搜索的结果赋给变量`match`来创建变量。第二个例子将变量`value`赋给在`while`循环表达式中调用函数的结果。最后，将调用`f(x)`的结果赋给列表理解中的变量`y`。

看看不使用赋值表达式的代码和使用赋值表达式的代码之间的区别可能会有所帮助。下面是一个分块读取文件的示例:

```py
with open(some_file) as file_obj:
    while True:
        chunk_size = 1024
        data = file_obj.read(chunk_size)
        if not data:
            break
        if 'something' in data:
            # process the data somehow here

```

这段代码将打开一个大小不确定的文件，一次处理 1024 个字节。您会发现这在处理非常大的文件时非常有用，因为它可以防止您将整个文件加载到内存中。如果这样做，可能会耗尽内存，导致应用程序甚至计算机崩溃。

您可以使用赋值表达式将这段代码缩短一点:

```py
with open(some_file) as file_obj:
    chunk_size = 1024
    while data := file_obj.read(chunk_size):
        if 'something' in data:
            # process the data somehow here

```

在这里，你在`while`循环的表达式中将`read()`的结果赋值给`data`。这允许您在`while`循环的代码块中使用该变量。它还检查是否返回了一些数据，这样就不必使用`if not data: break`节了。

PEP 572 中提到的另一个好例子取自 Python 自己的`site.py`。代码最初是这样的:

```py
env_base = os.environ.get("PYTHONUSERBASE", None)
if env_base:
    return env_base

```

这就是如何通过使用赋值表达式来简化它:

```py
if env_base := os.environ.get("PYTHONUSERBASE", None):
    return env_base

```

您将赋值移动到`conditional`语句的表达式中，这很好地缩短了代码。

现在让我们发现一些不能使用赋值表达式的情况。

## 你不能用赋值表达式做什么

有几种情况不能使用赋值表达式。

赋值表达式最有趣的特性之一是它们可以用在赋值语句不能用的上下文中，比如在`lambda`或前面提到的理解中。但是，它们不支持赋值语句可以做的一些事情。例如，您不能执行多个目标分配:

```py
x = y = z = 0  # Equivalent, but non-working: (x := (y := (z := 0)))

```

另一个被禁止的用例是在表达式语句的顶层使用赋值表达式。这里有一个来自 PEP 572 的例子:

```py
y := f(x)  # INVALID
(y := f(x))  # Valid, though not recommended

```

PEP 中有一个详细的列表，列出了禁止或不鼓励使用赋值表达式的其他情况。如果您计划经常使用赋值表达式，那么您应该查阅该文档。

## 包扎

赋值表达式是清理代码某些部分的一种优雅方式。该特性的语法有点类似于类型提示变量。一旦你掌握了其中的一个，另一个也会变得更容易。

在本文中，您看到了一些使用“walrus 操作符”的真实例子。您还了解了什么时候不应该使用赋值表达式。该语法仅在 Python 3.8 或更新版本中可用，因此如果您碰巧被迫使用 Python 的旧版本，该功能对您来说不会有太大用处。

## 相关阅读

这篇文章基于 **Python 101 第二版**中的一章，你可以在 [Leanpub](https://leanpub.com/py101) 或[亚马逊](https://amzn.to/2Zo1ARG)上购买。

如果你想学习更多的 Python，那么看看这些教程:

*   python 101—[如何处理图像](https://www.blog.pythonlibrary.org/2021/09/14/python-101-how-to-work-with-images/)

*   python 101-[记录你的代码](https://www.blog.pythonlibrary.org/2021/09/12/documenting-code/)

*   Python 101: [使用 JSON 的介绍](https://www.blog.pythonlibrary.org/2020/09/15/python-101-an-intro-to-working-with-json/)

*   python 101-[创建多个流程](https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/)