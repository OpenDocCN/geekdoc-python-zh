# Python 中的垃圾收集

> 原文：<https://www.askpython.com/python-modules/garbage-collection-in-python>

在本文中，我将向您介绍 Python 中垃圾收集的概念。垃圾收集是 Python 自动管理其内存的一种方式。

它是通过使用参考计数器来实现的。所以在我们进入垃圾收集的概念之前，让我们先了解一下什么是引用计数器。

## Python 中的引用计数器是什么？

引用计数器是在运行的程序中对对象的引用次数。它允许 Python 编译器知道什么时候使用了一个[变量](https://www.askpython.com/python/python-variables)，什么时候从内存中移除一个对象是安全的。

这减少了程序员跟踪填充系统资源的对象的工作，并允许他们专注于创建程序。

## Python 中的垃圾收集是如何工作的？

让我们理解 Python 如何使用引用计数器在后端执行垃圾收集。我们可以用一个简单的例子来理解这一点。

我们将首先介绍引用是如何计数的，然后看看 Python 如何识别对象没有引用的情况。

看看下面的代码:

```py
# Increasing reference count as more variables link to it

reference1 = 9 # Reference count for the value 9, becomes 1
reference2 = reference1 # Reference count for value 9 becomes 2
reference3 = reference1 # Reference count for value 9 becomes 3

# Decreasing reference count as the variable values change
reference2 = 10 # Reference count for value 9 decreases to 2
reference3 = 5 # Reference count for value 9 decreases to 1
reference1 = 1 # Reference count for value 9 decreases to 0

# After the reference value becomes 0, the object is deleted from memory

```

从上面可以清楚地看到，一旦最后一个引用变量“reference1”的值更改为 1，值 9 在内存中就不再有引用了。

一旦 Python 解释器在整个代码中没有发现对某个值的引用，垃圾收集器就会将内存释放给该值以释放空间。

## 什么是参考周期？

让我们看看另一个概念，称为参考循环。在这里，我们只是简单地引用一个对象本身。看看下面的示例代码:

```py
>>> a = []
>>> a.append(a)
>>> print a
[[...]]

```

此外，我们将执行 a=[]，并创建一个空列表。`a.append()`意味着我们将向列表中添加一些东西。

在这种情况下:a .所以我们要向这个对象添加另一个空列表。这是怎么回事？

如果我们调用`a`会看到这里有两个列表。

所以我们已经创建了一个空列表，然后我们把这个列表附加到对象本身。所以在这个对象中，我们得到了一个列表，然后在这个对象中，这个列表被再次调用，所以引用计数器上升到 1。

但是我们不再使用`a`，我们的程序不再调用它，但是引用计数器是 1。

Python 有一种移除引用循环的方法，但它不会立即这么做。在引用了很多次之后，它会引用一些东西，然后不引用一些东西，这就是一个事件。

所以在这种情况下，在多次出现之后，python 将运行它的垃圾收集，它将进入内存并查看每个对象。

当它进入内存并查看每个对象时，它会看到这个对象在引用自己，我们的程序不再调用它，但它的引用计数为 1，但没有任何对象调用它。

因此，它将继续删除它。

## 我们如何知道垃圾收集何时运行？

嗯，我们可以通过使用一个名为`garbage collection`的 Python 模块来看这个问题。我们将通过导入 gc 来导入垃圾收集模块。

然后，我们获得阈值，以了解垃圾收集将在何时进行，并捕获这些引用周期。

我们可以通过键入 gc.get_threshold()来获取这些信息。

```py
import gc
gc.get_threshold()

```

上面两行代码显示了下面的输出。

```py
(700,10,10)

```

让我们仔细看看输出。值“700”的意思是，在引用某个对象 700 次后，Python 将继续收集引用周期。

简单来说，在出现 700 次后，Python 将运行一个脚本或算法，遍历并清理你的内存。

虽然当引用计数器由于引用周期而停留在 1 时，Python 会在引用计数器达到 0 时自动执行此操作。那么只有在 700 次之后，Python 才会运行垃圾收集来捕捉这些循环。

## 手动使用垃圾收集

我们可以通过使用模块来改变这一点。我们不会在本文中详细讨论这个问题，但是请注意您可以更改它。

相同的代码如下所示。

用户也可以打开或关闭垃圾收集。使用该模块，您可以做很多事情。

```py
import gc
gc.disable()  

class Track:
    def __init__(self):
        print("Intitialisting your object here")
    def __del__(self):
        print("Deleting and clearing memory")

print("A")
A = Track()
print("B")
B = Track()

print("deleting here...")
del A
del B  

gc.collect() 

```

为了解释上面的代码，简而言之，我已经导入了垃圾收集器模块，但是在代码的开头使用`gc.disable()`禁用了垃圾收集。

这是为了确保不进行自动垃圾收集。然后，用一个构造函数和析构函数定义一个类轨迹。两个对象被定义为 A 和 B，它们在定义后在控制台中打印`Initialising your object here`。

然后使用`del`方法删除对象，并在成功删除对象后在控制台中打印`Deleting and clearing memory`。

`gc.collect()`方法确保垃圾收集器释放对象 A 和 b 占用的内存空间。

所以当我们到了那里，你会看到我们能用它做多少事。但是现在，只需要知道 python 在维护和管理我们的内存方面做得非常好。

## 如果没有进行垃圾收集，原因可能是什么？

我想指出的另一件事是，如果你的内存快满了并且用完了，垃圾收集将不会运行，因为垃圾收集运行需要内存。

假设你的程序很大，占用了很多内存，没有足够的内存来运行垃圾收集，那么你会得到一堆异常，你会有一堆问题。

所以要注意，如果你有很多这样的问题，那么你可能要习惯这个模块，在你的程序中早一点运行它。

## 结论

希望这篇文章有深刻的见解。请在下面的反馈部分告诉我们您的想法。

## 参考

[https://docs.python.org/3/library/gc.html](https://docs.python.org/3/library/gc.html)