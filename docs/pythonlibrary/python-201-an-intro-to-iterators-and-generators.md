# Python 201:迭代器和生成器介绍

> 原文：<https://www.blog.pythonlibrary.org/2016/05/03/python-201-an-intro-to-iterators-and-generators/>

自从开始用 Python 编程以来，你可能一直在使用迭代器和生成器，但你可能没有意识到这一点。在这篇文章中，我们将学习什么是迭代器和生成器。我们也将学习它们是如何被创建的，这样我们就可以在需要的时候创建我们自己的。

### 迭代程序

迭代器是一个允许你遍历容器的对象。Python 中的迭代器通过两种不同的方法实现: **__iter__** 和 **__next__** 。您的容器需要 **__iter__** 方法来提供迭代支持。它将返回迭代器对象本身。但是如果您想要创建一个迭代器对象，那么您还需要定义 **__next__** ，这将返回容器中的下一个项目。

*注意:在 Python 2 中，命名约定略有不同。你仍然需要 **__iter__** ，但是 **__next__** 被称为 **next** 。* 
为了让事情更加清楚，让我们回顾一下几个定义:

*   iterable -定义了 __iter__ 方法的对象
*   iterator -同时定义了 __iter__ 和 __next__ 的对象，其中 __iter__ 将返回迭代器对象，而 __next__ 将返回迭代中的下一个元素。

与大多数神奇的方法(带有双下划线的方法)一样，您不应该直接调用 __iter__ 或 __next__。相反，你可以使用一个**进行**循环或列表理解，Python 会自动为你调用这些方法。在某些情况下，您可能需要调用它们，但是您可以使用 Python 的内置函数来这样做: **iter** 和 **next** 。

在我们继续之前，我想提一下序列。Python 3 有列表、元组、范围等几种序列类型。该列表是可迭代的，但不是迭代器，因为它不实现 __next__。这在下面的例子中很容易看出:

```py

>>> my_list = [1, 2, 3]
>>> next(my_list)
Traceback (most recent call last):
  Python Shell, prompt 2, line 1
builtins.TypeError: 'list' object is not an iterator

```

在上面的例子中，当我们试图调用 list 的 next 方法时，我们收到了一个 **TypeError** ，并被告知 list 对象不是迭代器。

```py

>>> iter(my_list)
 >>> list_iterator = iter(my_list)
>>> next(list_iterator)
1
>>> next(list_iterator)
2
>>> next(list_iterator)
3
>>> next(list_iterator)
Traceback (most recent call last):
  Python Shell, prompt 8, line 1
builtins.StopIteration: 
```

要将列表转换成迭代器，只需将其封装在对 Python 的 **iter** 方法的调用中。然后你可以调用**的下一个**，直到迭代器用完所有条目，并且**的 StopIteration** 被抛出。让我们试着把这个列表变成一个迭代器，并用一个循环对它进行迭代:

```py

>>> for item in iter(my_list):
...     print(item)
... 
1
2
3

```

当使用循环迭代迭代器时，不需要调用 next，也不必担心会引发 StopIteration 异常。

* * *

### 创建你自己的迭代器

偶尔你会想要创建你自己的自定义迭代器。Python 让这变得非常容易。如前一节所述，您需要做的就是在您的类中实现 __iter__ 和 __next__ 方法。让我们创建一个迭代器，它可以迭代一串字母:

```py

class MyIterator:

    def __init__(self, letters):
        """
        Constructor
        """
        self.letters = letters
        self.position = 0

    def __iter__(self):
        """
        Returns itself as an iterator
        """
        return self

    def __next__(self):
        """
        Returns the next letter in the sequence or 
        raises StopIteration
        """
        if self.position >= len(self.letters):
            raise StopIteration
        letter = self.letters[self.position]
        self.position += 1
        return letter

if __name__ == '__main__':
    i = MyIterator('abcd')
    for item in i:
        print(item)

```

对于这个例子，我们的类中只需要三个方法。在我们的初始化中，我们传入字母字符串并创建一个类变量来引用它们。我们还初始化了一个位置变量，这样我们总是知道我们在字符串中的位置。__iter__ 方法只返回它自己，这是它真正需要做的。__next__ 方法是这个类中最重要的部分。在这里，我们根据字符串的长度检查位置，如果我们试图超过它的长度，就引发 StopIteration。否则，我们提取我们所在的字母，增加位置并返回字母。

让我们花点时间来创建一个无限迭代器。无限迭代器是可以永远迭代的迭代器。在调用这些函数时你需要小心，因为如果你不确定给它们加一个界限，它们会导致一个无限循环。

```py

class Doubler:
    """
    An infinite iterator
    """

    def __init__(self):
        """
        Constructor
        """
        self.number = 0

    def __iter__(self):
        """
        Returns itself as an iterator
        """
        return self

    def __next__(self):
        """
        Doubles the number each time next is called
        and returns it. 
        """
        self.number += 1
        return self.number * self.number

if __name__ == '__main__':
    doubler = Doubler()
    count = 0

    for number in doubler:
        print(number)
        if count > 5:
            break
        count += 1         

```

在这段代码中，我们不向迭代器传递任何东西。我们只是实例化它。然后，为了确保我们不会陷入无限循环，我们在开始迭代自定义迭代器之前添加了一个计数器。最后，当计数器超过 5 时，我们开始迭代并中断。

* * *

### 发电机

一个普通的 Python 函数将总是返回一个值，无论它是一个列表、一个整数还是一些其他对象。但是，如果您希望能够调用一个函数并让它产生一系列值，该怎么办呢？这就是发电机的用武之地。生成器的工作原理是“保存”它最后停止的地方(或产出)，并给调用函数一个值。因此，它不是将执行返回给调用者，而是将临时控制权交还给调用者。要做到这一点，生成器函数需要 Python 的 **yield** 语句。

附注:在其他语言中，生成器可能被称为协程。

让我们花点时间创建一个简单的生成器！

```py

>>> def doubler_generator():
...     number = 2
...     while True:
...         yield number
...         number *= number
>>> doubler = doubler_generator()
>>> next(doubler)
2
>>> next(doubler)
4
>>> next(doubler)
16
>>> type(doubler)

```

这个特殊的生成器基本上会创建一个无限序列。你可以整天在它上面调用**下一个**，它将永远不会失去价值。因为可以在生成器上迭代，所以生成器被认为是迭代器的一种，但是没有人真正这样称呼它们。但是在幕后，生成器也定义了我们在上一节中看到的 **__next__** 方法，这就是为什么我们刚刚使用的 **next** 关键字有效。

让我们看看另一个例子，它只产生 3 项，而不是一个无限序列！

```py

>>> def silly_generator():
...     yield "Python"
...     yield "Rocks"
...     yield "So do you!"
>>> gen = silly_generator()
>>> next(gen)
'Python'
>>> next(gen)
'Rocks'
>>> next(gen)
'So do you!'
>>> next(gen)
Traceback (most recent call last):
  Python Shell, prompt 21, line 1
builtins.StopIteration:

```

这里我们有一个生成器，它使用了 3 次 **yield** 语句。在每种情况下，它都会产生不同的字符串。你可以把**收益率**想象成一个发电机的**收益**语句。无论何时调用 yield，该函数都会停止并保存其状态。然后它输出值 out，这就是为什么在上面的例子中你会看到一些东西被输出到终端。如果我们的函数中有变量，这些变量也会被保存。

当你看到 **StopIteration** 的时候，你就知道你已经穷尽了迭代器。这意味着它用完了项目。这是所有迭代器的正常行为，正如你在迭代器部分看到的一样。

无论如何，当我们再次调用 **next** 时，生成器从它停止的地方开始，产生下一个值，或者我们完成函数，生成器停止。另一方面，如果你不再调用 next，那么这个状态最终会消失。

让我们重新实例化生成器，并尝试遍历它！

```py

>>> gen = silly_generator()
>>> for item in gen:
...     print(item)
... 
Python
Rocks
So do you!

```

我们创建生成器的一个新实例的原因是，如果我们试图对它进行循环，将不会产生任何结果。这是因为我们已经遍历了生成器的特定实例中的所有值。因此，在本例中，我们创建了新的实例，对其进行循环，并打印出生成的值。循环的**再次为我们处理 **StopIteration** 异常，并在生成器耗尽时退出循环。**

生成器的最大好处之一是它可以迭代大型数据集，并一次返回一部分。当我们打开一个文件并逐行返回时，就会发生这种情况:

```py

with open('/path/to/file.txt') as fobj:
    for line in fobj:
        # process the line

```

当我们以这种方式迭代文件对象时，Python 基本上将它变成了一个生成器。这允许我们处理太大而无法加载到内存中的文件。您会发现生成器对于您需要成块处理的任何大型数据集都很有用，或者当您需要生成一个大型数据集，否则它会填满您所有的计算机内存。

* * *

### 包扎

至此，你应该明白什么是迭代器以及如何使用迭代器。您还应该知道 iterable 和 iterator 之间的区别。最后，我们学习了什么是发电机，以及为什么你可能想要使用发电机。例如，生成器非常适合内存高效的数据处理。编码快乐！