# 如何在 Python 中使用生成器和 yield

> 原文：<https://realpython.com/introduction-to-python-generators/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 生成器 101**](/courses/python-generators/)

您是否曾经不得不处理如此大的数据集，以至于超出了计算机的内存？或者，您可能有一个复杂的函数，每次调用它时都需要维护一个内部状态，但该函数太小，不值得创建自己的类。在这些情况下以及更多的情况下，生成器和 Python yield 语句可以提供帮助。

到本文结束时，你会知道:

*   什么是**发电机**以及如何使用它们
*   如何创建**生成器函数和表达式**
*   Python yield 语句的工作原理
*   如何在生成器函数中使用多个 Python yield 语句
*   如何使用**高级生成器方法**
*   如何**用多个生成器构建数据管道**

如果你是 Pythonic 的初学者或中级用户，并且对学习如何以更 Pythonic 化的方式处理大型数据集感兴趣，那么这就是适合你的教程。

您可以通过单击下面的链接获得本教程中使用的数据集的副本:

**下载数据集:** [单击此处下载您将在本教程中使用的数据集](https://realpython.com/bonus/generators-yield/)以了解 Python 中的生成器和 yield。

## 使用发电机

随着 [PEP 255](https://www.python.org/dev/peps/pep-0255) 、**的引入，生成器函数**是一种特殊的函数，它返回一个[惰性迭代器](https://en.wikipedia.org/wiki/Lazy_evaluation)。这些是你可以像列表一样循环的对象。然而，与列表不同，惰性迭代器不将内容存储在内存中。关于 Python 中迭代器的概述，请看一下[Python“for”循环(明确迭代)](https://realpython.com/python-for-loop/)。

现在您对发电机有了一个大致的概念，您可能想知道它们在运行时是什么样子。我们来看两个例子。首先，您将从鸟瞰图中看到发电机是如何工作的。然后，您将放大并更彻底地检查每个示例。

[*Remove ads*](/account/join/)

### 示例 1:读取大文件

生成器的一个常见用例是[处理数据流或大文件](https://realpython.com/working-with-files-in-python/)，比如 [CSV 文件](https://realpython.com/courses/reading-and-writing-csv-files/)。这些文本文件使用逗号将数据分隔成列。这种格式是共享数据的常用方式。现在，如果您想计算 CSV 文件中的行数，该怎么办呢？下面的代码块显示了计算这些行的一种方法:

```py
csv_gen = csv_reader("some_csv.txt")
row_count = 0

for row in csv_gen:
    row_count += 1

print(f"Row count is {row_count}")
```

看这个例子，你可能会认为`csv_gen`是一个列表。为了填充这个列表，`csv_reader()`打开一个文件并将其内容加载到`csv_gen`中。然后，程序遍历列表，并为每一行增加`row_count`。

这是一个合理的解释，但是如果文件非常大，这种设计还有效吗？如果文件比您可用的内存大怎么办？为了回答这个问题，让我们假设`csv_reader()`只是打开文件并把它读入一个数组:

```py
def csv_reader(file_name):
    file = open(file_name)
    result = file.read().split("\n")
    return result
```

这个函数打开一个给定的文件，使用`file.read()`和`.split()`将每一行作为一个单独的元素添加到一个列表中。如果您在前面看到的行计数代码块中使用这个版本的`csv_reader()`,那么您将得到以下输出:

>>>

```py
Traceback (most recent call last):
  File "ex1_naive.py", line 22, in <module>
    main()
  File "ex1_naive.py", line 13, in main
    csv_gen = csv_reader("file.txt")
  File "ex1_naive.py", line 6, in csv_reader
    result = file.read().split("\n")
MemoryError
```

在这种情况下，`open()`返回一个生成器对象，您可以一行一行地进行惰性迭代。然而，`file.read().split()`一次将所有内容加载到内存中，导致了`MemoryError`。

在这之前，你可能会注意到你的电脑慢如蜗牛。你甚至可能需要用一个`KeyboardInterrupt`来终止程序。那么，如何处理这些庞大的数据文件呢？看看`csv_reader()`的新定义:

```py
def csv_reader(file_name):
    for row in open(file_name, "r"):
        yield row
```

在这个版本中，您打开文件，遍历它，并产生一行。此代码应该产生以下输出，没有内存错误:

```py
Row count is 64186394
```

这里发生了什么事？嗯，你实际上已经把`csv_reader()`变成了一个生成器函数。这个版本打开一个文件，遍历每一行，产生每一行，而不是返回它。

您还可以定义一个**生成器表达式**(也称为**生成器理解**)，其语法与[列表理解](https://realpython.com/courses/using-list-comprehensions-effectively/)非常相似。这样，您可以在不调用函数的情况下使用生成器:

```py
csv_gen = (row for row in open(file_name))
```

这是创建列表`csv_gen`的一种更简洁的方式。您将很快了解到关于 Python yield 语句的更多信息。现在，只要记住这个关键区别:

*   使用`yield`将产生一个生成器对象。
*   使用`return`将导致文件*的第一行只有*。

### 示例 2:生成无限序列

让我们换个话题，看看无限序列生成。在 Python 中，为了得到一个有限序列，你调用 [`range()`](https://realpython.com/python-range/) 并在一个列表上下文中对其求值:

>>>

```py
>>> a = range(5)
>>> list(a)
[0, 1, 2, 3, 4]
```

然而，生成一个**无限序列**需要使用一个生成器，因为你的计算机内存是有限的:

```py
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
```

这个代码块又短又甜。首先，你初始化[变量](https://realpython.com/python-variables/) `num`并开始一个无限循环。然后，你立即`yield num`这样你就可以捕捉到初始状态。这模仿了`range()`的动作。

在`yield`之后，你将`num`增加 1。如果你用一个 [`for`回路](https://realpython.com/python-for-loop/)试一下，你会发现它看起来确实是无限的:

>>>

```py
>>> for i in infinite_sequence():
...     print(i, end=" ")
...
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
30 31 32 33 34 35 36 37 38 39 40 41 42
[...]
6157818 6157819 6157820 6157821 6157822 6157823 6157824 6157825 6157826 6157827
6157828 6157829 6157830 6157831 6157832 6157833 6157834 6157835 6157836 6157837
6157838 6157839 6157840 6157841 6157842
KeyboardInterrupt
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
```

该程序将继续执行，直到您手动停止它。

除了使用`for`循环，还可以直接在生成器对象上调用`next()`。这对于在控制台中测试生成器特别有用:

>>>

```py
>>> gen = infinite_sequence()
>>> next(gen)
0
>>> next(gen)
1
>>> next(gen)
2
>>> next(gen)
3
```

这里，您有一个名为`gen`的生成器，您通过重复调用`next()`来手动迭代它。这是一个很好的健全检查，确保您的生成器产生您期望的输出。

**注意:**当你使用`next()`时，Python 在你作为参数传入的函数上调用`.__next__()`。这种参数化允许一些特殊的效果，但这超出了本文的范围。尝试改变你传递给`next()`的参数，看看会发生什么！

[*Remove ads*](/account/join/)

### 示例 3:检测回文

您可以在许多方面使用无限序列，但它们的一个实际用途是构建回文检测器。一个回文检测器将定位所有回文的字母序列或 T2 数字序列。这些是向前和向后读都一样的单词或数字，比如 121。首先，定义您的数字回文检测器:

```py
def is_palindrome(num):
    # Skip single-digit inputs
    if num // 10 == 0:
        return False
    temp = num
    reversed_num = 0

    while temp != 0:
        reversed_num = (reversed_num * 10) + (temp % 10)
        temp = temp // 10

    if num == reversed_num:
        return num
    else:
        return False
```

不要太担心理解这段代码中的底层数学。请注意，该函数接受一个输入数字，将其反转，并检查反转后的数字是否与原始数字相同。现在你可以使用你的无限序列生成器得到一个所有数字回文的运行列表:

>>>

```py
>>> for i in infinite_sequence():
...     pal = is_palindrome(i)
...     if pal:
...         print(i)
...
11
22
33
[...]
99799
99899
99999
100001
101101
102201
KeyboardInterrupt
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "<stdin>", line 5, in is_palindrome
```

在这种情况下，[打印](https://realpython.com/python-print/)到控制台的唯一数字是那些向前或向后相同的数字。

**注意:**实际上，你不太可能编写自己的无限序列生成器。 [`itertools`](https://realpython.com/python-itertools/) 模块提供了一个非常高效的带`itertools.count()`的无限序列发生器。

现在您已经看到了一个无限序列生成器的简单用例，让我们更深入地了解生成器是如何工作的。

## 了解发电机

到目前为止，您已经了解了创建生成器的两种主要方式:使用生成器函数和生成器表达式。你甚至可能对发电机如何工作有一个直观的理解。让我们花一点时间让这些知识更清晰一点。

生成器函数的外观和行为就像常规函数一样，但有一个定义特征。生成器函数使用 Python [`yield`关键字](https://realpython.com/python-keywords/#returning-keywords-return-yield)而不是`return`。回想一下您之前编写的生成器函数:

```py
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
```

这看起来像一个典型的[函数定义](https://realpython.com/defining-your-own-python-function/)，除了 Python yield 语句和它后面的代码。`yield`表示在哪里将一个值发送回调用者，但是与`return`不同的是，您不能在之后退出该函数。

相反，功能的**状态**被记住。这样，当在生成器对象上调用`next()`(在`for`循环中显式或隐式调用)时，先前产生的变量`num`递增，然后再次产生。由于生成器函数看起来像其他函数，并且行为与它们非常相似，您可以假设生成器表达式与 Python 中其他可用的理解非常相似。

**注:**你对 Python 的列表、集合、字典理解生疏了吗？你可以有效地使用列表理解来检查。

### 用生成器表达式构建生成器

像列表理解一样，生成器表达式允许您用几行代码快速创建一个生成器对象。它们在使用列表理解的相同情况下也很有用，还有一个额外的好处:您可以创建它们，而无需在迭代之前构建整个对象并将其保存在内存中。换句话说，当您使用生成器表达式时，您不会有内存损失。举个平方一些数字的例子:

>>>

```py
>>> nums_squared_lc = [num**2 for num in range(5)]
>>> nums_squared_gc = (num**2 for num in range(5))
```

`nums_squared_lc`和`nums_squared_gc`看起来基本相同，但有一个关键的区别。你能发现它吗？看看当你检查这些物体时会发生什么:

>>>

```py
>>> nums_squared_lc
[0, 1, 4, 9, 16]
>>> nums_squared_gc
<generator object <genexpr> at 0x107fbbc78>
```

第一个对象使用括号构建一个列表，而第二个对象使用括号创建一个生成器表达式。输出确认您已经创建了一个生成器对象，并且它不同于列表。

[*Remove ads*](/account/join/)

### 剖析发生器性能

您之前已经了解到生成器是优化内存的一个很好的方法。虽然无限序列生成器是这种优化的一个极端例子，但让我们放大刚才看到的数字平方例子，并检查结果对象的大小。您可以通过调用`sys.getsizeof()`来做到这一点:

>>>

```py
>>> import sys
>>> nums_squared_lc = [i ** 2 for i in range(10000)]
>>> sys.getsizeof(nums_squared_lc)
87624
>>> nums_squared_gc = (i ** 2 for i in range(10000))
>>> print(sys.getsizeof(nums_squared_gc))
120
```

在这个例子中，你从 list comprehension 得到的列表是 87，624 字节，而 generator 对象只有 120 字节。这意味着列表比生成器对象大 700 多倍！

不过，有一件事要记住。如果列表小于运行机器的可用内存，那么列表理解可以比等价的生成器表达式更快地评估 T2。为了探索这一点，让我们总结以上两种理解的结果。您可以使用`cProfile.run()`生成读数:

>>>

```py
>>> import cProfile
>>> cProfile.run('sum([i * 2 for i in range(10000)])')
 5 function calls in 0.001 seconds

 Ordered by: standard name

 ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 1    0.001    0.001    0.001    0.001 <string>:1(<listcomp>)
 1    0.000    0.000    0.001    0.001 <string>:1(<module>)
 1    0.000    0.000    0.001    0.001 {built-in method builtins.exec}
 1    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
 1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

>>> cProfile.run('sum((i * 2 for i in range(10000)))')
 10005 function calls in 0.003 seconds

 Ordered by: standard name

 ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 10001    0.002    0.000    0.002    0.000 <string>:1(<genexpr>)
 1    0.000    0.000    0.003    0.003 <string>:1(<module>)
 1    0.000    0.000    0.003    0.003 {built-in method builtins.exec}
 1    0.001    0.001    0.003    0.003 {built-in method builtins.sum}
 1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

在这里，您可以看到对列表中的所有值求和所用的时间大约是对生成器求和所用时间的三分之一。如果速度是一个问题，而记忆不是，那么列表理解可能是一个更好的工作工具。

**注意:**这些测量不仅仅对使用生成器表达式制作的对象有效。它们对于由模拟生成器函数生成的对象也是一样的，因为生成的生成器是等价的。

记住，列表理解返回完整列表，而生成器表达式返回生成器。无论是从函数还是从表达式构建，生成器的工作方式都是一样的。使用表达式只允许您在一行中定义简单的生成器，并在每个内部迭代的末尾假定一个`yield`。

Python yield 语句当然是生成器所有功能的关键所在，所以让我们深入了解一下`yield`在 Python 中是如何工作的。

## 理解 Python Yield 语句

总的来说，`yield`是一个相当简单的说法。它的主要工作是以类似于`return`语句的方式控制生成器函数的流程。正如上面简要提到的，Python yield 语句有一些技巧。

当调用生成器函数或使用生成器表达式时，会返回一个称为生成器的特殊迭代器。您可以将该生成器分配给一个变量，以便使用它。当您调用生成器上的特殊方法时，比如`next()`，函数中的代码执行到`yield`。

当 Python yield 语句被命中时，程序会暂停函数执行，并将生成的值返回给调用者。(相反，`return`完全停止功能执行。)当函数被挂起时，该函数的状态被保存。这包括生成器本地的任何变量绑定、指令指针、内部堆栈和任何异常处理。

这允许您在调用生成器的某个方法时恢复函数执行。通过这种方式，所有的函数计算在`yield`之后立即恢复。通过使用多个 Python yield 语句，您可以看到这一点:

>>>

```py
>>> def multi_yield():
...     yield_str = "This will print the first string"
...     yield yield_str
...     yield_str = "This will print the second string"
...     yield yield_str
...
>>> multi_obj = multi_yield()
>>> print(next(multi_obj))
This will print the first string
>>> print(next(multi_obj))
This will print the second string
>>> print(next(multi_obj))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

仔细看看最后一次对`next()`的调用。你可以看到执行已经被一个[回溯](https://realpython.com/python-traceback/)放大了。这是因为生成器和所有迭代器一样，可能会耗尽。除非你的生成器是无限的，否则你只能迭代一次。一旦评估完所有值，迭代将停止，并且`for`循环将退出。如果您使用了`next()`，那么您将得到一个显式的`StopIteration`异常。

**注意:** `StopIteration`是一个自然的异常，引发它是为了表示迭代器的结束。例如，`for`循环是围绕`StopIteration`建立的。您甚至可以通过使用 [`while`循环](https://realpython.com/python-while-loop/)来实现自己的`for`循环:

>>>

```py
>>> letters = ["a", "b", "c", "y"]
>>> it = iter(letters)
>>> while True:
...     try:
...         letter = next(it)
...     except StopIteration:
...         break
...     print(letter)
...
a
b
c
y
```

你可以在关于[异常](https://docs.python.org/3.6/library/exceptions.html)的 Python 文档中阅读更多关于`StopIteration`的内容。关于迭代的更多信息，请查看[Python“for”循环(有限迭代)](https://realpython.com/python-for-loop/)和[Python“while”循环(无限迭代)](https://realpython.com/python-while-loop/)。

`yield`可以用多种方式来控制你的生成器的执行流程。只要您的创造力允许，就可以使用多个 Python yield 语句。

## 使用高级生成器方法

您已经了解了发生器最常见的用途和构造，但是还需要了解一些技巧。除了`yield`，生成器对象可以利用以下方法:

*   `.send()`
*   `.throw()`
*   `.close()`

[*Remove ads*](/account/join/)

### 如何使用`.send()`

在下一节中，您将构建一个利用这三种方法的程序。这个程序将像以前一样打印数字回文，但是做了一些调整。当遇到回文时，你的新程序会添加一个数字，并从那里开始搜索下一个数字。您还将使用`.throw()`处理异常，并使用`.close()`在给定的位数之后停止生成器。首先，让我们回忆一下回文检测器的代码:

```py
def is_palindrome(num):
    # Skip single-digit inputs
    if num // 10 == 0:
        return False
    temp = num
    reversed_num = 0

    while temp != 0:
        reversed_num = (reversed_num * 10) + (temp % 10)
        temp = temp // 10

    if num == reversed_num:
        return True
    else:
        return False
```

这是您之前看到的相同代码，只是现在程序严格返回`True`或`False`。您还需要修改原来的无限序列生成器，如下所示:

```py
 1def infinite_palindromes():
 2    num = 0
 3    while True:
 4        if is_palindrome(num):
 5            i = (yield num)
 6            if i is not None:
 7                num = i
 8        num += 1
```

这里变化很大！您将看到的第一个在第 5 行，这里是`i = (yield num)`。尽管您之前了解到`yield`是一个陈述，但这并不是全部。

从 Python 2.5(引入了您现在正在学习的方法的同一个版本)开始，`yield`是一个**表达式**，而不是一个语句。当然，你还是可以用它来做陈述。但是现在，您也可以像在上面的代码块中看到的那样使用它，其中`i`获取产生的值。这允许您操作产生的值。更重要的是，它允许你`.send()`一个值回生成器。当`yield`之后执行拾取时，`i`将获取发送的值。

您还将检查`if i is not None`，如果在生成器对象上调用`next()`，这可能会发生。(当你用`for`循环迭代时，这也会发生。)如果`i`有一个值，那么用新值更新`num`。但是不管`i`是否有值，你都要增加`num`并再次开始循环。

现在，看一下主函数代码，它将最低的数字和另一个数字发送回生成器。例如，如果回文是 121，那么它将`.send()` 1000:

```py
pal_gen = infinite_palindromes()
for i in pal_gen:
    digits = len(str(i))
    pal_gen.send(10 ** (digits))
```

使用这段代码，您可以创建生成器对象并遍历它。程序只在找到一个回文时才给出一个值。它使用`len()`来确定回文中的位数。然后，它将`10 ** digits`发送给生成器。这将执行带回生成器逻辑，并将`10 ** digits`分配给`i`。由于`i`现在有了一个值，程序更新`num`，递增，并再次检查回文。

一旦您的代码找到并产生另一个回文，您将通过`for`循环进行迭代。这和用`next()`迭代是一样的。发电机也在 5 号线与`i = (yield num)`接通。但是，现在`i`是 [`None`](https://realpython.com/null-in-python/) ，因为你没有显式发送一个值。

您在这里创建的是一个**协程**，或者一个您可以向其中传递数据的生成器函数。这些对于构建数据管道很有用，但是您很快就会看到，它们对于构建数据管道并不是必需的。(如果你想更深入地学习，那么[这门关于协程和并发性的课程](http://www.dabeaz.com/coroutines/)是最全面的课程之一。)

既然你已经了解了`.send()`，那我们就来看看`.throw()`。

### 如何使用`.throw()`

允许你用生成器抛出异常。在下面的例子中，您在第 6 行引发了异常。一旦`digits`达到 5:

```py
 1pal_gen = infinite_palindromes()
 2for i in pal_gen:
 3    print(i)
 4    digits = len(str(i))
 5    if digits == 5:
 6        pal_gen.throw(ValueError("We don't like large palindromes"))
 7    pal_gen.send(10 ** (digits))
```

这与前面的代码相同，但现在您将检查`digits`是否等于 5。如果是这样，那么你就`.throw()`一个`ValueError`。要确认这是否如预期的那样工作，请看一下代码的输出:

>>>

```py
11
111
1111
10101
Traceback (most recent call last):
  File "advanced_gen.py", line 47, in <module>
    main()
  File "advanced_gen.py", line 41, in main
    pal_gen.throw(ValueError("We don't like large palindromes"))
  File "advanced_gen.py", line 26, in infinite_palindromes
    i = (yield num)
ValueError: We don't like large palindromes
```

`.throw()`在你可能需要捕捉[异常](https://realpython.com/courses/introduction-python-exceptions/)的任何领域都很有用。在这个例子中，您使用了`.throw()`来控制何时停止遍历生成器。你可以用`.close()`更优雅地做到这一点。

[*Remove ads*](/account/join/)

### 如何使用`.close()`

顾名思义，`.close()`允许您停止发电机。这在控制无限序列发生器时尤其方便。让我们通过将`.throw()`改为`.close()`来更新上面的代码，以停止迭代:

```py
 1pal_gen = infinite_palindromes()
 2for i in pal_gen:
 3    print(i)
 4    digits = len(str(i))
 5    if digits == 5:
 6        pal_gen.close() 7    pal_gen.send(10 ** (digits))
```

不调用`.throw()`，而是在第 6 行使用`.close()`。使用`.close()`的好处是它会引发`StopIteration`，这是一个异常，用来表示有限迭代器的结束:

>>>

```py
11
111
1111
10101
Traceback (most recent call last):
  File "advanced_gen.py", line 46, in <module>
    main()
  File "advanced_gen.py", line 42, in main
    pal_gen.send(10 ** (digits))
StopIteration
```

现在，您已经了解了更多关于生成器附带的特殊方法，让我们来谈谈使用生成器来构建数据管道。

## 用生成器创建数据管道

数据管道允许你将代码串在一起处理大型数据集或数据流，而不会耗尽你机器的内存。假设您有一个很大的 CSV 文件:

```py
permalink,company,numEmps,category,city,state,fundedDate,raisedAmt,raisedCurrency,round
digg,Digg,60,web,San Francisco,CA,1-Dec-06,8500000,USD,b
digg,Digg,60,web,San Francisco,CA,1-Oct-05,2800000,USD,a
facebook,Facebook,450,web,Palo Alto,CA,1-Sep-04,500000,USD,angel
facebook,Facebook,450,web,Palo Alto,CA,1-May-05,12700000,USD,a
photobucket,Photobucket,60,web,Palo Alto,CA,1-Mar-05,3000000,USD,a
```

这个例子来自 TechCrunch Continental USA 集合，它描述了美国各种初创公司的融资轮次和金额。单击下面的链接下载数据集:

**下载数据集:** [单击此处下载您将在本教程中使用的数据集](https://realpython.com/bonus/generators-yield/)以了解 Python 中的生成器和 yield。

是时候用 Python 做一些处理了！为了演示如何使用生成器构建管道，您将分析该文件以获得数据集中所有 A 轮的总数和平均值。

让我们想一个策略:

1.  阅读文件的每一行。
2.  将每一行拆分成一个值列表。
3.  提取列名。
4.  使用列名和列表创建词典。
5.  过滤掉你不感兴趣的回合。
6.  计算您感兴趣的轮次的总值和平均值。

通常，你可以用一个像 [`pandas`](https://realpython.com/pandas-python-explore-dataset/) 这样的包来实现这个功能，但是你也可以只用几个生成器来实现这个功能。您将从使用生成器表达式读取文件中的每一行开始:

```py
 1file_name = "techcrunch.csv"
 2lines = (line for line in open(file_name))
```

然后，您将使用另一个生成器表达式与前一个相配合，将每一行拆分成一个列表:

```py
 3list_line = (s.rstrip().split(",") for s in lines)
```

这里，您创建了生成器`list_line`，它迭代第一个生成器`lines`。这是设计生成器管道时常用的模式。接下来，您将从`techcrunch.csv`中提取列名。因为列名往往构成 CSV 文件的第一行，所以您可以通过一个简短的`next()`调用来获取它:

```py
 4cols = next(list_line)
```

对`next()`的调用使迭代器在`list_line`生成器上前进一次。将所有这些放在一起，您的代码应该如下所示:

```py
 1file_name = "techcrunch.csv"
 2lines = (line for line in open(file_name))
 3list_line = (s.rstrip().split(",") for s in lines)
 4cols = next(list_line)
```

综上所述，首先创建一个生成器表达式`lines`来生成文件中的每一行。接下来，在另一个名为`list_line`的生成器表达式*的定义内遍历该生成器，将每一行转换成一个值列表。然后，用`next()`将`list_line`的迭代向前推进一次，从 CSV 文件中获得列名列表。*

**注意**:注意尾随换行符！这段代码利用`list_line`生成器表达式中的`.rstrip()`来确保没有尾随换行符，这些字符可能出现在 CSV 文件中。

为了帮助您对数据进行过滤和执行操作，您将创建字典，其中的键是来自 CSV:

```py
 5company_dicts = (dict(zip(cols, data)) for data in list_line)
```

这个生成器表达式遍历由`list_line`生成的列表。然后，它使用 [`zip()`](https://realpython.com/python-zip-function/) 和`dict()`创建如上指定的字典。现在，您将使用一个*第四个*生成器来过滤您想要的融资回合，并拉动`raisedAmt`:

```py
 6funding = (
 7    int(company_dict["raisedAmt"])
 8    for company_dict in company_dicts
 9    if company_dict["round"] == "a"
10)
```

在这段代码中，您的生成器表达式遍历`company_dicts`的结果，并为任何`company_dict`获取`raisedAmt`，其中`round`键为`"a"`。

请记住，您不是在生成器表达式中一次遍历所有这些。事实上，在你真正使用一个`for`循环或者一个作用于可迭代对象的函数，比如`sum()`之前，你不会迭代任何东西。事实上，现在调用`sum()`来迭代生成器:

```py
11total_series_a = sum(funding)
```

将所有这些放在一起，您将生成以下脚本:

```py
 1file_name = "techcrunch.csv"
 2lines = (line for line in open(file_name))
 3list_line = (s.rstrip()split(",") for s in lines)
 4cols = next(list_line)
 5company_dicts = (dict(zip(cols, data)) for data in list_line)
 6funding = (
 7    int(company_dict["raisedAmt"])
 8    for company_dict in company_dicts
 9    if company_dict["round"] == "a"
10)
11total_series_a = sum(funding)
12print(f"Total series A fundraising: ${total_series_a}")
```

这个脚本将您构建的每个生成器集合在一起，它们都作为一个大数据管道运行。下面是一行行的分析:

*   **第 2 行**读入文件的每一行。
*   **第 3 行**将每一行拆分成值，并将这些值放入一个列表中。
*   **第 4 行**使用`next()`将列名存储在一个列表中。
*   **第 5 行**创建字典并用一个`zip()`调用将它们联合起来:
    *   **键**是第 4 行的列名`cols`。
    *   **值**是列表形式的行，在第 3 行创建。
*   **第 6 行**获得每家公司的 A 轮融资金额。它还会过滤掉任何其他增加的金额。
*   **第 11 行**通过调用`sum()`来获取 CSV 中的首轮融资总额，从而开始迭代过程。

当您在`techcrunch.csv`上运行这段代码时，您应该会发现在首轮融资中总共筹集了 4，376，015，000 美元。

**注意:**本教程中开发的处理 CSV 文件的方法对于理解如何使用生成器和 Python yield 语句非常重要。但是，当您在 Python 中处理 CSV 文件时，您应该使用 Python 标准库中包含的 [`csv`](https://docs.python.org/3/library/csv.html) 模块。该模块优化了有效处理 CSV 文件的方法。

为了更深入地挖掘，试着计算出每家公司在首轮融资中的平均融资额。这有点棘手，所以这里有一些提示:

*   生成器在被完全迭代后会耗尽自己。
*   您仍然需要`sum()`函数。

祝你好运！

[*Remove ads*](/account/join/)

## 结论

在本教程中，你已经学习了**生成器函数**和**生成器表达式**。

**你现在知道了:**

*   如何使用和编写生成器函数和生成器表达式
*   最重要的 **Python yield 语句**如何启用生成器
*   如何在生成器函数中使用多个 Python yield 语句
*   如何使用`.send()`向发生器发送数据
*   如何使用`.throw()`引发发电机异常
*   如何使用`.close()`来停止生成器的迭代
*   如何**构建一个生成器管道**来高效地处理大型 CSV 文件

您可以通过下面的链接获得本教程中使用的数据集:

**下载数据集:** [单击此处下载您将在本教程中使用的数据集](https://realpython.com/bonus/generators-yield/)以了解 Python 中的生成器和 yield。

发电机对您的工作或项目有什么帮助？如果你只是刚刚了解它们，那么你打算将来如何使用它们呢？找到解决数据管道问题的好办法了吗？请在下面的评论中告诉我们！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 生成器 101**](/courses/python-generators/)********