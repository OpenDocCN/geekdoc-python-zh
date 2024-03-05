# PyPy:用最少的努力更快的 Python

> 原文：<https://realpython.com/pypy-faster-python/>

Python 是开发人员中最流行的编程语言之一，但它有一定的局限性。例如，根据应用的不同，它可以比一些低级语言慢 100 倍。这就是为什么一旦 Python 的速度成为用户的瓶颈，许多公司就会用另一种语言重写他们的应用程序。但是，如果有一种方法既能保留 Python 的出色特性，又能提高它的速度，那会怎么样呢？输入 [PyPy](https://doc.pypy.org/en/latest/introduction.html) 。

PyPy 是一个非常兼容的 Python 解释器，是 CPython 2.7、3.6 以及即将到来的 3.7 的有价值的替代品。通过使用它安装和运行您的应用程序，您可以获得显著的速度提升。您将看到多少改进取决于您运行的应用程序。

在本教程中，您将学习:

*   如何用 **PyPy** 安装并运行你的代码
*   PyPy 在速度方面与 **CPython** 相比如何
*   PyPy 的**特性**是什么，它们如何让你的 Python 代码运行**更快**
*   PyPy 的**限制**是什么

本教程中的例子使用 Python 3.6，因为这是 PyPy 兼容的 Python 的最新版本。

## Python 和 PyPy

Python [语言规范](https://docs.python.org/3/reference/index.html)在许多实现中使用，如 [CPython](https://realpython.com/cpython-source-code-guide/) (用 [C](https://realpython.com/c-for-python-programmers/) 编写)、Jython(用 [Java](https://realpython.com/oop-in-python-vs-java/) 编写)、IronPython(用。NET)，还有 PyPy(用 Python 写的)。

CPython 是 Python 的原始实现，是迄今为止最受欢迎和维护最多的。当人们提到 Python 时，他们通常指的是 CPython。你可能正在使用 CPython ！

但是因为是高级解释语言，CPython 有一定的局限性，不会因为速度而得什么奖牌。这就是 PyPy 可以派上用场的地方。由于它符合 Python 语言规范，PyPy 不需要对您的代码库进行任何更改，并且由于您将在下面看到的特性，它可以提供显著的速度改进。

现在，您可能想知道为什么 CPython 不实现 PyPy 的出色功能，如果它们使用相同的语法。原因是实现这些特性需要对源代码进行巨大的修改，这将是一项艰巨的任务。

不要深入理论，让我们看看 PyPy 的行动。

[*Remove ads*](/account/join/)

### 安装

您的操作系统可能已经提供了 PyPy 包。比如在 macOS 上，你可以借助[家酿](https://brew.sh/)来安装:

```py
$ brew install pypy3
```

如果没有，您可以[下载一个预构建的二进制文件用于您的操作系统和架构](https://www.pypy.org/download.html)。完成下载后，只需解压缩 tarball 或 ZIP 文件。然后，您可以执行 PyPy，而不需要在任何地方安装它:

```py
$ tar xf pypy3.6-v7.3.1-osx64.tar.bz2
$ ./pypy3.6-v7.3.1-osx64/bin/pypy3
Python 3.6.9 (?, Jul 19 2020, 21:37:06)
[PyPy 7.3.1 with GCC 4.2.1]
Type "help", "copyright", "credits" or "license" for more information.
```

在执行上面的代码之前，您需要进入下载二进制文件的文件夹。参考[安装文件](https://doc.pypy.org/en/latest/install.html)获取完整说明。

### PyPy 在行动

现在，您已经安装了 PyPy，并准备好查看它的运行情况了！为此，创建一个名为`script.py`的 Python 文件，并将以下代码放入其中:

```py
 1total = 0
 2for i in range(1, 10000):
 3    for j in range(1, 10000):
 4        total += i + j
 5
 6print(f"The result is {total}")
```

这是一个脚本，在两个嵌套的 [`for`循环](https://realpython.com/python-for-loop/)中，将从`1`到`9,999`的数字相加，[打印结果](https://realpython.com/python-print/)。

要查看运行该脚本需要多长时间，请编辑它以添加突出显示的行:

```py
 1import time 2
 3start_time = time.time() 4
 5total = 0
 6for i in range(1, 10000):
 7    for j in range(1, 10000):
 8        total += i + j
 9
10print(f"The result is {total}") 11
12end_time = time.time() 13print(f"It took {end_time-start_time:.2f} seconds to compute")
```

代码现在执行以下操作:

*   **第 3 行**保存当前时间到变量`start_time`。
*   **线 5 到线 8** 运行[循环](https://realpython.com/courses/how-to-write-pythonic-loops/)。
*   **第 10 行**打印结果。
*   **第 12 行**保存当前时间到`end_time`。
*   **第 13 行**打印出`start_time`和`end_time`之间的差异，以显示运行脚本花了多长时间。

尝试用 Python 运行它。这是我在 2015 年 MacBook Pro 上得到的结果:

```py
$ python3.6 script.py
The result is 999800010000
It took 20.66 seconds to compute
```

现在用 PyPy 运行它:

```py
$ pypy3 script.py
The result is 999800010000
It took 0.22 seconds to compute
```

在这个小的合成基准测试中，PyPy 的速度大约是 Python 的 94 倍！

对于更严肃的基准测试，您可以看看 PyPy [速度中心](https://speed.pypy.org/)，在那里开发人员使用不同的可执行文件运行夜间基准测试。

请记住，PyPy 如何影响代码的性能取决于您的代码在做什么。在某些情况下，PyPy 实际上要慢一些，稍后您会看到。但是，在几何平均上，它是 Python 的 4.3 倍。

[*Remove ads*](/account/join/)

## PyPy 及其特性

历史上，PyPy 提到了两件事:

1.  用于生成动态语言解释器的动态语言框架
2.  使用该框架的 Python 实现

通过安装 PyPy 并使用它运行一个小脚本，您已经看到了第二种含义。你用的 Python 实现是用一个叫做 [RPython](https://rpython.readthedocs.io/en/latest/index.html) 的动态语言框架写的，就像 CPython 是用 C 写的，Jython 是用 Java 写的。

但是之前不是告诉你 PyPy 是用 Python 写的吗？嗯，这有点简化了。PyPy 之所以被称为用 Python(而不是用 RPython)编写的 Python 解释器，是因为 RPython 使用与 Python 相同的语法。

为了澄清一切，下面是 PyPy 的生产过程:

1.  源代码是用 RPython 写的。

2.  将 [RPython 翻译工具链](https://rpython.readthedocs.io/en/latest/translation.html)应用于代码，这基本上使代码更加高效。它还将代码编译成机器码，这就是为什么 Mac、Windows 和 Linux 用户必须下载不同版本的代码。

3.  生成二进制可执行文件。这是您用来运行小脚本的 Python 解释器。

请记住，使用 PyPy 不需要经历所有这些步骤。该可执行文件已经可供您安装和使用。

此外，由于在框架和实现中使用同一个词非常令人困惑，PyPy 背后的团队决定不再使用这种双重用法。现在，PyPy 仅指 Python 实现。该框架被称为 **RPython 翻译工具链**。

接下来，您将了解在某些情况下使 PyPy 比 Python 更好更快的特性。

### 实时(JIT)编译器

在进入什么是 JIT 编译之前，让我们后退一步，回顾一下 C 等[编译的](https://en.wikipedia.org/wiki/Compiled_language)语言和 JavaScript 等[解释的](https://en.wikipedia.org/wiki/Interpreted_language)语言的属性。

编译过的编程语言性能更好，但是更难移植到不同的 CPU 架构和操作系统。**解释型**编程语言的可移植性更强，但性能却比编译型语言差很多。这是光谱的两个极端。

还有像 Python 这样的混合编译和解释的编程语言。具体来说，Python 首先被编译成一个**中间字节码**，然后由 CPython 解释。这使得代码比用纯解释编程语言编写的代码性能更好，并且保持了可移植性的优势。

然而，其性能仍然与编译版本相差甚远。原因是编译后的代码可以做很多优化，而字节码是不可能做到的。

这就是**实时(JIT)编译器**的用武之地。它试图通过编译成机器码和一些解释来获得两个世界的更好的部分。简而言之，以下是 JIT 编译提供更快性能的步骤:

1.  确定代码中最常用的部分，例如循环中的函数。
2.  在运行时将这些部分转换成机器代码。
3.  优化生成的机器码。
4.  用优化的机器码版本替换以前的实现。

还记得教程开头的两个嵌套循环吗？PyPy 检测到相同的操作被反复执行，将其编译成机器码，优化机器码，然后交换实现。这就是为什么你会看到速度有如此大的提高。

### 垃圾收集

每当你创建变量、[函数](https://realpython.com/defining-your-own-python-function/)或任何其他对象时，你的计算机都会给它们分配内存。最终，这些对象中的一些将不再需要。如果你不清理它们，那么你的计算机可能会耗尽内存，使你的程序崩溃。

在 C 和 C++等编程语言中，通常要手动处理这个问题。Python 和 Java 等其他编程语言会自动为您完成这项工作。这被称为**自动垃圾收集**，有几种技术可以实现它。

CPython 使用一种叫做[引用计数](https://realpython.com/cpython-source-code-guide/#reference-counting)的技术。本质上，Python 对象的引用计数在对象被引用时递增，在对象被取消引用时递减。当引用计数为零时，CPython 会自动调用该对象的内存释放函数。这是一种简单有效的技术，但是有一个问题。

当一个大对象树的引用计数变为零时，*所有相关对象都被释放。结果，你有一个潜在的长暂停，在此期间你的程序根本没有进展。*

此外，在一个用例中，引用计数根本不起作用。考虑以下代码:

```py
 1class A(object):
 2    pass
 3
 4a = A()
 5a.some_property = a
 6del a
```

在上面的代码中，您定义了新的类。然后，创建该类的一个实例，并将其分配为自身的一个属性。最后，删除实例。

此时，该实例不再可访问。但是，引用计数不会从内存中删除实例，因为它有对自身的引用，所以引用计数不为零。这个问题叫做**引用周期**，用引用计数是解决不了的。

这就是 CPython 使用另一个叫做[循环垃圾收集器](https://docs.python.org/3/c-api/gcsupport.html)的工具的地方。它从已知的根开始遍历内存中的所有对象，比如`type`对象。然后，它识别所有可到达的对象，并释放不可到达的对象，因为它们已经不存在了。这解决了参考循环问题。但是，当内存中有大量对象时，它会造成更明显的停顿。

另一方面，PyPy 不使用引用计数。相反，它只使用第二种技术，即循环查找器。也就是说，它周期性地从根开始遍历活的对象。这使得 PyPy 比 CPython 有一些优势，因为它不需要进行引用计数，使得花在内存管理上的总时间比 CPython 少。

此外，PyPy 不是像 CPython 那样在一个主要的项目中做所有的事情，而是将工作分成可变数量的部分，并运行每个部分，直到一个也不剩。这种方法在每次小的收集后只增加几毫秒，而不是像 CPython 那样一次性增加数百毫秒。

垃圾收集很复杂，并且有更多的细节超出了本教程的范围。您可以在[文档](https://doc.pypy.org/en/latest/gc_info.html)中找到关于 PyPy 垃圾收集的更多信息。

[*Remove ads*](/account/join/)

## PyPy 的局限性

PyPy 不是灵丹妙药，可能并不总是最适合您的任务的工具。它甚至可能使您的应用程序执行速度比 CPython 慢得多。这就是为什么记住以下限制很重要。

### 它不能很好地与 C 扩展一起工作

PyPy 最适合纯 Python 应用程序。无论何时你使用一个 [C 扩展模块](https://realpython.com/build-python-c-extension-module/)，它的运行速度都比在 CPython 中慢得多。原因是 PyPy 不能优化 C 扩展模块，因为它们不被完全支持。此外，PyPy 必须模拟这部分代码的引用计数，这使得它更慢。

在这种情况下，PyPy 团队建议去掉 CPython 扩展，用一个纯 Python 版本替换它，这样 JIT 就可以看到它并进行优化。如果这不是一个选项，那么您将不得不使用 CPython。

也就是说，核心团队正在开发 C 扩展。有些包已经移植到 PyPy 上，运行速度也一样快。

### 它只适用于长时间运行的程序

假设你想去一家离家很近的商店。你可以步行去，也可以开车去。

你的车显然比你的脚快得多。然而，想想它会要求你做什么:

1.  去你的车库。
2.  发动你的车。
3.  把车预热一下。
4.  开车去商店。
5.  找个停车位。
6.  回来的路上重复这个过程。

开车有很多开销，如果你想去的地方就在附近，那就不值得了！

现在想想如果你想去五十英里外的邻近城市会发生什么。开车去那里而不是步行肯定是值得的。

虽然速度上的差异不像上面的类比那样明显，但是 PyPy 和 CPython 也是如此。

当你用 PyPy 运行一个脚本时，它会做很多事情来使你的代码运行得更快。如果脚本太小，那么开销会导致脚本运行速度比在 CPython 中慢。另一方面，如果您有一个长时间运行的脚本，那么这种开销可以带来显著的性能收益。

要亲自查看，请在 CPython 和 PyPy 中运行以下小脚本:

```py
 1import time
 2
 3start_time = time.time()
 4
 5for i in range(100):
 6    print(i)
 7
 8end_time = time.time()
 9print(f"It took {end_time-start_time:.10f} seconds to compute")
```

当你用 PyPy 运行它时，开始会有一点延迟，而 CPython 会立即运行它。确切地说，在 2015 款 MacBook Pro 上运行它需要`0.0004873276`秒，在 PyPy 上运行它需要`0.0019447803`秒。

### 它不做提前编译

正如您在本教程开始时看到的，PyPy 不是一个完全编译的 Python 实现。它*编译* Python 代码，但它不是 Python 代码的*编译器*。由于 Python 固有的动态性，不可能将 Python 编译成独立的二进制文件并重用它。

PyPy 是一个**运行时解释器**，它比完全解释的语言快，但比完全编译的语言如 c 慢。

[*Remove ads*](/account/join/)

## 结论

PyPy 是 CPython 的快速而强大的替代品。通过用它来运行您的脚本，您可以在不对代码做任何改动的情况下获得很大的速度提升。但这不是银弹。它有一些限制，您需要测试您的程序，看看 PyPy 是否能有所帮助。

**在本教程中，您学习了:**

*   什么是 PyPy
*   如何**安装** PyPy 和**用它运行你的脚本**
*   PyPy 在速度方面与 **CPython** 相比如何
*   PyPy 有什么特性以及它如何提高你程序的速度
*   PyPy 有哪些限制使它不适合某些情况

如果您的 Python 脚本需要一点速度提升，那么试试 PyPy 吧。根据你的程序，你可能会得到一些明显的速度提高！

如果你有任何问题，请在下面的评论区联系我们。****