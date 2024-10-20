# Python 201:生成器简介

> 原文：<https://www.blog.pythonlibrary.org/2014/01/27/python-201-an-intro-to-generators/>

关于发电机的话题之前已经讨论过无数次了。然而，这仍然是一个许多新程序员都有困难的话题，我大胆猜测，即使是有经验的用户也不会真正使用它们。

Python 生成器允许开发人员懒散地评估数据。这在你处理所谓的“大数据”时非常有帮助。它们的主要用途是产生价值，并以有效的方式产生价值。在本文中，我们将讨论如何使用生成器，并看看生成器表达式。希望到最后你能在自己的项目中自如地使用生成器。

生成器的典型用例是展示如何以一系列块或行的形式读取一个大文件。这个想法没有错，所以我们也把它用在第一个例子中。要创建一个生成器，我们需要做的就是使用 Python 的 **yield** 关键字。yield 语句将把一个函数变成一个迭代器。要将一个常规函数变成迭代器，你所要做的就是用一个**产生**语句替换**返回**语句。让我们来看一个例子:

```py

#----------------------------------------------------------------------
def read_large_file(file_object):
    """
    Uses a generator to read a large file lazily
    """
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

#----------------------------------------------------------------------
def process_file(path):
    """"""
    try:
        with open(path) as file_handler:
            for line in read_large_file(file_handler):
                # process line
                print(line)
    except (IOError, OSError):
        print("Error opening / processing file")

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "TB_burden_countries_2014-01-23.csv"
    process_file(path)

```

为了让测试更容易，我去了世界卫生组织(世卫组织)的网站，下载了一个关于结核病的 CSV 文件。具体来说，我从[这里](http://www.who.int/tb/country/data/download/en/)抓取了“世卫组织结核病负担估计[csv 890kb]”文件。如果您已经有了一个大文件，请随意适当地编辑代码。总之，在这段代码中，我们创建了一个名为 **read_large_file** 的函数，通过让它返回数据，将它变成一个生成器。

魔术是这样工作的:我们为循环创建一个**，它在我们的生成器函数上循环。对于每次迭代，generator 函数将产生一个包含一行数据的 generator 对象，For 循环将处理它。在这种情况下,“过程”只是将这一行打印到 stdout，但是您可以根据需要进行修改。在真实的程序中，您可能会将数据保存到数据库中，或者用数据创建 PDF 或其他报告。当生成器返回时，它会挂起函数的执行状态，以便保留局部变量。这允许我们在不丢失位置的情况下继续下一个循环。**

无论如何，当生成器函数用完数据时，我们会中断它，这样循环就不会无限地继续下去。生成器允许我们一次只处理一个数据块，这样可以节省大量内存。

**更新 2014/01/28** :我的一位读者指出，文件首先返回惰性迭代器，这是我认为他们做的事情。奇怪的是，每个人和他们的狗都推荐使用生成器来读取文件，但是仅仅迭代文件就足够了。因此，让我们重写上面的例子来利用这个概念:

```py

#----------------------------------------------------------------------
def process_file_differently(path):
    """
    Process the file line by line using the file's returned iterator
    """
    try:
        with open(path) as file_handler:
            while True:
                print next(file_handler)
    except (IOError, OSError):
        print("Error opening / processing file")
    except StopIteration:
        pass

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "TB_burden_countries_2014-01-23.csv"
    process_file_differently(path)

```

在这段代码中，我们创建了一个无限循环，它将在文件处理程序对象上调用 Python 的 **next** 函数。这将导致 Python 逐行返回文件以供使用。当文件用完数据时，会引发 StopIteration 异常，所以我们要确保捕捉到它并忽略它。

### 生成器表达式

Python 有生成器表达式的概念。生成器表达式的语法非常类似于列表理解。让我们来看看两者的区别:

```py

# list comprehension
lst = [ord(i) for i in "ABCDEFGHI"]

# equivalent generator expression
gen = list(ord(i) for i in "ABCDEFGHI")

```

这个例子是基于 Python 的 [HOWTO 章节](https://wiki.python.org/moin/Generators)中关于生成器的一个例子，坦率地说，我觉得它有点迟钝。生成器表达式和列表理解之间的主要区别在于包含表达式的内容。对于列表理解，是方括号；对于生成器表达式，它是常规括号。让我们创建生成器表达式本身，而不把它变成列表:

```py

gen = (ord(i) for i in "ABCDEFGHI")
while True:
    print gen.next()

```

如果您运行这段代码，您将看到它打印出字符串中每个成员的每个序数值，然后您将看到一个回溯，表明 StopIteration 已经发生。这意味着发电机本身已经耗尽(即它是空的)。到目前为止，我还没有在自己的工作中发现 generator 表达式的用途，但是我很想知道您用它来做什么。

### 包扎

现在你知道了发电机的用途和它最普遍的用途之一。您还了解了生成器表达式及其工作原理。我个人曾使用一个生成器来解析那些应该成为“大数据”的数据文件。你用这些做什么？

*   Python [关于生成器的文档](http://docs.python.org/2/tutorial/classes.html#generators)
*   Python [关于生成器的 wiki 条目](https://wiki.python.org/moin/Generators)
*   Python 如何进行函数编程- [生成器部分](http://docs.python.org/2/howto/functional.html#generators)
*   改进你的 Python:[‘yield’和 Generators 讲解](http://www.jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)
*   [Python 生成器函数可以用来做什么？](https://stackoverflow.com/questions/102535/what-can-you-use-python-generator-functions-for)
*   [zetcode 的生成器页面](http://zetcode.com/lang/python/itergener/)
*   Python 的历史- [从列表理解到生成器表达式](http://python-history.blogspot.com/2010/06/from-list-comprehensions-to-generator.html)