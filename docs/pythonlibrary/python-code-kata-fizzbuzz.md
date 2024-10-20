# Python 代码 Kata: Fizzbuzz

> 原文：<https://www.blog.pythonlibrary.org/2019/09/18/python-code-kata-fizzbuzz/>

代码形是计算机程序员练习编码的一种有趣方式。他们也经常被用来学习如何在编写代码时实现测试驱动开发(TDD)。其中一个流行的编程招式叫做 [FizzBuzz](https://en.wikipedia.org/wiki/Fizz_buzz) 。这也是计算机程序员普遍的面试问题。

FizzBuzz 背后的概念如下:

*   写一个程序打印数字 1-100，每一行一个
*   对于每个是 3 的倍数的数字，打印“Fizz ”,而不是数字
*   对于每个是 5 的倍数的数字，打印“Buzz”而不是数字
*   对于每个都是 3 和 5 的倍数的数字，打印“FizzBuzz”而不是数字

现在你知道你需要写什么了，你可以开始了！

* * *

### 创建工作空间

第一步是在您的机器上创建一个工作空间或项目文件夹。例如，你可以创建一个 **katas** 文件夹，里面有一个 **fizzbuzz** 。

下一步是安装源代码控制程序。最流行的一个是 Git，但是您也可以使用像 Mercurial 这样的东西。出于本教程的目的，您将使用 Git。可以从 [Git 网站](https://git-scm.com/)获取。

如果你是 Windows 用户，现在打开一个终端或者运行 cmd.exe。然后在终端中导航到您的 **fizzbuzz** 文件夹。你可以使用 **cd** 命令来实现。进入文件夹后，运行以下命令:

 `git init` 

这将把 **fizzbuzz** 文件夹初始化为 Git 存储库。你在 **fizzbuzz** 文件夹中添加的任何文件或文件夹现在都可以添加到 Git 并进行版本控制。

* * *

### 嘶嘶测试

为了简单起见，您可以在 **fizzbuzz** 文件夹中创建您的测试文件。许多人会将他们的测试保存在名为**测试**或**测试**的子文件夹中，并告诉他们的测试运行人员将顶层文件夹添加到 **sys.path** 中，以便测试可以导入它。

**注意:**如果你需要温习如何使用 Python 的 unittest 库，那么你可能会发现 [Python 3 测试:unittest 简介](https://www.blog.pythonlibrary.org/2016/07/07/python-3-testing-an-intro-to-unittest/)很有帮助。

继续在你的 **fizzbuzz** 文件夹中创建一个名为 **test_fizzbuzz.py** 的文件。

现在，在 Python 文件中输入以下内容:

```py

import fizzbuzz
import unittest

class TestFizzBuzz(unittest.TestCase):

    def test_multiple_of_three(self):
       self.assertEqual(fizzbuzz.process(6), 'Fizz')

if __name__ == '__main__':
    unittest.main()

```

Python 附带了内置的 [unittest](https://docs.python.org/3/library/unittest.html) 库。要使用它，你需要做的就是导入它并子类化 **unittest。测试用例**。然后，您可以创建一系列函数来表示您想要运行的测试。

请注意，您还导入了 **fizzbuzz** 模块。您还没有创建那个模块，所以当您运行这个测试代码时，您将收到一个 **ModuleNotFoundError** 。除了导入之外，您甚至不需要添加任何代码就可以创建这个文件，并且测试会失败。但是为了完整起见，您继续断言 **fizzbuzz.process(6)** 返回正确的字符串。

修复方法是创建一个空的 **fizzbuzz.py** 文件。这只会修复 **ModuleNotFoundError** ，但是它将允许您运行测试并查看其输出。

您可以通过以下方式运行测试:

 `python test_fizzbuzz.py` 

输出将如下所示:

 `ERROR: test_multiple_of_three (__main__.TestFizzBuzz)
----------------------------------------------------------------------
Traceback (most recent call last):
File "/Users/michael/Dropbox/code/fizzbuzz/test_fizzbuzz.py", line 7, in test_multiple_of_three
self.assertEqual(fizzbuzz.process(6), 'Fizz')
AttributeError: module 'fizzbuzz' has no attribute 'process'`

-
在 0.001 秒内完成一项测试

失败(错误=1)

所以这告诉你你的 **fizzbuzz** 模块缺少一个叫做**进程**的属性。

您可以通过在您的 **fizzbuzz.py** 文件中添加一个 **process()** 函数来解决这个问题:

```py

def process(number):
    if number % 3 == 0:
        return 'Fizz'

```

该函数接受一个数字，并使用模数运算符将该数字除以 3，并检查是否有余数。如果没有余数，那么你知道这个数可以被 3 整除，所以你可以返回字符串“Fizz”。

现在，当您运行测试时，输出应该如下所示:

 `.
----------------------------------------------------------------------
Ran 1 test in 0.000s`

好的

上面第一行中的句点表示您运行了一个测试，它通过了。

让我们快速回到这里。当测试失败时，它被认为处于“红色”状态。当测试通过时，这是一个“绿色”状态。这指的是红色/绿色/重构的测试驱动开发(TDD)咒语。大多数开发人员会通过创建一个失败的测试(红色)来开始一个新项目。然后他们会编写代码使测试通过，通常是以最简单的方式(绿色)。

当您的测试为绿色时，这是提交您的测试和代码变更的好时机。这允许您拥有一段可以回滚的工作代码。现在，您可以编写一个新的测试或重构代码来使它变得更好，而不用担心您会丢失您的工作，因为现在您有一个简单的方法来回滚到代码的前一个版本。

要提交代码，可以执行以下操作:

 `git add fizzbuzz.py test_fizzbuzz.py
git commit -m "First commit"` 

第一个命令将添加两个新文件。不需要提交 ***。pyc** 文件，只是 Python 文件。有一个方便的文件叫做**。gitignore** 您可以添加到您的 Git 存储库中，您可以使用它来排除某些文件类型或文件夹，如* . pyc。Github 有一些默认的 gitignore 文件，用于各种语言，如果您想查看示例，您可以获得这些文件。

第二个命令是如何将代码提交到本地存储库。“-m”表示消息，后面是关于您正在提交的更改的描述性消息。如果你也想把你的修改保存到 Github(这对于备份来说很好)，你应该看看这篇文章。

现在我们准备编写另一个测试了！

* * *

### 嗡嗡声测试

你可以写的第二个测试可以是 5 的倍数。要添加一个新的测试，您可以在 **TestFizzBuzz** 类中创建另一个方法:

```py

import fizzbuzz
import unittest

class TestFizzBuzz(unittest.TestCase):

    def test_multiple_of_three(self):
        self.assertEqual(fizzbuzz.process(6), 'Fizz')

    def test_multiple_of_five(self):
        self.assertEqual(fizzbuzz.process(20), 'Buzz')

if __name__ == '__main__':
    unittest.main()

```

这一次，您希望使用只能被 5 整除的数字。当您调用 **fizzbuzz.process()** 时，应该会得到“buzz”的返回。但是，当您运行测试时，您将收到以下内容:

 `F.
======================================================================
FAIL: test_multiple_of_five (__main__.TestFizzBuzz)
----------------------------------------------------------------------
Traceback (most recent call last):
File "test_fizzbuzz.py", line 10, in test_multiple_of_five
self.assertEqual(fizzbuzz.process(20), 'Buzz')
AssertionError: None != 'Buzz'`

-
在 0.000 秒内运行 2 次测试

失败(失败次数=1)

哎呀！现在，您的代码使用模数运算符来检查除以 3 后的余数。如果数字 20 有余数，这个语句就不会运行。函数的默认返回值是 **None** ，所以这就是为什么你会得到上面的失败。

继续将 **process()** 函数更新如下:

```py

def process(number):
    if number % 3 == 0:
        return 'Fizz'
    elif number % 5 == 0:
        return 'Buzz'

```

现在你可以用 3 和 5 来检查余数。当您这次运行测试时，输出应该如下所示:

 `..
----------------------------------------------------------------------
Ran 2 tests in 0.000s`

好的

耶！您的测试通过，现在是绿色的！这意味着您可以将这些更改提交到您的 Git 存储库中。

现在你已经准备好为 FizzBuzz 添加一个测试了！

* * *

### 嘶嘶声测试

你能写的下一个测试将是当你想要得到“FizzBuzz”的时候。你可能还记得，只要这个数能被 3 和 5 整除，你就会得到 FizzBuzz。继续添加第三个测试:

```py

import fizzbuzz
import unittest

class TestFizzBuzz(unittest.TestCase):

    def test_multiple_of_three(self):
        self.assertEqual(fizzbuzz.process(6), 'Fizz')

    def test_multiple_of_five(self):
        self.assertEqual(fizzbuzz.process(20), 'Buzz')

    def test_fizzbuzz(self):
        self.assertEqual(fizzbuzz.process(15), 'FizzBuzz')

if __name__ == '__main__':
    unittest.main()

```

对于这个测试， **test_fizzbuzz** ，您要求您的程序处理数字 15。这应该还不能正常工作，但是继续运行测试代码来检查:

 `F..
======================================================================
FAIL: test_fizzbuzz (__main__.TestFizzBuzz)
----------------------------------------------------------------------
Traceback (most recent call last):
File "test_fizzbuzz.py", line 13, in test_fizzbuzz
self.assertEqual(fizzbuzz.process(15), 'FizzBuzz')
AssertionError: 'Fizz' != 'FizzBuzz'`

-
在 0.000 秒内运行了 3 次测试

失败(失败次数=1)

进行了三次测试，只有一次失败。你现在回到红色。这次的错误是**‘嘶嘶’！= 'FizzBuzz'** 而不是拿 None 和 FizzBuzz 比较。原因是你的代码检查 15 是否能被 3 整除，所以它返回“Fizz”。

因为这不是您想要发生的，所以您将需要更新您的代码，在检查 3:

```py

def process(number):
    if number % 3 == 0 and number % 5 == 0:
        return 'FizzBuzz'
    elif number % 3 == 0:
        return 'Fizz'
    elif number % 5 == 0:
        return 'Buzz'

```

在这里，首先对 3 和 5 进行整除检查。然后像以前一样检查另外两个。

现在，如果您运行您的测试，您应该得到以下输出:

 `...
----------------------------------------------------------------------
Ran 3 tests in 0.000s`

好的

到目前为止一切顺利。然而，你没有返回不能被 3 或 5 整除的数字的代码。该进行另一项测试了！

* * *

### 最终测试

您的代码需要做的最后一件事是，当该数除以 3 和 5 后还有余数时，返回该数。让我们用几种不同的方法来测试它:

```py

import fizzbuzz
import unittest

class TestFizzBuzz(unittest.TestCase):

    def test_multiple_of_three(self):
        self.assertEqual(fizzbuzz.process(6), 'Fizz')

    def test_multiple_of_five(self):
        self.assertEqual(fizzbuzz.process(20), 'Buzz')

    def test_fizzbuzz(self):
        self.assertEqual(fizzbuzz.process(15), 'FizzBuzz')

    def test_regular_numbers(self):
        self.assertEqual(fizzbuzz.process(2), 2)
        self.assertEqual(fizzbuzz.process(98), 98)

if __name__ == '__main__':
    unittest.main()

```

对于这个测试，您将使用 **test_regular_numbers()** 测试来测试正常数字 2 和 98。这些数字在被 3 或 5 除时总会有余数，所以它们应该被返回。

当您现在运行测试时，您应该得到类似这样的结果:

 `...F
======================================================================
FAIL: test_regular_numbers (__main__.TestFizzBuzz)
----------------------------------------------------------------------
Traceback (most recent call last):
File "test_fizzbuzz.py", line 16, in test_regular_numbers
self.assertEqual(fizzbuzz.process(2), 2)
AssertionError: None != 2`

-
在 0.000 秒内运行了 4 次测试

失败(失败次数=1)

这一次，您又回到了将 None 与 number 进行比较，这就是您可能怀疑的输出。

继续更新**进程()**函数，如下所示:

```py

def process(number):
    if number % 3 == 0 and number % 5 == 0:
        return 'FizzBuzz'
    elif number % 3 == 0:
        return 'Fizz'
    elif number % 5 == 0:
        return 'Buzz'
    else:
        return number

```

那很容易！此时您需要做的就是添加一个返回数字的 **else** 语句。

现在，当您运行测试时，它们应该都通过了:

 `....
----------------------------------------------------------------------
Ran 4 tests in 0.000s`

好的

干得好！现在你的代码工作了。您可以通过将以下内容添加到您的 **fizzbuzz.py** 模块来验证它是否适用于 1-100 的所有数字:

```py

if __name__ == '__main__':
    for i in range(1, 101):
        print(process(i))

```

现在，当您自己使用 **python fizzbuzz.py** 运行 fizzbuzz 时，您应该会看到本教程开头指定的适当输出。

这是提交您的代码并将其推送到云的好时机。

* * *

### 包扎

现在你知道了使用测试驱动开发来驱动你解决编码问题的基础。Python 的 **unittest** 模块拥有比这篇简短教程中所涵盖的更多类型的断言和功能。您还可以修改本教程以使用 pytest，这是另一个流行的第三方 Python 包，您可以用它来代替 Python 自己的 unittest 模块。

进行这些测试的好处是，现在您可以重构代码，并通过运行测试来验证您没有破坏任何东西。这也允许您在不破坏现有功能的情况下更容易地添加新功能。只要确保在添加更多特性时添加更多测试。

* * *

### 相关阅读

*   Python 的[单元测试模块文档](https://docs.python.org/3/library/unittest.html)
*   pytest [网站](https://docs.pytest.org/en/latest/)
*   Python [使用 doctest](https://www.blog.pythonlibrary.org/2014/03/17/python-testing-with-doctest/) 进行测试
*   Python 3 测试:[单元测试简介](https://www.blog.pythonlibrary.org/2016/07/07/python-3-testing-an-intro-to-unittest/)
*   python 102:[TDD 和单元测试简介](https://www.blog.pythonlibrary.org/2011/03/09/python-102-an-intro-to-tdd-and-unittest/)