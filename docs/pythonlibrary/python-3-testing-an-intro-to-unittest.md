# Python 3 测试:单元测试简介

> 原文：<https://www.blog.pythonlibrary.org/2016/07/07/python-3-testing-an-intro-to-unittest/>

unittest 模块实际上是一个测试框架，最初是受 JUnit 的启发。它目前支持测试自动化、设置和关闭代码的共享、将测试聚集到集合中以及测试与报告框架的独立性。

单元测试框架支持以下概念:

*   测试夹具——夹具是用来设置测试的，这样它就可以运行，并且在测试结束时也可以拆除。例如，您可能需要在测试运行之前创建一个临时数据库，并在测试完成后销毁它。
*   测试用例——测试用例是你实际的测试。它通常会检查(或断言)特定的响应来自一组特定的输入。unittest 框架提供了一个名为**TestCase**的基类，您可以使用它来创建新的测试用例。
*   测试套件——测试套件是测试用例、测试套件或两者的集合。
*   测试运行者——运行者是控制或协调测试或套件运行的人。它还将向用户提供结果(例如，他们是通过还是失败)。跑步者可以使用图形用户界面，也可以是简单的文本界面。

* * *

### 简单的例子

我总是发现一两个代码示例是学习新事物如何工作的最快方法。所以让我们创建一个小模块，我们称之为 **mymath.py** 。然后将以下代码放入其中:

```py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(numerator, denominator):
    return float(numerator) / denominator

```

这个模块定义了四个数学函数:加、减、乘、除。它们不做任何错误检查，它们实际上并不完全符合您的预期。例如，如果您要用两个字符串调用 **add** 函数，它会很高兴地将它们连接在一起并返回它们。但是为了便于说明，这个模块将创建一个测试用例。所以让我们实际上为 add 函数编写一个测试用例吧！我们将这个脚本命名为 **test_mymath.py** ，并保存在包含 **mymath.py** 的同一个文件夹中。

```py

import mymath
import unittest

class TestAdd(unittest.TestCase):
    """
    Test the add function from the mymath library
    """

    def test_add_integers(self):
        """
        Test that the addition of two integers returns the correct total
        """
        result = mymath.add(1, 2)
        self.assertEqual(result, 3)

    def test_add_floats(self):
        """
        Test that the addition of two floats returns the correct result
        """
        result = mymath.add(10.5, 2)
        self.assertEqual(result, 12.5)

    def test_add_strings(self):
        """
        Test the addition of two strings returns the two string as one
        concatenated string
        """
        result = mymath.add('abc', 'def')
        self.assertEqual(result, 'abcdef')

if __name__ == '__main__':
    unittest.main()

```

让我们花一点时间来看看这段代码是如何工作的。首先，我们导入 mymath 模块和 Python 的 **unittest** 模块。然后我们子类化**测试用例**并添加三个测试，这转化为三个方法。第一个函数测试两个整数的相加；第二个函数测试两个浮点数的相加；最后一个函数将两个字符串连接在一起。最后我们在最后调用 unittest 的 **main** 方法。

您会注意到每个方法都以字母“test”开头。这个其实很重要！它告诉测试运行者哪些方法是它应该运行的测试。每个测试应该至少有一个断言来验证结果是否符合我们的预期。unittest 模块支持许多不同类型的断言。您可以测试异常、布尔条件和许多其他条件。

让我们尝试运行测试。打开终端，导航到包含 mymath 模块和测试模块的文件夹:

```py

python test_mymath.py 

```

这将执行我们的测试，我们应该得到以下输出:

```py

...
----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK

```

你会注意到有三个时期。每个时期代表一个已经通过的测试。然后它告诉我们它运行了 3 次测试，花费的时间和结果:ok。这告诉我们所有的测试都成功通过了。

您可以通过传入 **-v** 标志使输出更加详细:

```py

python test_mymath.py -v

```

这将导致以下输出被打印到 stdout:

```py

test_add_floats (__main__.TestAdd) ... ok
test_add_integers (__main__.TestAdd) ... ok
test_add_strings (__main__.TestAdd) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK

```

正如您所看到的，这向我们准确地显示了运行了哪些测试以及每个测试的结果。这也引导我们进入下一节，我们将学习一些可以在命令行上与 unittest 一起使用的命令。

* * *

### 命令行界面

unittest 模块附带了一些其他命令，您可能会发现这些命令很有用。要找出它们是什么，您可以直接运行 unittest 模块并传递给它 **-h** ，如下所示:

```py

python -m unittest -h

```

这将导致以下输出被打印到 stdout。请注意，为了简洁起见，我已经删除了包含**测试发现**命令行选项的输出的一部分:

```py

usage: python -m unittest [-h] [-v] [-q] [--locals] [-f] [-c] [-b]
                           [tests [tests ...]]

positional arguments:
  tests           a list of any number of test modules, classes and test
                  methods.

optional arguments:
  -h, --help      show this help message and exit
  -v, --verbose   Verbose output
  -q, --quiet     Quiet output
  --locals        Show local variables in tracebacks
  -f, --failfast  Stop on first fail or error
  -c, --catch     Catch ctrl-C and display results so far
  -b, --buffer    Buffer stdout and stderr during tests

Examples:
  python -m unittest test_module               - run tests from test_module
  python -m unittest module.TestClass          - run tests from module.TestClass
  python -m unittest module.Class.test_method  - run specified test method

```

现在我们有了一些想法，如果测试代码的底部没有对 **unittest.main()** 的调用，我们将如何调用测试代码。实际上，继续用不同的名称重新保存代码，比如删除最后两行的 **test_mymath2.py** 。然后运行以下命令:

```py

python -m unittest test_mymath2.py

```

这应该会产生与之前相同的输出:

```py

...
----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK

```

在命令行上使用 unittest 模块最酷的一点是，我们可以在测试中使用它来调用特定的函数。这里有一个例子:

```py

python -m unittest test_mymath2.TestAdd.test_add_integers

```

该命令将只运行运行测试，因此该命令的输出应该如下所示:

```py

.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

或者，如果您在这个测试模块中有多个测试用例，那么您可以一次只调用一个测试用例，就像这样:

```py

python -m unittest test_mymath2.TestAdd

```

这只是调用我们的 **TestAdd** 子类，并运行其中的所有测试方法。因此，结果应该与我们在第一个示例中运行的结果相同:

```py

...
----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK

```

这个练习的要点是，如果你在这个测试模块中有额外的测试用例，那么这个方法给你一个方法来运行单独的测试用例，而不是所有的测试用例。

* * *

### 创建更复杂的测试

大多数代码比我们的例子要复杂得多。因此，让我们创建一段依赖于现有数据库的代码。我们将创建一个简单的脚本，它可以创建带有一些初始数据的数据库(如果它不存在的话),以及一些允许我们查询、删除和更新行的函数。我们将这个脚本命名为 **simple_db.py** 。这是一个相当长的例子，所以请原谅我:

```py

import sqlite3

def create_database():
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    # create a table
    cursor.execute("""CREATE TABLE albums
                          (title text, artist text, release_date text,
                           publisher text, media_type text)
                       """)
    # insert some data
    cursor.execute("INSERT INTO albums VALUES "
                   "('Glow', 'Andy Hunter', '7/24/2012',"
                   "'Xplore Records', 'MP3')")

    # save data to database
    conn.commit()

    # insert multiple records using the more secure "?" method
    albums = [('Exodus', 'Andy Hunter', '7/9/2002',
               'Sparrow Records', 'CD'),
              ('Until We Have Faces', 'Red', '2/1/2011',
               'Essential Records', 'CD'),
              ('The End is Where We Begin', 'Thousand Foot Krutch',
               '4/17/2012', 'TFKmusic', 'CD'),
              ('The Good Life', 'Trip Lee', '4/10/2012',
               'Reach Records', 'CD')]
    cursor.executemany("INSERT INTO albums VALUES (?,?,?,?,?)",
                       albums)
    conn.commit()

def delete_artist(artist):
    """
    Delete an artist from the database
    """
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    sql = """
    DELETE FROM albums
    WHERE artist = ?
    """
    cursor.execute(sql, [(artist)])
    conn.commit()
    cursor.close()
    conn.close()

def update_artist(artist, new_name):
    """
    Update the artist name
    """
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    sql = """
    UPDATE albums
    SET artist = ?
    WHERE artist = ?
    """
    cursor.execute(sql, (new_name, artist))
    conn.commit()
    cursor.close()
    conn.close()

def select_all_albums(artist):
    """
    Query the database for all the albums by a particular artist
    """
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    sql = "SELECT * FROM albums WHERE artist=?"
    cursor.execute(sql, [(artist)])
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

if __name__ == '__main__':
    import os
    if not os.path.exists("mydatabase.db"):
        create_database()

    delete_artist('Andy Hunter')
    update_artist('Red', 'Redder')
    print(select_all_albums('Thousand Foot Krutch'))

```

您可以稍微试验一下这段代码，看看它是如何工作的。一旦你适应了，我们就可以继续测试了。

现在有些人可能会说，为每个测试创建一个数据库并销毁它是相当大的开销。他们可能说得有道理。然而，为了测试某些功能，您有时需要做这类事情。此外，您通常不需要仅仅为了健全性检查而创建整个生产数据库。

无论如何，这再次是为了说明的目的。单元测试模块允许我们为这些类型的事情覆盖**设置**和**拆卸**方法。因此，我们将创建一个创建数据库的 setUp 方法和一个在测试结束时删除数据库的 tearDown 方法。请注意，每次测试都会进行安装和拆卸。这可以防止测试以导致后续测试失败的方式更改数据库。

让我们来看看测试用例类的第一部分:

```py

import os
import simple_db
import sqlite3
import unittest

class TestMusicDatabase(unittest.TestCase):
    """
    Test the music database
    """

    def setUp(self):
        """
        Setup a temporary database
        """
        conn = sqlite3.connect("mydatabase.db")
        cursor = conn.cursor()

        # create a table
        cursor.execute("""CREATE TABLE albums
                          (title text, artist text, release_date text,
                           publisher text, media_type text)
                       """)
        # insert some data
        cursor.execute("INSERT INTO albums VALUES "
                       "('Glow', 'Andy Hunter', '7/24/2012',"
                       "'Xplore Records', 'MP3')")

        # save data to database
        conn.commit()

        # insert multiple records using the more secure "?" method
        albums = [('Exodus', 'Andy Hunter', '7/9/2002',
                   'Sparrow Records', 'CD'),
                  ('Until We Have Faces', 'Red', '2/1/2011',
                   'Essential Records', 'CD'),
                  ('The End is Where We Begin', 'Thousand Foot Krutch',
                   '4/17/2012', 'TFKmusic', 'CD'),
                  ('The Good Life', 'Trip Lee', '4/10/2012',
                   'Reach Records', 'CD')]
        cursor.executemany("INSERT INTO albums VALUES (?,?,?,?,?)",
                           albums)
        conn.commit()

    def tearDown(self):
        """
        Delete the database
        """
        os.remove("mydatabase.db")

```

**setUp** 方法将创建我们的数据库，然后用一些数据填充它。**拆卸**方法将删除我们的数据库文件。如果您使用类似 MySQL 或 Microsoft SQL Server 的数据库，那么您可能会删除该表，但使用 sqlite，我们可以删除整个表。

现在让我们在代码中添加一些实际的测试。您可以将这些添加到上面的测试类的末尾:

```py

def test_updating_artist(self):
    """
    Tests that we can successfully update an artist's name
    """
    simple_db.update_artist('Red', 'Redder')
    actual = simple_db.select_all_albums('Redder')
    expected = [('Until We Have Faces', 'Redder',
                 '2/1/2011', 'Essential Records', 'CD')]
    self.assertListEqual(expected, actual)

def test_artist_does_not_exist(self):
    """
    Test that an artist does not exist
    """
    result = simple_db.select_all_albums('Redder')
    self.assertFalse(result)

```

第一个测试将把其中一个艺术家的名字更新为字符串 **Redder** 。然后，我们进行查询，以确保新的艺术家姓名存在。下一个测试还将检查被称为“更红”的艺术家是否存在。这一次不应该，因为数据库在两次测试之间被删除并重新创建。让我们试着运行它，看看会发生什么:

```py

python -m unittest test_db.py 

```

上面的命令应该会产生下面的输出，尽管您的运行时可能会有所不同:

```py

..
----------------------------------------------------------------------
Ran 2 tests in 0.032s

OK

```

很酷，是吧？现在我们可以继续学习测试套件了！

* * *

### 创建测试套件

正如在开始提到的，一个测试套件仅仅是测试用例、测试套件或者两者的集合。大多数时候，当你调用 **unittest.main()** 时，它会做正确的事情，在执行之前为你收集所有模块的测试用例。但有时你会想成为掌控一切的人。在这种情况下，您可以使用**测试套件**类。这里有一个你可能如何使用它的例子:

```py

import unittest

from test_mymath import TestAdd

def my_suite():
    suite = unittest.TestSuite()
    result = unittest.TestResult()
    suite.addTest(unittest.makeSuite(TestAdd))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

my_suite()

```

创建自己的套件是一个有点复杂的过程。首先你需要创建一个**测试套件**的实例和一个**测试结果**的实例。TestResult 类只保存测试的结果。接下来，我们在 suite 对象上调用 **addTest** 。这就是有点奇怪的地方。如果您只是传入 **TestAdd** ，那么它必须是 TestAdd 的一个实例，并且 TestAdd 还必须实现一个 **runTest** 方法。因为我们没有这样做，所以我们使用 unittest 的 **makeSuite** 函数将我们的 TestCase 类转换成一个套件。

最后一步是实际运行套件，这意味着如果我们想要良好的输出，我们需要一个 runner。因此，我们创建了一个 **TextTestRunner** 的实例，并让它运行我们的套件。如果您这样做，并打印出它返回的内容，您应该会在屏幕上看到类似这样的内容:

```py

...
----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK

```

另一种方法是调用 **suite.run(result)** 并打印出结果。然而，您得到的只是一个 TestResult 对象，看起来与上面最后一行输出非常相似。如果你想要更平常的输出，那么你将会想要使用一个转轮。

* * *

### 如何跳过测试

从 Python 3.1 开始，unittest 模块支持跳过测试。有一些跳过测试的用例:

*   如果库的版本不支持您想要测试的内容，您可能想要跳过测试
*   该测试依赖于运行它的操作系统
*   或者你有一些跳过测试的其他标准

让我们改变我们的测试用例，这样就有几个测试将被跳过:

```py

import mymath
import sys
import unittest

class TestAdd(unittest.TestCase):
    """
    Test the add function from the mymath module
    """

    def test_add_integers(self):
        """
        Test that the addition of two integers returns the correct total
        """
        result = mymath.add(1, 2)
        self.assertEqual(result, 3)

    def test_add_floats(self):
        """
        Test that the addition of two floats returns the correct result
        """
        result = mymath.add(10.5, 2)
        self.assertEqual(result, 12.5)

    @unittest.skip('Skip this test')
    def test_add_strings(self):
        """
        Test the addition of two strings returns the two string as one
        concatenated string
        """
        result = mymath.add('abc', 'def')
        self.assertEqual(result, 'abcdef')

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_adding_on_windows(self):
        result = mymath.add(1, 2)
        self.assertEqual(result, 3)

```

这里我们演示两种不同的跳过测试的方法: **skip** 和 **skipUnless** 。你会注意到我们正在修饰需要跳过的功能。**跳过**装饰器可以用于以任何理由跳过任何测试。除非条件返回真，否则 **skipUnless** decorator 将跳过一个测试。因此，如果您在 Mac 或 Linux 上运行这个测试，它将被跳过。还有一个 **skipIf** decorator，如果条件为真，它将跳过一个测试。

您可以使用 verbose 标志运行这个脚本，看看它为什么跳过测试:

```py

python -m unittest test_mymath.py -v

```

该命令将产生以下输出:

```py

test_add_floats (test_mymath4.TestAdd) ... ok
test_add_integers (test_mymath4.TestAdd) ... ok
test_add_strings (test_mymath4.TestAdd) ... skipped 'Skip this test'
test_adding_on_windows (test_mymath4.TestAdd) ... skipped 'requires Windows'

----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK (skipped=2)

```

他的输出告诉我们，我们试图运行四个测试，但是跳过了其中的两个。

还有一个**expected failure**decorator，您可以将它添加到您知道会失败的测试中。我会把那个留给你自己去尝试。

* * *

### 与 doctest 集成

unittest 模块也可以与 Python 的 doctest 模块一起使用。如果您已经创建了许多包含文档测试的模块，那么您通常会希望能够系统地运行它们。这就是 unittest 的用武之地。从 Python 3.2 开始，unittest 模块支持**测试发现**。测试发现基本上允许 unittest 查看目录的内容，并根据文件名确定哪些目录可能包含测试。然后它通过导入它们来加载测试。

让我们创建一个新的空文件夹，并在其中创建一个名为 **my_docs.py** 的文件。它需要有以下代码:

```py

def add(a, b):
    """
    Return the addition of the arguments: a + b

    >>> add(1, 2)
    3
    >>> add(-1, 10)
    9
    >>> add('a', 'b')
    'ab'
    >>> add(1, '2')
    Traceback (most recent call last):
      File "test.py", line 17, in add(1, '2')
      File "test.py", line 14, in add
        return a + b
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    return a + b

def subtract(a, b):
    """
    Returns the result of subtracting b from a

    >>> subtract(2, 1)
    1
    >>> subtract(10, 10)
    0
    >>> subtract(7, 10)
    -3
    """
    return a - b 
```

现在我们需要在与这个模块相同的位置创建另一个模块，它将把我们的文档测试转换成单元测试。让我们称这个文件为 **test_doctests.py.** 将下面的代码放入其中:

```py

import doctest
import my_docs
import unittest

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(my_docs))
    return tests

```

根据 doctest 模块的文档，测试发现在这里实际上需要函数名。我们在这里所做的是向 **tests** 对象添加一个套件，方式与我们之前所做的非常相似。在本例中，我们使用的是 doctest 的 **DocTestSuite** 类。如果您的测试需要，您可以给这个类一个 setUp 和 tearDown 方法作为参数。要运行此代码，您需要在新文件夹中执行以下命令:

```py

python -m unittest discover

```

在我的机器上，我收到了以下输出:

```py

..
----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK

```

您会注意到，当您使用 unittest 运行 doctest 时，每个 docstring 都被视为一个单独的测试。如果您直接用 doctest 运行 docstrings，那么您会注意到 doctest 会说还有更多测试。除此之外，它的工作和预期的差不多。

* * *

### 包扎

我们在这篇文章中讨论了很多。您学习了使用 unittest 模块的基础知识。然后我们继续学习如何从命令行使用 unittest。我们还发现了如何建立和拆除测试。您还发现了如何创建测试套件。最后，我们学习了如何将一系列文档测试转化为单元测试。请务必花些时间阅读这两个优秀库的文档，因为还有很多额外的功能没有在这里介绍。

* * *

### 相关阅读

*   Python 3 文档:[单元测试](https://docs.python.org/3/library/unittest.html)
*   本周 Python 模块:[单元测试](https://pymotw.com/2/unittest/)
*   python 102:[TDD 和单元测试简介](https://www.blog.pythonlibrary.org/2011/03/09/python-102-an-intro-to-tdd-and-unittest/)
*   使用 [doctest](https://www.blog.pythonlibrary.org/2014/03/17/python-testing-with-doctest/) 进行 Python 测试