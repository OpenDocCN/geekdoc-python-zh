# python 102:TDD 和单元测试简介

> 原文：<https://www.blog.pythonlibrary.org/2011/03/09/python-102-an-intro-to-tdd-and-unittest/>

Python 代码测试对我来说是新事物。我工作的地方并不要求这样，所以除了读一本关于这个主题的书和读几个博客之外，我没有花太多时间去研究它。然而，我决定是时候来看看这个，看看到底是怎么回事。在本文中，您将使用 Python 的内置 unittest 模块了解 Python 的测试驱动开发(TDD)。这实际上是基于我的一次 TDD 和结对编程的经验(感谢 Matt 和 Aaron！).在这篇文章中，我们将学习如何用 Python 打保龄球！

## 入门指南

我在网上搜索了如何给保龄球评分。我找到了一本关于 About.com 的有用教程，欢迎你也来读。一旦你知道了规则，就该写一些测试了。如果你不知道，测试驱动开发背后的思想是在你写实际代码之前写测试。在本文中，我们将编写一个测试，然后通过测试的一些代码。我们将在编写测试和代码之间来回迭代，直到完成为止。对于本文，我们将只编写三个测试。我们开始吧！

## 第一次测试

我们的第一个测试将是测试我们的游戏对象，看看如果我们掷骰子 11 次，每次只打翻一个瓶子，它是否能计算出正确的总数。这应该给我们总共 11 个。

```py

import unittest

########################################################################
class TestBowling(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_all_ones(self):
        """Constructor"""
        game = Game()
        game.roll(11, 1)
        self.assertEqual(game.score, 11)

```

这是一个非常简单的测试。我们创建一个游戏对象，然后调用它的 **roll** 方法 11 次，每次得分为 1。然后我们使用来自 **unittest** 模块的 **assertEqual** 方法来测试游戏对象的分数是否正确(即十一)。下一步是编写您能想到的最简单的代码来通过测试。这里有一个例子:

```py

########################################################################
class Game:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.score = 0

    #----------------------------------------------------------------------
    def roll(self, numOfRolls, pins):
        """"""
        for roll in numOfRolls:
            self.score += pins

```

为了简单起见，您可以将它复制并粘贴到与您的测试相同的文件中。为了下一次测试，我们将把它们分成两个文件。反正你也看到了，我们的**游戏**类超级简单。通过测试所需要的只是一个 score 属性和一个可以更新它的 **roll** 方法。如果你不知道循环的**是如何工作的，那么你需要去读一读 [Python 教程](http://docs.python.org/tutorial/)。**

让我们运行测试，看看它是否通过！运行测试最简单的方法是将下面两行代码添加到文件的底部:

```py

if __name__ == '__main__':
    unittest.main()

```

然后，只需通过命令行运行 Python 文件，如果这样做，您应该会得到如下所示的内容:

```py

E
======================================================================
ERROR: test_all_ones (__main__.TestBowling)
Constructor
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\Mike\Documents\My Dropbox\Scripts\Testing\bowling\test_one.py",
 line 27, in test_all_ones
    game.roll(11, 1)
  File "C:\Users\Mike\Documents\My Dropbox\Scripts\Testing\bowling\test_one.py",
 line 15, in roll
    for roll in numOfRolls:
TypeError: 'int' object is not iterable

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)

```

哎呀！我们在某个地方弄错了。看起来我们在传递一个整数，然后试图迭代它。那不行！我们需要将我们的游戏对象的 roll 方法更改为下面的方法来使它工作:

```py

#----------------------------------------------------------------------
def roll(self, numOfRolls, pins):
    """"""
    for roll in range(numOfRolls):
        self.score += pins

```

如果您现在运行测试，您应该得到以下结果:

```py

.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

请注意“.”因为这很重要。那个小点意味着一个测试已经运行并且通过了。结尾的“OK”也暗示了你这个事实。如果您研究原始输出，您会注意到它以“E”开头表示错误，并且没有点！让我们继续测试#2。

## 第二个测试

对于第二个测试，我们将测试当我们得到一个罢工会发生什么。我们需要改变第一个测试，使用一个列表来显示每一帧中被击倒的球瓶数量，所以我们将在这里查看两个测试。您可能会发现这是一个相当常见的过程，由于您测试的内容发生了根本性的变化，您可能需要编辑几个测试。通常这只会发生在你编码的开始阶段，你会在以后变得更好，这样你就不需要这样做了。因为这是我第一次这么做，所以我想得不够多。无论如何，让我们看看代码:

```py

from game import Game
import unittest

########################################################################
class TestBowling(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_all_ones(self):
        """Constructor"""
        game = Game()
        pins = [1 for i in range(11)]
        game.roll(11, pins)
        self.assertEqual(game.score, 11)

    #----------------------------------------------------------------------
    def test_strike(self):
        """
        A strike is 10 + the value of the next two rolls. So in this case
        the first frame will be 10+5+4 or 19 and the second will be
        5+4\. The total score would be 19+9 or 28.
        """
        game = Game()
        game.roll(11, [10, 5, 4])
        self.assertEqual(game.score, 28)

if __name__ == '__main__':
    unittest.main()

```

让我们看看我们的第一个测试，以及它是如何变化的。是的，当谈到 TDD 时，我们在这里打破了一些规则。放心，不要改变第一个测试，看看有什么问题。在 **test_all_ones** 方法中，我们将 **pins** 变量设置为等于 list comprehension，它创建了一个包含 11 个 1 的列表。然后我们把它和掷骰数一起传递给我们的**游戏**对象的**掷骰子**方法。

在第二个测试中，我们在第一轮投了一个好球，第二轮投了五个好球，第三轮投了四个好球。请注意，我们去了头，告诉它我们通过了 11 卷，但我们只通过了 3 卷。这意味着我们需要将其他八个卷设置为零。最后，我们使用可靠的 **assertEqual** 方法来检查我们是否得到了正确的总数。最后，注意我们现在正在导入游戏类，而不是在测试中保留它。现在我们需要实现通过这两个测试所必需的代码。让我们来看看一个可能的解决方案:

```py

########################################################################
class Game:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.score = 0
        self.pins = [0 for i in range(11)]

    #----------------------------------------------------------------------
    def roll(self, numOfRolls, pins):
        """"""
        x = 0
        for pin in pins:
            self.pins[x] = pin
            x += 1
        x = 0
        for roll in range(numOfRolls):
            if self.pins[x] == 10:
                self.score = self.pins[x] + self.pins[x+1] + self.pins[x+2]
            else:
                self.score += self.pins[x]
            x += 1
        print self.score

```

马上，你会注意到我们有了一个新的类属性叫做 **self.pins** ，它保存了默认的管脚列表，有十一个零。然后在我们的 **roll** 方法中，我们在第一个循环中将正确的分数添加到 self.pins 列表中的正确位置。然后在第二个循环中，我们检查被击倒的瓶子是否等于 10。如果是的话，我们把它和接下来的两个分数加起来计分。否则，我们做我们以前做的。在方法的最后，我们打印出分数来检查它是否是我们所期望的。在这一点上，我们已经准备好编码我们的最终测试。

## 第三个(也是最后一个)测试

我们的最后一个测试将测试正确的分数，如果有人掷出一个备用。测试很容易，解决方法稍微难一点。当我们这样做的时候，我们将对测试代码进行一点重构。和往常一样，我们先来看看测试。

```py

from game_v2 import Game
import unittest

########################################################################
class TestBowling(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def setUp(self):
        """"""
        self.game = Game()

    #----------------------------------------------------------------------
    def test_all_ones(self):
        """
        If you don't get a strike or a spare, then you just add up the 
        face value of the frame. In this case, each frame is worth
        one point, so the total is eleven.
        """
        pins = [1 for i in range(11)]
        self.game.roll(11, pins)
        self.assertEqual(self.game.score, 11)

    #----------------------------------------------------------------------
    def test_spare(self):
        """
        A spare is worth 10, plus the value of your next roll. So in this
        case, the first frame will be 5+5+5 or 15 and the second will be
        5+4 or 9\. The total is 15+9, which equals 24,
        """
        self.game.roll(11, [5, 5, 5, 4])
        self.assertEqual(self.game.score, 24)

    #----------------------------------------------------------------------
    def test_strike(self):
        """
        A strike is 10 + the value of the next two rolls. So in this case
        the first frame will be 10+5+4 or 19 and the second will be
        5+4\. The total score would be 19+9 or 28.
        """
        self.game.roll(11, [10, 5, 4])
        self.assertEqual(self.game.score, 28)

if __name__ == '__main__':
    unittest.main()

```

首先，我们添加了一个 **setUp** 方法，它将为我们的每个测试创建一个 self.game 对象。如果我们访问数据库或类似的东西，我们可能会有一个关闭连接或文件或类似的东西的方法。如果存在的话，它们分别在每个测试的开始和结束时运行。 **test_all_ones** 和 **test_strike** 测试除了现在用的是“self.game”之外，基本相同。唯一的新测试是**测试 _ 备用**。docstring 解释了备件如何工作，代码只有两行。是的，你可以解决这个问题。让我们看看通过这些测试所需的代码:

```py

# game_v2.py
########################################################################
class Game:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.score = 0
        self.pins = [0 for i in range(11)]

    #----------------------------------------------------------------------
    def roll(self, numOfRolls, pins):
        """"""
        x = 0
        for pin in pins:
            self.pins[x] = pin
            x += 1
        x = 0
        spare_begin = 0
        spare_end = 2
        for roll in range(numOfRolls):
            spare = sum(self.pins[spare_begin:spare_end])
            if self.pins[x] == 10:
                self.score = self.pins[x] + self.pins[x+1] + self.pins[x+2]
            elif spare == 10:
                self.score = spare + self.pins[x+2]
                x += 1
            else:
                self.score += self.pins[x]
            x += 1
            if x == 11: 
                break
            spare_begin += 2
            spare_end += 2
        print self.score

```

对于谜题的这一部分，我们在循环中添加了条件语句。为了计算备件的值，我们使用 **spare_begin** 和 **spare_end** 列表位置从我们的列表中获得正确的值，然后对它们求和。这就是**备用**变量的用途。这可能更适合放在 elif 中，但是我将留给读者去试验。严格来说，那只是备用分数的前半部分。后半部分是接下来的两卷，这是你在当前代码的 elif 部分的计算中会发现的。代码的其余部分是相同的。

## 其他注释

正如您可能已经猜到的那样，unittest 模块的内容远不止于此。有很多其他的断言可以用来测试结果。您可以跳过测试，从命令行运行测试，使用 TestLoader 创建测试套件等等。当你有机会的时候一定要阅读完整的文档,因为本教程仅仅触及了这个库的表面。

## 包扎

我对测试还不是很了解，但是我从这篇文章中学到了很多，希望没有做得太糟糕。如果有，请告诉我，我会更新帖子。也可以随时让我知道我将来应该涉及的测试的其他方面，我会看一看。除了 mock 和 nose 之外，我还想解决一些我在本教程中没有提到的其他单元测试特性。你有什么建议？