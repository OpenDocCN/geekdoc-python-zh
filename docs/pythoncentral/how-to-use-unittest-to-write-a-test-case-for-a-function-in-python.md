# 如何使用 Unittest 为 Python 中的函数编写测试用例

> 原文：<https://www.pythoncentral.io/how-to-use-unittest-to-write-a-test-case-for-a-function-in-python/>

从 Python 2.1 开始，unittest 模块就是 Python 标准库的一部分。顾名思义，它帮助开发人员为他们的代码编写和运行测试。

单元测试包括通过测试最小的可测试代码片段来检查代码的缺陷。这些微小的代码块被称为“单元”以这种方式测试代码有助于验证程序的每一部分——包括用户可能没有意识到的助手函数——都做了它应该做的事情。

除了帮助发现 bug，单元测试还有助于防止代码随着时间的推移而退化。

由于这些原因，TDD 驱动的开发团队发现模块是必不可少的，并且这些团队确保所有的代码都有一些测试。

请记住，单元测试不同于集成和回归测试，在集成和回归测试中，测试的目的是检查程序的不同部分是否能一起工作并产生预期的结果。

简单地说，编写单元测试是防止代码失败的好方法。

在本指南中，我们将带领您使用 unittest 模块编写测试。您需要对 Python 函数有一个基本的了解，才能有效地从本指南中学习。

## **单元测试的主要思想**

回想高中:

数学问题涉及使用一套不同的算术程序来得出不同的解决方案。然后将这些解决方案结合起来，得到正确的最终答案。

我们中的许多人会再次通读解决方案，以确保每一步都做对了。更重要的是，检查我们没有写错任何东西或犯错误，并且每一步结束时的计算都是正确的。

对我们的步骤和计算的检查捕捉到了我们在单元测试中所做的事情。在编码过程中犯错误是写出好代码的一部分——就像在数学中得到正确答案的一部分一样。

但是回到旧代码并疯狂地检查它来验证它的正确性并不是正确的方法。这可能会非常令人沮丧。

假设你写了一小段简单的代码来确定一个矩形的面积。为了检查代码是否正确工作，您可以传递各种数字，并查看计算是否正确。

单元测试会为你带来同样的检查过程。

单元测试是回归测试的一个标准且关键的组成部分，有助于确保代码在发生变化时能按预期执行。此外，测试有助于确保程序的稳定性。

因此，当对程序进行更改时，您可以运行预先编写的相应单元测试，以检查它现有的功能不会影响代码库的其他部分。

单元测试的主要好处之一是它有助于隔离错误。如果您要运行整个项目的代码并收到几个错误，调试代码将会非常乏味。

单元测试的输出使得确定代码的任何部分是否抛出错误变得简单。调试过程可以从那里开始。

这并不是说单元测试总是有助于发现错误。但有一点是肯定的。在您检查集成测试中的集成组件之前，这些测试为寻找 bug 提供了一个方便的起点。

现在你已经完全理解了使用单元测试背后的动机，让我们来探索如何实现它们，并使它们成为你开发管道的一部分。

## **Python 中单元测试的基础知识**

也称为 PyUnit，单元测试模块基于 Erich Gamma 和 Kent Beck 创建的 XUnit 框架设计。

你会在其他几种编程语言中找到类似的单元测试模块，包括 Java 和 c

正如你可能知道的，单元测试既可以手动完成，也可以在工具支持下自动完成。自动化测试更快，更可靠，并且减少了测试的人力资源成本，这也是为什么大多数开发团队更喜欢单元测试的原因。

unittest 模块的框架支持测试套件、夹具和测试运行程序，从而实现自动化测试。

### **单元测试的结构**

在单元测试中，一个测试有两个部分。一部分涉及管理测试“夹具”的代码，第二部分是测试本身。

通过子类化 TestCase 并添加一个适当的方法或覆盖来编写一个测试。这里有一个例子:

```py
import unittest

class SimpleTest(unittest.TestCase):

    def test(self):

        self.failUnless(True)

if __name__ == '__main__':
    unittest.main()
```

上面的 SimpleTest 类有一个 Test()方法，如果 True 变为 False，该方法将失败。

### **运行单元测试**

运行单元测试的直接方法是在测试文件的底部写下如下内容:

```py
if __name__ == '__main__':
    unittest.main()
```

然后，您可以从命令行运行该脚本。您可以期待这样的输出:

| 。-在 0.000 秒内运行 1 次测试OK |

输出开始处的单点表示测试已通过。

单元测试的输出将总是包括测试运行的次数。它还将包括每个测试的状态指示器。但是如果您想要更多关于测试的细节，您可以使用-v 选项。

### **单元测试的结果**

单元测试的输出可以表明以下三种情况之一:

*   好的——这意味着你的测试已经通过。
*   失败——这意味着存在 AssertionError 异常，测试没有通过。
*   ERROR——这意味着测试引发了除 AssertionError 之外的异常。

现在你已经熟悉了 Python 中单元测试的基础知识。我们现在将学习如何通过例子实现单元测试。

## **定义测试用例**

TestCase 类是 unittest 中最重要的类之一。类提供了基本的代码结构，您可以使用它在 Python 中测试函数。

这里有一个例子:

```py
import unittest

def put_fish_in_aquarium(list_of_fishes):
    if len(list_of_fishes) > 10:
        raise ValueError("The aquarium cannot hold more than ten fish")
    return {"tank_1": list_of_fishes}

class TestPutFishInAquarium(unittest.TestCase):
    def test_put_fish_in_aquarium_success(self):
        actual = put_fish_in_aquarium(list_of_fishes=["guppy", "goldfish"])
        expected = {"tank_1": ["guppy", "goldfish"]}
        self.assertEqual(actual, expected)
```

下面是这段代码中发生的事情的分类:

unittest 模块被导入以使其对代码可用。接下来，定义需要测试的功能。在这种情况下，该函数被称为 put_fish_in_aquarium。

编写这个函数是为了接受一个鱼的列表，如果这个列表有十个以上的条目，它就会产生一个错误。

接下来，该函数返回鱼缸的字典映射，在代码中定义为“tank_1”，到鱼的列表。

TestPutFishInAquarium 类被定义为 unittest.TestCase 的子类，类中定义了 test _ put _ fish _ in _ aquarium _ success 方法。该方法使用某个输入调用 put_fish_in_aquarium 函数，然后验证返回值是否与预期的返回值相同。

让我们继续探索执行这个测试的步骤。

### **执行测试用例**

假设上一节中的代码保存为“test_fish_in_aquarium.py”。要执行它，请在命令行中导航到包含 Python 文件的目录，然后运行命令:

```py
python -m unittest test_put_fish_in_aquarium.py
```

在上面的代码行中，我们用代码行的“python -m unittest”部分调用了 unittest Python 模块。这一行中的下一个字符定义了文件的路径，将之前编写的测试用例作为一个参数。

当您运行该命令时，命令行将提供以下输出:

| 输出。-在 0.000 秒内运行 1 次测试OK |

如你所知，测试已经通过了。

让我们来看一个失败的测试。

我们已经更改了下面代码中“expected”变量的值，以使测试失败:

```py
import unittest

def put_fish_in_aquarium(list_of_fishes):
    if len(list_of_fishes) > 10:
        raise ValueError("The aquarium cannot hold more than ten fish")
    return {"tank_1": list_of_fishes}

class TestPutFishInAquarium(unittest.TestCase):
    def test_put_fish_in_aquarium_success(self):
        actual = put_fish_in_aquarium(list_of_fishes=["guppy", "goldfish"])
        expected = {"tank_1": ["tiger"]}
        self.assertEqual(actual, expected)
```

要运行代码，您可以运行我们之前描述的相同命令行条目。输出将如下所示:

| 输出F= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =FAIL:test _ put _ fish _ in _ aquarium _ success(test _ put _ fish _ in _ aquarium。-回溯(最近呼叫最后一次):文件“test_put_fish_in_aquarium.py”，第 13 行，在 test _ put _ fish _ in _ aquarium _ success 中self.assertEqual(实际，预期)assertion error:{ ' tank _ 1 ':[' guppy '，'金鱼']}！= {'tank_1': ['tiger']}- {'tank_1': ['孔雀鱼'，'金鱼']}+ { '坦克 _1': ['老虎']}-在 0.001 秒内完成 1 项测试失败(失败次数=1) |

注意现在输出的开头是“F”而不是“a”它表示测试失败。

### **测试出现异常的功能**

unittest 模块的一个好处是，你可以用它来验证一个测试是否会引发 ValueError 异常。当列表中有太多条目时，将引发异常。

让我们通过添加一个新的测试方法来扩展我们已经讨论过的同一个例子:

```py
import unittest

def put_fish_in_aquarium(list_of_fishes):
    if len(list_of_fishes) > 10:
        raise ValueError("The aquarium cannot hold more than ten fish")
    return {"tank_1": list_of_fishes}

class TestPutFishInAquarium(unittest.TestCase):
    def test_put_fish_in_aquarium_success(self):
        actual = put_fish_in_aquarium(list_of_fishes=["guppy", "goldfish"])
        expected = {"tank_1": ["tiger"]}
        self.assertEqual(actual, expected)

    def test_put_fish_in_aquarium_exception(self):
        excess_fish = ["guppy"] * 25
        with self.assertRaises(ValueError) as exception_context:
            put_fish_in_aquarium(list_of_fishes=excess_fish)
        self.assertEqual(
            str(exception_context.exception),
            " The aquarium cannot hold more than ten fish."
        )
```

代码中定义的新测试方法调用了 put_fish_in_aquarium 方法，就像我们定义的第一个方法一样。但是，该方法调用包含 25 个“guppy”字符串的函数。

新方法使用 with self.assertRaises(...)上下文管理器，它是 TestCase 方法的默认部分，用于检查 put_fish_in_aquarium 方法是否因为这个字符串列表太长而拒绝它。

我们期望出现的异常类 ValueError 是 self.assertRaises 的第一个参数，涉及的上下文管理器绑定到某个名为 exception_context 的变量。

此变量有一个异常属性，其基础值 Error 由 put_fish_in_aquarium 方法引发。当调用 ValueError 上的 str()来检索它所拥有的消息时，它会返回预期的异常。

再次运行测试，您将看到以下输出:

| 输出..-在 0.000 秒内运行 2 次测试OK |

因此，测试通过了。重要的是要记住单元测试。除了 assertRaises 和 assertEqual，TestCase 还提供了几个方法。我们在下面定义了一些值得注意的方法，但是你可以在文档 中找到支持的断言方法 [的完整列表。](https://docs.python.org/3/library/unittest.html)

| **方法** | **断言** |
| assertEqual(a，b) | a == b |
| assertFalse(a) | bool(a)为假 |
| assertIn(a，b) | a 在 b 中 |
| 阿松酮(a) | a 无 |
| assertinotone(a) | a 不是无 |
| assertNotEqual(a，b) | 答！= b |
| assertNotIn(a，b) | a 不在 b |
| assertTrue(a) | bool(a)为真 |

到目前为止，我们已经介绍了一些可以引入到代码中的基本单元测试。让我们继续探索 TestCase 提供的其他工具，您可以使用它们来测试您的代码。

## **设置方法**

TestCase 支持的最有用的方法之一是 setUp 方法。它允许您为单独的测试创建资源。当您有一组在每次测试之前都需要运行的公共准备代码时，使用这种方法是理想的。

通过设置，您可以将所有的准备代码放在一个地方，无需在每次测试中反复重复。这里有一个例子:

```py
import unittest

class FishBowl:
    def __init__(self):
        self.holding_water = False

    def put_water(self):
        self.holding_water = True

class TestFishBowl(unittest.TestCase):
    def setUp(self):
        self.fish_bowl = FishBowl()

    def test_fish_bowl_empty_by_default(self):
        self.assertFalse(self.fish_bowl.holding_water)

    def test_fish_bowl_can_be_filled(self):
        self.fish_bowl.put_water()
        self.assertTrue(self.fish_bowl.holding_water)
```

上面的代码有一个鱼缸类，holding_water 实例最初设置为 False。但是，您可以通过调用 put_water()方法将其设置为 True。

TestFishBowl 方法在 TestCase 子类中定义，并定义设置。这个方法实例化了 FishBowl 的一个新实例，分配给 self.fish_bowl。

如前所述，设置方法在每个单独的方法之前运行，因此，为 test _ fish _ bowl _ empty _ by _ default 和 test_fish_bowl_can_be_filled 创建一个新的 FishBowl 实例。

这两个中的第一个验证 holding_water 为假，第二个验证调用 put_water()后 holding_water 为真。

运行这段代码，我们将看到输出:

| 输出..-在 0.000 秒内运行 2 次测试OK |

如你所见，两项测试都通过了。

-

**注意:** 如果有几个测试文件，一次运行完会更方便。您可以在命令行中使用“python -m unittest discover”来运行多个文件。

-

## **拆卸方法**

tear down 方法是 setUp 方法的对应物，使您能够在测试后删除数据库连接并重置对文件系统所做的修改。这里有一个例子:

```py
import os
import unittest

class AdvancedFishBowl:
    def __init__(self):
        self.fish_bowl_file_name = "fish_bowl.txt"
        default_contents = "guppy, goldfish"
        with open(self.fish_bowl_file_name, "w") as f:
            f.write(default_contents)

    def empty_tank(self):
        os.remove(self.fish_bowl_file_name)

class TestAdvancedFishBowl(unittest.TestCase):
    def setUp(self):
        self.fish_bowl = AdvancedFishBowl()

    def tearDown(self):
        self.fish_bowl.empty_bowl()

    def test_fish_bowl_writes_file(self):
        with open(self.fish_bowl.fish_bowl_file_name) as f:
            contents = f.read()
        self.assertEqual(contents, "guppy, goldfish")
```

上面代码中的 AdvancedFishBowl 类创建了一个名为 fish_bowl.txt 的文件，然后将字符串“guppy，金鱼”写入其中。该类还有一个 empty_bowl 方法，用于删除 fish_bowl.txt 文件。

在代码中，您还可以找到 TestAdvancedFishBowl TestCase 子类的 setUp 和 tearDown 方法。

您可能会猜到 setUp 方法创建了 AdvancedFishBowl 的一个新实例，并将其分配给 self.fish_bowl。

另一方面，tearDown 方法调用 self.fish_bowl 上的 empty_bowl 方法。通过这种方式，它可以确保每次测试结束后都删除 fish_bowl.txt 文件。因此，每个测试都重新开始，而不会受到前一个测试所做的更改和结果的任何影响。

test _ fish _ bowl _ writes _ file 方法验证“孔雀鱼，金鱼”被写入 fish _ bowl . txt。

运行代码将得到以下输出:

| 输出。-在 0.000 秒内运行 1 次测试OK |

## **如何写出好的单元测试？**

有一些提示要记住，这将帮助你写出好的单元测试:

### **#1 仔细命名你的测试**

需要注意的是，在 Python 中，当测试方法以单词“test”开头时，TestCase 会识别它们

因此，以下测试将不起作用:

```py
class TestAddition(unittest.TestCase):     
   def add_test(self):
   self.assertEqual(functions.add(6, 1), 9)
```

你可以写 test_add，这样就可以了。但是 add_test 就不行了。这很重要，因为如果你写的方法前面没有“test ”,你的测试总是会通过的。这是因为它根本没有运行。

有一个你认为已经通过的测试比根本没有测试要糟糕得多。犯这种错误会彻底搞乱 bug 修复。

使用长名字是最好的方法。名字越具体，以后就越容易找到 bug。

### **#2 开始时做简单的测试，慢慢积累**

编写首先想到的单元测试是正确的方法，因为它们肯定很重要。

这个想法是编写测试来检查你的功能是否正常工作。如果这些测试通过了，你就可以开始编写更复杂的测试了。但是永远不要急着写复杂的测试，直到你写了检查代码基本功能的测试。

### **#3 边缘情况**

为边缘案例编写测试是进行单元测试的一个很好的方法。假设你在一个程序中处理数字。如果有人输入否定会怎么样？还是浮点数？

更奇怪的是，如果输入为零，会发生什么吗？

Zero 以破解代码而闻名，所以如果你在和数字打交道，测试它是明智的。

### **#4 编写相互独立的测试**

永远不要编写相互依赖的测试。它忽略了单元测试的要点，并且很有可能会使以后的 bug 测试变得乏味。

由于这个原因，unittest 模块有一个内置的特性来防止开发人员这样做。同样需要注意的是，该模块并不保证测试会按照您指定的顺序运行。

然而，您可以使用 setUp()和 tearDown()方法来编写在每次测试之前和之后执行的代码。

### 避开断言。IsTrue

使用断言。IsTrue 不是一个好主意，因为它没有提供足够的信息。它只告诉你值是真还是假。使用 assertEqual 方法会更有帮助，因为它会给你更多的细节。

### **#6 你的一些测试会漏掉的东西**

编写好的测试需要练习，即使你很擅长，也有可能会遗漏细节。

事实是，即使是最好的开发人员也无法想到程序失败的所有方式。然而，编写好的测试提高了在代码破坏之前捕捉到代码破坏错误的机会。

因此，如果你在代码中遗漏了什么，也不用担心。回到测试文件，为您偶然发现的情况编写一个新的断言。这确保您在将来不会错过它——即使是在您复制函数和测试文件的项目中。

今天编写的有用的测试类和将来编写的好函数一样有用。

## **结论**

使用 unittest 模块是开始测试的最好方法之一，因为它涉及到编写简短的、可维护的方法来验证你的代码。

有了这个模块，编写测试不再费时费力，因为测试对所有 CI 渠道都至关重要。测试有助于在事情发展到修复 bug 成为主要的时间和成本支出之前捕捉 bug。

该模块无疑使测试变得更加容易和容易。开发人员不需要花时间学习或依赖外部测试框架。Python 内置的几个命令和库有助于确保应用程序按预期工作。

然而，随着你积累了更多编写测试的经验，你可以考虑转向其他更强大的测试框架，比如 pytest。这些框架往往具有高级特性，可以帮助你充实你的测试。