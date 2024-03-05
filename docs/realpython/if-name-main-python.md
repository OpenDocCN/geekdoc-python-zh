# Python 中 if __name__ == "__main__ "是做什么的？

> 原文：<https://realpython.com/if-name-main-python/>

你可能在阅读别人的代码时遇到过 Python 的`if __name__ == "__main__"`习语。难怪——[流传甚广](https://github.com/search?q=__name__+%3D%3D+%22__main__%22&type=code)！您甚至可以在自己的脚本中使用`if __name__ == "__main__"`。但是你用对了吗？

也许你以前已经用类似于 [Java](https://realpython.com/java-vs-python/) 的 [C 族语言](https://en.wikipedia.org/wiki/List_of_C-family_programming_languages)编程过，你想知道这个构造是否是使用`main()`函数作为[入口点](https://en.wikipedia.org/wiki/Entry_point#Contemporary)的笨拙附件。

从语法上来说，Python 的`if __name__ == "__main__"`习语只是一个普通的[条件块](https://realpython.com/python-conditional-statements/):

```py
 1if __name__ == "__main__":
 2    ...
```

从第 2 行开始的缩进块包含了当第 1 行的条件语句求值为`True`时 Python 将执行的所有代码。在上面的代码示例中，您放在条件块中的特定代码逻辑用占位符[省略号](https://realpython.com/python-ellipsis/) ( `...`)表示。

那么——如果`if __name__ == "__main__"`习语没有什么特别的，那么为什么*看起来*令人困惑，为什么它继续在 Python 社区引发讨论？

如果习语看起来仍然有点神秘，并且你不完全确定**它是做什么的**，**为什么**你可能想要它，以及**什么时候**使用它，那么你来对地方了！在本教程中，您将了解 Python 的`if __name__ == "__main__"`习语——从它在 Python 中的真正作用开始，以一个更快速引用它的建议结束。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/if-name-main-python-code/)，你将使用它来了解习语这个名字。

## 简而言之:当文件作为脚本运行时，它允许您执行代码，但当它作为模块导入时，则不允许

出于最实际的目的，您可以将使用`if __name__ == "__main__"`打开的条件块视为一种存储代码的方式，这些代码只应在您的文件作为脚本执行时运行。

你马上就会明白这意味着什么。现在，假设您有以下文件:

```py
 1# echo.py
 2
 3def echo(text: str, repetitions: int = 3) -> str:
 4    """Imitate a real-world echo."""
 5    echoed_text = ""
 6    for i in range(repetitions, 0, -1):
 7        echoed_text += f"{text[-i:]}\n"
 8    return f"{echoed_text.lower()}."
 9
10if __name__ == "__main__":
11    text = input("Yell something at a mountain: ")
12    print(echo(text))
```

在本例中，您定义了一个函数`echo()`，它通过逐渐打印输入文本越来越少的最后几个字母来模拟真实世界的回声。

接下来，在第 10 到 12 行，您使用了`if __name__ == "__main__"`习语。这段代码从第 10 行的条件语句`if __name__ == "__main__"`开始。在缩进的第 11 行和第 12 行中，您收集用户输入并用该输入调用`echo()`。当您从命令行将`echo.py`作为脚本运行时，这两行将会执行:

```py
$ python echo.py
Yell something at a mountain: HELLOOOO ECHOOOOOOOOOO
ooo
oo
o
.
```

当您通过将文件对象传递给 Python 解释器将文件作为脚本运行时，表达式`__name__ == "__main__"`返回`True`。然后运行`if`下的代码块，因此 Python 收集用户输入并调用`echo()`。

自己试试吧！您可以从下面的链接下载您将在本教程中使用的所有代码文件:

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/if-name-main-python-code/)，你将使用它来了解习语这个名字。

同时，如果您在另一个模块或控制台会话中导入`echo()`，那么嵌套代码将不会运行:

>>>

```py
>>> from echo import echo
>>> print(echo("Please help me I'm stuck on a mountain"))
ain
in
n
.
```

在这种情况下，您希望在另一个脚本或解释器会话的上下文中使用`echo()`,因此您不需要收集用户输入。运行`input()`会在导入`echo`时产生副作用，从而扰乱你的代码。

当您将特定于文件脚本用法的代码嵌套在`if __name__ == "__main__"`习语下时，您可以避免运行与导入的模块无关的代码。

在`if __name__ == "__main__"`下嵌套代码允许你迎合不同的用例:

*   **脚本:**当作为脚本运行时，您的代码提示用户输入，调用`echo()`，并打印结果。
*   **模块:**当你把`echo`作为一个模块导入时，那么`echo()`被定义，但是没有代码执行。您向主代码会话提供了`echo()`,而没有任何副作用。

通过在代码中实现`if __name__ == "__main__"`习语，您设置了一个额外的入口点，允许您直接从命令行使用`echo()`。

这就对了。您现在已经了解了关于这个主题的最重要的信息。尽管如此，还有更多东西需要了解，其中一些细微之处可以帮助您更深入地理解这段代码，更全面地理解 Python。

继续读下去，了解更多关于[主习语](#is-the-idiom-boilerplate-code-that-should-be-simplified)的信息，因为本教程简称它。

[*Remove ads*](/account/join/)

## 域名习语是如何运作的？

在其核心，习语是一个[条件语句](https://realpython.com/python-conditional-statements/)，它检查变量`__name__`的值是否等于字符串`"__main__"`:

*   如果`__name__ == "__main__"`表达式是`True`，那么执行条件语句后面的缩进代码。
*   如果`__name__ == "__main__"`表达式是`False`，那么 Python 会跳过缩进的代码。

但是什么时候`__name__`等于字符串`"__main__"`？在上一节中，您了解了在命令行中将 Python 文件作为脚本运行时的情况。虽然这涵盖了大多数真实生活中的用例，但也许您想更深入一些。

如果 Python 解释器在**顶级代码环境**中运行您的代码，Python 会将模块的[全局`__name__`](https://realpython.com/python-scope-legb-rule/#globals) 设置为等于`"__main__"`:

> “顶层代码”是开始运行的第一个用户指定的 Python 模块。它是“顶级”的，因为它导入了程序需要的所有其他模块。([来源](https://docs.python.org/3/library/__main__.html#what-is-the-top-level-code-environment))

为了更好地理解这意味着什么，您将设置一个小的实际例子。创建一个 Python 文件，将其命名为`namemain.py`，并添加一行代码:

```py
# namemain.py

print(__name__, type(__name__))
```

您的新文件只包含一行代码，它将全局`__name__`的值和[类型](https://realpython.com/python-data-types/)打印到控制台。

启动您的终端，[将 Python 文件作为脚本运行](https://realpython.com/run-python-scripts/):

```py
$ python namemain.py
__main__ <class 'str'>
```

输出显示，如果您将文件作为脚本运行，`__name__`的值就是 [Python 字符串](https://realpython.com/python-strings/) `"__main__"`。

**注意:**在顶层代码环境中，`__name__`的值始终是`"__main__"`。顶层代码环境通常是作为文件参数传递给 Python 解释器的模块，如上所述。但是，还有其他选项可以构成顶级代码环境:

*   交互式提示的范围
*   用[和`-m`选项](https://docs.python.org/3/using/cmdline.html?highlight=command%20line#cmdoption-m)将 Python 模块或包传递给 Python 解释器，选项【】代表*模块*
*   Python 解释器从标准输入中读取 Python 代码
*   Python 代码通过[的`-c`选项](https://docs.python.org/3/using/cmdline.html?highlight=command%20line#cmdoption-c)传递给 Python 解释器，代表*命令*

如果您想了解更多关于这些选项的信息，那么请查看关于[的 Python 文档，什么是顶级代码环境](https://docs.python.org/3/library/__main__.html#what-is-the-top-level-code-environment)。文档用简明的代码片段说明了每一个要点。

现在，当您的代码在顶级代码环境中执行时，您知道了`__name__`的值。

但是，只有当条件有机会以不同的方式评估时，条件语句才能产生不同的结果。那么，什么时候你的代码*不是*运行在顶级代码环境中，在那种情况下`__name__`的值会发生什么变化呢？

如果你[导入](https://realpython.com/python-import/)你的模块，你的文件中的代码不会在顶级代码环境中运行。在这种情况下，Python 将`__name__`设置为模块的名称。

为了测试这一点，启动一个 Python 控制台并从`namemain.py`导入代码作为一个模块:

>>>

```py
>>> import namemain
namemain <class 'str'>
```

Python 在导入过程中执行存储在全局名称空间`namemain.py`中的代码，这意味着它将调用`print(__name__, type(__name__))`并将输出写入控制台。

然而，在这种情况下，模块的`__name__`的值是不同的。它指向`"namemain"`，一个等于模块名称的字符串。

**注意:**你可以导入任何包含 Python 代码的文件作为模块，Python 会在导入过程中运行你文件中的代码。模块的名称通常是没有 Python 文件扩展名(`.py`)的文件名。

您刚刚了解到，对于您的顶级代码环境，`__name__`始终是`"__main__"`，所以请继续在您的解释器会话中确认这一点。还要检查字符串`"namemain"`来自哪里:

>>>

```py
>>> __name__
'__main__'

>>> namemain.__name__
'namemain'
```

全局`__name__`的值为`"__main__"`，导入的`namemain`模块的`.__name__`的值为`"namemain"`，这是模块的字符串名称。

**注意:**大多数时候，顶层代码环境是您执行的 Python 脚本，也是您导入其他模块的地方。然而，在这个例子中，您可以看到顶级代码环境并不严格地与脚本运行相关联，例如，它也可以是一个解释器会话。

现在您知道了`__name__`的值将根据它所在的位置有两个值:

*   在**顶级代码环境**中，`__name__`的值为 **`"__main__"`** 。
*   在一个**导入的模块**中，`__name__`的值是**模块的名字**作为一个字符串。

因为 Python 遵循这些规则，所以您可以发现一个模块是否正在顶级代码环境中运行。您可以通过使用条件语句检查`__name__`的值来实现这一点，这将带您回到主名称习语:

```py
# namemain.py

print(__name__, type(__name__))

if __name__ == "__main__":
 print("Nested code only runs in the top-level code environment")
```

有了这种条件检查，您就可以声明仅当模块在顶级代码环境中运行时才执行的代码。

将习语添加到`namemain.py`中，如上面的代码块所示，然后再次将该文件作为脚本运行:

```py
$ python namemain.py
__main__ <class 'str'>
Nested code only runs in the top-level code environment
```

当您的代码作为脚本运行时，对`print()`的两个调用都会执行。

接下来，启动一个新的解释器会话，并再次将`namemain`作为模块导入:

>>>

```py
>>> import namemain
namemain <class 'str'>
```

当您将文件作为模块导入时，您嵌套在`if __name__ == "__main__"`下的代码不会执行。

现在您已经知道了名称主习语在 Python 中是如何工作的，您可能想知道您应该何时以及如何在您的代码中使用它——以及何时避免它！

[*Remove ads*](/account/join/)

## 在 Python 中什么时候应该使用主习语这个名字？

当您想要为脚本创建一个额外的入口点时，可以使用这个习语，这样您的文件就可以作为一个独立的脚本以及一个可导入的模块来访问。当您的脚本需要收集用户输入时，您可能需要这样做。

在本教程的[第一部分](#in-short-it-allows-you-to-execute-code-when-the-file-runs-as-a-script-but-not-when-its-imported-as-a-module)中，您使用了 name-main 习语和`input()`来收集运行脚本`echo.py`时的用户输入。这是使用“主习语”这个名字的一个很好的理由！

还有其他方法可以直接从命令行收集用户输入。例如，您可以使用`sys.argv`和名称 main 习语为一个小 Python 脚本创建一个命令行入口点:

```py
 1# echo.py
 2
 3import sys 4
 5def echo(text: str, repetitions: int = 3) -> str:
 6    """Imitate a real-world echo."""
 7    echoed_text = ""
 8    for i in range(repetitions, 0, -1):
 9        echoed_text += f"{text[-i:]}\n"
10    return f"{echoed_text.lower()}."
11
12if __name__ == "__main__":
13    text = " ".join(sys.argv[1:]) 14    print(echo(text))
```

您没有使用`input()`收集用户输入，而是更改了`echo.py`中的代码，以便用户可以直接从命令行提供文本作为参数:

```py
$ python echo.py HELLOOOOO ECHOOOO
ooo
oo
o
.
```

Python 将任意数量的单词收集到`sys.argv`中，这是一个表示所有输入的字符串列表。当空白字符将每个单词与其他单词分开时，每个单词都被视为一个新的参数。

通过执行处理用户输入的代码并将其嵌套在 name-main 习语中，您为脚本提供了一个额外的入口点。

如果您想为一个[包](https://realpython.com/python-modules-packages/#python-packages)创建一个入口点，那么您应该为此创建一个专用的 [`__main__.py`](https://docs.python.org/3/library/__main__.html#main-py-in-python-packages) 文件。这个文件表示当您使用 [`-m`选项](https://docs.python.org/3/using/cmdline.html#cmdoption-m)运行您的包时 Python 调用的入口点:

```py
$ python -m venv venv
```

当您使用`venv`模块创建一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)时，如上所示，然后运行在`__main__.py`文件中定义的代码。模块名`venv`后面的`-m`选项从`venv`模块中调用 [`__main__.py`。](https://github.com/python/cpython/blob/3.10/Lib/venv/__main__.py)

因为`venv`是一个包而不是一个小的命令行界面(CLI)脚本，所以它有一个专用的`__main__.py`文件作为它的入口点。

**注意:**在 name-main 习语下嵌套代码还有额外的优点，比如用户输入集合。因为嵌套代码不会在模块导入期间执行，所以您可以从一个单独的测试模块中对您的函数运行单元测试，而不会产生副作用。

否则会产生副作用，因为测试模块需要导入您的模块来针对您的代码运行测试。

在野外，您可能会遇到更多在 Python 代码中使用 name-main 习语的原因。然而，通过标准输入或命令行收集用户输入是使用它的主要原因。

## 什么时候你应该避免习语这个名字？

既然你已经学会了什么时候使用主习语这个名字，是时候找出什么时候使用它是最好的主意了。您可能会惊讶地发现，在许多情况下，有比将代码嵌套在 Python 中的`if __name__ == "__main__"`下更好的选择。

有时，开发人员使用主习语将**测试运行**添加到一个脚本中，该脚本将代码功能和测试组合在同一个文件中:

```py
# adder.py

import unittest

def add(a: int, b: int) -> int:
    return a + b

class TestAdder(unittest.TestCase):
    def test_add_adds_two_numbers(self):
        self.assertEqual(add(1, 2), 3)

if __name__ == "__main__":
    unittest.main()
```

通过这种设置，当您将代码作为脚本执行时，您可以对代码运行测试:

```py
$ python adder.py
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

因为你将文件作为脚本运行，`__name__`等于`"__main__"`，条件表达式返回`True`，Python 调用`unittest.main()`。小测试套件运行了，您的测试成功了。

同时，在将代码作为模块导入时，您没有创建任何意外的代码执行:

>>>

```py
>>> import adder
>>> adder.add(1, 2)
3
```

仍然可以导入模块并使用您在那里定义的函数。除非在顶级代码环境中执行模块，否则单元测试不会运行。

虽然这适用于小文件，但通常不被认为是好的做法。不建议在同一个文件中混合测试和代码。相反，在单独的文件中编写测试。遵循这个建议通常会使代码库更有条理。这种方法还消除了任何开销，比如需要在主脚本文件中导入`unittest`。

一些程序员使用 name-main 习语的另一个原因是为了包含一个关于他们的代码能做什么的演示:

```py
# echo_demo.py

def echo(text: str, repetitions: int = 3) -> str:
    """Imitate a real-world echo."""
    echoed_text = ""
    for i in range(repetitions, 0, -1):
        echoed_text += f"{text[-i:]}\n"
    return f"{echoed_text.lower()}."

if __name__ == "__main__":
    print('Example call: echo("HELLO", repetitions=2)', end=f"\n{'-' * 42}\n")
    print(echo("HELLO", repetitions=2))
```

同样，您的用户仍然可以导入该模块，而不会有任何副作用。此外，当他们将`echo_demo.py`作为脚本运行时，他们可以看到它的功能:

```py
$ python echo_demo.py
Example call: echo("HELLO", repetitions=2)
------------------------------------------
lo
o
.
```

您可能会在习语中找到这样的演示代码执行，但是可以说还有更好的方式来演示如何使用您的程序。您可以使用可以兼作文档测试的示例运行编写详细的文档字符串，并且可以为您的项目编写适当的文档。

前面的两个例子涵盖了名称为 main 的习语的两个常见的次优用例。还有其他一些场景是最好避免使用 Python 中的名称主习语:

*   **一个纯脚本:**如果你写了一个*的脚本，意味着*作为一个脚本运行，那么你可以把你的代码执行放到全局名称空间中，而不用把它嵌套在主习语中。你*可以*使用 Python 作为脚本语言，因为它没有强制实施强[面向对象模式](https://realpython.com/oop-in-python-vs-java/)。当你用 Python 编程时，你不必拘泥于其他语言的设计模式。

*   **一个复杂的命令行程序:**如果你写一个*更大的*命令行应用，那么最好创建一个单独的文件作为你的切入点。然后，从模块中导入代码，而不是处理名为 main 习语的用户输入。对于更复杂的命令行程序，使用内置的 [`argparse`模块](https://realpython.com/command-line-interfaces-python-argparse/)而不是`sys.argv`也会让你受益。

也许你以前曾经为了这些次优的目的使用过习语这个名字。如果你想了解更多关于如何[为这些场景中的每一个编写更地道的 Python](https://realpython.com/courses/writing-idiomatic-python/) 的知识，请点击提供的链接:

| ❌次优用例 | ✅更好的选择 |
| --- | --- |
| 测试代码执行 | [创建专用测试模块](https://realpython.com/python-testing/#where-to-write-the-test) |
| 演示代码 | 构建[项目文档](https://realpython.com/python-project-documentation-with-mkdocs/)并将示例包含在[您的文档字符串](https://realpython.com/documenting-python-code/)中 |
| 创建一个纯脚本 | [作为脚本运行](https://realpython.com/run-python-scripts/) |
| 提供复杂的 CLI 程序入口点 | [创建专用 CLI 模块](https://realpython.com/site-connectivity-checker-python/#step-3-create-your-website-connectivity-checkers-cli) |

尽管您现在知道了何时应该避免使用主习语，但是您可能仍然想知道如何在有效的场景中最好地使用它。

[*Remove ads*](/account/join/)

## 你应该以何种方式包括名称-主要习语？

Python 中的 name-main 习语[只是一个条件语句](#how-does-the-name-main-idiom-work)，所以你可以在文件中的任何地方使用它——甚至不止一次！然而，对于大多数用例，你将把*的一个*名字——主习语放在脚本的*底部*:

```py
# All your code

if __name__ == "__main__":
    ...
```

您将名称 main 习语放在脚本的末尾，因为 Python 脚本的入口点总是在文件的顶部。如果你把主习语放在文件的底部，那么在 Python 计算条件表达式之前，你所有的函数和类都已经定义好了。

**注意:**在 Python 中，函数或类体中的代码在定义过程中不会运行。只有当你[调用一个函数](https://realpython.com/defining-your-own-python-function/#function-calls-and-definition)或者[实例化一个类](https://realpython.com/python3-object-oriented-programming/#instantiate-an-object-in-python)时，这些代码才会执行。

然而，尽管在一个脚本中使用多个主习语并不常见，但在某些情况下这样做可能是有原因的。Python 的风格指南文档 [PEP 8](https://realpython.com/python-pep8/) 清楚地说明了将所有导入语句放在哪里:

> 导入总是放在文件的顶部，就在任何模块注释和文档字符串之后，模块全局变量和常量之前。([来源](https://peps.python.org/pep-0008/#imports))

这就是为什么您在文件顶部的`echo.py`中导入了`sys`:

```py
# echo.py

import sys 
def echo(text: str, repetitions: int = 3) -> str:
    """Imitate a real-world echo."""
    echoed_text = ""
    for i in range(repetitions, 0, -1):
        echoed_text += f"{text[-i:]}\n"
    return f"{echoed_text.lower()}."

if __name__ == "__main__":
    text = " ".join(sys.argv[1:])
    print(echo(text))
```

然而，当您只是想将`echo`作为一个模块导入时，您甚至根本不需要导入`sys`。

为了解决这个问题，并仍然坚持在 PEP 8 中定义的风格建议，您可以使用第二个名字-主习语。通过将`sys`的导入嵌套在名称主习语中，您可以将所有导入保存在文件的顶部，但避免在不需要使用`sys`时导入它:

```py
# echo.py

if __name__ == "__main__":
 import sys 
def echo(text: str, repetitions: int = 3) -> str:
    """Imitate a real-world echo."""
    echoed_text = ""
    for i in range(repetitions, 0, -1):
        echoed_text += f"{text[-i:]}\n"
    return f"{echoed_text.lower()}."

if __name__ == "__main__":
    text = " ".join(sys.argv[1:])
    print(echo(text))
```

您将`sys`的导入嵌套在*的另一个*名称下——主习语。这样，您可以将导入语句放在文件的顶部，但是当您将`echo`用作模块时，可以避免导入`sys`。

**注意:**你可能不会经常遇到这种情况，但是它可以作为一个例子，说明*在一个文件中使用多个主名习惯用法*可能会有所帮助。

尽管如此，可读性还是很重要的，所以将 import 语句放在顶部，而不使用第二个名字——main 习语通常是更好的选择。然而，如果你在一个[资源](https://realpython.com/micropython/)有限的环境中工作，第二个域名习语可能会派上用场。

正如你在前面的教程中了解到的，使用主习语的场合[比你想象的要少。对于大多数用例，将这些条件检查之一放在脚本的底部将是您的最佳选择。](#when-should-you-use-the-name-main-idiom-in-python)

最后，您可能想知道什么代码应该放入条件代码块。在这方面，Python 文档提供了关于名为 main 的习语的习惯用法的明确指导:

> 在`if __name___ == '__main__'`下面的块中放置尽可能少的语句可以提高代码的清晰性和正确性。([来源](https://docs.python.org/3/library/__main__.html#idiomatic-usage))

尽量少用你的名字——主习语——写代码！当你开始将多行代码嵌套在名字 main 习语下时，你应该用[定义一个`main()`函数](https://realpython.com/python-main-function/)并调用这个函数:

```py
# echo.py

import sys

def echo(text: str, repetitions: int = 3) -> str:
    """Imitate a real-world echo."""
    echoed_text = ""
    for i in range(repetitions, 0, -1):
        echoed_text += f"{text[-i:]}\n"
    return f"{echoed_text.lower()}."

def main() -> None:
 text = " ".join(sys.argv[1:]) print(echo(text)) 
if __name__ == "__main__":
 main()
```

这种设计模式的优势在于——main 习语名下的代码清晰简洁。此外，它使得调用`main()`成为可能，即使你已经将你的代码作为一个模块导入，例如单元测试它的功能。

**注意:**在 Python 中定义`main()`的意思和在其他语言中不同，比如 [Java](https://realpython.com/oop-in-python-vs-java/) 和 c .在 Python 中，将这个函数命名为 *main* 只是一个约定。您可以给这个函数起任何名字——正如您之前所看到的，您甚至根本不需要使用它。

其他面向对象的语言将`main()`函数定义为程序的入口点。在这种情况下，解释器隐式调用一个名为`main()`的函数，没有它你的程序就无法运行。

在这一节中，你已经了解到你应该在剧本的底部写上名字“主习语”。

如果您计划将多行代码嵌套在`if __name__ == "__main__"`下，那么最好将这些代码重构为一个`main()`函数，您可以从名为 main 习语的条件块中调用该函数。

既然您已经知道如何使用主名称习语，您可能会奇怪为什么它看起来比您熟悉的其他 Python 代码更神秘。

[*Remove ads*](/account/join/)

## 是应该简化的习语样板代码吗？

如果您来自不同的面向对象编程语言，您可能会认为 Python 的 name-main 习语是一个入口点，类似于 Java 或 C 中的`main()`函数，但是更笨拙:

[![The Power Rangers and Teletubby meme with text about main functions and Python's if __name__ equal to "__main__"](img/95fdf73d3f3526ab0f2ce86d057139d8.png)](https://files.realpython.com/media/namemain.19d27b02755e.a38f654f963f.jpg)

<figcaption class="figure-caption text-center">Meme based on a web comic (Image: [Mike Organisciak](https://www.instagram.com/p/CH-kflgjCCZ/))</figcaption>

虽然肯定有趣且相关，但这个迷因具有误导性，因为它暗示了名称主习语类似于其他语言中的入口点函数。

Python 的名字——main 习语并不特别。这只是一个条件检查。乍一看，它可能有点神秘，尤其是当您开始使用 Python，并且已经习惯了 Python 纤细优雅的语法时。毕竟，name-main 习语包括一个来自全局名称空间的 dunder 变量，以及一个也是 dunder 值的字符串。

所以它不是其他语言中 *main* 表示的入口点类型。但是为什么看起来是这样的呢？您可能已经多次复制和粘贴了习语，或者甚至将它打了出来，并且想知道为什么 Python 没有更简洁的语法。

如果你浏览 [Python-ideas 邮件列表](https://mail.python.org/archives/list/python-ideas@python.org/)、[被拒 PEPs](https://peps.python.org/#abandoned-withdrawn-and-rejected-peps) 和 [Python 讨论论坛](https://discuss.python.org/search?q=if%20__name__%20%3D%3D%20__main__%20%23ideas)的档案，你会发现许多改变习语的尝试。

如果你阅读了其中的一些讨论，那么你会注意到许多经验丰富的毕达哥拉斯派认为习语并不神秘，不应该被改变。他们给出了多种理由:

*   **很短:**大多数建议的修改只保存两行代码。
*   它有一个有限的用例:你应该只在你需要既作为模块又作为脚本运行一个文件的时候使用它。你应该不需要经常使用它。
*   **它暴露了复杂性:**一旦你[看得更深一点](https://realpython.com/cpython-source-code-guide/)，变量和函数是 Python 的一大部分。这可以使当前的习语成为激发学习者好奇心的一个切入点，并让他们初步了解 Python 的语法。
*   它保持向后兼容性:名字-main 习语长期以来一直是这种语言中事实上的标准，这意味着改变它会破坏向后兼容性。

好吧，所以你现在只能用`if __name__ == "__main__"`了。似乎找到一种好的方法来一致而简洁地引用它会很有帮助！

展开下面的部分，了解一些背景知识和一些建议，告诉你如何在不扭曲舌头或打结的情况下谈论习语这个名字:



在 Python 生涯的某个阶段，您可能会讨论使用主习语这个名字。写出来很长的表达，大声说出来就更繁琐了，不妨找个好的方式说说。

在 Python 社区中有不同的方式来引用它。大多数网上提及都包括整个`if __name__ == "__main__"`表达式，后跟一个词:

*   `if __name__ == "__main__"`大会([来源](https://mail.python.org/pipermail/python-dev/2006-March/062965.html)
*   `if __name__ == "__main__"`表情([来源](https://docs.python.org/3/library/__main__.html)
*   `if __name__ ...`习语([来源](https://mail.python.org/archives/list/python-ideas@python.org/message/JHW7WPSBNRXE4CQZ4PUPBHBPKICH7IDO/)
*   `if __name__ == "__main__": ...`习语([来源](https://peps.python.org/pep-3122/)
*   `if __name__ == "__main__"`习语([来源](https://mail.python.org/archives/list/python-ideas@python.org/message/I736HMSWPQ4O7HXL3LFGVCJ56GIEELRN/)
*   可执行节([来源](https://mail.python.org/pipermail/python-dev/2006-March/062967.html))

你可能会注意到，关于如何谈论`if __name__ == "__main__"`，并没有严格的约定，但是如果你按照普遍的共识称之为**`if __name__ == "__main__"`习语**，你可能不会做错。

如果你想推广标准化的习语的简称，那么告诉你的朋友称它为**主习语**。在真正的 Python 中，我们将这样称呼它！如果你觉得这个术语有用，也许它会流行起来。

如果你很好奇，可以深入到各种 Python 社区渠道中的一些相关讨论中，了解更多关于开发人员为什么主张保持以域名为主的习语不变的信息。

## 结论

你已经学习了 Python 中的`if __name__ == "__main__"`习语做了什么。它允许您编写将文件作为脚本运行时执行的代码，但不允许您将其作为模块导入时执行。当您希望在脚本运行期间收集用户输入并避免导入模块时的副作用(例如，对其功能进行单元测试)时，最好使用它。

您还了解了一些常见但次优的用例，并了解了在这些场景中可以采用的更好、更习惯的方法。也许在了解了 Python 之后，你已经接受了它的名字——main 习语，但是如果你仍然不喜欢它，那么很高兴知道在大多数情况下你可能可以代替它的使用。

你什么时候在你的 Python 代码中使用名称主习语？在阅读本教程时，您是否发现了替换它的方法，或者是否有我们错过的好用例？请在下面的评论中分享你的想法。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/if-name-main-python-code/)，你将使用它来了解习语这个名字。****