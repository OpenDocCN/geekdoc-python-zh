# 测试 Jupyter 笔记本电脑

> 原文：<https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/>

你做的编程越多，你就越需要了解如何测试你的代码。你会听说极限编程和测试驱动开发(TDD)之类的东西。这些都是创建高质量代码的好方法。但是测试如何适应 Jupyter 呢？坦白说，真的没有。如果您想正确地测试您的代码，您应该在 Jupyter 之外编写代码，并在需要时将其导入单元格。这允许你使用 Python 的 **unittest** 模块或者 **py.test** 来为你的代码编写测试，与 Jupyter 分开。这也可以让你添加像 **nose** 这样的测试人员，或者使用 Travis CI 或 Jenkins 这样的工具将你的代码放入一个持续集成设置中。

然而，现在一切都失去了。您可以对您的 Jupyter 笔记本进行一些测试，即使您没有通过保持代码独立而获得的全部灵活性。我们将会看到一些想法，你可以用它们来做一些基本的测试。

* * *

### 执行并检查

一种流行的“测试”笔记本的方法是从命令行运行它，并将其输出发送到一个文件中。如果您想在命令行上执行，可以使用以下示例语法:

```py

jupyter-nbconvert --to notebook --execute --output output_file_path input_file_path

```

当然，我们希望通过编程来实现这一点，并且希望能够捕获错误。为此，我们将从我的 [exporting Jupyter Notebook 文章](https://www.blog.pythonlibrary.org/2018/10/09/how-to-export-jupyter-notebooks-into-other-formats/)中获取我们的笔记本 runner 代码，并重用它。为了您的方便，再次提醒您:

```py

# notebook_runner.py

import nbformat
import os

from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python3')
    proc.allow_errors = True

    proc.preprocess(nb, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb, f)

    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    return nb, errors

if __name__ == '__main__':
    nb, errors = run_notebook('Testing.ipynb')
    print(errors)

```

你会注意到我已经更新了运行新笔记本的代码。让我们继续创建一个包含两个代码单元的笔记本。创建笔记本后，将标题改为**测试**并保存。这将导致 Jupyter 将文件保存为 **Testing.ipynb** 。现在，在第一个单元格中输入以下代码:

```py

def add(a, b):
    return a + b

add(5, 6)

```

并在单元格#2 中输入以下代码:

```py

1 / 0

```

现在您可以运行笔记本跑步者代码了。当您这样做时，您应该得到以下输出:

```py

[{'ename': 'ZeroDivisionError',
  'evalue': 'integer division or modulo by zero',
  'output_type': 'error',
  'traceback': ['\x1b[0;31m\x1b[0m',
                '\x1b[0;31mZeroDivisionError\x1b[0mTraceback (most recent call '
                'last)',
                '\x1b[0;32m\x1b[0m in '
                '\x1b[0;36m<module>\x1b[0;34m()\x1b[0m\n'
                '\x1b[0;32m----> 1\x1b[0;31m \x1b[0;36m1\x1b[0m '
                '\x1b[0;34m/\x1b[0m '
                '\x1b[0;36m0\x1b[0m\x1b[0;34m\x1b[0m\x1b[0m\n'
                '\x1b[0m',
                '\x1b[0;31mZeroDivisionError\x1b[0m: integer division or '
                'modulo by zero']}]
```

这表明我们有一些输出错误的代码。在这种情况下，我们确实预料到了，因为这是一个非常人为的例子。在您自己的代码中，您可能不希望任何代码输出错误。无论如何，这个笔记本跑步者脚本不足以进行真正的测试。您需要用测试代码包装这些代码。因此，让我们创建一个新文件，我们将把它保存到与笔记本 runner 代码相同的位置。我们将把这个脚本保存为“test_runner.py”。将以下代码放入新脚本中:

```py

import unittest

import runner

class TestNotebook(unittest.TestCase):

    def test_runner(self):
        nb, errors = runner.run_notebook('Testing.ipynb')
        self.assertEqual(errors, [])

if __name__ == '__main__':
    unittest.main()

```

这段代码使用了 Python 的 **unittest** 模块。这里我们创建了一个测试类，其中有一个名为 **test_runner** 的测试函数。这个函数调用我们的笔记本 runner 并断言错误列表应该是空的。要运行此代码，请打开终端并导航到包含您的代码的文件夹。然后运行以下命令:

```py

python test_runner.py

```

当我运行它时，我得到了以下输出:

```py

F
======================================================================
FAIL: test_runner (__main__.TestNotebook)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_runner.py", line 10, in test_runner
    self.assertEqual(errors, [])
AssertionError: Lists differ: [{'output_type': u'error', 'ev... != []

First list contains 1 additional elements.
First extra element 0:
{'ename': 'ZeroDivisionError',
 'evalue': 'integer division or modulo by zero',
 'output_type': 'error',
 'traceback': ['\x1b[0;31m---------------------------------------------------------------------------\x1b[0m',
               '\x1b[0;31mZeroDivisionError\x1b[0m                         '
               'Traceback (most recent call last)',
               '\x1b[0;32m\x1b[0m in '
               '\x1b[0;36m<module>\x1b[0;34m()\x1b[0m\n'
               '\x1b[0;32m----> 1\x1b[0;31m \x1b[0;36m1\x1b[0m '
               '\x1b[0;34m/\x1b[0m \x1b[0;36m0\x1b[0m\x1b[0;34m\x1b[0m\x1b[0m\n'
               '\x1b[0m',
               '\x1b[0;31mZeroDivisionError\x1b[0m: integer division or modulo '
               'by zero']}

Diff is 677 characters long. Set self.maxDiff to None to see it.

----------------------------------------------------------------------
Ran 1 test in 1.463s

FAILED (failures=1)
```

这清楚地表明我们的代码失败了。如果您移除有被零除问题的储存格，并重新执行您的测试，您应该会得到这个结果:

```py

.
----------------------------------------------------------------------
Ran 1 test in 1.324s

OK

```

通过移除单元格(或者只是纠正该单元格中的错误)，您可以通过测试。

* * *

### py.test 插件

我发现了一个很好的插件，你可以使用它来帮助你简化工作流程。我指的是 Jupyter 的 py.test 插件，你可以在这里了解更多关于[的内容。](https://github.com/computationalmodelling/nbval)

基本上，它使 py.test 能够识别 Jupyter 笔记本，并检查存储的输入是否与存储的输出相匹配，以及笔记本是否运行无误。安装完 **nbval** 包后，可以像这样用 py.test 运行它(假设你已经安装了 py.test):

```py

py.test --nbval

```

坦白地说，您可以在我们已经创建的测试文件上运行 py.test，而不需要任何命令，它将使用我们的测试代码。添加 nbval 的主要好处是，如果您这样做的话，就不需要在 Jupyter 周围添加包装器代码。

* * *

### 笔记本电脑内的测试

运行测试的另一种方法是在笔记本中包含一些测试。让我们向测试笔记本添加一个新单元，它包含以下代码:

```py

import unittest

class TestNotebook(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)

```

这将最终在第一个单元格中测试 **add** 函数。我们可以在这里添加一些不同的测试。例如，我们可能想测试如果我们添加一个字符串类型和一个 None 类型会发生什么。但是你可能已经注意到，如果你试图运行这个单元格，你会得到输出。原因是我们还没有实例化这个类。我们需要调用 **unittest.main** 来完成这项工作。因此，虽然运行该单元格以将其放入 Jupyter 的内存是很好的，但我们实际上需要使用以下代码再添加一个单元格:

```py

unittest.main(argv=[''], verbosity=2, exit=False)

```

这段代码应该放在您笔记本的最后一个单元格中，这样它就可以运行您添加的所有测试。它基本上是告诉 Python 以 2 的详细级别运行，并且不要退出。当您运行此代码时，您应该在笔记本中看到以下输出:

```py

test_add (__main__.TestNotebook) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.003s

OK

```

你也可以在 Jupyter 笔记本里用 Python 的 **doctest** 模块做类似的事情。

* * *

### 包扎

正如我在开始时提到的，虽然您可以在 Jupyter 笔记本中测试您的代码，但是如果您只是在它之外测试您的代码，实际上会好得多。然而，有一些变通办法，因为有些人喜欢使用 Jupyter 进行文档记录，所以有一种方法来验证他们是否正常工作是很好的。在本章中，您学习了如何以编程方式运行笔记本，并验证输出是否符合您的预期。如果您愿意，还可以增强代码来验证某些错误是否存在。

您还学习了如何在笔记本单元格中直接使用 Python 的 unittest 模块。这确实提供了一些很好的灵活性，因为您现在可以在一个地方运行您的代码。明智地使用这些工具，它们会很好地为你服务。

* * *

### 相关阅读

*   测试 [Jupyter 笔记本](https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/)
*   nbval [包](https://github.com/computationalmodelling/nbval)
*   [测试和调试](https://kolesnikov.ga/Testing_and_Debugging_Jupyter_Notebooks/) Jupyter 笔记本
*   Jupyter 笔记本中函数的单元测试？