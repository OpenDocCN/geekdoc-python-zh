# 在 Python 中定义主函数

> 原文：<https://realpython.com/python-main-function/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**在 Python 中定义主要函数**](/courses/python-main-function/)

许多编程语言都有一个特殊的功能，当操作系统开始运行程序时，这个功能会自动执行。这个函数通常被称为`main()`，根据语言标准，它必须有一个特定的[返回](https://realpython.com/python-return-statement/)类型和参数。另一方面，Python 解释器从文件顶部开始执行脚本，没有 Python 自动执行的特定函数。

然而，为程序的执行定义一个起点对于理解程序如何工作是有用的。Python 程序员想出了几个约定来定义这个起点。

到本文结束时，你会明白:

*   什么是特殊的`__name__`变量，Python 如何定义它
*   为什么要在 Python 中使用`main()`
*   在 Python 中定义`main()`有什么约定
*   将什么代码放入您的`main()`中的最佳实践是什么

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 一个基本的 Python main()

在某些 Python 脚本中，您可能会看到类似于以下示例的函数定义和条件语句:

```py
def main():
    print("Hello World!")

if __name__ == "__main__":
    main()
```

在这段代码中，有一个名为`main()`的函数，当 Python 解释器执行它时，它打印出短语`Hello World!`。还有一个条件(或`if`)语句检查`__name__`的值，并将其与字符串`"__main__"`进行比较。当`if`语句评估为`True`时，Python 解释器执行`main()`。你可以在 Python 的[条件语句中阅读更多关于条件语句的内容。](https://realpython.com/python-conditional-statements/)

这种代码模式在 Python 文件中很常见，你希望**作为脚本**执行，而**在另一个模块**中导入。为了帮助理解这段代码将如何执行，您应该首先理解 Python 解释器如何根据代码如何执行来设置`__name__`。

[*Remove ads*](/account/join/)

## Python 中的执行模式

有两种主要方法可以指示 Python 解释器执行或使用代码:

1.  您可以使用命令行将 Python 文件作为**脚本**来执行。
2.  您可以**将代码从一个 Python 文件导入**到另一个文件或交互式解释器中。

你可以在[如何运行你的 Python 脚本](https://realpython.com/run-python-scripts/)中读到更多关于这些方法的内容。无论您使用哪种方式运行代码，Python 都定义了一个名为`__name__`的特殊变量，该变量包含一个字符串，其值取决于代码的使用方式。

我们将使用这个保存为`execution_methods.py`的示例文件，探索代码的行为如何根据上下文而变化:

```py
print("This is my file to test Python's execution methods.")
print("The variable __name__ tells me which context this file is running in.")
print("The value of __name__ is:", repr(__name__))
```

在这个文件中，有三个对 [`print()`](https://realpython.com/python-print/) 的调用被定义。前两个打印一些介绍性短语。第三个`print()`将首先打印短语`The value of __name__ is`，然后它将使用 Python 的内置`repr()`打印`__name__`变量的表示。

在 Python 中，`repr()`显示对象的可打印表示。这个例子使用了`repr()`来强调`__name__`的值是一个字符串。你可以在 [Python 文档](https://docs.python.org/3/library/functions.html#repr)中阅读更多关于`repr()`的内容。

您将看到贯穿本文的词语**文件**、**模块**和**脚本**。实际上，他们之间没有太大的区别。然而，在强调一段代码的目的时，在含义上有细微的差别:

1.  **文件:**通常，Python 文件是任何包含代码的文件。大多数 Python 文件都有扩展名`.py`。

2.  **脚本:**Python 脚本是您打算从命令行执行以完成任务的文件。

3.  **模块:**Python 模块是您打算从另一个模块或脚本中，或者从交互式解释器中导入的文件。你可以在 [Python 模块和包——简介](https://realpython.com/python-modules-packages/)中阅读更多关于模块的内容。

在[如何运行您的 Python 脚本](https://realpython.com/run-python-scripts/)中也讨论了这种区别。

### 从命令行执行

在这种方法中，您希望从命令行执行 Python 脚本。

当您执行脚本时，您将无法以交互方式定义 Python 解释器正在执行的代码。对于本文的目的来说，如何从命令行执行 Python 的细节并不重要，但是您可以展开下面的框来阅读更多关于 Windows、Linux 和 macOS 上命令行之间的差异。



根据操作系统的不同，从命令行告诉计算机执行代码的方式会略有不同。

在 Linux 和 macOS 上，命令行通常如下例所示:

```py
eleanor@realpython:~/Documents$
```

美元符号(`$`)之前的部分可能看起来不同，这取决于您的用户名和电脑名称。您输入的命令将在`$`之后。在 Linux 或 macOS 上，Python 3 可执行文件的名称是`python3`，所以您应该通过在`$`后面键入`python3 script_name.py`来运行 Python 脚本。

在 Windows 上，命令提示符通常如下例所示:

```py
C:\Users\Eleanor\Documents>
```

`>`之前的部分可能看起来不同，这取决于您的用户名。您输入的命令将在`>`之后。在 Windows 上，Python 3 可执行文件的名字通常是`python`，所以您应该通过在`>`后面键入`python script_name.py`来运行 Python 脚本。

无论您使用什么操作系统，您在本文中使用的 Python 脚本的输出都是相同的，因此本文只显示 Linux 和 macOS 风格的输入，输入行将从`$`开始。

现在您应该从命令行执行`execution_methods.py`脚本，如下所示:

```py
$ python3 execution_methods.py
This is my file to test Python's execution methods.
The variable __name__ tells me which context this file is running in.
The value of __name__ is: '__main__'
```

在本例中，您可以看到`__name__`的值为`'__main__'`，其中引号(`'`)告诉您该值为字符串类型。

记住，在 Python 中，用单引号(`'`)和双引号(`"`)定义的字符串没有区别。你可以阅读更多关于在 Python 的[基本数据类型中定义字符串的内容。](https://realpython.com/python-data-types/#strings)

如果您在脚本中包含一个 [shebang 行](https://en.wikipedia.org/wiki/Shebang_(Unix))并直接执行它(`./execution_methods.py`)，或者使用 IPython 或 Jupyter 笔记本中的`%run`魔法，您会发现相同的输出。

通过在命令中添加`-m`参数，您还可以看到从包中执行的 Python 脚本。大多数情况下，当你使用`pip` : `python3 -m pip install package_name`时，你会看到这个推荐。

添加`-m`参数运行包的`__main__.py`模块中的代码。你可以在[How to Publish a Open-Source Python Package to PyPI](https://realpython.com/pypi-publish-python-package/#different-ways-of-calling-a-package)中找到更多关于`__main__.py`文件的信息。

在所有这三种情况下，`__name__`都有相同的值:字符串`'__main__'`。

**技术细节:**Python 文档明确定义了`__name__`何时拥有值`'__main__'`:

> 当从标准输入、脚本或交互式提示中读取时，模块的`__name__`被设置为等于`'__main__'`。([来源](https://docs.python.org/3/library/__main__.html)

`__name__`与`__doc__`、`__package__`和其他属性一起存储在模块的全局名称空间中。你可以在 [Python 数据模型文档](https://docs.python.org/3/reference/datamodel.html)中读到更多关于这些属性的内容，尤其是模块和包，在 [Python 导入文档](https://docs.python.org/3/reference/import.html#import-related-module-attributes)中。

[*Remove ads*](/account/join/)

### 导入模块或交互式解释器

现在让我们看看 Python 解释器执行代码的第二种方式:导入。当你在开发一个模块或脚本时，你很可能想要利用别人已经构建好的模块，你可以用 [`import`关键字](https://realpython.com/python-keywords/#import-keywords-import-from-as)来实现。

在导入过程中，Python 会执行指定模块中定义的语句(但只在第一次导入模块时*)。为了演示导入`execution_methods.py`文件的结果，启动交互式 Python 解释器，然后导入`execution_methods.py`文件:*

>>>

```py
>>> import execution_methods
This is my file to test Python's execution methods.
The variable __name__ tells me which context this file is running in.
The value of __name__ is: 'execution_methods'
```

在这段代码输出中，您可以看到 Python 解释器执行了对 [`print()`](https://realpython.com/python-print/) 的三次调用。输出的前两行与您在命令行上将该文件作为脚本执行时完全相同，因为前两行都没有变量。但是，第三个`print()`的输出存在差异。

当 Python 解释器导入代码时，`__name__`的值被设置为与正在导入的模块的名称相同。您可以在上面的第三行输出中看到这一点。`__name__`的值为`'execution_methods'`，这是 Python 从中导入的`.py`文件的名称。

请注意，如果您在没有退出 Python 的情况下再次`import`该模块，将不会有输出。

**注意:**关于 Python 中导入如何工作的更多信息，请查看 [Python 导入:高级技术和技巧](https://realpython.com/python-import/)以及[Python 中的绝对与相对导入](https://realpython.com/absolute-vs-relative-python-imports/)。

## Python 主函数的最佳实践

既然您已经看到了 Python 处理不同执行模式的不同之处，那么了解一些可以使用的最佳实践是很有用的。每当您想要编写可以作为脚本*和*在另一个模块或交互式会话中导入运行的代码时，这些都适用。

您将了解四种最佳实践，以确保您的代码可以服务于双重目的:

1.  将大部分代码放入函数或类中。
2.  使用`__name__`来控制代码的执行。
3.  创建一个名为`main()`的函数来包含您想要运行的代码。
4.  从`main()`调用其他函数。

### 将大部分代码放入函数或类

请记住，Python 解释器在导入模块时会执行模块中的所有代码。有时，您编写的代码会有您希望用户控制的副作用，例如:

*   运行需要很长时间的计算
*   写入磁盘上的文件
*   打印会扰乱用户终端的信息

在这些情况下，您希望用户控制触发代码的执行，而不是让 Python 解释器在导入您的模块时执行代码。

因此，最佳实践是**将大多数代码包含在一个函数或一个类**中。这是因为当 Python 解释器遇到 [`def`或`class`关键字](https://realpython.com/python-keywords/#structure-keywords-def-class-with-as-pass-lambda)时，它只存储那些定义供以后使用，并不实际执行它们，直到你告诉它这样做。

将下面的代码保存到一个名为`best_practices.py`的文件中来演示这个想法:

```py
 1from time import sleep
 2
 3print("This is my file to demonstrate best practices.")
 4
 5def process_data(data):
 6    print("Beginning data processing...")
 7    modified_data = data + " that has been modified"
 8    sleep(3)
 9    print("Data processing finished.")
10    return modified_data
```

在这段代码中，首先从 [`time`模块](https://realpython.com/python-time-module/)中导入 [`sleep()`](https://realpython.com/python-sleep/) 。

暂停解释器，无论你给定多少秒作为一个参数，它将产生一个函数，这个函数需要很长时间来运行。接下来，使用`print()`打印一个句子，描述这段代码的用途。

然后，定义一个名为`process_data()`的函数，它做五件事:

1.  打印一些输出，告诉用户数据处理正在开始
2.  修改输入数据
3.  使用`sleep()`暂停执行三秒钟
4.  打印一些输出，告诉用户处理已经完成
5.  返回修改后的数据

**在命令行上执行最佳实践文件**

现在，当您在命令行上将这个文件作为脚本执行时，会发生什么呢？

Python 解释器将执行函数定义之外的`from time import sleep`和`print()`行，然后创建名为`process_data()`的函数定义。然后，脚本将退出，不再做任何事情，因为脚本没有任何执行`process_data()`的代码。

下面的代码块显示了将该文件作为脚本运行的结果:

```py
$ python3 best_practices.py
This is my file to demonstrate best practices.
```

我们在这里看到的输出是第一个`print()`的结果。注意，从`time`导入并定义`process_data()`不会产生输出。具体来说，在`process_data()`定义内的`print()`调用的输出不会被打印出来！

**在另一个模块或交互式解释器中导入最佳实践文件**

当您在交互式会话(或另一个模块)中导入此文件时，Python 解释器将执行与将文件作为脚本执行时完全相同的步骤。

一旦 Python 解释器导入了文件，您就可以使用在您导入的模块中定义的任何变量、类或函数。为了演示这一点，我们将使用交互式 Python 解释器。启动交互式解释器，然后键入`import best_practices`:

>>>

```py
>>> import best_practices
This is my file to demonstrate best practices.
```

导入`best_practices.py`文件的唯一输出来自在`process_data()`之外定义的第一个`print()`调用。从`time`导入并定义`process_data()`不会产生输出，就像从命令行执行代码一样。

[*Remove ads*](/account/join/)

### 使用`if __name__ == "__main__"`来控制代码的执行

如果您希望在从命令行运行脚本时执行`process_data()`,而不是在 Python 解释器导入文件时执行，该怎么办？

您可以使用 [`if __name__ == "__main__"`习语](https://realpython.com/if-name-main-python/)来确定**执行上下文**，并且只有当`__name__`等于`"__main__"`时，才有条件运行`process_data()`。将下面的代码添加到您的`best_practices.py`文件的底部:

```py
11if __name__ == "__main__":
12    data = "My data read from the Web"
13    print(data)
14    modified_data = process_data(data)
15    print(modified_data)
```

在这段代码中，您添加了一个条件语句来检查`__name__`的值。当`__name__`等于字符串`"__main__"`时，该条件将评估为`True`。记住，变量`__name__`的特殊值`"__main__"`意味着 Python 解释器正在执行你的脚本，而不是导入它。

在条件块中，您添加了四行代码(第 12、13、14 和 15 行):

*   **第 12 行和第 13 行:**您正在创建一个变量`data`，它存储您从 Web 上获取的数据并打印出来。
*   **第 14 行:**你在处理数据。
*   **第 15 行:**您正在打印修改后的数据。

现在，从命令行运行您的`best_practices.py`脚本，看看输出将如何变化:

```py
$ python3 best_practices.py
This is my file to demonstrate best practices.
My data read from the Web
Beginning data processing...
Data processing finished.
My data read from the Web that has been modified
```

首先，输出显示了在`process_data()`之外`print()`调用的结果。

之后，打印出`data`的值。这是因为当 Python 解释器将文件作为脚本执行时，变量`__name__`的值为`"__main__"`，所以条件语句的值为`True`。

接下来，您的脚本调用`process_data()`并传入`data`进行修改。当`process_data()`执行时，它在输出中打印一些状态信息。最后，打印出`modified_data`的值。

现在，您应该检查当您从交互式解释器(或另一个模块)导入`best_practices.py`文件时发生了什么。下面的示例演示了这种情况:

>>>

```py
>>> import best_practices
This is my file to demonstrate best practices.
```

请注意，您将获得与在文件末尾添加条件语句之前相同的行为！这是因为`__name__`变量有值`"best_practices"`，所以 Python 不执行块内的代码，包括`process_data()`，因为条件语句求值为`False`。

### 创建一个名为 main()的函数来包含您想要运行的代码

现在，您可以编写 Python 代码，这些代码可以作为脚本从命令行运行，并且在导入时不会产生不必要的副作用。接下来，您将学习如何编写代码，使其他 Python 程序员能够轻松理解您的意思。

很多语言，比如 [C](https://realpython.com/c-for-python-programmers/) 、 [C++](https://realpython.com/python-vs-cpp/) 、 [Java](https://realpython.com/oop-in-python-vs-java/) 等，都定义了一个必须被调用的特殊函数`main()`，操作系统在执行编译好的程序时自动调用这个函数。这个函数通常被称为**入口点**，因为它是执行进入程序的地方。

相比之下，Python 没有作为脚本入口点的特殊函数。实际上，您可以给 Python 脚本中的入口点函数起任何您想要的名字！

尽管 Python 没有给名为`main()`的函数赋予任何意义，但最佳实践是**将入口点函数命名为`main()`** 。这样，任何阅读您的脚本的其他程序员都会立即知道这个函数是完成脚本主要任务的代码的起点。

此外，`main()`应该包含 Python 解释器执行文件时想要运行的任何代码。这比直接将代码放入条件块要好，因为如果用户导入您的模块，他们可以重用`main()`。

更改`best_practices.py`文件，使其看起来像下面的代码:

```py
 1from time import sleep
 2
 3print("This is my file to demonstrate best practices.")
 4
 5def process_data(data):
 6    print("Beginning data processing...")
 7    modified_data = data + " that has been modified"
 8    sleep(3)
 9    print("Data processing finished.")
10    return modified_data
11
12def main():
13    data = "My data read from the Web"
14    print(data)
15    modified_data = process_data(data)
16    print(modified_data)
17
18if __name__ == "__main__":
19    main()
```

在这个例子中，您添加了`main()`的定义，它包含了之前在条件块中的代码。然后，您更改了条件块，使其执行`main()`。如果您将此代码作为脚本运行或导入，您将获得与上一节相同的输出。

[*Remove ads*](/account/join/)

### 从 main() 调用其他函数

Python 中另一个常见的做法是**让`main()`执行其他函数**，而不是将完成任务的代码包含在`main()`中。当您可以将整个任务由几个可以独立执行的更小的子任务组成时，这尤其有用。

例如，您可能有一个执行以下操作的脚本:

1.  从可能是数据库、磁盘上的文件或 web API 的源中读取数据文件
2.  处理数据
3.  将处理过的数据写入另一个位置

如果您在单独的功能中实现这些子任务中的每一个，那么您(或另一个用户)很容易重用其中的一些步骤，而忽略那些您不想要的步骤。然后可以在`main()`中创建一个默认的工作流，两全其美。

是否将这种实践应用到您的代码中是您自己的判断。将工作分成几个功能使得重用更容易，但是增加了其他人试图解释你的代码的难度，因为他们必须跟随程序流中的几个跳跃。

修改您的`best_practices.py`文件，使其看起来像下面的代码:

```py
 1from time import sleep
 2
 3print("This is my file to demonstrate best practices.")
 4
 5def process_data(data):
 6    print("Beginning data processing...")
 7    modified_data = data + " that has been modified"
 8    sleep(3)
 9    print("Data processing finished.")
10    return modified_data
11
12def read_data_from_web():
13    print("Reading data from the Web")
14    data = "Data from the web"
15    return data
16
17def write_data_to_database(data):
18    print("Writing data to a database")
19    print(data)
20
21def main():
22    data = read_data_from_web()
23    modified_data = process_data(data)
24    write_data_to_database(modified_data)
25
26if __name__ == "__main__":
27    main()
```

在这个示例代码中，文件的前 10 行与之前的内容相同。第 12 行的第二个函数定义创建并返回一些样本数据，第 17 行的第三个函数定义模拟将修改后的数据写入数据库。

第 21 行定义了`main()`。在本例中，您修改了`main()`,使其依次调用数据读取、数据处理和数据写入函数。

首先，从`read_data_from_web()`创建`data`。这个`data`被传递给`process_data()`，T3 返回`modified_data`。最后将`modified_data`传入`write_data_to_database()`。

脚本的最后两行是条件块，它检查`__name__`并在`if`语句为`True`时运行`main()`。

现在，您可以从命令行运行整个处理管道，如下所示:

```py
$ python3 best_practices.py
This is my file to demonstrate best practices.
Reading data from the Web
Beginning data processing...
Data processing finished.
Writing processed data to a database
Data from the web that has been modified
```

在这个执行的输出中，您可以看到 Python 解释器执行了`main()`，它执行了`read_data_from_web()`、`process_data()`和`write_data_to_database()`。但是，您也可以导入`best_practices.py`文件，并对不同的输入数据源重复使用`process_data()`，如下所示:

>>>

```py
>>> import best_practices as bp
This is my file to demonstrate best practices.
>>> data = "Data from a file"
>>> modified_data = bp.process_data(data)
Beginning data processing...
Data processing finished.
>>> bp.write_data_to_database(modified_data)
Writing processed data to a database
Data from a file that has been modified
```

在这个例子中，您导入了`best_practices`并将这个代码的名称缩短为`bp`。

导入过程导致 Python 解释器执行`best_practices.py`文件中的所有代码行，因此输出显示了解释文件用途的代码行。

然后，您将文件中的数据存储在`data`中，而不是从 Web 上读取数据。然后，您重用了`best_practices.py`文件中的`process_data()`和`write_data_to_database()`。在这种情况下，您利用了重用代码的优势，而不是在`main()`中定义所有的逻辑。

### Python 主函数最佳实践总结

这里是你刚刚看到的关于 Python 中的`main()`的四个关键最佳实践:

1.  将需要长时间运行或对计算机有其他影响的代码放在函数或类中，这样您就可以精确地控制代码的执行时间。

2.  使用不同的值`__name__`来确定上下文，并用条件语句改变代码的行为。

3.  您应该将您的入口点函数命名为`main()`，以便传达该函数的意图，尽管 Python 并没有赋予名为`main()`的函数任何特殊的意义。

4.  如果您想重用代码中的功能，请在`main()`之外的函数中定义逻辑，并在`main()`内调用这些函数。

[*Remove ads*](/account/join/)

## 结论

恭喜你！您现在知道了如何创建 Python `main()`函数。

您学到了以下内容:

*   了解变量`__name__`的值对于编写具有可执行脚本和可导入模块双重用途的代码非常重要。

*   `__name__`根据您执行 Python 文件的方式呈现不同的值。`__name__`将等于:

    *   `"__main__"`从命令行或用`python -m`执行文件时(执行一个包的`__main__.py`文件)
    *   模块的名称(如果正在导入模块)
*   当您想要开发可重用代码时，Python 程序员已经开发了一套好的实践。

现在您已经准备好编写一些令人敬畏的 Python `main()`函数代码了！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**在 Python 中定义主要函数**](/courses/python-main-function/)*******