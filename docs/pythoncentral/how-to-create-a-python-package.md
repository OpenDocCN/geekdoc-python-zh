# 如何创建 Python 包

> 原文：<https://www.pythoncentral.io/how-to-create-a-python-package/>

使用包是 Python 编程的基本部分。如果包不存在，程序员将需要花费大量时间重写以前编写过的代码。想象一下这样一个场景，每当你想使用一个解析器时，你都必须编写一个解析器——这将浪费大量的时间和精力，程序员也不会做任何其他事情。

当您拥有大量 Python 类(或“模块”)时，您会希望将它们组织成包。当任何项目中的模块数量(简单地说，一个模块可能只是包含一些类的文件)显著增长时，将它们组织成包是更明智的——也就是说，将功能相似的模块/类放在同一个目录中。这篇文章将向你展示如何创建一个 Python 包。

## **Python 中的包是什么？**

在你理解什么是 Python 包之前，你必须对什么是脚本和模块有一个概念。脚本由您在 shell 中运行以完成特定任务的代码组成。程序员可以在自己选择的文本编辑器中编写脚本，并将其保存在。py 扩展名。运行脚本就像在终端中使用 Python 命令一样简单。

相比之下，模块是一个 Python 程序，程序员可以将其导入到其他程序中，或者直接在 Python shell 的交互模式下导入。术语“模块”在 Python 中没有严格的定义；它是可重用代码的总称。

Python 包通常包含几个模块。实际上，一个包是一个模块文件夹，它可能包含更多的文件夹，这些文件夹包含更多的文件夹和模块。

从概念上讲，Python 包是一个名称空间，这意味着包中的模块被包的名称所绑定，并且可以被该名称所引用。

由于模块被定义为可重用、可导入的代码，每个包都可以被定义为一个模块。但是，不能将每个模块都定义为一个包。

包文件夹通常包含一个“__init__”。py”文件，它向 Python 表明该目录是一个包。该文件可能为空，或者包含需要在包初始化时执行的代码。

如果你有一些 Python 编程的经验，你可能对术语“库”很熟悉在 Python 语言中，“库”并不像包或模块那样明确定义。然而，当一个包被发布时，它可以被称为一个库。

## 创建 Python 包的步骤

使用 Python 包非常简单。你需要做的就是:

1.  创建一个目录，并以您的包名命名。
2.  把你的班级放进去。
3.  创建一个 **__init__。目录中的 py** 文件

仅此而已！为了创建一个 Python 包，这是非常容易的。`__init__.py`文件是必需的，因为有了这个文件，Python 将知道这个目录是一个 Python 包目录，而不是一个普通的目录(或文件夹——无论你想叫它什么)。无论如何，就是在这个文件中，我们将从我们全新的包中为 *import* 类编写一些导入语句。

## 如何创建 Python 包的示例

在本教程中，我们将创建一个`Animals`包——它只包含两个名为`Mammals`和`Birds`的模块文件，分别包含`Mammals`和`Birds`类。

### 步骤 1:创建包目录

因此，首先我们创建一个名为`Animals`的目录。

### 步骤 2:添加类

现在，我们为我们的包创建两个类。首先，在`Animals`目录中创建一个名为`Mammals.py`的文件，并将以下代码放入其中:

```py
[python]
class Mammals:
def __init__(self):
''' Constructor for this class. '''
# Create some member animals
self.members = ['Tiger', 'Elephant', 'Wild Cat']

def printMembers(self):
print('Printing members of the Mammals class')
for member in self.members:
print('\t%s ' % member)
[/python]
```

代码几乎是不言自明的！该类有一个名为`members`的属性——这是我们可能感兴趣的一些哺乳动物的列表。它还有一个名为`printMembers`的方法，简单地打印这个类的哺乳动物列表！记住，当你创建一个 Python 包时，所有的类都必须能够被导入，并且不会被直接执行。

接下来我们创建另一个名为`Birds`的类。在`Animals`目录下创建一个名为`Birds.py`的文件，并将以下代码放入其中:

```py
[python]
class Birds:
def __init__(self):
''' Constructor for this class. '''
# Create some member animals
self.members = ['Sparrow', 'Robin', 'Duck']

def printMembers(self):
print('Printing members of the Birds class')
for member in self.members:
print('\t%s ' % member)
[/python]
```

这段代码类似于我们为`Mammals`类提供的代码。

## 第三步:添加 __init__。py 文件

最后，我们在`Animals`目录中创建一个名为`__init__.py`的文件，并将以下代码放入其中:

```py
[python]
from Mammals import Mammals
from Birds import Birds
[/python]
```

就是这样！这就是创建 Python 包的全部内容。为了测试，我们在`Animals`目录所在的同一个目录中创建一个名为`test.py`的简单文件。我们将下面的代码放在`test.py`文件中:

```py
[python]
# Import classes from your brand new package
from Animals import Mammals
from Animals import Birds

# Create an object of Mammals class & call a method of it
myMammal = Mammals()
myMammal.printMembers()

# Create an object of Birds class & call a method of it
myBird = Birds()
myBird.printMembers()
[/python]
```

## **如何使用 Python 包**

如果你以前没有使用过 Python 包，我们已经涵盖了你需要知道的所有东西，以理解在你的脚本中使用 Python 包的过程:

### **导入 Python 包**

您可以使用 import 关键字将包导入到您的 Python 程序中。假设您没有安装任何包，Python 在标准安装中包含了大量的包。预安装软件包的集合被称为 Python 标准库。

标准库装载了各种用例的工具，包括文本处理和数学。您可以通过运行语句:来导入一个用于计算的库

```py
import math
```

你可以把 import 语句想象成 Python 查找模块的搜索触发器。搜索器是严格组织的，Python 从在缓存中查找指定的模块开始。接下来，Python 在标准库中查找模块，最后，它搜索路径列表。

将 sys 模块(标准库中的另一个模块)导入程序后，可以访问路径列表。

```py
import sys
sys.path
```

前面提到的代码中的 sys.path 行返回 Python 试图在其中找到一个包的所有目录。有时，当程序员下载一个包并试图导入它时，他们会遇到导入错误。这里有一个例子:

```py
>>> import gensim
Traceback (most recent call last)
   File "<stdin>", line 1, in <module>
ImportError: No module named genism
```

如果发生这种情况，您必须检查您导入的包是否位于 Python 的搜索路径之一。如果包不在这些路径中，您将需要扩展搜索路径列表以包含包的位置:

```py
>>> sys.path.append('/home/monty/gensim-package')
```

运行上面那行代码将为解释器提供一个额外的位置来查找您导入的包。

### **名称空间和别名的相关性**

例如，当您将数学模块导入 Python 时，您实际上是在初始化数学名称空间。换句话说，您可以使用点符号引用 math 模块中的函数和类。这里有一个例子:

```py
>>> import math
>>> math.factorial(3)
6
```

如果程序员只对模块的特定函数感兴趣，比如说数学模块的阶乘函数，他们可以使用 import 语句只导入相关的函数，就像这样:

```py
>>> from math import factorial
```

也可以从同一个包中导入多个资源，用逗号分隔:

```py
>>> from math import factorial, log
```

需要注意的是，每当你导入一个包时，总会有一些变量冲突的小风险。例如，如果脚本中的一个变量名为 log，并且您从 math 包中导入了 log 函数，Python 将使用您的变量覆盖该函数。这样会产生 bug。

如前所述，通过导入整个包可以避免这些错误。如果您想节省每次使用模块时键入包名的时间，您可以通过以下方式导入包:

```py
>>> import math as m
>>> m.factorial(3)
```

这被称为混叠。一些常用的包有众所周知的别名。例如，NumPy 库通常作为“np”导入。

从一个包中导入所有的资源到你的名字空间也可以避免变量冲突的错误，就像这样:

```py
>>> from math import *
```

但是需要注意的是，上述方法带来了严重的风险，因为你不知道包中使用的所有名字，因此你的变量被覆盖的可能性很高。