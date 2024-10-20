# Python 中的递归文件和目录操作(第 1 部分)

> 原文：<https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/>

如果您希望利用 Python 来操作系统上的目录树或文件，有许多工具可以提供帮助，包括 Python 的标准操作系统模块。下面是一个简单/基本的方法，可以帮助您通过文件扩展名找到系统中的某些文件。

如果您有在系统中“丢失”文件的经历，您不记得它的位置，甚至不确定它的名称，尽管您记得它的类型，这就是您可能会发现这个方法有用的地方。

在某种程度上，这份食谱结合了 Python 中的[如何遍历目录树](https://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/ "How to Traverse a Directory Tree")和[递归目录遍历:制作你的电影列表！](https://www.pythoncentral.io/recursive-python-function-example-make-list-movies/ " Recursive Directory Traversal in Python: Make a list of your movies!")，但我们会对其稍作调整，并在第二部分中对其进行改进。

要编写这个任务的脚本，我们可以使用`os.path`模块中的`walk`函数或`os`模块中的`walk`函数(分别使用 Python 3.x 版或 Python 3.x 版)。

## **用 Python 2.x 中的 os.path.walk 进行递归**

`os.path.walk`函数有 3 个参数:

1.  一个武断的(但却是强制性的)论点。
2.  `visit` -每次迭代时执行的函数。
3.  `top` -目录树的顶端行走。

然后*遍历*顶部的目录树，在每一步执行功能。让我们检查一下函数(我们将其定义为“step”)，我们使用它来打印 top 下的文件的路径名，这些文件的文件扩展名可以通过`arg`提供。

下面是 step 的定义:

```py
def step(ext, dirname, names):
ext = ext.lower()

for name in names:
if name.lower().endswith(ext):
print os.path.join(dirname, name)
```

现在让我们一行一行地分解它，但首先要指出的是，给 step 的参数是由用户直接通过`os.path.walk`函数、**而不是**传递的，这一点非常重要。walk 在每次迭代中传递的三个参数是:

1.  `ext` -赋予`os.path.walk`的任意自变量。
2.  `dirname` -该迭代的目录名。
3.  `names`-`dirname`下所有文件的名称。

我们的 step 函数的第一行当然是我们的函数声明，包括将由`os.path.walk`直接传递的默认参数。

第二行确保我们的`ext`字符串是小写的。第三行开始我们的参数名循环，这是一个列表类型。第四行是我们如何检索带有我们想要的扩展名的文件名，使用字符串方法`endswith`来测试后缀。

最后一行打印通过后缀(扩展名)测试的任何文件的路径，将`dirname`参数连接到名称(带有适当的系统相关分隔符)。

现在，将我们的 step 函数与 walk 函数结合后，脚本看起来类似于这样:

```py
# We only need to import this module
import os.path

# The top argument for walk. The
# Python27/Lib/site-packages folder in my case

topdir = '.'

# The arg argument for walk, and subsequently ext for step
exten = '.txt'

def step(ext, dirname, names):
ext = ext.lower()

for name in names:
if name.lower().endswith(ext):
print(os.path.join(dirname, name))

# Start the walk
os.path.walk(topdir, step, exten)

```

对于我的系统，我在 Python 2.7 的站点包中安装了`wx_py`，输出如下:

```py

.\README.txt

.\wx-2.8-msw-unicode\docs\CHANGES.txt

.\wx-2.8-msw-unicode\docs\MigrationGuide.txt

.\wx-2.8-msw-unicode\docs\README.win32.txt

......

.\wx-2.8-msw-unicode\wx\tools\XRCed\TODO.txt

```

## **用 Python 3.x 中的 os.walk 进行递归**

现在让我们用 Python 3.x 做同样的事情。

Python 3.x 中的`os.walk`函数工作方式不同，提供了比其他函数更多的选项。它需要 4 个参数，只有第一个是强制的。参数(及其默认值)依次为:

`top`

`topdown(=True)`

*布尔型*

`onerror(=None)`

`followlinks(=False)`

*布尔型*

我们现在唯一关心的是第一个。除了参数之外，walk 函数的两个版本的最大区别可能是 Python 2.x 版本自动遍历目录树，而 Python 3.x 版本生成一个生成器函数。这意味着 Python 3.x 版本只有在我们告诉它的时候才会进行下一次迭代，我们这样做的方式是通过一个循环。

我们将把`os.walk`生成器写入进入`step`函数的循环中，而不是像*步骤*那样定义一个单独的函数来调用。像 Python 2.x 版本一样，`os.walk`产生了 3 个值，我们可以在每次迭代中使用(目录路径、目录名和文件名)，但是这次它们是三元组的形式，所以我们必须相应地调整我们的方法。除此之外，我们根本不会改变扩展名后缀测试，所以脚本最终看起来像这样:

```py
import os

# The top argument for walk
topdir = '.'

# The extension to search for
exten = '.txt'

for dirpath, dirnames, files in os.walk(topdir):
for name in files:
if name.lower().endswith(exten):
print(os.path.join(dirpath, name))


```

因为我的系统的 Python32/Lib/site-packages 文件夹不包含任何特殊的内容，所以这个文件夹的输出结果只是:

```py

.\README.txt

```

无论“topdir”和“exten”字符串被设置为什么，这都将以相同的方式工作；然而，这个脚本只是将文件名打印到窗口(在我们的例子中是 Python 的空闲窗口)，如果有许多文件要打印，这会使我们的解释器(或 shell)窗口多行高——滚动起来有点麻烦。如果我们知道是这种情况，那么将结果写入我们可以随时查看的文本文件就容易多了。如果我们像这样加入一个`with`语句(比如在 Python 中的[读写文件),我们可以很容易做到:](https://www.pythoncentral.io/reading-and-writing-to-files-in-python/)

```py

with open(logpath, 'a') as logfile:

logfile.write('%s\n' % os.path.join(dirname, name))

```

让我们先看看如何将它合并到 Python 2.x 版本的脚本中:

```py

# We only need to import this module
import os.path
# The top argument for walk. The
# Python27/Lib/site-packages folder in my case.
topdir = '.'

# The arg argument for walk, and subsequently ext for step
exten = '.txt'

logname = 'findfiletype.log'

def step((ext, logpath), dirname, names):
ext = ext.lower()

for name in names:
if name.lower().endswith(ext):
# Instead of printing, open up the log file for appending
with open(logpath, 'a') as logfile:
logfile.write('%s\n' % os.path.join(dirname, name))

# Change the arg to a tuple containing the file
# extension and the log file name. Start the walk.
os.path.walk(topdir, step, (exten, logname))

```

正如我们在上面看到的，除了第三个变量`logname`和第三个参数`os.path.walk`之外，没有什么变化。with 语句已经取代了`print`语句。由于`os.path.walk`函数的性质，`step`需要打开日志文件，写入日志文件，每找到一个文件名就关闭日志文件；这不会导致任何错误，但有点尴尬。我们还必须注意，因为日志文件是在追加模式下打开的，所以它将**而不是**覆盖已经存在的日志文件，它将**只将**追加到文件中。这意味着如果我们在不改变`logname`的情况下连续运行脚本 2 次或更多次，每次运行的结果将被添加到同一个文件中，这可能是不希望的。

修改版 Python 3.x 脚本就没那么别扭了:

```py
import os

# The top argument for walk
topdir = '.'
# The extension to search for
exten = '.txt'
logname = 'findfiletype.log'
# What will be logged
results = str()
or dirpath, dirnames, files in os.walk(topdir):
for name in files:
if name.lower().endswith(exten):
# Save to results string instead of printing
results += '%s\n' % os.path.join(dirpath, name)

# Write results to logfile
with open(logname, 'w') as logfile:
logfile.write(results)
```

在这个版本中，每个找到的文件的名称被附加到`results`字符串，然后当搜索结束时，结果被写入日志文件。与 Python 2.x 版本不同，日志文件以*写*模式打开，这意味着任何现有的日志文件**都将被**覆盖。在这两种情况下，日志文件都将被写入与脚本相同的目录中(因为我们没有指定完整的路径名)。

有了它，我们就有了一个简单的脚本，可以在文件树下找到某个扩展名的文件，并记录这些结果。在接下来的部分中，我们将在此基础上增加搜索多种文件类型、避免特定路径等功能。