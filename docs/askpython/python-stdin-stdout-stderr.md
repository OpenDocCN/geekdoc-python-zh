# Python – stdin, stdout, and stderr

> 原文： [https://www.askpython.com/python/python-stdin-stdout-stderr](https://www.askpython.com/python/python-stdin-stdout-stderr)

在阅读本文之前，让我们先了解一下什么是术语`stdin`、`stdout`和`stderr`。

**标准输入**–这是一个用户程序从用户那里读取信息的*文件句柄*。我们给标准输入( **stdin** )输入。

**标准输出**–用户程序将正常信息写入该文件句柄。输出通过标准输出( **stdout** )返回。

**标准错误**–用户程序将错误信息写入该文件句柄。错误通过标准错误( **stderr** )返回。

Python 为我们提供了**类文件对象**，分别代表 **stdin** 、 **stdout、**和 **stderr** 。让我们看看如何使用这些对象来处理程序的输入和输出。

* * *

## 1\. sys.stdin

Python 的 [`sys`模块](https://www.askpython.com/python-modules/python-sys-module)为我们提供了 stdin、stdout 和 stderr 这三个文件对象。对于输入文件对象，我们使用`sys.stdin`。这类似于一个文件，你可以打开和关闭它，就像任何其他文件一样。

让我们通过一个基本的例子来理解这一点:

```py
import sys

stdin_fileno = sys.stdin

# Keeps reading from stdin and quits only if the word 'exit' is there
# This loop, by default does not terminate, since stdin is open
for line in stdin_fileno:
    # Remove trailing newline characters using strip()
    if 'exit' == line.strip():
        print('Found exit. Terminating the program')
        exit(0)
    else:
        print('Message from sys.stdin: ---> {} <---'.format(line))

```

**输出**

```py
Hi
Message from sys.stdin: ---> Hi
 <---
Hello from AskPython
Message from sys.stdin: ---> Hello from AskPython
 <---
exit
Found exit. Terminating the program

```

上面的代码片段一直从`stdin`读取输入，并将消息打印到控制台(`stdout`)，直到遇到单词`exit`为止。

**注意**:我们通常不关闭默认的`stdin`文件对象，尽管它是允许的。所以`stdin_fileno.close()`是有效的 Python 代码。

现在我们对`stdin`有了一点了解，让我们转到`stdout`。

* * *

## 2\. sys.stdout

对于输出文件对象，我们使用`sys.stdout`。它类似于`sys.stdin`，但是它直接将写入其中的任何内容显示到控制台。

下面的代码片段显示，如果我们向`sys.stdout`写入数据，就会得到控制台的输出。

```py
import sys

stdout_fileno = sys.stdout

sample_input = ['Hi', 'Hello from AskPython', 'exit']

for ip in sample_input:
    # Prints to stdout
    stdout_fileno.write(ip + '\n')

```

**输出**

```py
Hi
Hello from AskPython
exit

```

* * *

## 3\. sys.stderr

这与`sys.stdout`类似，因为它也直接打印到控制台。但不同的是，它*只*打印**异常**和**错误信息**。(这就是为什么它被称为**标准误差**)。

让我们举个例子来说明这一点。

```py
import sys

stdout_fileno = sys.stdout
stderr_fileno = sys.stderr

sample_input = ['Hi', 'Hello from AskPython', 'exit']

for ip in sample_input:
    # Prints to stdout
    stdout_fileno.write(ip + '\n')
    # Tries to add an Integer with string. Raises an exception
    try:
        ip = ip + 100
    # Catch all exceptions
    except:
        stderr_fileno.write('Exception Occurred!\n')

```

**输出**

```py
Hi
Exception Occurred!
Hello from AskPython
Exception Occurred!
exit
Exception Occurred!

```

正如您所观察到的，对于所有的输入字符串，我们试图添加一个整数，这将引发一个异常。我们捕捉所有这样的异常，并使用`sys.stderr`打印另一个调试消息。

* * *

## 重定向到文件

我们可以将`stdin`、`stdout`和`stderr`文件句柄重定向到任何其他文件(文件句柄)。如果您想在不使用任何其他模块(如日志记录)的情况下将事件记录到文件中，这可能很有用。

下面的代码片段将输出(`stdout`)重定向到一个名为`Output.txt`的文件。

因此，我们不会看到任何打印到控制台的内容，因为它现在被打印到文件本身！这就是输出重定向的本质。您将输出“重定向”到其他地方。(这一次，改为`Output.txt`，而不是控制台)

```py
import sys

# Save the current stdout so that we can revert sys.stdou after we complete
# our redirection
stdout_fileno = sys.stdout

sample_input = ['Hi', 'Hello from AskPython', 'exit']

# Redirect sys.stdout to the file
sys.stdout = open('Output.txt', 'w')

for ip in sample_input:
    # Prints to the redirected stdout (Output.txt)
    sys.stdout.write(ip + '\n')
    # Prints to the actual saved stdout handler
    stdout_fileno.write(ip + '\n')

# Close the file
sys.stdout.close()
# Restore sys.stdout to our old saved file handler
sys.stdout = stdout_fileno

```

**输出**

```py
[email protected]:~# python3 output_redirection.py
Hi
Hello from AskPython
exit
[email protected]:~# cat Output.txt
Hi
Hello from AskPython
exit

```

如您所见，我们已经将输出打印到控制台和`Output.txt`。

我们首先将原始的`sys.stdout`文件处理程序对象保存到另一个变量中。我们不仅需要这个来将`sys.stdout`恢复到旧的处理程序(指向控制台)，而且我们还可以使用这个变量打印到控制台！

注意，在写入文件之后，我们关闭它，类似于我们关闭一个文件，因为该文件仍然是打开的。

我们最后使用变量`stdout_fileno`将`sys.stdout`的处理程序恢复到控制台。

对于输入和错误重定向，可以遵循类似的过程，用`sys.stdin`或`sys.stderr`代替`sys.stdout`，并处理输入和异常而不是输出。

* * *

## 结论

在本文中，我们通过使用`sys`模块，学习了如何在 Python 中使用`stdin`、`stdout`和`stderr`。我们还学习了如何操作相应的文件处理程序来重定向到文件或从文件重定向。

## 参考

*   关于从 stdin 读取输入的 JournalDev 文章
*   [关于标准输入、标准输出和标准错误的堆栈溢出问题](https://stackoverflow.com/questions/3385201/confused-about-stdin-stdout-and-stderr)