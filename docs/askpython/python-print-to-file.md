# python–打印到文件

> 原文：<https://www.askpython.com/python/built-in-methods/python-print-to-file>

在本文中，我们将研究使用 Python 打印到文件的一些方法。

* * *

## 方法 1:使用 Write()打印到文件

我们可以使用我们在文件处理教程中学到的[内置函数 write()](https://www.askpython.com/python/python-file-handling) 直接写入文件。

```py
with open('output.txt', 'a') as f:
    f.write('Hi')
    f.write('Hello from AskPython')
    f.write('exit')

```

**输出**(假设`output.txt`是新创建的文件)

```py
[email protected]:~# python output_redirection.py
Hi
Hello from AskPython
exit
[email protected]:~# cat output.txt
Hi
Hello from AskPython
exit

```

* * *

## 方法 2:将 sys.stdout 重定向到文件

通常，当我们使用 **[打印功能](https://www.askpython.com/python/built-in-methods/python-print-function)** 时，输出会显示到控制台上。

但是，由于标准输出流也是文件对象的处理程序，我们可以将标准输出`sys.stdout`改为指向目标文件。

下面的代码摘自我们之前关于[标准输入、标准输出和标准错误](https://www.askpython.com/python/python-stdin-stdout-stderr)的文章。这会将`print()`重定向到文件。

```py
import sys

# Save the current stdout so that we can revert sys.stdou after we complete
# our redirection
stdout_fileno = sys.stdout

sample_input = ['Hi', 'Hello from AskPython', 'exit']

# Redirect sys.stdout to the file
sys.stdout = open('output.txt', 'w')

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

**输出**(假设`output.txt`是新创建的文件)

```py
[email protected]:~# python output_redirection.py
Hi
Hello from AskPython
exit
[email protected]:~# cat output.txt
Hi
Hello from AskPython
exit

```

* * *

## 方法 3:显式打印到文件

我们可以在对`print()`的调用中，通过提及**文件的**关键字参数，直接指定要打印的文件。

例如，下面的代码片段打印到文件`output.txt`。

```py
print('Hi', file=open('output.txt', 'a'))
print('Hello from AskPython', file=open('output.txt', 'a'))
print('exit', file=open('output.txt', 'a'))

```

该文件现在附加了这三行，并且我们已经成功地打印到了`output.txt`！

### 使用上下文管理器

然而，这种方法并不是解决这种情况的最佳方法，因为在同一个文件上重复调用了`open()`。这是浪费时间，我们可以做得更好！

更好的方法是显式使用上下文管理器`with`语句，它负责自动关闭文件并直接使用文件对象。

```py
with open("output.txt", "a") as f:
    print('Hi', file=f)
    print('Hello from AskPython', file=f)
    print('exit', file=f)

```

这给出了与之前相同的结果，将这三行追加到`output.txt`，但是现在更快了，因为我们不再一次又一次地打开同一个文件。

* * *

## 方法 4:使用日志模块

我们可以使用 Python 的[日志模块](https://www.askpython.com/python-modules/python-logging-module)打印到文件。这优于方法 2，在方法 2 中，显式更改文件流不是最佳解决方案。

```py
import logging

# Create the file
# and output every level since 'DEBUG' is used
# and remove all headers in the output
# using empty format=''
logging.basicConfig(filename='output.txt', level=logging.DEBUG, format='')

logging.debug('Hi')
logging.info('Hello from AskPython')
logging.warning('exit')

```

默认情况下，这将把这三行追加到`output.txt`。因此，我们使用`logging`打印到文件，这是推荐的打印到文件的方法之一。

* * *

## 参考

*   关于打印到文件的文章

* * *