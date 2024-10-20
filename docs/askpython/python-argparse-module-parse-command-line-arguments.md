# Python argparse 模块——轻松解析命令行参数

> 原文：<https://www.askpython.com/python-modules/python-argparse-module-parse-command-line-arguments>

在用 Python 编写命令行脚本时，我们可能会遇到这样的情况:我们需要为程序添加更多的命令行选项。我们自己进行参数解析往往是非常乏味和耗时的，并且经常会大大降低我们的速度。Python 的`argparse`模块提供了这个问题的解决方案。

`argparse`模块为程序员提供了一种快速简单地编写好的命令行界面的方法。让我们看看如何使用这个库为我们编写的任何脚本提供良好的命令行界面选项。

* * *

## Python argparse 库方法

该库为我们提供了各种方法来解析和处理参数字符串，并添加用户友好的命令行选项。

### 1.创建参数解析器

为了处理参数字符串，我们首先需要构建一个解析器。该库为我们提供了`argparse.ArgumentParser()`来构建一个参数解析器。

格式:`parser = argparse.ArgumentParser(description)`

### 2.向解析器对象添加参数

下一步是为命令行界面的解析器添加参数/选项( **CLI** )。我们使用`parser.add_argument()`来做这件事。

格式:`parser.add_argument(name, metavar, type, help)`

*   ***名称*** - >解析对象的属性名称
*   ***metavar*** - >它为帮助消息中的可选参数提供了不同的名称
*   ***类型*** - >变量的数据类型(可能是`int`、`str`等)
*   ***帮助*** - >帮助消息中参数的描述

一个例子来说明上述概念

```py
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='A Simple Program which prints to the console.')

parser.add_argument('integer', metavar='N', type=int, help='an integer to be printed')

args = parser.parse_args()

a = args.integer
print('Printing the integer')
print(a)

```

输出

```py
[email protected] $ python argparse_example.py
usage: argparse_example.py [-h] N
argparse_example.py: error: the following arguments are required: N

[email protected] $ python argparse_example.py 10
Printing the integer
10

[email protected] $ python argparse_example.py -h
usage: argparse_example.py [-h] N

A Simple Program which prints to the console.

positional arguments:
  N           an integer to be printed

optional arguments:
  -h, --help  show this help message and exit

[email protected] $ python argparse_example.py hi
usage: argparse_example.py [-h] N
argparse_example.py: error: argument N: invalid int value: 'hi'

```

请注意，该模块负责参数的类型检查，确保`a`必须是整数，并且必须传递正确的参数，程序才能运行。这就是`type`参数的意义。

* * *

## 程序界面的其他选项

通过在创建解析器对象时指定两个可选参数，即`prog`和`usage`，我们可以向程序添加/修改更多选项。

格式:`argparse.ArgumentParser(prog, usage, description)`

*   `**prog**` - >指定程序的名称(通常默认为`sys.argv[0]`，但可以通过该参数修改。
*   `**usage**` - >指定帮助字符串的使用格式。
*   `**prefix_chars**` - >指定可选参数的前缀字符(对于 Unix 系统为`-`，对于 Windows 为`/`

为了将所有这些放在一起，让我们基于前面的代码片段编写一些简单的代码来说明这个概念。

```py
import argparse

# Create the parser
parser = argparse.ArgumentParser(prog='simple_printer',
                                usage='%(prog)s [options] integer string',
                                description='A Simple Program which prints to the Console',
                                prefix_chars='-')

# Add an integer argument
parser.add_argument('--integer', metavar='N',
                    type=int, help='an integer to be printed')

# Add a second string argument
parser.add_argument('--string', metavar='S',
                    type=str, help='a string to be printed')

# Parse the list of arguments into an object
# called 'args'
args = parser.parse_args()

print('Argument Object:', args)
print('Type of the Argument Object:', type(args))

first_argument = args.integer
second_argument = args.string

print('Printing the integer')
print(first_argument)

print('Printing the string')
print(second_argument)

```

### 1.传递可选参数

请注意，我们已经将参数的名称更改为`--integer`和`--string`。这是因为这是为 Python 脚本指定可选参数的标准格式。(`python script.py -o --option`)

`argparse`通过照顾`--`来自动为我们处理这个问题，确保我们只需要输入一次。下面的输出说明了使用 argparse 解析这些参数的便利性。

可选参数的输出

```py
[email protected] $ python3 argparse_example.py --integer=10
Argument Object: Namespace(integer=10, string=None)
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the integer
10
Printing the string
None

```

其他案例的输出，展示了`argparse`如何为您处理一切。

```py
[email protected] $ python3 argparse_example.py 10 Hello
Argument Object: Namespace(integer=10, string='Hello')
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the integer
10
Printing the string
Hello

[email protected] $ python3 argparse_example.py 10
usage: simple_printer [options] --integer --string
simple_printer: error: the following arguments are required: S

[email protected] $ python3 argparse_example.py -h
usage: simple_printer [options] integer string

A Simple Program which prints to the Console

optional arguments:
  -h, --help   show this help message and exit
  --integer N  an integer to be printed
  --string S   a string to be printed

[email protected] $ python3 argparse_example.py --integer 10 --string Hi
Argument Object: Namespace(integer=10, string='Hi')
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the integer
10
Printing the string
Hi

[email protected] $ python3 argparse_example.py --integer=10 --string=Hi
Argument Object: Namespace(integer=10, string='Hi')
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the integer
10
Printing the string
Hi

```

### 2.对可选参数使用短名称

为了避免为每个可选参数写完整的参数名，可以在我们的脚本中使用一个带连字符的选项(`-o`而不是`--option`)。`argparse`允许我们在向解析器添加参数时，只需将短选项名称作为第一个参数。

格式:`parser.add_args('-o', '--option', help='a simple option')`

在我们之前的代码片段中，我们简单地在`--integer`和`--string`选项中添加了两个小的变化:

```py
# Add an integer argument
parser.add_argument('-i', '--integer', metavar='N',
                    type=int, help='an integer to be printed')

# Add a second string argument
parser.add_argument('-s', '--string', metavar='S',
                    type=str, help='a string to be printed')

```

以简短形式指定可选参数时的输出:

```py
roo[email protected] $ python3 argparse_example.py -s=Hi
Argument Object: Namespace(integer=None, string='Hi')
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the Integer
None
Printing the string
Hi

[email protected] $ python3 argparse_example.py -s=Hi -i=10
Argument Object: Namespace(integer=10, string='Hi')
Type of the Argument Object: <class 'argparse.Namespace'>
Printing the integer
10
Printing the string
Hi

```

* * *

## 结论

在本文中，我们了解了`argparse`库解析参数的基本用法，以及如何通过`type`参数利用类型检查。我们还学习了如何使用`usage`和带连字符的参数名，向脚本中添加可选参数，并使其更加用户友好。

## 参考

*   [arg parse 库文档](https://docs.python.org/3/library/argparse.html)
*   [RealPython 在 argparse 上的帖子](https://realpython.com/command-line-interfaces-python-argparse/)