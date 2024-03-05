# 比较 Python 命令行解析库 Argparse、Docopt 和 Click

> 原文：<https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/>

大约一年前，我开始了一项构建命令行应用程序是家常便饭的工作。那时我已经用了 [argparse](https://realpython.com/command-line-interfaces-python-argparse/) 很多次，并想探索一下还有什么其他的选择。

我发现最受欢迎的选择是 [click](http://click.pocoo.org/5/) 和 [docopt](http://docopt.org/) 。在我的探索过程中，我还发现除了每个库的“为什么使用我”部分之外，没有太多关于这三个库的完整比较。现在有了——这篇博文！

如果您愿意，您可以直接前往[源](https://github.com/kpurdon/greeters)，尽管如果没有本文中介绍的比较和逐步构建，它真的没有多大用处。

本文使用以下版本的库:

```py
$ python --version
Python 3.4.3
# argparse is a Python core library

$ pip list | grep click
click (5.1)

$ pip list | grep docopt
docopt (0.6.2)

$ pip list | grep invoke
invoke (0.10.1)
```

(暂时忽略`invoke`，这是给后面的特别惊喜！)

## 命令行示例

我们正在创建的命令行应用程序将具有以下界面:

```py
python [file].py [command] [options] NAME
```

[*Remove ads*](/account/join/)

### 基本用法*

```py
$ python [file].py hello Kyle
Hello, Kyle!

$ python [file].py goodbye Kyle
Goodbye, Kyle!
```

### 带选项(标志)的用法

```py
$ python [file].py hello --greeting=Wazzup Kyle
Whazzup, Kyle!

$ python [file].py goodbye --greeting=Later Kyle
Later, Kyle!

$ python [file].py hello --caps Kyle
HELLO, KYLE!

$ python [file].py hello --greeting=Wazzup --caps Kyle
WAZZUP, KYLE!
```

本文将比较实现以下特性的每种库方法:

1.  命令(`hello`，`goodbye`)
2.  参数(名称)
3.  选项/标志(`--greeting=<str>`，`--caps`)

附加功能:

1.  版本打印(`-v/--version`)
2.  自动帮助消息
3.  错误处理

正如您所料， *argparse* 、 *docopt* 和 *click* 实现了所有这些特性(就像任何完整的命令行库一样)。这个事实意味着我们将比较的是这些特性的实际实现。每个库都采用了非常不同的方法，这带来了一个非常有趣的比较- *argparse=standard* ， *docopt=docstrings* ， *click=decorators* 。

## 奖金部分

1.  我一直对使用像 [fabric](https://fabric.readthedocs.org/en/latest/) 这样的任务运行器库感到好奇，它是 python3 的替代品 [invoke](https://invoke.readthedocs.org/en/latest/) 来创建简单的命令行接口，所以我将尝试把相同的接口与 *invoke* 放在一起。
2.  打包命令行应用程序时需要一些额外的步骤，所以我也将介绍这些步骤！

## 命令

让我们从为每个库建立基本框架(没有参数或选项)开始。

抱怨吗

```py
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

hello_parser = subparsers.add_parser('hello')
goodbye_parser = subparsers.add_parser('goodbye')

if __name__ == '__main__':
    args = parser.parse_args()
```

这样我们现在有了两个命令(`hello`和`goodbye`)和一个内置的帮助消息。请注意，当作为 hello 命令上的一个选项运行时，帮助消息会发生变化。

```py
$ python argparse/commands.py --help
usage: commands.py [-h] {hello,goodbye} ...

positional arguments:
 {hello,goodbye}

optional arguments:
 -h, --help       show this help message and exit

$ python argparse/commands.py hello --help
usage: commands.py hello [-h]

optional arguments:
 -h, --help  show this help message and exit
```

Docopt

```py
"""Greeter.

Usage:
 commands.py hello
 commands.py goodbye
 commands.py -h | --help

Options:
 -h --help     Show this screen.
"""
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__)
```

同样，我们有两个命令(`hello`、`goodbye`)和一个内置的帮助消息。注意，当作为`hello`命令的一个选项运行时，帮助信息**没有**改变。此外，我们**不需要**在`Options`部分明确指定`commands.py -h | --help`来获得帮助命令。但是，如果我们不这样做，它们将不会作为选项出现在输出帮助消息中。

```py
$ python docopt/commands.py --help
Greeter.

Usage:
 commands.py hello
 commands.py goodbye
 commands.py -h | --help

Options:
 -h --help     Show this screen.

$ python docopt/commands.py hello --help
Greeter.

Usage:
 commands.py hello
 commands.py goodbye
 commands.py -h | --help

Options:
 -h --help     Show this screen.
```

点击

```py
import click

@click.group()
def greet():
    pass

@greet.command()
def hello(**kwargs):
    pass

@greet.command()
def goodbye(**kwargs):
    pass

if __name__ == '__main__':
    greet()
```

这样我们现在有了两个命令(`hello`、`goodbye`)和一个内置的帮助消息。注意，当作为`hello`命令的一个选项运行时，帮助信息会发生变化。

```py
$ python click/commands.py --help
Usage: commands.py [OPTIONS] COMMAND [ARGS]...

Options:
 --help  Show this message and exit.

Commands:
 goodbye
 hello

$ python click/commands.py hello --help
Usage: commands.py hello [OPTIONS]

Options:
 --help  Show this message and exit.
```

即使在这一点上，你可以看到我们有非常不同的方法来构建一个基本的命令行应用程序。接下来让我们添加 *NAME* 参数，以及从每个工具输出结果的逻辑。

[*Remove ads*](/account/join/)

## 参数

在这一节中，我们将向上一节中显示的相同代码添加新的逻辑。我们将在新行中添加注释，说明它们的用途。参数(也称为位置参数)是命令行应用程序的必需输入。在本例中，我们添加了一个必需的`name`参数，这样工具就可以问候一个特定的人。

抱怨吗

为了给子命令添加一个参数，我们使用了`add_argument`方法。为了执行正确的逻辑，当一个命令被调用时，我们使用`set_defaults`方法来设置一个默认函数。最后，在运行时解析参数后，我们通过调用`args.func(args)`来执行默认函数。

```py
import argparse

def hello(args):
    print('Hello, {0}!'.format(args.name))

def goodbye(args):
    print('Goodbye, {0}!'.format(args.name))

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

hello_parser = subparsers.add_parser('hello')
hello_parser.add_argument('name')  # add the name argument
hello_parser.set_defaults(func=hello)  # set the default function to hello

goodbye_parser = subparsers.add_parser('goodbye')
goodbye_parser.add_argument('name')
goodbye_parser.set_defaults(func=goodbye)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)  # call the default function
```

```py
$ python argparse/arguments.py hello Kyle
Hello, Kyle!

$ python argparse/arguments.py hello --help
usage: arguments.py hello [-h] name

positional arguments:
 name

optional arguments:
 -h, --help  show this help message and exit
```

Docopt

为了添加一个选项，我们向 docstring 添加了一个`<name>`。`<>`用于指定位置参数。为了执行正确的逻辑，我们必须检查命令(作为参数处理)在运行时是否是`True``if arguments['hello']:`，然后调用正确的函数。

```py
"""Greeter.

Usage:
 basic.py hello <name>
 basic.py goodbye <name>
 basic.py (-h | --help)

Options:
 -h --help     Show this screen.

"""
from docopt import docopt

def hello(name):
    print('Hello, {0}'.format(name))

def goodbye(name):
    print('Goodbye, {0}'.format(name))

if __name__ == '__main__':
    arguments = docopt(__doc__)

    # if an argument called hello was passed, execute the hello logic.
    if arguments['hello']:
        hello(arguments['<name>'])
    elif arguments['goodbye']:
        goodbye(arguments['<name>'])
```

```py
$ python docopt/arguments.py hello Kyle
Hello, Kyle

$ python docopt/arguments.py hello --help
Greeter.

Usage:
 basic.py hello <name>
 basic.py goodbye <name>
 basic.py (-h | --help)

Options:
 -h --help     Show this screen.
```

> 请注意，帮助消息并不特定于子命令，而是程序的整个文档字符串。

点击

为了给*点击*命令添加一个参数，我们使用了`@click.argument`装饰器。在这种情况下，我们只是传递参数名，但是还有更多选项，其中一些我们稍后会用到。因为我们用参数来修饰逻辑(函数),所以我们不需要做任何事情来设置或调用正确的逻辑。

```py
import click

@click.group()
def greet():
    pass

@greet.command()
@click.argument('name')  # add the name argument
def hello(**kwargs):
    print('Hello, {0}!'.format(kwargs['name']))

@greet.command()
@click.argument('name')
def goodbye(**kwargs):
    print('Goodbye, {0}!'.format(kwargs['name']))

if __name__ == '__main__':
    greet()
```

```py
$ python click/arguments.py hello Kyle
Hello, Kyle!

$ python click/arguments.py hello --help
Usage: arguments.py hello [OPTIONS] NAME

Options:
 --help  Show this message and exit.
```

## 标志/选项

在本节中，我们将再次向上一节中显示的相同代码添加新的逻辑。我们将在新行中添加注释来说明目的。选项是非必需的输入，可以用来改变命令行应用程序的执行。标志仅是选项的[布尔](https://realpython.com/python-boolean/)(`True`/`False`)子集。例如:`--foo=bar`将传递`bar`作为`foo`选项的值，如果给定选项，则`--baz`(如果定义为标志)将传递`True`的值，否则传递`False`。

对于本例，我们将添加`--greeting=[greeting]`选项和`--caps`标志。`greeting`选项将有默认值`Hello`和`Goodbye`，并允许用户传入自定义的问候。例如，给定`--greeting=Wazzup`，工具将响应`Wazzup, [name]!`。如果给出的话,`--caps`标志将大写整个响应。例如，给定`--caps`，工具将响应`HELLO, [NAME]!`。

抱怨吗

```py
import argparse

# since we are now passing in the greeting
# the logic has been consolidated to a single greet function
def greet(args):
    output = '{0}, {1}!'.format(args.greeting, args.name)
    if args.caps:
        output = output.upper()
    print(output)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

hello_parser = subparsers.add_parser('hello')
hello_parser.add_argument('name')
# add greeting option w/ default
hello_parser.add_argument('--greeting', default='Hello')
# add a flag (default=False)
hello_parser.add_argument('--caps', action='store_true')
hello_parser.set_defaults(func=greet)

goodbye_parser = subparsers.add_parser('goodbye')
goodbye_parser.add_argument('name')
goodbye_parser.add_argument('--greeting', default='Goodbye')
goodbye_parser.add_argument('--caps', action='store_true')
goodbye_parser.set_defaults(func=greet)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
```

```py
$ python argparse/options.py hello --greeting=Wazzup Kyle
Wazzup, Kyle!

$ python argparse/options.py hello --caps Kyle
HELLO, KYLE!

$ python argparse/options.py hello --greeting=Wazzup --caps Kyle
WAZZUP, KYLE!

$ python argparse/options.py hello --help
usage: options.py hello [-h] [--greeting GREETING] [--caps] name

positional arguments:
 name

optional arguments:
 -h, --help           show this help message and exit
 --greeting GREETING
 --caps
```

Docopt

一旦我们遇到添加默认选项的情况，我们就遇到了在 *docopt* 中命令的基本实现的障碍。让我们继续来说明这个问题。

```py
"""Greeter.

Usage:
 basic.py hello <name> [--caps] [--greeting=<str>]
 basic.py goodbye <name> [--caps] [--greeting=<str>]
 basic.py (-h | --help)

Options:
 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].

"""
from docopt import docopt

def greet(args):
    output = '{0}, {1}!'.format(args['--greeting'],
                                args['<name>'])
    if args['--caps']:
        output = output.upper()
    print(output)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    greet(arguments)
```

现在，看看当我们运行以下命令时会发生什么:

```py
$ python docopt/options.py hello Kyle
Hello, Kyle!

$ python docopt/options.py goodbye Kyle
Hello, Kyle!
```

什么？！因为我们只能为`--greeting`选项设置一个缺省值，所以我们的`Hello`和`Goodbye`命令现在都用`Hello, Kyle!`来响应。为了让我们完成这项工作，我们需要遵循 docopt 提供的 [git 示例](https://github.com/docopt/docopt/tree/master/examples/git)。重构后的代码如下所示:

```py
"""Greeter.

Usage:
 basic.py hello <name> [--caps] [--greeting=<str>]
 basic.py goodbye <name> [--caps] [--greeting=<str>]
 basic.py (-h | --help)

Options:
 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].

Commands:
 hello       Say hello
 goodbye     Say goodbye

"""

from docopt import docopt

HELLO = """usage: basic.py hello [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].
"""

GOODBYE = """usage: basic.py goodbye [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Goodbye].
"""

def greet(args):
    output = '{0}, {1}!'.format(args['--greeting'],
                                args['<name>'])
    if args['--caps']:
        output = output.upper()
    print(output)

if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True)

    if arguments['<command>'] == 'hello':
        greet(docopt(HELLO))
    elif arguments['<command>'] == 'goodbye':
        greet(docopt(GOODBYE))
    else:
        exit("{0} is not a command. \
 See 'options.py --help'.".format(arguments['<command>']))
```

如你所见，`hello` | `goodbye`子命令现在有了自己的文档字符串，与变量`HELLO`和`GOODBYE`相关联。当这个工具被执行时，它使用一个新的参数`command`来决定解析哪个。这不仅纠正了我们只有一个默认值的问题，而且我们现在还有子命令特定的帮助消息。

```py
$ python docopt/options.py --help
usage: greet [--help] <command> [<args>...]

options:
 -h --help         Show this screen.

commands:
 hello       Say hello
 goodbye     Say goodbye

$ python docopt/options.py hello --help
usage: basic.py hello [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].

$ python docopt/options.py hello Kyle
Hello, Kyle!

$ python docopt/options.py goodbye Kyle
Goodbye, Kyle!
```

此外，我们所有的新选项/标志都工作正常:

```py
$ python docopt/options.py hello --greeting=Wazzup Kyle
Wazzup, Kyle!

$ python docopt/options.py hello --caps Kyle
HELLO, KYLE!

$ python docopt/options.py hello --greeting=Wazzup --caps Kyle
WAZZUP, KYLE!
```

点击

为了添加`greeting`和`caps`选项，我们使用了`@click.option`装饰器。同样，因为我们现在有了默认的问候，所以我们将逻辑提取到一个函数中(`def greeter(**kwargs):`)。

```py
import click

def greeter(**kwargs):
    output = '{0}, {1}!'.format(kwargs['greeting'],
                                kwargs['name'])
    if kwargs['caps']:
        output = output.upper()
    print(output)

@click.group()
def greet():
    pass

@greet.command()
@click.argument('name')
# add an option with 'Hello' as the default
@click.option('--greeting', default='Hello')
# add a flag (is_flag=True)
@click.option('--caps', is_flag=True)
# the application logic has been refactored into a single function
def hello(**kwargs):
    greeter(**kwargs)

@greet.command()
@click.argument('name')
@click.option('--greeting', default='Goodbye')
@click.option('--caps', is_flag=True)
def goodbye(**kwargs):
    greeter(**kwargs)

if __name__ == '__main__':
    greet()
```

```py
$ python click/options.py hello --greeting=Wazzup Kyle
Wazzup, Kyle!

$ python click/options.py hello --greeting=Wazzup --caps Kyle
WAZZUP, KYLE!

$ python click/options.py hello --caps Kyle
HELLO, KYLE!
```

[*Remove ads*](/account/join/)

## 版本选项(`--version` )

在这一节中，我们将展示如何给每个工具添加一个`--version`参数。为了简单起见，我们将把版本号硬编码为 *1.0.0* 。请记住，在生产应用程序中，您会希望从已安装的应用程序中提取它。实现这一点的一种方法是使用这个简单的过程:

>>>

```py
>>> import pkg_resources
>>> # Replace click with the name of your tool:
>>> pkg_resources.get_distribution("click").version
>>> '5.1'
```

确定版本的第二种选择是，当发布新版本时，让自动版本碰撞软件改变文件中定义的版本号。这可以通过 [bumpversion](https://pypi.python.org/pypi/bumpversion) 实现。但是不推荐这种方法，因为它很容易失去同步。通常，最好的做法是在尽可能少的地方保存版本号。

由于添加硬编码版本选项的实现相当简单，我们将使用`...`来表示从上一部分代码中跳过的部分。

抱怨吗

对于 *argparse* ，我们再次需要使用`add_argument`方法，这一次传递了`action='version'`参数和`version`的值。我们将这个方法应用于根解析器(而不是`hello`或`goodbye`子解析器)。

```py
...
parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
...
```

```py
$ python argparse/version.py --version
1.0.0
```

docopt

为了将`--version`添加到 *docopt* 中，我们将它作为一个选项添加到主 docstring 中。此外，我们将`version`参数添加到对 docopt 的第一次调用中(解析主 docstring)。

```py
"""usage: greet [--help] <command> [<args>...]

options:
 -h --help         Show this screen.
 --version         Show the version.

commands:
 hello       Say hello
 goodbye     Say goodbye

"""

from docopt import docopt

...

if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True, version='1.0.0')
    ...
```

```py
$ python docopt/version.py --version
1.0.0
```

点击

*点击*为我们提供了一个方便的`@click.version_option`装饰器。为了增加这一点，我们修饰了我们的`greet`函数(主`@click.group`函数)。

```py
...
@click.group()
@click.version_option(version='1.0.0')
def greet():
    ...
```

```py
$ python click/version.py --version
version.py, version 1.0.0
```

## 改善帮助(`-h` / `--help` )

完成我们的应用程序的最后一步是改进每个工具的帮助文档。我们要确保我们可以访问关于`-h`和`--help`的帮助，并且每个*参数*和*选项*都有一定程度的描述。

抱怨吗

默认情况下 *argparse* 为我们提供了`-h`和`--help`，所以我们不需要为此添加任何东西。然而，我们当前的子命令帮助文档缺少关于`--caps`和`--greeting`做什么以及`name`参数是什么的信息。

```py
$ python argparse/version.py hello -h
usage: version.py hello [-h] [--greeting GREETING] [--caps] name

positional arguments:
 name

optional arguments:
 -h, --help           show this help message and exit
 --greeting GREETING
 --caps
```

为了添加更多的信息，我们使用了`add_argument`方法的`help`参数。

```py
...

hello_parser = subparsers.add_parser('hello')
hello_parser.add_argument('name', help='name of the person to greet')
hello_parser.add_argument('--greeting', default='Hello', help='word to use for the greeting')
hello_parser.add_argument('--caps', action='store_true', help='uppercase the output')
hello_parser.set_defaults(func=greet)

goodbye_parser = subparsers.add_parser('goodbye')
goodbye_parser.add_argument('name', help='name of the person to greet')
goodbye_parser.add_argument('--greeting', default='Hello', help='word to use for the greeting')
goodbye_parser.add_argument('--caps', action='store_true', help='uppercase the output')

...
```

现在，当我们提供帮助标志时，我们会得到一个更加完整的结果:

```py
$ python argparse/help.py hello -h
usage: help.py hello [-h] [--greeting GREETING] [--caps] name

positional arguments:
 name                 name of the person to greet

optional arguments:
 -h, --help           show this help message and exit
 --greeting GREETING  word to use for the greeting
 --caps               uppercase the output
```

Docopt

这部分是 *docopt* 的亮点。因为我们将文档编写为命令行界面本身的定义，所以我们已经完成了帮助文档。此外，已经提供了`-h`和`--help`。

```py
$ python docopt/help.py hello -h
usage: basic.py hello [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].
```

点击

向*点击*添加帮助文档与 *argparse* 非常相似。我们需要向所有的`@click.option`装饰器添加`help`参数。

```py
...

@greet.command()
@click.argument('name')
@click.option('--greeting', default='Hello', help='word to use for the greeting')
@click.option('--caps', is_flag=True, help='uppercase the output')
def hello(**kwargs):
    greeter(**kwargs)

@greet.command()
@click.argument('name')
@click.option('--greeting', default='Goodbye', help='word to use for the greeting')
@click.option('--caps', is_flag=True, help='uppercase the output')
def goodbye(**kwargs):
    greeter(**kwargs)

...
```

但是，*点击*T5 默认不提供给我们`-h`。我们需要使用`context_settings`参数来覆盖默认的`help_option_names`。

```py
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

...

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def greet():
    pass
```

现在*点击*帮助文档已经完成。

```py
$ python click/help.py hello -h
Usage: help.py hello [OPTIONS] NAME

Options:
 --greeting TEXT  word to use for the greeting
 --caps           uppercase the output
 -h, --help       Show this message and exit.
```

[*Remove ads*](/account/join/)

## 错误处理

错误处理是任何应用程序的重要组成部分。本节将探讨每个应用程序的默认错误处理，并在需要时实现附加逻辑。我们将探讨三种错误情况:

1.  没有给出足够的必需参数。
2.  给定的选项/标志无效。
3.  给出了一个带有值的标志。

抱怨吗

```py
$ python argparse/final.py hello
usage: final.py hello [-h] [--greeting GREETING] [--caps] name
final.py hello: error: the following arguments are required: name

$ python argparse/final.py --badoption hello Kyle
usage: final.py [-h] [--version] {hello,goodbye} ...
final.py: error: unrecognized arguments: --badoption

$ python argparse/final.py hello --caps=notanoption Kyle
usage: final.py hello [-h] [--greeting GREETING] [--caps] name
final.py hello: error: argument --caps: ignored explicit argument 'notanoption'
```

不是很令人兴奋，因为 *argparse* 开箱即用地处理我们所有的错误案例。

Docopt

```py
$ python docopt/final.py hello
Hello, None!

$ python docopt/final.py hello --badoption Kyle
usage: basic.py hello [options] [<name>]
```

不幸的是，我们需要做一些工作来让 *docopt* 达到可接受的最低错误处理水平。在 *docopt* 中推荐的验证方法是[模式](https://pypi.python.org/pypi/schema)模块。*确保安装- `pip install schema`。此外，它们还提供了一个非常基本的[验证示例](https://github.com/docopt/docopt/blob/master/examples/validation_example.py)。下面是我们的模式验证应用程序:

```py
...
from schema import Schema, SchemaError, Optional
...
    schema = Schema({
        Optional('hello'): bool,
        Optional('goodbye'): bool,
        '<name>': str,
        Optional('--caps'): bool,
        Optional('--help'): bool,
        Optional('--greeting'): str
    })

    def validate(args):
        try:
            args = schema.validate(args)
            return args
        except SchemaError as e:
            exit(e)

    if arguments['<command>'] == 'hello':
        greet(validate(docopt(HELLO)))
    elif arguments['<command>'] == 'goodbye':
        greet(validate(docopt(GOODBYE)))
...
```

有了这个验证，我们现在得到一些错误消息。

```py
$ python docopt/validation.py hello
None should be instance of <class 'str'>

$ python docopt/validation.py hello --greeting Kyle
None should be instance of <class 'str'>

$ python docopt/validation.py hello --caps=notanoption Kyle
--caps must not have an argument
usage: basic.py hello [options] [<name>]
```

虽然这些消息不是非常具有描述性，并且对于较大的应用程序来说,[可能很难调试](https://realpython.com/python-debugging-pdb/),但总比没有验证好。schema 模块确实提供了其他机制来添加更具描述性的错误消息，但是我们不会在这里讨论这些。

点击

```py
$ python click/final.py hello
Usage: final.py hello [OPTIONS] NAME

Error: Missing argument "name".

$ python click/final.py hello --badoption Kyle
Error: no such option: --badoption

$ python click/final.py hello --caps=notanoption Kyle
Error: --caps option does not take a value
```

和 *argparse* 一样， *click* 默认处理错误输入。

* * *

至此，我们已经完成了我们要构建的命令行应用程序的构建。在我们结束之前，让我们看看另一个可能的选择。

## 调用

我们可以使用简单的任务运行库 [invoke](https://invoke.readthedocs.org/en/latest/) 来构建欢迎命令行应用程序吗？让我们来了解一下！

首先，让我们从最简单的欢迎界面开始:

*tasks.py*

```py
from invoke import task

@task
def hello(name):
    print('Hello, {0}!'. format(name))

@task
def goodbye(name):
    print('Goodbye, {0}!'.format(name))
```

通过这个非常简单的文件，我们得到了两个任务和非常少的帮助。从与 *tasks.py* 相同的目录中，我们得到以下结果:

```py
$ invoke -l
Available tasks:

 goodbye
 hello

$ invoke hello Kyle
Hello, Kyle!

$ invoke goodbye Kyle
Goodbye, Kyle!
```

现在让我们添加我们的选项/标志- `--greeting`和`--caps`。此外，我们可以将问候逻辑提取到它自己的功能中，就像我们对其他工具所做的那样。

```py
from invoke import task

def greet(name, greeting, caps):
    output = '{0}, {1}!'.format(greeting, name)
    if caps:
        output = output.upper()
    print(output)

@task
def hello(name, greeting='Hello', caps=False):
    greet(name, greeting, caps)

@task
def goodbye(name, greeting='Goodbye', caps=False):
    greet(name, greeting, caps)
```

现在我们实际上拥有了我们在开始时指定的完整接口！

```py
$ invoke hello Kyle
Hello, Kyle!

$ invoke hello --greeting=Wazzup Kyle
Wazzup, Kyle!

$ invoke hello --greeting=Wazzup --caps Kyle
WAZZUP, KYLE!

$ invoke hello --caps Kyle
HELLO, KYLE!
```

[*Remove ads*](/account/join/)

### 帮助文档

为了与 *argparse* 、 *docopt* 和 *click* 竞争，我们还需要能够添加完整的帮助文档。幸运的是，这也可以在 *invoke* 中通过使用`@task`装饰器的`help`参数并向被装饰的函数添加文档字符串来实现。

```py
...

HELP = {
    'name': 'name of the person to greet',
    'greeting': 'word to use for the greeting',
    'caps': 'uppercase the output'
}

@task(help=HELP)
def hello(name, greeting='Hello', caps=False):
    """
 Say hello.
 """
    greet(name, greeting, caps)

@task(help=HELP)
def goodbye(name, greeting='Goodbye', caps=False):
    """
 Say goodbye.
 """
    greet(name, greeting, caps)
```

```py
$ invoke --help hello
Usage: inv[oke] [--core-opts] hello [--options] [other tasks here ...]

Docstring:
 Say hello.

Options:
 -c, --caps                     uppercase the output
 -g STRING, --greeting=STRING   word to use for the greeting
 -n STRING, --name=STRING       name of the person to greet
 -v, --version
```

### 版本选项

实现一个`--version`选项并不那么简单，并且有一个警告。基本的是，我们将添加`version=False`作为每个任务的选项，如果`True`调用新的`print_version`函数。为了实现这一点，我们不能有任何没有默认值的位置参数，否则我们会得到:

```py
$ invoke hello --version
'hello' did not receive all required positional arguments!
```

还要注意，我们在命令`hello`和`goodbye`上调用`--version`，因为*调用*本身有一个版本命令:

```py
$ invoke --version
Invoke 0.10.1
```

版本命令的完整实现如下:

```py
...

def print_version():
    print('1.0.0')
    exit(0)

@task(help=HELP)
def hello(name='', greeting='Hello', caps=False, version=False):
    """
 Say hello.
 """
    if version:
        print_version()
    greet(name, greeting, caps)

...
```

现在，我们能够请求 invoke 提供我们工具的版本:

```py
$ invoke hello --version
1.0.0
```

## 结论

回顾一下，让我们看看我们创建的每个工具的最终版本。

抱怨吗

```py
import argparse

def greet(args):
    output = '{0}, {1}!'.format(args.greeting, args.name)
    if args.caps:
        output = output.upper()
    print(output)

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
subparsers = parser.add_subparsers()

hello_parser = subparsers.add_parser('hello')
hello_parser.add_argument('name', help='name of the person to greet')
hello_parser.add_argument('--greeting', default='Hello', help='word to use for the greeting')
hello_parser.add_argument('--caps', action='store_true', help='uppercase the output')
hello_parser.set_defaults(func=greet)

goodbye_parser = subparsers.add_parser('goodbye')
goodbye_parser.add_argument('name', help='name of the person to greet')
goodbye_parser.add_argument('--greeting', default='Hello', help='word to use for the greeting')
goodbye_parser.add_argument('--caps', action='store_true', help='uppercase the output')
goodbye_parser.set_defaults(func=greet)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
```

Docopt

```py
"""usage: greet [--help] <command> [<args>...]

options:
 -h --help         Show this screen.
 --version         Show the version.

commands:
 hello       Say hello
 goodbye     Say goodbye

"""

from docopt import docopt
from schema import Schema, SchemaError, Optional

HELLO = """usage: basic.py hello [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Hello].
"""

GOODBYE = """usage: basic.py goodbye [options] [<name>]

 -h --help         Show this screen.
 --caps            Uppercase the output.
 --greeting=<str>  Greeting to use [default: Goodbye].
"""

def greet(args):
    output = '{0}, {1}!'.format(args['--greeting'],
                                args['<name>'])
    if args['--caps']:
        output = output.upper()
    print(output)

if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True, version='1.0.0')

    schema = Schema({
        Optional('hello'): bool,
        Optional('goodbye'): bool,
        '<name>': str,
        Optional('--caps'): bool,
        Optional('--help'): bool,
        Optional('--greeting'): str
    })

    def validate(args):
        try:
            args = schema.validate(args)
            return args
        except SchemaError as e:
            exit(e)

    if arguments['<command>'] == 'hello':
        greet(validate(docopt(HELLO)))
    elif arguments['<command>'] == 'goodbye':
        greet(validate(docopt(GOODBYE)))
    else:
        exit("{0} is not a command. See 'options.py --help'.".format(arguments['<command>']))
```

点击

```py
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

def greeter(**kwargs):
    output = '{0}, {1}!'.format(kwargs['greeting'],
                                kwargs['name'])
    if kwargs['caps']:
        output = output.upper()
    print(output)

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def greet():
    pass

@greet.command()
@click.argument('name')
@click.option('--greeting', default='Hello', help='word to use for the greeting')
@click.option('--caps', is_flag=True, help='uppercase the output')
def hello(**kwargs):
    greeter(**kwargs)

@greet.command()
@click.argument('name')
@click.option('--greeting', default='Goodbye', help='word to use for the greeting')
@click.option('--caps', is_flag=True, help='uppercase the output')
def goodbye(**kwargs):
    greeter(**kwargs)

if __name__ == '__main__':
    greet()
```

引起

```py
from invoke import task

def greet(name, greeting, caps):
    output = '{0}, {1}!'.format(greeting, name)
    if caps:
        output = output.upper()
    print(output)

HELP = {
    'name': 'name of the person to greet',
    'greeting': 'word to use for the greeting',
    'caps': 'uppercase the output'
}

def print_version():
    print('1.0.0')
    exit(0)

@task(help=HELP)
def hello(name='', greeting='Hello', caps=False, version=False):
    """
 Say hello.
 """
    if version:
        print_version()
    greet(name, greeting, caps)

@task(help=HELP)
def goodbye(name='', greeting='Goodbye', caps=False, version=False):
    """
 Say goodbye.
 """
    if version:
        print_version()
    greet(name, greeting, caps)
```

[*Remove ads*](/account/join/)

### 我的建议

现在，为了解决这个问题，我的个人首选库是 *click* 。去年我一直在大型、多命令、复杂的界面上使用它。(感谢 [@kwbeam](https://twitter.com/kwbeam) 把我介绍给*点击*)。我更喜欢装饰方法，认为它提供了一个非常干净、可组合的界面。话虽如此，还是让我们公平地评价一下每个选项吧。

抱怨吗

***arperse*是用于创建命令行实用程序的标准库(包含在 Python 中)。**就这一点而言，它可以说是本文研究的最常用的工具。Argparse 使用起来也非常简单，因为大量的*魔法*(在幕后发生的隐式工作)被用来构建接口。例如，参数和选项都是使用`add_arguments`方法定义的，并且 *argparse* 在幕后判断出哪个是哪个。

Docopt

如果你认为写文档很棒， *docopt* 适合你！此外 *docopt* 有很多其他语言[的实现](https://github.com/docopt)——这意味着你可以学习一个库并在多种语言中使用它。docopt 的缺点是它非常结构化，你必须定义你的命令行界面。(有人可能会说这是好事！)

点击

我已经说过我非常喜欢 *click* 并且已经在生产中使用了一年多。**我鼓励你阅读非常完整的[为什么点击？](https://click.palletsprojects.com/en/latest/why/)文献资料。**事实上，正是这些文档激发了这篇博文的灵感！ *click* 的装饰风格实现使用起来非常简单，因为你正在装饰你想要执行的函数，这使得阅读代码和判断将要执行什么变得非常容易。此外， *click* 支持回调、命令嵌套等高级功能。 *Click* 基于现已废弃的 [optparse](https://docs.python.org/2/library/optparse.html) 库的一个分支。

引起

**Invoke 这个对比让我很惊讶。**我认为一个为任务执行而设计的库可能无法轻松匹配完整的命令行库——但它做到了！也就是说，我不建议在这种类型的工作中使用它，因为对于比这里给出的例子更复杂的事情，您肯定会遇到限制。

## 奖励:包装

由于不是每个人都用 [setuptools](https://pypi.python.org/pypi/setuptools) (或其他解决方案)打包 python 源代码，我们决定不把它作为本文的核心部分。此外，我们不想将*包装*作为一个完整的话题。如果你想了解更多关于 setuptools [包装的信息，请点击这里](https://packaging.python.org/en/latest/)或者 conda [包装的信息，请点击这里](http://conda.pydata.org/docs/building/build.html)或者你可以阅读我之前关于 conda 包装的[博文](http://kylepurdon.com/blog/packaging-python-basics-with-continuum-analytics-conda.html)。**我们将在这里介绍如何使用 *entry_points* 选项使命令行应用程序成为安装**时的可执行命令。

### 入口点基础知识

一个[入口点](https://packaging.python.org/en/latest/distributing.html?highlight=entry_points#entry-points)本质上是你的代码中一个单一函数的映射，这个函数将在你的系统路径上被给予一个命令。入口点的形式为- `command = package.module:function`

解释这一点的最佳方式是查看我们的 *click* 示例并添加一个入口点。

### 打包点击命令

*单击*使打包变得简单，因为默认情况下，我们在执行程序时调用一个函数:

```py
if __name__ == '__main__':
    greet()
```

除了其余的 *setup.py* (此处未涉及)之外，我们将添加以下内容来为我们的 *click* 应用程序创建一个*入口点*。

假设以下目录结构-

```py
greeter/
├── greet
│   ├── __init__.py
│   └── cli.py       <-- the same as our final.py
└── setup.py
```

-我们将创建以下*入口点*:

```py
entry_points={
    'console_scripts': [
        'greet=greet.cli:greet',  # command=package.module:function
    ],
},
```

当用户安装用这个*入口点*创建的包时， *setuptools* 会创建下面的可执行脚本(名为`greet`)并放在用户系统的路径上。

```py
#!/usr/bin/python
if __name__ == '__main__':
    import sys
    from greet.cli import greet

    sys.exit(greet())
```

安装后，用户现在可以运行以下程序:

```py
$ greet --help
Usage: greet [OPTIONS] COMMAND [ARGS]...

Options:
 --version   Show the version and exit.
 -h, --help  Show this message and exit.

Commands:
 goodbye
 hello
```

[*Remove ads*](/account/join/)

### 打包 Argparse 命令

我们需要做的唯一不同于 *click* 的事情是将所有的应用程序初始化都放到一个函数中，我们可以在我们的*入口点*中调用这个函数。

这个:

```py
if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
```

变成了:

```py
def greet():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    greet()
```

现在我们可以为*点击*定义的*入口点*使用相同的模式。

### 打包 Docopt 命令

打包 *docopt* 命令需要与 *argparse* 相同的过程。

这个:

```py
if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True, version='1.0.0')

    if arguments['<command>'] == 'hello':
        greet(docopt(HELLO))
    elif arguments['<command>'] == 'goodbye':
        greet(docopt(GOODBYE))
    else:
        exit("{0} is not a command. See 'options.py --help'.".format(arguments['<command>']))
```

变成了:

```py
def greet():
    arguments = docopt(__doc__, options_first=True, version='1.0.0')

    if arguments['<command>'] == 'hello':
        greet(docopt(HELLO))
    elif arguments['<command>'] == 'goodbye':
        greet(docopt(GOODBYE))
    else:
        exit("{0} is not a command. See 'options.py --help'.".format(arguments['<command>']))

if __name__ == '__main__':
    greet()
```
*******