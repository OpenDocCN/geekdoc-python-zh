# 如何使用 Argparse 编写命令行程序:示例和提示

> 原文：<https://www.pythoncentral.io/how-to-use-argparse-to-write-command-line-programs-examples-and-tips/>

命令行界面是最简单的界面，它的使用可以追溯到几十年前第一台现代计算机诞生的时候。

虽然图形用户界面已经变得很普通，但 CLI 今天仍在继续使用。它们促进了一些 Python 操作，例如系统管理、开发和数据科学。

如果你正在构建一个命令行应用程序，你还需要一个用户友好的命令行界面来促进与应用程序的交互。Python 的好处在于，其标准库中的 argparse 模块使开发人员能够构建全功能的 CLI。

在本指南中，我们将带领您使用 argparse 构建 CLI。

需要注意的是，除了理解 OOP、包和模块等概念之外，你还需要了解终端或命令行界面的基本工作原理。

## **如何使用 Argparse 编写命令行程序**

最早出现在[Python 3.2](https://docs.python.org/3/whatsnew/3.2.html#pep-389-argparse-command-line-parsing-module)中，argparse 可以接受可变数量的参数，解析命令行参数和选项。

使用该模块非常简单，只需导入它，制作一个参数解析器，将参数和选项放入图片中，然后调用。parse_args()从解析器获取参数的名称空间。

让我们看一个例子来理解 argparse 是如何工作的。

这里有一个程序列出了目录中的文件，就像 Linux 上的 ls 命令一样:

```py
# ls.py v1

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("path")

args = parser.parse_args()

target_dir = Path(args.path)

if not target_dir.exists():
    print("Target directory doesn't exist")
    raise SystemExit(1)

for entry in target_dir.iterdir():
    print(entry.name)
```

如您所见，导入模块后，我们用 ArgumentParser 类创建了一个解析器。然后，定义“path”参数，获取目标目录。

接下来，程序调用。parse_args()，解析输入参数并获取带有所有用户参数的名称空间对象。

在我们深入剖析 argparse 模块之前，您必须注意到它可以识别两种命令行参数。

第一种是位置论元，或简称为论点。之所以这样称呼它，是因为它的用途是由它在命令结构中的相对位置定义的。上例中的“path”参数是一个位置参数。

第二种是可选参数。它也被称为选项、开关或标志。这种类型的参数使您能够修改命令的工作方式。顾名思义，不需要使用它来运行命令。

### **编写一个参数解析器**

参数解析器是用 argparse 构建的任何 CLI 的基本组件，因为它处理传递的每个参数和选项。

您必须通过实例化 ArgumentParser 类来编写一个:

```py
>>> from argparse import ArgumentParser

>>> parser = ArgumentParser()
>>> parser
ArgumentParser(
    prog='',
    usage=None,
    description=None,
    formatter_class=<class 'argparse.HelpFormatter'>,
    conflict_handler='error',
    add_help=True
)
```

ArgumentParser 构造函数接受几个参数，您可以用它们来调整 CLI 的功能集。

它接受的参数是可选的，所以如果实例化不带任何参数的 ArgumentParser 类，您将得到一个简洁的解析器。

### **添加参数和选项**

向您的 CLI 添加参数和选项就像使用。您创建的 ArgumentParser 实例上的 add_argument()方法。

你添加到。add_argument()方法对比了选项和参数的区别。第一个参数称为“名称”或“标志”，这取决于您是在定义参数还是选项。

让我们给上面创建的 CLI 添加一个“-l”选项:

```py
# ls.py version2

import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("path")

parser.add_argument("-l", "--long", action="store_true") #Creating an option with flags -l and --long

args = parser.parse_args()

target_dir = Path(args.path)

if not target_dir.exists():
    print("The target directory doesn't exist")
    raise SystemExit(1)

def build_output(entry, long=False):
    if long:
        size = entry.stat().st_size
        date = datetime.datetime.fromtimestamp(
            entry.stat().st_mtime).strftime(
            "%b %d %H:%M:%S"
        )
        return f"{size:>6d} {date} {entry.name}"
    return entry.name

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

在第 11 行创建选项时(突出显示为黄色)，我们还将“action”参数设置为“store_true”这指示 Python 在命令行中提供选项时存储布尔值“True”。

如您所料，如果没有提供选项，该值将保持“False”

在程序的后面，我们使用 Path.stat()和 datetime.datetime 对象等工具来定义 build_output()函数。

当“long”为“True”时，它会产生一个详细的输出该输出包括目录中每个项目的名称、修改日期和大小。

当该值为“假”时，它生成一个基本输出。

### **解析参数和选项**

需要对提供的参数进行解析，以便您的 CLI 可以相应地使用这些操作。在前面的程序中，这发生在第一次定义“args”变量的那一行。

该语句调用. parse.args()方法。然后，它的返回值被赋给 args 变量，这是一个包含命令行提供的所有参数和选项的名称空间对象。该对象还通过点符号存储其相应的值。

这个名称空间对象在你的应用程序的主要代码中很方便——就像前面提到的程序中的 for 循环一样。

## **使用 Argparse** 设置描述、Epilog 消息和分组帮助消息

### **定义描述和结尾消息**

对您的 CLI 应用程序进行描述是一种很好的方式，可以让应用程序的功能更容易理解。结尾信息或结束语通常用来感谢用户。

你可以使用“描述”和“结尾”参数来完成这些事情。让我们继续我们的定制 ls 命令示例，看看参数是如何工作的:

```py
# ls.py version3

import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ls", # This argument sets the program's name
    description=" This program lists a directory's content",
    epilog="Thank you for using %(prog)s",
)

# ...

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

我们设置的描述将出现在帮助信息的开头。

仔细看看 epilog 参数，您会看到使用了“%”格式说明符。这可能会让您感到惊讶，但是帮助消息确实支持这种格式说明符。

由于 f 字符串说明符在运行时用值替换名称，所以它们不受支持。因此，如果你尝试将 prog 放入 epilog 并在调用 ArgumentParser 时使用 f 字符串，应用程序将抛出一个 NameError。

执行该应用程序将得到以下输出:

| $ python ls.py -h用法:ls [-h] [-l] path这个程序列出了一个目录的内容立场论点:路径选项:-h，- help 显示该帮助信息并退出-l，- long感谢您使用 ls |

### **显示分组帮助信息**

argparse 模块拥有帮助组功能，允许您将相关命令和参数组织成组，并使帮助消息更容易阅读。

创建帮助组需要使用 ArgumentParser 的. add_argument_group()方法，如下:

```py
# ls.py version4
# ...

parser = argparse.ArgumentParser(
    prog="ls", # This argument sets the program's name
    description=" This program lists a directory's content",
    epilog="Thank you for using %(prog)s",
)

general = arg_parser.add_argument_group("general output")
general.add_argument("path")

detailed = arg_parser.add_argument_group("detailed output")
detailed.add_argument("-l", "--long", action="store_true")

args = arg_parser.parse_args()

# ...

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

虽然在这样一个简单的例子中，对参数进行分组似乎是不必要的，但这样做的目的是让你改善应用程序的用户体验。

如果你的应用程序有几个参数和选项，这种帮助信息的分组方式对 UX 来说尤其有效。

使用-h 选项执行应用程序将提供以下结果:

| python ls.py -h用法:ls [-h] [-l] path这个程序列出了一个目录的内容选项:-h，- help 显示该帮助信息并退出一般输出:路径详细输出:-l，- long感谢您使用 ls |

## **改进您的命令行参数和选项**

### **设置选项后的动作**

向 CLI 添加标志或选项时，在大多数情况下，您必须定义选项的值必须如何存储在其对应的名称空间对象中。

要做到这一点，需要使用。add_argument()方法。有问题的参数存储为选项提供的值，就像它在命名空间中一样。这是因为默认情况下，参数的值为“store”。

然而，自变量可以有其他值。让我们看看 action 参数可以存储的所有值及其含义。

| **允许值** | **描述** |
| 追加 | 当提供选项时，将当前值追加到列表中。 |
| 追加常量 | 当提供选项时，将常数值追加到列表中。 |
| 计数 | 统计当前选项被提供的次数，并存储该值。 |
| 商店 | 将输入放入命名空间对象。 |
| store_const | 如果指定了选项，存储一个常数值。 |
| store_false | 默认值为“真”如果指定了选项，它将存储“False”。 |
| store_true | 默认值为“假”如果指定了选项，它将存储“True”。 |
| 版本 | 显示应用程序的版本，然后应用程序终止。 |

如果您提供任何带有“_const”后缀的值，您将需要提供常数值。这样做很简单，只需在调用。add_argument()。

出于同样的原因，您需要在对的调用中使用 version 参数向“version”操作提供应用程序的版本。add_argument()

这里有一个例子，让我们更深入地了解这些动作是如何工作的:

```py
# Toy app
# actions.py 

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name", action="store"
)  
# This option will store the passed value without any considerations

parser.add_argument("--pi", action="store_const", const=3.14) # When the option is supplied, this option will automatically store the target constant 

parser.add_argument("--is-valid", action="store_true")
# This option will store True when supplied and False otherwise

parser.add_argument("--is-invalid", action="store_false")
# This option will store False when supplied and True otherwise

parser.add_argument("--item", action="append")
# This option will help you create a list of all values, but you will need to repeat the option for every value. The argparse method will append the supplied items to a list with the same name as the option. 

parser.add_argument("--repeated", action="append_const", const=42)
# This option works similarly to the "--item" option. The only difference being it'll append the same constant value you provide with the const argument. 

parser.add_argument("--add-one", action="count")
# It will count the frequency of the option's total usage in the command line. 

parser.add_argument(
    "--version", action="version", version="%(prog)s 0.1.0"
)
# It'll show the app's version before terminating it. For this option to work, you will need to supply the version number beforehand. 

args = parser.parse_args()

print(args)
```

如你所见，我们已经在上面的程序中使用了 action 参数的所有可能值。我们还评论了每个已定义选项的功能。

当你运行它时，它将打印所有使用的动作参数的名称空间对象。

尽管 argparse 提供的一组默认动作没有缺陷，但值得注意的是，您可以通过对 argparse 进行子类化来创建自定义动作。动作类。

## **为你的命令行应用定制输入值**

有时，你的应用程序可能会要求所提供的参数接受一个字符串、值列表或其他类型的值。默认情况下，命令行将提供的参数视为字符串。

但是 argparse 模块具有允许它检查参数是否是有效的列表、字符串、整数等的机制。让我们来看看定制输入值的不同方式。

### **定义输入值的类型**

您可以使用。add_argument()方法来定义要存储在 Namespace 对象中的输入。

假设你正在开发一个将两个数相除的 CLI 应用程序。为了让它像预期的那样工作，它将接受两个选项:-被除数和-除数。

为了使这些选项正确工作，它们必须在命令行中只接受整数。你可以这样做:

```py
# divide.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dividend", type=int)# Setting "int" as the acceptable input type
parser.add_argument("--divisor", type=int) # Setting "int" as the acceptable input type

args = parser.parse_args()

print(args.dividend / args.divisor)
```

当使用选项时，此 CLI 现在将只接受整数。它将尝试将提供的值转换为整数，但如果提供了文本字符串和浮点值，将会失败。

### **接受多个输入值**

argparse 模块假设您接受每个选项和参数的单个值。要改变这种行为，可以使用“nargs”参数。

该参数向模块表明，所讨论的参数(或选项)可以接受零个或几个输入值，这取决于您分配给 nargs 的具体值。

nargs 参数可以接受以下值:

| **允许值** | **意为** |
| * | 接受零个或多个值，并将它们存储在一个列表中 |
| ？ | 接受零或一个值 |
| + | 接受一个或多个值，并将它们存储在一个列表中 |
| argparse。余数 | 收集命令行中剩余的所有值 |

让我们通过一个例子来看看允许的值是如何工作的。假设您创建了一个命令行选项“- coordinates”的 CLI，它接受两个值。如你所想，这个想法是接受笛卡尔平面的 x 和 y 坐标。

下面是代码的样子:

```py
# point.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--coordinates", nargs=2) # The option will only accept two arguments

args = parser.parse_args()

print(args)
```

程序中的最后一条语句指示 Python 打印名称空间对象。如果您为它提供两个值，您将看到如下预期结果:

| $ python point . py-coordinates 2 3名称空间(坐标=['2 '，' 3']) |

但是如果提供了零个、一个或两个以上的参数，程序将抛出一个错误。

现在，我们来看一个使用 nargs 的*值的例子。假设您构建了一个接受数字并返回其总和的 CLI 应用程序。它看起来像这样:

```py
# sum.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("numbers", nargs="*", type=float)# The numbers argument will only accept floating point numbers 

args = parser.parse_args()

print(sum(args.numbers))
```

因为我们已经使用了*值，所以不管你传递了多少个值(或者没有传递任何值)，程序都将返回总和。

相比之下，将+值赋给 nargs 只会强制参数接受最少一个值。下面是一个 CLI 应用程序的例子，它接受一个或多个文件并打印它们的名称。

```py
# files.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("files", nargs="+") # The files argument needs a minimum of one value

args = parser.parse_args()

print(args)
```

如果你给这个程序提供一个或多个文件，它将返回一个带有文件名的名称空间，比如:

| $ python files . py hello world . txt命名空间(files=['helloWorld.txt']) |

但是，如果没有提供文件，它将抛出如下错误:

| $ python files.py用法:files.py [-h] files [files...]files.py: error:需要以下参数:files |

余数值使 nargs 能够在命令行捕获剩余的输入值。让我们来看看它是如何工作的:

```py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('value')
parser.add_argument('remaining', nargs=argparse.REMAINDER)

args = parser.parse_args()

print(f'Initial value: {args.value}')
print(f'Leftover values: {args.remaining}')
```

向这个程序传递一些字符串将导致第一个字符串被第一个参数捕获。第二个参数将捕获剩余的字符串。下面是运行它时的输出结果:

| > python nargs.py 你好读者第一个值:良好其他值:['天'，'到'，'你'，'读者'] |

nargs 参数提供了很大的灵活性，但是当你的应用程序有多个命令行选项和参数时，它可能很难使用。

当涉及到不同的 nargs 值时，您可能很难组合选项和参数。让我们用一个例子来看看如何:

```py
# cooking.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("vegetables", nargs="+")
parser.add_argument("spices", nargs="*")

args = parser.parse_args()

print(args)
```

你会期望“蔬菜”接受一个或多个项目，而“香料”接受零个或多个项目。但这并不是你运行程序时会发生的:

| $ python cooking.py 茄子辣椒黄瓜名称空间(蔬菜=['茄子'，'辣椒'，'黄瓜']，调料=[]) |

我们得到这个输出，因为解析器没有任何方法来确定哪个值需要分配给哪个参数或选项。

在本例中，您可以通过将两个参数都转换为选项来解决这个问题。这样做很简单，只需在参数后面添加两个连字符。

然后，您可以运行以下命令来获得正确的输出:

| $ python cooking.py -蔬菜茄子黄瓜-香料辣椒名称空间(蔬菜=['茄子'，'黄瓜']，香料=['辣椒']) |

这个例子表明，在用 nargs 组合参数和选项集时必须小心。

### **设置默认值**

将“默认”参数与。add_argument()方法，可以为参数和选项提供适当的默认值。

当您的目标选项或参数需要一个有效值时，即使用户没有提供输入，使用该参数也会非常有用。

让我们回到我们在这篇文章前面写的自定义 ls 命令。假设应用程序现在需要让命令列表显示当前目录的内容，如果用户没有提供目标目录的话。

下面是这个程序的样子:

```py
# ls.py version5

import argparse
import datetime
from pathlib import Path

# ...

general = parser.add_argument_group("general output")
general.add_argument("path", nargs="?", default=".")

# ...
```

如你所见，我们已经将“默认”参数设置为“.”字符串，表示当前目录。另外，nargs 被设置为“？”它删除了对输入值的约束，并且只接受单个值(如果提供了任何值的话)。

运行程序，看看它是如何工作的！

### **提及允许输入值的列表**

argparse 模块提供了提供值列表的可能性，并且只允许这些值与特定的参数和选项一起工作。

该模块通过您可以提供给。add_argument()方法。您需要为此参数提供您的可接受值列表。

假设您正在构建一个需要接受 t 恤尺寸的 CLI 应用程序。这里有一个程序将定义一个"- size "选项和一些可接受的值:

```py
# size.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--size", choices=["S", "M", "L", "XL"], default="M")

args = parser.parse_args()

print(args)
```

如您所见，我们使用了“choices”参数来提供可接受值的列表。如您所料，CLI 应用程序将只接受这些值。

如果用户试图提供一个不在列表中的值，会发生什么情况:

| $ python choices.py - size A用法:choices.py [-h] [ - size {S，M，L，XL}]choices . py:error:argument-size:无效选择:' A'(从‘S’、‘M’、‘L’、‘XL’中选择) |

但是如果提供了一个有效值，程序将打印名称空间对象，就像前面讨论的例子一样。

更有趣的是，“choices”参数可以接受不同数据类型的值。因此，如果您需要接受整数值，您可以定义一个可接受值的范围。

让我们看看如何使用 range()来做到这一点:

```py
# weekdays.py

import argparse

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--weekday", type=int, choices=range(1, 8))

args = my_parser.parse_args()

print(args)
```

这个应用程序将自动检查用户以“选择”参数形式提供给命令行的 range 对象的任何值。

如果提供的数字超出范围，应用程序将抛出如下错误:

| $ python days.py - weekday 9用法:days.py [-h] [ - weekday {1，2，3，4，5，6，7}]days.py:错误:参数-工作日:无效选择:9(从 1、2、3、4、5、6、7 中选择) |

### **在 Argparse 中创建和定制帮助消息**

argparse 模块最好的特性之一是为您的应用程序生成帮助消息和自动使用消息。

用户可以使用-h 或- help 标志来访问这些消息。默认情况下，这些标志集成到所有 argparse CLIs 中。

在本文的前几部分，你看到了如何将描述和结束消息集成到你的 CLI 应用程序中。

现在，让我们继续我们的定制 ls 命令示例，看看如何通过“help”和“metavar”参数为各个参数和选项提供增强的消息。

```py
# ls.py version6

import argparse
import datetime
from pathlib import Path

# ...

general = parser.add_argument_group("general output")
general.add_argument(
    "path",
    nargs="?",
    default=".",
    help="take the path to the target directory (default: %(default)s)",
)

detailed = parser.add_argument_group("detailed output")
detailed.add_argument(
    "-l",
    "--long",
    action="store_true",
    help="display detailed directory content",
)

# ...
```

我们已经讨论过像%(prog)这样的格式说明符在 argparse 中没有任何问题。但是在这个阶段，同样值得注意的是，您可以使用 add_argument()的大部分参数作为格式说明符。这些包括但不限于%(类型)和%(默认)

执行我们现在重新设计的应用程序会产生以下输出:

| $ python ls.py -h用法:ls [-h] [-l] [path]这个程序列出了一个目录的内容选项:-h，- help 显示该帮助信息并退出一般输出:path 取目标目录的路径(默认:。)详细输出:-l，- long 显示详细目录内容感谢您使用 ls |

正如你所看到的，当运行带有-h 标志的应用程序时，现在-l 和 path 都会显示描述性的帮助消息。尽管“path”在它的帮助信息中有默认值，但它仍然是有用的。

argparse 中的默认用法消息已经足够好了，但是如果您愿意，可以使用 metavar 参数对其进行改进。

当选项或参数接受输入值时，参数变得特别方便。您可以为解析器用来生成帮助消息的输入值指定描述性名称。

让我们后退一步，用-h 开关运行前几节中的 point.py 示例。下面是输出:

| $ python point.py -h用法:point . py[-h][-COORDINATES COORDINATES]选项:-h，- help 显示此帮助信息并退出-坐标坐标坐标 |

argparse 模块使用选项的原始名称在帮助和用法消息中指定相应的输入值。

但是在上面的输出中，“坐标”出现了两次，这可能会使用户误以为坐标需要提供两次。

你可以这样处理这种不确定性:

```py
# point.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--coordinates",
    nargs=2,
    metavar=("X", "Y"),
    help="take the Cartesian coordinates %(metavar)s",
)

args = parser.parse_args()

print(args)
```

在上面的代码中，一个具有两个坐标名的元组值被赋给了 metavar。另外，我们还为适当的选项添加了自定义消息。

运行上面的代码将得到以下输出:

| $ python coordinates.py -h用法:coordinates . py[-h][-coordinates X Y]选项:-h，- help 显示此帮助信息并退出-坐标 X Y 取笛卡尔坐标(' X '，' Y') |

## **处理您的 CLI 应用程序的执行如何终止**

CLI 应用程序必须在特定情况下终止，例如出现错误和异常时。

处理异常和错误的最常见方式是退出应用程序并显示退出状态或错误代码。状态或代码向操作系统或其他应用程序指示应用程序由于某些执行错误而终止。

一般来说，当一个命令以零代码退出时，它在终止前成功地完成了它的任务。另一方面，非零代码表示命令未能完成其任务。

这个表示成功的系统非常简单。但是，这使得指示命令失败的任务变得复杂。退出状态和错误代码没有明确的标准。

操作系统或编程语言可能使用简单的小数、十六进制、字母数字串或完整短语来描述错误。

Python 中使用整数值来描述 CLI 应用程序的系统退出状态。如果返回“无”退出状态，则意味着退出状态为零，终止成功。

如你所料，非零值表示异常终止。大多数系统需要范围在 0 到 127 之间的退出代码。如果该值超出范围，则结果未定义。

使用 argparse 构建 CLI 应用程序时，您不必考虑返回成功操作的退出代码和命令语法错误。

然而，当应用程序由于其他错误而突然终止时，您需要返回适当的退出代码。

ArgumentParser 类提供了两种在出错时终止应用程序的方法。

你可以使用。exit(status=0，message=None)方法，该方法结束应用程序并返回指定的状态和消息。或者，您可以使用。error(message)方法，该方法打印提供的消息，并使用状态代码“2”终止应用程序

无论您使用哪种方法，状态都将打印到标准错误流中，这是一个用于错误报告的专用流。

使用。当您想自己指定状态代码时，exit()方法是正确的方法。您可以使用。error()方法在任何情况下。

让我们稍微编辑一下我们的自定义 ls 命令程序，看看如何退出:

```py
# ls.py version7

import argparse
import datetime
from pathlib import Path

# ...

target_dir = Path(args.path)

if not target_dir.exists():
    parser.exit(1, message="The target directory doesn't exist")

# ...
```

代码末尾的条件语句检查目标字典是否存在。如您所见，我们没有使用“raise SystemExit(1)”，而是使用了“ArgumentParser.exit()”

这个简单的决定使得代码更加关注 argparse 框架。

当你运行程序时，你会发现当目标目录不存在时，程序就会终止。

如果您使用的是 Linux 或 macOS，您可以检查$?Shell 变量，并确认您的应用程序已返回“1”，表示执行中有错误。在 Windows 上，您需要检查$LASTEXITCODE 变量的内容。

在您构建的 CLI 应用程序中保持状态代码的一致性是一个很好的方法，可以确保您和您的用户轻松地将您的应用程序集成到命令管道和 shell 脚本中。