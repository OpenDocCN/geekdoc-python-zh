# 下载教程

> 原文：<https://www.pythonforbeginners.com/argparse/argparse-tutorial>

## 这是什么？

命令行选项、参数和子命令的解析器

## 为什么要用？

argparse 模块使得编写用户友好的命令行界面变得容易。

## 它是怎么做到的？

该程序定义了它需要什么参数，argparse 将计算出如何从 sys.argv 中解析这些参数，argparse 模块还会自动生成帮助和用法消息，并在用户向程序提供无效参数时发出错误消息。

## 概念

当您在没有任何选项的情况下运行“ls”命令时，它将默认显示当前目录的内容。如果您在当前所在的不同目录上运行“ls”，您应该键入“ls directory_name”。

“目录名”是一个“位置参数”，这意味着程序知道如何处理这个值。

要获得关于文件的更多信息，我们可以使用“-l”开关。

“-l”被称为“可选参数”。如果您想显示 ls 命令的帮助文本，可以键入“ls–help”

## 抱怨吗

要开始使用 argparse 模块，我们首先必须导入它。

```py
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()

```

#### 运行代码

使用–help 选项运行代码(不使用任何选项运行脚本将不会在 stdout 中显示任何内容)

```py
python program.py --help (or python program.py -h) 
usage: program.py [-h]

optional arguments:
  -h, --help  show this help message and exit 
```

正如上面所看到的，即使我们没有在脚本中指定任何帮助参数，它仍然给了我们一个很好的帮助信息。这是我们唯一免费的选择。

## 位置参数

在上面的“ls”示例中，我们使用了位置参数“ls directory_name”。每当我们想要指定程序将接受哪个命令行选项时，我们使用“add_argument()”方法。

```py
parser.add_argument("echo") 	# naming it "echo"
args = parser.parse_args()	# returns data from the options specified (echo)
print(args.echo) 
```

如果我们现在运行代码，我们可以看到它要求我们指定一个选项

```py
$ python program.py

usage: program.py [-h] echo
program.py: error: too few arguments 
```

当我们指定回声选项时，它将显示“回声”

```py
$ python program.py echo
echo

#Using the --help option
$ python program.py --help
usage: program.py [-h] echo

positional arguments:
  echo

optional arguments:
  -h, --help  show this help message and exit 
```

## 扩展帮助文本

为了获得关于我们的位置论点(echo)的更多帮助，我们必须改变我们的脚本。

```py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo", help="echo the string you use here")
args = parser.parse_args()
print(args.echo) 
```

导致以下结果:

```py
$ python program.py --help
usage: program.py [-h] echo

positional arguments:
  echo        echo the string you use here

optional arguments:
  -h, --help  show this help message and exit 
```

注意:Argparse 将我们给出的选项视为一个字符串，但是我们可以改变它。

## 运行类型设置为整数的代码

这段代码将把输入视为一个整数。

```py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=int)
args = parser.parse_args()
print(args.square**2) 
```

如果我们使用–help 选项运行程序，我们可以看到:

```py
$ python program.py -h
usage: program.py [-h] square

positional arguments:
  square      display a square of a given number

optional arguments:
  -h, --help  show this help message and exit 
```

## 运行程序

从帮助文本中，我们可以看到，如果我们给程序一个数字，它会把方块还给我们。

酷，让我们试试吧:

```py
$ python program.py 4
16

$ python program.py 10
100 
```

如果我们使用字符串而不是数字，程序将返回一个错误

```py
$ python program.py four

usage: program.py [-h] square

program.py: error: argument square: invalid int value: 'four' 
```

## 可选参数

在上面的“ls”示例中，我们使用了可选参数“-l”来获取关于文件的更多信息。当指定了–verbosity 时，下面的程序将显示一些内容，而当没有指定时，则不显示任何内容。

```py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on") 
```

可选参数(或选项)在不使用时(默认情况下)被赋予 None 值。使用–verbosity 选项，实际上只有两个值是有用的，True 或 False。

关键字“action”被赋予值“store_true ”,这意味着如果指定了该选项，则将值“true”赋给 args，verbose 不指定该选项意味着 False。

如果我们使用–help 选项运行程序，我们可以看到:

```py
$ python program.py -h
usage: program.py [-h] [--verbose]

optional arguments:
  -h, --help  show this help message and exit
  --verbose   increase output verbosity 
```

使用–verbose 选项运行程序

```py
$ python program.py --verbose
verbosity turned on 
```

## 排序选项

使用这些选项的简短版本非常简单:

```py
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true") 
```

帮助文本将用简短版本更新。

##### 来源

```py
 [http://docs.python.org/dev/library/argparse.html](https://docs.python.org/dev/library/argparse.html "argparse") 
```