# 如何使用 argparse 创建命令行应用程序

> 原文：<https://www.blog.pythonlibrary.org/2022/05/19/how-to-create-a-command-line-application-with-argparse/>

当你创建一个应用程序时，你通常会希望能够告诉你的应用程序如何去做一些事情。有两种流行的方法来完成这项任务。您可以让您的应用程序接受命令行参数，也可以创建一个图形用户界面。有些应用程序两者都支持。

当您需要在服务器上运行代码时，命令行界面非常有用。大多数服务器没有连接监视器，尤其是如果它们是 Linux 服务器。在这些情况下，即使您想运行图形用户界面，也可能无法运行。

Python 附带了一个名为`argparse`的内置库，可以用来创建命令行界面。在本文中，您将了解以下内容:

*   Parsing Arguments
*   Creating Helpful Messages
*   Adding Aliases
*   Using Mutually Exclusive Arguments
*   Creating a Simple Search Utility

`argparse`模块的内容远不止本文将介绍的内容。如果你想了解更多，你可以查看一下[文档。](https://docs.python.org/3/library/argparse.html)

现在是时候开始从命令行解析参数了！

## 解析参数

在学习如何使用`argparse`之前，最好知道还有另一种方法可以将参数传递给 Python 脚本。您可以向 Python 脚本传递任何参数，并通过使用`sys`模块来访问这些参数。

要查看它是如何工作的，创建一个名为`sys_args.py`的文件，并在其中输入以下代码:

```py
# sys_args.py

import sys

def main():
    print('You passed the following arguments:')
    print(sys.argv)

if __name__ == '__main__':
    main()
```

这段代码导入`sys`并打印出`sys.argv`中的内容。`argv`属性包含传递给脚本的所有内容的列表，第一项是脚本本身。

下面是一个示例，说明了在运行这段代码和几个示例参数时会发生什么:

```py
$ python3 sys_args.py --s 45
You passed the following arguments:
['sys_args.py', '--s', '45']
```

使用`sys.argv`的问题是您无法控制可以传递给应用程序的参数:

*   You can't ignore arguments
*   You can't create default arguments
*   You can't really tell what is a valid argument at all

这就是为什么在使用 Python 的标准库时使用`argparse`是正确的方法。`argparse`模块非常强大和有用。让我们考虑命令行应用程序遵循的一个常见过程:

*   **pass** in a file
*   **do** something to that file in your program
*   **output** the result

这是一个如何工作的一般例子。继续创建`file_parser.py`并添加以下代码:

```py
# file_parser.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser('File parser')
    parser.add_argument('--infile', help='Input file')
    parser.add_argument('--out', help='Output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

`file_parser()`函数是解析的逻辑所在。对于这个例子，它只接受一个文件名并打印出来。`output_file`参数默认为空字符串。

尽管如此，这个项目的核心仍在`main()`中。在这里，您创建了一个`argparse.ArgumentParser()`的实例，并为您的解析器命名。然后你添加两个参数，`--infile`和`--out`。要使用解析器，您需要调用`parse_args()`，它将返回传递给程序的任何有效参数。最后，检查用户是否使用了`--infile`标志。如果是，那么你运行`file_parser()`。

以下是您在终端中运行代码的方式:

```py
$ python file_parser.py --infile something.txt
Processing something.txt
Finished processing
```

在这里，您使用带有文件名的`--infile`标志运行您的脚本。这将运行`main()`，而 T1 又调用`file_parser()`。

下一步是使用您在代码中声明的两个命令行参数来尝试您的应用程序:

```py
$ python file_parser.py --infile something.txt --out output.txt
Processing something.txt
Finished processing
Creating output.txt
```

这一次，您会得到一行额外的输出，其中提到了输出文件名。这代表了代码逻辑中的一个分支。当指定输出文件时，可以使用新的代码块或函数让代码经历生成该文件的过程。如果不指定输出文件，那么该代码块将不会运行。

当你使用`argparse`创建你的命令行工具时，当你的用户不确定如何正确地与你的程序交互时，你可以很容易地添加信息来帮助他们。

现在是时候了解如何从您的应用程序中获得帮助了！

## 创建有用的信息

`argparse`库将使用您在创建每个参数时提供的信息，自动为您的应用程序创建一条有用的消息。下面是您的代码:

```py
# file_parser.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser('File parser')
    parser.add_argument('--infile', help='Input file')
    parser.add_argument('--out', help='Output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

现在尝试运行带有`-h`标志的代码，您应该看到以下内容:

```py
$ file_parser.py -h
usage: File parser [-h] [--infile INFILE] [--out OUT]

optional arguments:
  -h, --help       show this help message and exit
  --infile INFILE  Input file
  --out OUT        Output file
```

`add_argument()`的`help`参数用于创建上面的帮助消息。`-h`和`--help`选项由`argparse`自动添加。你可以通过给你的帮助加上一个`description`和一个`epilog`来增加它的信息量。

让我们使用它们来改进您的帮助信息。首先将上面的代码复制到一个名为`file_parser_with_description.py`的新文件中，然后将其修改为如下所示:

```py
# file_parser_with_description.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser(
            'File parser',
            description='PyParse - The File Processor',
            epilog='Thank you for choosing PyParse!',
            )
    parser.add_argument('--infile', help='Input file for conversion')
    parser.add_argument('--out', help='Converted output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

在这里，您将`description`和`epilog`参数传递给`ArgumentParser`。您还可以将`help`参数更新为`add_argument()`，使其更具描述性。

在进行这些更改后，当您使用`-h`或`--help`运行这个脚本时，您将看到以下输出:

```py
$ python file_parser_with_description.py -h
usage: File parser [-h] [--infile INFILE] [--out OUT]

PyParse - The File Processor

optional arguments:
  -h, --help       show this help message and exit
  --infile INFILE  Input file for conversion
  --out OUT        Converted output file

Thank you for choosing PyParse!
```

现在，您可以在帮助输出中看到新的描述和附录。这让您的命令行应用程序更加完美。

您也可以通过`add_help`到`ArgumentParser`的参数来完全禁用应用程序中的帮助。如果您认为您的帮助文本过于冗长，您可以像这样禁用它:

```py
# file_parser_no_help.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser(
            'File parser',
            description='PyParse - The File Processor',
            epilog='Thank you for choosing PyParse!',
            add_help=False,
            )
    parser.add_argument('--infile', help='Input file for conversion')
    parser.add_argument('--out', help='Converted output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

通过将`add_help`设置为`False`，可以禁用`-h`和`--help`标志。

You can see this demonstrated below:

```py
$ python file_parser_no_help.py --help
usage: File parser [--infile INFILE] [--out OUT]
File parser: error: unrecognized arguments: --help
```

In the next section, you'll learn about adding aliases to your arguments!

## Adding Aliases

An alias is a fancy word for using an alternate flag that does the same thing. For example, you learned that you can use both `-h` and `--help` to access your program's help message. `-h` is an alias for `--help`, and vice-versa

Look for the changes in the `parser.add_argument()` methods inside of `main()`:

```py
# file_parser_aliases.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser(
            'File parser',
            description='PyParse - The File Processor',
            epilog='Thank you for choosing PyParse!',
            add_help=False,
            )
    parser.add_argument('-i', '--infile', help='Input file for conversion')
    parser.add_argument('-o', '--out', help='Converted output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

Here you change the first `add_argument()` to accept `-i` in addition to `--infile` and you also added `-o` to the second `add_argument()`. This allows you to run your code using two new shortcut flags.

Here's an example:

```py
$ python3 file_parser_aliases.py -i something.txt -o output.txt
Processing something.txt
Finished processing
Creating output.txt
```

If you go looking through the `argparse` documentation, you will find that you can add aliases to subparsers too. A subparser is a way to create sub-commands in your application so that it can do other things. A good example is Docker, a virtualization or container application. It has a series of commands that you can run under `docker` as well as `docker compose` and more. Each of these commands has separate sub-commands that you can use.

Here is a typical docker command to run a container:

```py
docker exec -it container_name bash
```

This will launch a container with docker. Whereas if you were to use `docker compose`, you would use a different set of commands. The `exec` and `compose` are examples of subparsers.

The topic of subparsers are outside the scope of this tutorial. If you are interested in more details dive right into the [documentation.](https://docs.python.org/3/library/argparse.html#sub-commands)

## Using Mutually Exclusive Arguments

Sometimes you need to have your application accept some arguments but not others. For example, you might want to limit your application so that it can only create *or* delete files, but not both at once.

The `argparse` module provides the `add_mutually_exclusive_group()` method that does just that!

Change your two arguments to be mutually exclusive by adding them to a `group` object like in the example below:

```py
# file_parser_exclusive.py

import argparse

def file_parser(input_file, output_file=''):
    print(f'Processing {input_file}')
    print('Finished processing')
    if output_file:
        print(f'Creating {output_file}')

def main():
    parser = argparse.ArgumentParser(
            'File parser',
            description='PyParse - The File Processor',
            epilog='Thank you for choosing PyParse!',
            add_help=False,
            )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--infile', help='Input file for conversion')
    group.add_argument('-o', '--out', help='Converted output file')
    args = parser.parse_args()
    if args.infile:
        file_parser(args.infile, args.out)

if __name__ == '__main__':
    main()
```

First, you created a mutually exclusive group. Then, you added the `-i` and `-o` arguments to the group instead of to the `parser` object. Now these two arguments are mutually exclusive.

Here is what happens when you try to run your code with both arguments:

```py
$ python3 file_parser_exclusive.py -i something.txt -o output.txt
usage: File parser [-i INFILE | -o OUT]
File parser: error: argument -o/--out: not allowed with argument -i/--infile
```

Running your code with both arguments causes your parser to show the user an error message that explains what they did wrong.

After covering all this information related to using `argparse`, you are ready to apply your new skills to create a simple search tool!

## Creating a Simple Search Utility

Before starting to create an application, it is always good to figure out what you are trying to accomplish. The application you want to build in this section should be able to search for files of a specific file type. To make it more interesting, you can add an additional argument that allows you to optionally search for specific file sizes as well.

You can use Python's `glob` module for searching for file types. You can read all about this module here:

*   [https://docs.python.org/3/library/glob.html](https://docs.python.org/3/library/glob.html)

There is also the `fnmatch` module, which `glob` itself uses. You should use `glob` for now as it is easier to use, but if you're interested in writing something more specialized, then `fnmatch` may be what you are looking for.

However, since you want to be able to optionally filter the files returned by the file size, you can use `pathlib` which includes a `glob`-like interface. The `glob` module itself does not provide file size information.

You can start by creating a file named `pysearch.py` and entering the following code:

```py
# pysearch.py

import argparse
import pathlib

def search_folder(path, extension, file_size=None):
    """
    Search folder for files
    """
    folder = pathlib.Path(path)
    files = list(folder.rglob(f'*.{extension}'))

    if not files:
        print(f'No files found with {extension=}')
        return

    if file_size is not None:
        files = [
                f
                for f in files
                if f.stat().st_size >= file_size
                ]

    print(f'{len(files)} *.{extension} files found:')
    for file_path in files:
        print(file_path)
```

You start the code snippet above by importing `argparse` and `pathlib`. Next you create the `search_folder()` function which takes in three arguments:

*   `path` - The folder to search within
*   `extension` - The file extension to look for
*   `file_size` - What file size to filter on in bytes

You turn the `path` into a `pathlib.Path` object and then use its `rglob()` method to search in the folder for the extension that the user passed in. If no files are found, you print out a meaningful message to the user and exit.

If any files are found, you check to see whether `file_size` has been set. If it was set, you use a list comprehension to filter out the files that are smaller than the specified `file_size`.

Next, you print out the number of files that were found and finally loop over these files to print out their names.

To make this all work correctly, you need to create a command-line interface. You can do that by adding a `main()` function that contains your `argparse` code like this:

```py
def main():
    parser = argparse.ArgumentParser(
            'PySearch',
            description='PySearch - The Python Powered File Searcher',
            )
    parser.add_argument('-p', '--path',
                        help='The path to search for files',
                        required=True,
                        dest='path')
    parser.add_argument('-e', '--ext',
                        help='The extension to search for',
                        required=True,
                        dest='extension')
    parser.add_argument('-s', '--size',
                        help='The file size to filter on in bytes',
                        type=int,
                        dest='size',
                        default=None)

    args = parser.parse_args()
    search_folder(args.path, args.extension, args.size)

if __name__ == '__main__':
    main()
```

This `ArgumentParser()` has three arguments added to it that correspond to the arguments that you pass to `search_folder()`. You make the `--path` and `--ext` arguments required while leaving the `--size` argument optional. Note that the `--size` argument is set to `type=int`, which means that you cannot pass it a string.

There is a new argument to the `add_argument()` function. It is the `dest` argument which you use to tell your argument parser where to save the arguments that are passed to them.

Here is an example run of the script:

```py
$ python3 pysearch.py -p /Users/michael/Dropbox/python101code/chapter32_argparse -e py -s 650
6 *.py files found:
/Users/michael/Dropbox/python101code/chapter32_argparse/file_parser_aliases2.py
/Users/michael/Dropbox/python101code/chapter32_argparse/pysearch.py
/Users/michael/Dropbox/python101code/chapter32_argparse/file_parser_aliases.py
/Users/michael/Dropbox/python101code/chapter32_argparse/file_parser_with_description.py
/Users/michael/Dropbox/python101code/chapter32_argparse/file_parser_exclusive.py
/Users/michael/Dropbox/python101code/chapter32_argparse/file_parser_no_help.py
```

That worked quite well! Now try running it with `-s` and a string:

```py
$ python3 pysearch.py -p /Users/michael/Dropbox/python101code/chapter32_argparse -e py -s python
usage: PySearch [-h] -p PATH -e EXTENSION [-s SIZE]
PySearch: error: argument -s/--size: invalid int value: 'python'
```

This time, you received an error because `-s` and `--size` only accept integers. Go try this code on your own machine and see if it works the way you want when you use `-s` with an integer.

Here are some ideas you can use to improve your version of the code:

*   Handle the extensions better. Right now it will accept `*.py` which won't work the way you might expect
*   Update the code so you can search for multiple extensions at once
*   Update the code to filter on a range of file sizes (Ex. 1 MB - 5MB)

There are lots of other features and enhancements you can add to this code, such as adding error handling or unittests.

## Wrapping Up

The `argparse` module is full featured and can be used to create great, flexible command-line applications. In this chapter, you learned about the following:

*   Parsing Arguments
*   Creating Helpful Messages
*   Adding Aliases
*   Using Mutually Exclusive Arguments
*   Creating a Simple Search Utility

You can do a lot more with the `argparse` module than what was covered in this chapter. Be sure to check out the documentation for full details. Now go ahead and give it a try yourself. You will find that once you get the hang of using `argparse`, you can create some really neat applications!