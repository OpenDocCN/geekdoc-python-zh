# Python 中何时使用注释与文档字符串

> 原文：<https://www.pythonforbeginners.com/comments/when-to-use-comments-vs-docstrings-in-python>

给你的程序添加注释将有助于其他程序员理解你的工作。没有注释，理解别人的代码可能会很困难。这就是为什么作为 Python 程序员，知道什么时候使用注释而不是文档字符串是很重要的。

知道何时使用注释以及如何使用注释可能会令人困惑。你知道文档字符串和字符串注释的区别吗？

在本教程中，我们将介绍用 Python 编写注释的最佳实践。我们还将向您展示如何使用 docstrings，并解释它们与普通注释的不同之处。希望我们能够揭开 Python 代码文档化过程的神秘面纱。

到本文结束时，您应该已经牢牢掌握了在 Python 中什么时候使用注释，什么时候使用文档字符串。

## 使用 Python 块注释

用 Python 做注释有多种方式。这对初学者来说可能有点困惑。理解块注释和字符串注释之间的区别会使情况变得更加清晰。

在 Python 中，大部分注释都是使用*块注释*编写的。块注释前面有一个 **#** 符号。Python 会在程序运行时忽略这些语句。使用块注释来提供简短的解释性注释。

#### 示例 1:如何用 Python 编写块注释

```py
# this is a Python block comment 
```

也可以在 Python 语句的末尾写一个注释。这些注释被称为*行内注释*。内联注释非常适合解释不明显的代码。

#### 示例 Python 中的行内注释

```py
offset = width + 1 # compensate for the screen border
```

行内注释非常有用，但是最好少用。在这里，少即是多是一个很好的规则。

## 使用 Python 字符串注释

块注释只占一行。如果你需要一个跨越多行的注释，你需要使用更多的块注释。如果你有一个很长的注释，或者需要注释掉一段代码，你可以使用*字符串注释*。字符串注释是用引号写的。

#### 示例 3:如何用 Python 编写字符串注释

```py
"""
This is an example of a string comment.
String comments can be many lines long.

Python will ignore them when you run the program.

""" 
```

有了字符串注释，你的注释长度没有限制。但是需要谨慎。如果将字符串注释放在错误的位置，可能会被误认为是文档字符串。

例如，函数后面的字符串注释将被解释为 docstring。由于这个原因，一些 Python 程序员根本不使用字符串注释。

### 用 Python 字符串注释有错吗？

一些程序员不赞成使用字符串作为 Python 注释。因为它们可能与 docstrings 混淆，一些 Python 程序员甚至认为最好完全避免它们。

但是官方的说法是使用字符串注释完全没问题。甚至 Python 的创造者 Guido Van Rossum 也认可了这种技术。

## 何时使用 Python 注释

使用注释为他人和自己留下关于代码的注释。注释有多种用途，但它们通常是为开发人员准备的。

与 docstrings 不同，注释不会被转换成文档。写评论没有规则，只有准则。如果你环顾四周，你会发现关于什么时候和什么时候不做评论有不同的观点。

一般来说，大多数程序员会告诉你，在下列情况下，注释是必要的:

*   当你需要解释复杂的算法时。
*   如果您想留下关于未完成代码的待办事项或注释。
*   如果你需要警告。
*   当您想要包含可选代码时。

以下是你在评论时要记住的一些小技巧:

*   尽可能保持评论简短。
*   永远尊重和帮助他人。
*   用评论来解释**为什么**做出决定。
*   使用注释来解释**某事如何**工作。

## 什么是 Python Docstring？

**Docstrings** 代表*文档字符串*，它们提供了一种记录 Python 程序的方法。使用 docstring，程序可以提供其他程序员可能想要使用的函数、类和模块的描述。

使用 docstrings，Python 开发人员可以简单解释函数或类的用途。没有这样的文档，学习新的 Python 模块将非常困难——如果不是不可能的话。

docstring 也可以用来生成 API(**A**应用 **P** 编程 **I** 接口)。API 用于通过网络发送数据。要使用 API，有必要查阅文档并了解它是如何工作的。Docstrings 允许程序将文档直接包含在代码中，从而使生成文档变得更加容易。

## 如何在 Python 中使用 Docstrings

既然我们已经在 Python 中区分了文档字符串和注释，那么让我们更仔细地看看文档字符串是如何工作的以及如何使用它们。

**如何声明一个 docstring** : 使用三重引号创建一个新的 docstring。因此，文档字符串可能会与字符串注释混淆。

**如何访问文档字符串**:可以使用 __doc__ 属性读取文档字符串。

## Docstring 应该是什么样子？

与注释不同，docstrings 有一些正式的要求。首先，每个 docstring 都以与其相关的函数或类的简短摘要开始。

总结应该清晰简洁。程序员立即掌握一个函数做什么是很重要的。然而，这有时说起来容易做起来难，但应该永远是目标。

可以在摘要下面的空行中提供进一步的文档。docstring 的细节应该作为函数和参数的快速参考。

## 记录 Python 函数

在用 Python 编写函数文档时，一定要包括函数参数的描述。此外，给出函数返回值的详细解释，如果有的话。

#### 示例 4:如何用 Python 编写函数文档

```py
def get_neighbors(some_list, index):
    """Returns the neighbors of a given index in a list.

        Parameters:
            some_list (list): Any basic list.
            index (int): Position of an item in some_list.

        Returns:
            neighbors (list): A list of the elements beside the index in some_list.

    """
    neighbors = []

    if index - 1 >= 0:
        neighbors.append(some_list[index-1])
    if index < len(some_list):
        neighbors.append(some_list[index+1])

    return neighbors

some_list = [8,7,"car","banana",10]

print(get_neighbors(some_list, 2)) 
```

**输出**

```py
[7, 'banana']
```

## 在 Python 中访问文档字符串

要在 Python 中访问 docstring，请使用 **__doc__** 属性。这是一个特殊的属性，将检索 Python 函数或类的 docstring。我们可以使用 __doc__ 属性从命令行打印 docstring。

#### 示例 5:使用 __doc__ 属性访问文档字符串

```py
print(get_neighbors.__doc__)
```

**输出**

```py
Returns the neighbors of a given index in a list.

        Parameters:
            some_list (list): Any basic list.
            index (int): Position of an item in some_list.

        Returns:
            neighbors (list): A list of the neighboring elements of the index in some_list. 
```

## 用文档字符串记录 Python 类

在用 Python 编写类的文档时，包括类所代表的内容的摘要以及对类的属性和方法的解释是很重要的。

该类中的每个方法也需要一个 docstring。下面，你会发现一个博客文章类的例子。Python 文档字符串用于提供该类及其方法的详细描述。

#### 示例 5:如何用 Python 编写类的文档

```py
class BlogPost:
    """
    A generic blog post model.

    ...

    Attributes
    ----------
    author: str
        who wrote the blog post
    title: str
        the title of the blog post
    content: str
        the body content of the blog post

    Methods
    ----------
    description():
        Prints the author, title, and content of the blog post

    """

    def __init__(self, author, title, content):
        """Constructs all the necessary attributes of the blog post object.

        Parameters
        ----------
            author: str
                who wrote the blog post
            title: str
                the title of the blog post
            content: str
                the body content of the blog post
        """

        self.author = author
        self.title = title
        self.content = content

    def description(self):
        """Prints the author, title, and content of the blog post.

        Returns
        -------
        None
        """

        print(f'{self.author}\n{self.title}\n{self.content}') 
```

当谈到用于编写文档字符串的样式选择时，程序员会有所不同。稍加研究，你可以找到许多风格指南。选择如何格式化 docstring 将取决于许多因素，其中最不重要的可能是个人选择。记住你的文档是针对谁的。

## 摘要

知道在 Python 中什么时候使用注释，什么时候使用文档字符串，对于构建健壮、复杂的程序至关重要。学习如何以及何时发表评论不仅有助于你的合作，还能节省你的时间。

与字符串注释不同，文档字符串是供公众阅读的。它们提供了对如何使用函数、类和模块的深入了解。这使得其他程序员更容易学习如何在他们自己的项目中使用该模块。

#### 快速回顾一下注释与文档字符串:

*   使用注释来解释代码是如何工作的。
*   注释对于为你的程序工作的人留下注释是很棒的。
*   Docstrings 提供了关于函数、类和模块的文档。
*   使用 docstrings 来教其他开发者如何使用你的程序。

## 相关职位

*   如何使用 [Python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)函数打开文本文件？
*   使用 [Python split](https://www.pythonforbeginners.com/dictionary/python-split) ()拆分字符串。