# Python 程序员的 c 语言

> 原文：<https://realpython.com/c-for-python-programmers/>

本教程的目的是让一个有经验的 Python 程序员快速掌握 C 语言的基础知识，以及如何在 [CPython 源代码](https://realpython.com/cpython-source-code-guide/)中使用它。它假设您已经对 Python 语法有了初步的了解。

也就是说，C 是一种相当有限的语言，它在 CPython 中的大部分使用都属于一小组语法规则。与能够有效地编写 C 语言相比，理解代码是很小的一步。本教程针对的是第一个目标，而不是第二个目标。

在本教程中，您将学习:

*   C 预处理器是什么，它在构建 C 程序中起什么作用
*   如何使用**预处理指令**来操作源文件
*   C 语法与 **Python 语法**相比如何
*   如何在 C 语言中创建**循环**、**函数**、**字符串**以及其他特性

Python 和 C 之间最突出的区别之一是 C 预处理器。你先看看那个。

**注:**本教程改编自 [*CPython 内部:你的 Python 解释器指南*](https://realpython.com/products/cpython-internals-book/) 中的附录《Python 程序员 C 语言入门》。

**免费下载:** [从 CPython Internals:您的 Python 3 解释器指南](https://realpython.com/bonus/cpython-internals-sample/)获得一个示例章节，向您展示如何解锁 Python 语言的内部工作机制，从源代码编译 Python 解释器，并参与 CPython 的开发。

## C 预处理器

预处理程序，顾名思义，是在编译器运行之前在源文件上运行的。它的能力非常有限，但是您可以在构建 C 程序时充分利用它们。

预处理器生成一个新文件，这是编译器实际处理的内容。预处理程序的所有命令都从一行的开头开始，以一个`#`符号作为第一个非空白字符。

预处理程序的主要目的是在源文件中做文本替换，但是也会用`#if`或者类似的语句做一些基本的条件代码。

您将从最常用的预处理程序指令开始:`#include`。

[*Remove ads*](/account/join/)

### `#include`

`#include`用于将一个文件的内容拉进当前源文件。`#include`没什么高深的。它从文件系统中读取一个文件，对该文件运行预处理器，并将结果放入输出文件。对于每个`#include`指令，这是通过[递归](https://realpython.com/python-recursion/)完成的。

例如，如果您查看 CPython 的 [`Modules/_multiprocessing/semaphore.c`文件](https://github.com/python/cpython/blob/master/Modules/_multiprocessing/semaphore.c)，那么在顶部附近您会看到下面一行:

```py
#include  "multiprocessing.h"
```

这告诉预处理器获取`multiprocessing.h`的全部内容，并将它们放入输出文件的这个位置。

您会注意到`#include`语句的两种不同形式。其中一个使用引号(`""`)来指定包含文件的名称，另一个使用尖括号(`<>`)。不同之处在于在文件系统中查找文件时搜索的路径。

如果您使用`<>`作为文件名，那么预处理器将只查看系统包含文件。相反，在文件名周围使用引号会迫使预处理程序首先在本地目录中查找，然后返回到系统目录。

### `#define`

`#define`允许您进行简单的文本替换，也适用于您将在下面看到的`#if`指令。

最基本的是，`#define`允许您定义一个新符号，在预处理程序输出中用一个文本字符串替换它。

继续进入`semphore.c`，你会发现这一行:

```py
#define SEM_FAILED NULL
```

这告诉预处理器在代码被发送到编译器之前，用文字字符串`NULL`替换该点下面的每个`SEM_FAILED`实例。

`#define`项目也可以像在这个特定于 Windows 版本的`SEM_CREATE`中一样接受参数:

```py
#define SEM_CREATE(name, val, max) CreateSemaphore(NULL, val, max, NULL)
```

在这种情况下，预处理器会期望`SEM_CREATE()`看起来像一个函数调用，并且有三个参数。这通常被称为**宏**。它会直接将三个参数的文本替换到输出代码中。

例如，在`semphore.c`的第 460 行，`SEM_CREATE`宏是这样使用的:

```py
handle  =  SEM_CREATE(name,  value,  max);
```

当您为 Windows 编译时，该宏将被展开，如下所示:

```py
handle  =  CreateSemaphore(NULL,  value,  max,  NULL);
```

在后面的部分中，您将看到这个宏在 Windows 和其他操作系统上的不同定义。

[*Remove ads*](/account/join/)

### `#undef`

该指令从`#define`中删除任何先前的预处理器定义。这使得`#define`只对文件的一部分有效成为可能。

### `#if`

预处理器还允许条件语句，允许您根据特定条件包含或排除文本部分。条件语句以`#endif`指令结束，也可以利用`#elif`和`#else`进行微调。

您将在 CPython 源代码中看到三种基本形式的`#if`:

1.  **`#ifdef <macro>`** 如果定义了指定的宏，则包括后续的文本块。你也可以把它写成 **`#if defined(<macro>)`** 。
2.  如果指定的宏是**而不是**定义的， **`#ifndef <macro>`** 包括随后的文本块。
3.  如果宏定义了**和**，则`#if <macro>` 包括后续的文本块，其计算结果为`True`。

注意使用“文本”而不是“代码”来描述文件中包含或排除的内容。预处理器对 C 语法一无所知，也不关心指定的文本是什么。

### `#pragma`

编译指令是对编译器的指令或提示。一般来说，在阅读代码时可以忽略这些，因为它们通常处理的是代码如何编译，而不是代码如何运行。

### `#error`

最后，`#error`显示一条消息并使预处理器停止执行。同样，在阅读 CPython 源代码时，您可以安全地忽略这些。

## Python 程序员的基本 C 语法

本节不会涵盖 C 语言的所有方面，也不会教你如何编写 C 语言。它将集中在 Python 开发人员第一次看到 C 语言时感到不同或困惑的方面。

### 常规

与 Python 不同，空白对于 C 编译器并不重要。编译器并不关心你是否将语句跨行拆分，或者将整个程序挤在一个很长的行中。这是因为它对所有语句和块使用分隔符。

当然，解析器有非常具体的规则，但是一般来说，只要知道每个语句都以分号(`;`)结尾，并且所有代码块都用花括号(`{}`)括起来，您就能够理解 CPython 源代码。

这个规则的例外是，如果一个块只有一条语句，那么可以省略花括号。

C 中的所有变量都必须由**声明为**，这意味着需要有一个单独的语句来指示该变量的**类型**。注意，与 Python 不同，单个变量可以容纳的数据类型是不能改变的。

这里有几个例子:

```py
/* Comments are included between slash-asterisk and asterisk-slash */ /* This style of comment can span several lines -
 so this part is still a comment. */ // Comments can also come after two slashes
// This type of comment only goes until the end of the line, so new
// lines must start with double slashes (//).

int  x  =  0;  // Declares x to be of type 'int' and initializes it to 0

if  (x  ==  0)  { // This is a block of code
  int  y  =  1;  // y is only a valid variable name until the closing }
  // More statements here
  printf("x is %d y is %d\n",  x,  y); } // Single-line blocks do not require curly brackets
if  (x  ==  13) printf("x is 13!\n"); printf("past the if block\n");
```

一般来说，您会看到 CPython 代码的格式非常简洁，并且通常在给定的模块中坚持单一的风格。

[*Remove ads*](/account/join/)

### `if`报表

在 C 语言中，`if`通常像在 Python 中一样工作。如果条件为真，则执行下面的块。Python 程序员应该足够熟悉`else`和`else if`语法。注意，C `if`语句不需要`endif`，因为块是由`{}`分隔的。

C 语言中有一种简写`if` … `else`语句的方法，叫做**三元运算符**:

```py
condition  ?  true_result  :  false_result
```

您可以在`semaphore.c`中找到它，对于 Windows，它为`SEM_CLOSE()`定义了一个宏:

```py
#define SEM_CLOSE(sem) (CloseHandle(sem) ? 0 : -1)
```

如果函数`CloseHandle()`返回`true`，则该宏的返回值为`0`，否则返回`-1`。

**注意:**部分 CPython 源代码支持并使用布尔变量类型，但它们不是原始语言的一部分。c 使用一个简单的规则解释二元条件:`0`或`NULL`为假，其他都为真。

### `switch`报表

与 Python 不同，C 也支持`switch`。使用`switch`可视为扩展`if` … `elseif`链的快捷方式。这个例子来自`semaphore.c`:

```py
switch  (WaitForSingleObjectEx(handle,  0,  FALSE))  { case  WAIT_OBJECT_0: if  (!ReleaseSemaphore(handle,  1,  &previous)) return  MP_STANDARD_ERROR; *value  =  previous  +  1; return  0; case  WAIT_TIMEOUT: *value  =  0; return  0; default: return  MP_STANDARD_ERROR; }
```

这将对来自`WaitForSingleObjectEx()`的返回值执行切换。如果值为`WAIT_OBJECT_0`，则执行第一个程序块。`WAIT_TIMEOUT`值产生第二个块，其他任何东西都匹配`default`块。

注意，被测试的值，在这种情况下是来自`WaitForSingleObjectEx()`的返回值，必须是整数值或枚举类型，并且每个`case`必须是常量值。

### 循环

C 语言中有三种循环结构:

1.  `for`循环
2.  `while`循环
3.  `do` … `while`循环

循环的语法与 Python 完全不同:

```py
for  (  <initialization>;  <condition>;  <increment>)  { <code  to  be  looped  over> }
```

除了要在循环中执行的代码之外，还有三个控制`for`循环的代码块:

1.  当循环开始时，`<initialization>`段恰好运行一次。它通常用于将循环计数器设置为初始值(也可能用于声明循环计数器)。

2.  `<increment>`代码在每次通过循环的主程序块后立即运行。传统上，这将增加循环计数器。

3.  最后，`<condition>`在`<increment>`之后运行。将计算此代码的返回值，当此条件返回 false 时，循环中断。

这里有一个来自 [`Modules/sha512module.c`](https://github.com/python/cpython/blob/master/Modules/sha512module.c) 的例子:

```py
for  (i  =  0;  i  <  8;  ++i)  { S[i]  =  sha_info->digest[i]; }
```

该循环将运行`8`次，其中`i`从`0`增加到`7`，并且将在条件被检查并且`i`为`8`时终止。

`while`循环实际上与它们的 [Python 对应物](https://realpython.com/python-while-loop/)相同。然而，`do` … `while`的语法有点不同。在第一次执行循环体之后的*之前，不会检查`do` … `while`循环的条件。*

CPython 代码库中有很多`for`循环和`while`循环的实例，但是`do` … `while`没有使用。

[*Remove ads*](/account/join/)

### 功能

C 语言中函数的语法类似于 Python 中的[，但是必须指定返回类型和参数类型。C 语法看起来像这样:](https://realpython.com/defining-your-own-python-function/)

```py
<return_type>  function_name(<parameters>)  { <function_body> }
```

返回类型可以是 C 语言中的任何有效类型，包括像`int`和`double`这样的内置类型，以及像`PyObject`这样的自定义类型，如本例中的`semaphore.c`所示:

```py
static  PyObject  * semlock_release(SemLockObject  *self,  PyObject  *args) { <statements  of  function  body  here> }
```

这里您可以看到一些 C 语言特有的特性。首先，记住空白不重要。许多 CPython 源代码将函数的返回类型放在函数声明的其余部分之上。这就是`PyObject *`部分。稍后您将仔细查看`*`的用法，但是现在重要的是要知道您可以对函数和变量使用几个修饰符。

`static`就是这些修饰语之一。修改器的操作有一些复杂的规则。例如，`static`修饰符在这里的意思和你把它放在变量声明前面的意思完全不同。

幸运的是，在试图阅读和理解 CPython 源代码时，通常可以忽略这些修饰符。

函数的参数列表是逗号分隔的变量列表，类似于 Python 中使用的列表。同样，C 要求每个参数都有特定的类型，所以`SemLockObject *self`说第一个参数是一个指向`SemLockObject`的指针，被称为`self`。请注意，C 中的所有参数都是位置性的。

让我们来看看该语句的“指针”部分是什么意思。

举个例子，传递给 C 函数的参数都是通过值传递的**，这意味着函数操作的是值的副本，而不是调用函数中的原始值。为了解决这个问题，函数会频繁地传入它可以修改的一些数据的地址。**

这些地址被称为**指针**，并且有类型，所以`int *`是一个指向整数值的指针，与`double *`是不同的类型，后者是一个指向双精度浮点数的指针。

### 指针

如上所述，指针是保存值的地址的变量。这些在 C 中经常使用，如下例所示:

```py
static  PyObject  * semlock_release(SemLockObject  *self,  PyObject  *args) { <statements  of  function  body  here> }
```

这里，`self`参数将保存`SemLockObject`值的地址，或*指向*的指针。还要注意，该函数将返回一个指向`PyObject`值的指针。

**注:**要深入了解如何在 Python 中模拟指针，请查看[Python 中的指针:有什么意义？](https://realpython.com/pointers-in-python/)

C 语言中有一个特殊的值叫做 **`NULL`** ，它表示指针没有指向任何东西。在整个 CPython 源代码中，您将看到分配给`NULL`的指针，并对照`NULL`进行检查。这一点很重要，因为指针的取值没有什么限制，访问不属于程序的内存位置会导致非常奇怪的行为。

另一方面，如果你试图在`NULL`访问内存，那么你的程序将立即退出。这可能看起来不太好，但是如果访问了`NULL`,通常比修改随机内存地址更容易发现内存错误。

### 字符串

c 没有字符串类型。有一个惯例，许多标准库函数都是围绕这个惯例编写的，但是没有实际的类型。相反，C 语言中的字符串存储为由`char`(对于 ASCII)或`wchar`(对于 Unicode)值组成的数组，每个数组保存一个字符。字符串用一个**空终止符**标记，其值为`0`，通常在代码中显示为`\\0`。

像`strlen()`这样的基本字符串操作依靠这个空终止符来标记字符串的结尾。

因为字符串只是值的数组，所以不能直接复制或比较。标准库有`strcpy()`和`strcmp()`函数(以及它们的`wchar`表兄弟)来完成这些操作以及更多。

[*Remove ads*](/account/join/)

### 支柱

C 语言迷你之旅的最后一站是如何在 C: **structs** 中创建新类型。`struct`关键字允许您将一组不同的数据类型组合成一个新的自定义数据类型:

```py
struct  <struct_name>  { <type>  <member_name>; <type>  <member_name>; ... };
```

这个局部的例子从 [`Modules/arraymodule.c`](https://github.com/python/cpython/blob/master/Modules/arraymodule.c) 展示了一个`struct`的声明:

```py
struct  arraydescr  { char  typecode; int  itemsize; ... };
```

这创建了一个名为`arraydescr`的新数据类型，它有许多成员，前两个是`char typecode`和`int itemsize`。

结构经常被用作`typedef`的一部分，它为名字提供了一个简单的别名。在上面的例子中，所有新类型的变量都必须用全名`struct arraydescr x;`声明。

您会经常看到这样的语法:

```py
typedef  struct  { PyObject_HEAD SEM_HANDLE  handle; unsigned  long  last_tid; int  count; int  maxvalue; int  kind; char  *name; }  SemLockObject;
```

这将创建一个新的自定义结构类型，并将其命名为`SemLockObject`。要声明这种类型的变量，只需使用别名`SemLockObject x;`。

## 结论

这就结束了您对 C 语法的快速浏览。虽然这个描述仅仅触及了 C 语言的表面，但是您现在已经有足够的知识来阅读和理解 CPython 源代码了。

**在本教程中，您学习了:**

*   C 预处理器是什么，它在构建 C 程序中起什么作用
*   如何使用**预处理指令**来操作源文件
*   C 语法与 **Python 语法**相比如何
*   如何在 C 语言中创建**循环**、**函数**、**字符串**以及其他特性

既然您已经熟悉了 C，那么您可以通过探索 CPython 源代码来加深对 Python 内部工作方式的了解。快乐的蟒蛇！

**注意:**如果你喜欢从[*CPython Internals:Your Guide to the Python Interpreter*](https://realpython.com/products/cpython-internals-book/)中学到的东西，那么一定要看看本书的其余部分。*****