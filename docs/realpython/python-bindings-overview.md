# Python 绑定:从 Python 调用 C 或 C++

> 原文：<https://realpython.com/python-bindings-overview/>

你是一名 Python 开发人员，有一个想从 Python 中使用的 C 或 C++库吗？如果是这样，那么 **Python 绑定**允许你调用函数并将数据从 Python 传递到 [C](https://realpython.com/build-python-c-extension-module/) 或 [C++](https://realpython.com/python-vs-cpp/) ，让你利用两种语言的优势。在本教程中，您将看到一些可用于创建 Python 绑定的工具的概述。

在本教程中，您将了解到:

*   为什么要从 Python 中**调用 C 或 C++**
*   如何在 C 和 Python 之间传递数据
*   哪些**工具和方法**可以帮助你创建 Python 绑定

本教程面向中级 Python 开发人员。它假设[具备 Python](https://realpython.com/learning-paths/python3-introduction/) 的基础知识，并对 C 或 C++中的函数和数据类型有所了解。通过点击下面的链接，您可以获得本教程中的所有示例代码:

**获取示例代码:** [单击此处获取示例代码，您将在本教程中使用](https://realpython.com/bonus/python-bindings-code/)来学习 Python 绑定。

让我们深入了解一下 Python 绑定！

## Python 绑定概述

在深入研究*如何*从 Python 调用 C 之前，最好花点时间研究一下*为什么*。在几种情况下，创建 Python 绑定来调用 C 库是一个好主意:

1.  **你已经有了一个用 C++** 编写的大型的、经过测试的、稳定的库，你想在 Python 中加以利用。这可能是一个通信库，也可能是一个与特定硬件对话的库。它做什么并不重要。

2.  **你想通过将关键部分转换成 C 来加速你的 Python 代码**的特定部分。C 不仅具有更快的执行速度，而且如果你小心的话，它还允许你摆脱 [GIL](https://realpython.com/python-gil/) 的限制。

3.  **你想用 Python 测试工具**对他们的系统做大规模测试。

以上所有这些都是学习创建 Python 绑定来与 C 库接口的好理由。

**注意:**在整个教程中，您将创建到 C *和* C++的 Python 绑定。大多数一般概念都适用于这两种语言，所以除非这两种语言之间有特定的区别，否则将使用 C。一般来说，每个工具都支持 C *或* C++，但不是两者都支持。

我们开始吧！

[*Remove ads*](/account/join/)

### 编组数据类型

等等！在开始编写 Python 绑定之前，先看看 Python 和 C 如何存储数据，以及这会导致什么类型的问题。首先，我们来定义一下**编组**。维基百科对这一概念的定义如下:

> 将对象的内存表示转换为适合存储或传输的数据格式的过程。([来源](https://en.wikipedia.org/wiki/Marshalling_(computer_science)))

就您的目的而言，编组是 Python 绑定在准备将数据从 Python 移动到 C 时所做的事情，反之亦然。Python 绑定需要进行编组，因为 Python 和 C 以不同的方式存储数据。c 以最紧凑的形式在内存中存储数据。如果你使用一个`uint8_t`，那么它总共只会使用 8 位内存。

另一方面，在 Python 中，一切都是一个 [**对象**](https://realpython.com/python-data-types/) 。这意味着每个整数使用内存中的几个字节。多少将取决于您运行的 Python 版本、您的操作系统和其他因素。这意味着您的 Python 绑定需要将一个 **C 整数**转换成一个 **Python 整数**用于跨越边界的每个整数。

其他数据类型在两种语言之间也有类似的关系。让我们依次看一看:

*   [**整数**](https://realpython.com/python-data-types/#integers) 存储计数数字。Python 以[任意精度](https://mortada.net/can-integer-operations-overflow-in-python.html)存储整数，这意味着你可以存储非常非常大的数字。指定整数的精确大小。在不同语言之间转换时，您需要注意数据大小，以防止 Python 整数值溢出 C 整型变量。

*   [**浮点数**](https://realpython.com/python-data-types/#floating-point-numbers) 是带小数点的数字。Python 可以存储比 c 大得多(也小得多)的浮点数。这意味着您还必须注意这些值，以确保它们在范围内。

*   [**复数**](https://realpython.com/python-complex-numbers/) 是有虚数部分的数字。虽然 Python 有内置的复数，而 C 也有复数，但是它们之间没有内置的编组方法。为了整理复数，您需要在 C 代码中构建一个`struct`或`class`来管理它们。

*   [**字符串**](https://realpython.com/python-data-types/#strings) 是字符序列。作为一种如此常见的数据类型，当您创建 Python 绑定时，字符串将被证明是相当棘手的。与其他数据类型一样，Python 和 C 以完全不同的格式存储字符串。(与其他数据类型不同，这也是 C 和 C++不同的地方，这增加了乐趣！)您将研究的每种解决方案在处理字符串方面都有略微不同的方法。

*   [**布尔变量**](https://realpython.com/python-boolean/) 只能有两个值。因为它们在 C 中受支持，编组它们将被证明是相当简单的。

除了数据类型转换，在构建 Python 绑定时还需要考虑其他问题。让我们继续探索它们。

### 理解可变和不可变的值

除了所有这些数据类型，你还必须知道 Python 对象如何成为 [**可变**或**不可变**](https://realpython.com/courses/immutability-python/) 。c 在谈到**按值传递**或**按引用传递**时，对函数参数也有类似的概念。在 C 中，*所有的*参数都是传值的。如果你想让一个函数改变调用者中的一个变量，那么你需要传递一个指向那个变量的指针。

您可能想知道是否可以通过使用指针将一个不可变的对象传递给 C 来绕过不可变的限制。除非你走向丑陋和不可移植的极端， [Python 不会给你一个指向对象的指针](https://realpython.com/pointers-in-python/)，所以这个就是不行。如果你想在 C 中修改一个 Python 对象，那么你需要采取额外的步骤来实现。正如您将在下面看到的，这些步骤将取决于您使用的工具。

因此，在创建 Python 绑定时，您可以将不变性添加到您要考虑的项目清单中。创建这个清单的旅程的最后一站是如何处理 Python 和 C 处理内存管理的不同方式。

### 管理内存

c 和 Python **管理内存**的方式不同。在 C 语言中，开发人员必须管理所有的内存分配，并确保它们只被释放一次。Python 会使用一个[垃圾收集器](https://realpython.com/python-memory-management/)来帮你处理这个问题。

虽然每种方法都有其优点，但它确实给创建 Python 绑定增加了额外的麻烦。你需要知道**每个对象的内存是在哪里分配的**，并确保它只在语言障碍的同一边被释放。

例如，设置`x = 3`时会创建一个 Python 对象。这方面的内存是在 Python 端分配的，需要进行垃圾收集。幸运的是，有了 Python 对象，做其他事情就相当困难了。看看 C 语言中的相反情况，直接分配一块内存:

```py
int*  iPtr  =  (int*)malloc(sizeof(int));
```

当你这样做的时候，你需要确保这个指针在 c 中是自由的，这可能意味着手动添加代码到你的 Python 绑定中来完成这个任务。

这就完成了你的一般主题清单。让我们开始设置您的系统，以便您可以编写一些代码！

### 设置您的环境

对于本教程，您将使用来自真实 Python GitHub repo 的[预先存在的 C 和 C++库](https://github.com/realpython/materials/tree/master/python-bindings)来展示每个工具的测试。目的是让你能够将这些想法用于任何 C 库。要理解这里的所有示例，您需要具备以下条件:

*   安装了一个 **C++库**，并且知道命令行调用的路径
*   Python **开发工具**:
    *   对于 Linux，这是`python3-dev`或`python3-devel`包，取决于您的发行版。
    *   对于 Windows，有[多个选项](https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_10.0_standalone:_Windows_SDK_7.1_.28x86.2C_x64.2C_ia64.29)。
*   **Python 3.6** 或更高版本
*   一个 [**虚拟环境**](https://realpython.com/courses/working-python-virtual-environments/) (推荐，但不要求)
*   **`invoke`** 工具

最后一个对你来说可能是新的，所以让我们仔细看看。

[*Remove ads*](/account/join/)

### 使用`invoke`工具

[`invoke`](http://www.pyinvoke.org/) 是您将在本教程中用来构建和测试 Python 绑定的工具。它与`make`的目的相似，但是使用 Python 而不是 Makefiles。您需要使用 [`pip`](https://realpython.com/what-is-pip/) 在您的虚拟环境中安装`invoke`:

```py
$ python3 -m pip install invoke
```

要运行它，您可以键入`invoke`,后跟您希望执行的任务:

```py
$ invoke build-cmult
==================================================
= Building C Library
* Complete
```

要查看哪些任务可用，您可以使用`--list`选项:

```py
$ invoke --list
Available tasks:

 all              Build and run all tests
 build-cffi       Build the CFFI Python bindings
 build-cmult      Build the shared library for the sample C code
 build-cppmult    Build the shared library for the sample C++ code
 build-cython     Build the cython extension module
 build-pybind11   Build the pybind11 wrapper library
 clean            Remove any built objects
 test-cffi        Run the script to test CFFI
 test-ctypes      Run the script to test ctypes
 test-cython      Run the script to test Cython
 test-pybind11    Run the script to test PyBind11
```

注意，当您查看定义了`invoke`任务的`tasks.py`文件时，您会看到列出的第二个任务的名称是`build_cffi`。然而，`--list`的输出显示为`build-cffi`。减号(`-`)不能用作 Python 名称的一部分，因此该文件使用下划线(`_`)代替。

对于您将要研究的每个工具，都将定义一个`build-`和一个`test-`任务。例如，要运行`CFFI`的代码，您可以键入`invoke build-cffi test-cffi`。一个例外是`ctypes`，因为`ctypes`没有构建阶段。此外，为了方便起见，还添加了两个特殊任务:

*   **`invoke all`** 运行所有工具的构建和测试任务。
*   **`invoke clean`** 删除任何生成的文件。

现在您已经对如何运行代码有了一个感觉，在点击工具概述之前，让我们先看一下您将要包装的 C 代码。

### C 或 C++源代码

在下面的每个示例部分中，您将**用 C 或 C++为同一个函数创建 Python 绑定**。这些部分旨在让您初步了解每种方法的样子，而不是对该工具的深入教程，因此您要包装的函数很小。您将为其创建 Python 绑定的函数将一个`int`和一个`float`作为输入参数，并返回一个`float`，它是两个数字的乘积:

```py
// cmult.c
float  cmult(int  int_param,  float  float_param)  { float  return_value  =  int_param  *  float_param; printf("    In cmult : int: %d float %.1f returning  %.1f\n",  int_param, float_param,  return_value); return  return_value; }
```

C 和 C++函数几乎完全相同，只是在名称和字符串上有细微的差别。您可以通过点击下面的链接获得所有代码的副本:

**获取示例代码:** [单击此处获取示例代码，您将在本教程中使用](https://realpython.com/bonus/python-bindings-code/)来学习 Python 绑定。

现在您已经克隆了 repo 并安装了工具，您可以构建和测试工具了。因此，让我们深入下面的每一部分！

## `ctypes`

您将从 [`ctypes`](https://docs.python.org/3.8/library/ctypes.html) 开始，它是标准库中用于创建 Python 绑定的工具。它提供了一个低级工具集，用于在 Python 和 c 之间加载共享库和编组数据。

### 它是如何安装的

`ctypes`的一大优势是它是 Python 标准库的一部分。它是在 Python 版本 2.5 中添加的，所以很可能您已经拥有它了。您可以使用 [`import`](https://realpython.com/absolute-vs-relative-python-imports/) ，就像您使用`sys`或 [`time`](https://realpython.com/python-time-module/) 模块一样。

[*Remove ads*](/account/join/)

### 调用函数

加载 C 库和调用函数的所有代码都在 Python 程序中。这很好，因为在你的过程中没有额外的步骤。你只要运行你的程序，一切都会搞定。要在`ctypes`中创建 Python 绑定，您需要完成以下步骤:

1.  **加载**你的库。
2.  **包装**你的一些输入参数。
3.  **告诉** `ctypes`你的函数的返回类型。

您将依次查看这些内容。

#### 库加载

`ctypes`为你提供了几种[加载共享库](https://docs.python.org/3.5/library/ctypes.html#loading-shared-libraries)的方法，其中一些是特定于平台的。对于您的示例，您将通过传递您想要的共享库的完整路径来直接创建一个`ctypes.CDLL`对象:

```py
# ctypes_test.py
import ctypes
import pathlib

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "libcmult.so"
    c_lib = ctypes.CDLL(libname)
```

这将适用于共享库与 Python 脚本位于同一目录的情况，但当您试图从 Python 绑定之外的包中加载库时要小心。在`ctypes`文档中有许多关于加载库和查找路径的细节，这些细节是特定于平台和情况的。

**注意:**在库加载期间可能会出现许多特定于平台的问题。一旦你得到一个工作的例子，最好进行增量的改变。

现在您已经将这个库加载到 Python 中，您可以尝试调用它了！

#### 调用您的函数

请记住，C 函数的函数原型如下:

```py
// cmult.h
float  cmult(int  int_param,  float  float_param);
```

您需要传入一个整数和一个浮点，并期望得到一个浮点返回。整数和浮点数在 Python 和 C 中都有本地支持，所以您希望这种情况下可以得到合理的值。

一旦将库加载到 Python 绑定中，该函数将成为`c_lib`的一个属性，也就是您之前创建的`CDLL`对象。你可以试着这样称呼它:

```py
x, y = 6, 2.3
answer = c_lib.cmult(x, y)
```

哎呀！这不管用。这一行在示例 repo 中被注释掉，因为它失败了。如果您试图使用该调用运行，Python 将报错:

```py
$ invoke test-ctypes
Traceback (most recent call last):
 File "ctypes_test.py", line 16, in <module>
 answer = c_lib.cmult(x, y)
ctypes.ArgumentError: argument 2: <class 'TypeError'>: Don't know how to convert parameter 2
```

看起来你需要告诉`ctypes`任何非整数的参数。`ctypes`不知道这个函数，除非你明确地告诉它。任何没有另外标记的参数都被假定为整数。`ctypes`不知道如何将存储在`y`中的值`2.3`转换成整数，所以它失败了。

要解决这个问题，您需要从该号码创建一个`c_float`。您可以在调用函数的行中这样做:

```py
# ctypes_test.py
answer = c_lib.cmult(x, ctypes.c_float(y))
print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
```

现在，当您运行这段代码时，它会返回您传入的两个数字的乘积:

```py
$ invoke test-ctypes
 In cmult : int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 48.0
```

等一下… `6`乘以`2.3`不是`48.0`！

原来，很像输入参数，`ctypes` **假设**你的函数返回一个`int`。实际上，您的函数返回了一个`float`，它被错误地封送了。就像输入参数一样，你需要告诉`ctypes`使用不同的类型。这里的语法略有不同:

```py
# ctypes_test.py
c_lib.cmult.restype = ctypes.c_float
answer = c_lib.cmult(x, ctypes.c_float(y))
print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
```

这应该能行。让我们运行整个`test-ctypes`目标，看看您得到了什么。记住，输出的第一段是在之前的**，你把函数的`restype`固定为一个浮点:**

```py
$ invoke test-ctypes
==================================================
= Building C Library
* Complete
==================================================
= Testing ctypes Module
 In cmult : int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 48.0

 In cmult : int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 13.8
```

那更好！虽然第一个未修正的版本返回了错误的值，但是您的修正版本符合 C 函数。C 和 Python 得到的结果都一样！现在它工作了，看看为什么你可能想或不想使用`ctypes`。

[*Remove ads*](/account/join/)

### 优势和劣势

与您将在这里研究的其他工具相比，`ctypes`最大的优势是它被内置在标准库中。它也不需要额外的步骤，因为所有的工作都是作为 Python 程序的一部分来完成的。

此外，使用的概念是低级的，这使得像你刚才做的练习是可管理的。然而，由于缺乏自动化，更复杂的任务变得越来越麻烦。在下一节中，您将看到一个工具，它为这个过程增加了一些自动化。

## `CFFI`

[`CFFI`](https://cffi.readthedocs.io/en/latest/) 是 Python 的 **C 对外函数接口**。生成 Python 绑定需要更自动化的方法。`CFFI`有多种方式可以构建和使用 Python 绑定。有两个不同的选项可供选择，这为您提供了四种可能的模式:

*   **ABI vs API:** API 模式使用 C 编译器生成完整的 Python 模块，而 ABI 模式加载共享库并与之直接交互。如果不运行编译器，获得正确的结构和参数是容易出错的。文档强烈建议使用 API 模式。

*   **线内与线外:**这两种模式的区别在于速度和便利性之间的权衡:

    *   **内联模式**在每次脚本运行时编译 Python 绑定。这很方便，因为您不需要额外的构建步骤。然而，它确实会减慢你的程序。
    *   **离线模式**需要一个额外的步骤来一次性生成 Python 绑定，然后在程序每次运行时使用它们。这要快得多，但这对您的应用程序来说可能无关紧要。

对于本例，您将使用 API out-of-line 模式，该模式生成最快的代码，总体上看起来类似于您将在本教程稍后创建的其他 Python 绑定。

### 它是如何安装的

因为`CFFI`不是标准库的一部分，你需要在你的机器上安装它。建议您为此创建一个虚拟环境。好在`CFFI`装着 [`pip`](https://realpython.com/what-is-pip/) :

```py
$ python3 -m pip install cffi
```

这将把软件包安装到您的虚拟环境中。如果你已经从`requirements.txt`开始安装了，那么这个问题就应该解决了。您可以通过下面的链接访问回购来了解一下`requirements.txt`:

**获取示例代码:** [单击此处获取示例代码，您将在本教程中使用](https://realpython.com/bonus/python-bindings-code/)来学习 Python 绑定。

既然已经安装了`CFFI`，是时候带着它转一圈了！

### 调用函数

与`ctypes`不同，使用`CFFI`你可以创建一个完整的 Python 模块。你将能够 [`import`](https://realpython.com/python-import/) 这个模块就像标准库中的任何其他模块一样。构建 Python 模块还需要做一些额外的工作。要使用您的`CFFI` Python 绑定，您需要采取以下步骤:

*   **写一些描述绑定的 Python 代码。**
*   **运行**代码以生成可加载模块。
*   **修改**调用代码以导入并使用您新创建的模块。

这可能看起来工作量很大，但是您将通过这些步骤中的每一步来了解它是如何工作的。

#### 编写绑定

`CFFI`提供了读取 **C 头文件**的方法，以便在生成 Python 绑定时完成大部分工作。在`CFFI`的文档中，完成这项工作的代码放在一个单独的 Python 文件中。对于这个例子，您将把代码直接放入构建工具`invoke`，它使用 Python 文件作为输入。要使用`CFFI`，首先要创建一个`cffi.FFI`对象，它提供了您需要的三种方法:

```py
# tasks.py
import cffi
...
""" Build the CFFI Python bindings """
print_banner("Building CFFI Module")
ffi = cffi.FFI()
```

一旦有了 FFI，您将使用`.cdef()`来自动处理头文件的内容。这将为您创建包装器函数，以便从 Python 中整理数据:

```py
# tasks.py
this_dir = pathlib.Path().absolute()
h_file_name = this_dir / "cmult.h"
with open(h_file_name) as h_file:
    ffi.cdef(h_file.read())
```

读取和处理头文件是第一步。之后，你需要用`.set_source()`来描述`CFFI`将要生成的源文件:

```py
# tasks.py
ffi.set_source(
    "cffi_example",
    # Since you're calling a fully-built library directly, no custom source
    # is necessary. You need to include the .h files, though, because behind
    # the scenes cffi generates a .c file that contains a Python-friendly
    # wrapper around each of the functions.
    '#include "cmult.h"',
    # The important thing is to include the pre-built lib in the list of
    # libraries you're linking against:
    libraries=["cmult"],
    library_dirs=[this_dir.as_posix()],
    extra_link_args=["-Wl,-rpath,."],
)
```

这是您传递的参数的明细:

*   **`"cffi_example"`** 是将在您的文件系统上创建的源文件的基本名称。`CFFI`会生成一个`.c`文件，编译成一个`.o`文件，链接到一个`.<system-description>.so`或者`.<system-description>.dll`文件。

*   **`'#include "cmult.h"'`** 是自定义的 C 源代码，在编译之前会包含在生成的源代码中。这里，您只包含了您正在为其生成绑定的`.h`文件，但是这可以用于一些有趣的定制。

*   **`libraries=["cmult"]`** 告诉链接器你预先存在的 C 库的名字。这是一个[列表](https://realpython.com/courses/using-list-comprehensions-effectively/)，因此如果需要，您可以指定几个库。

*   **`library_dirs=[this_dir.as_posix(),]`** 是一个目录列表，告诉链接器在哪里寻找上面的库列表。

*   **`extra_link_args=['-Wl,-rpath,.']`** 是一组生成共享对象的选项，它会在当前路径(`.`)中查找它需要加载的其他库。

#### 构建 Python 绑定

调用`.set_source()`不会构建 Python 绑定。它只设置元数据来描述将要生成的内容。要构建 Python 绑定，您需要调用`.compile()`:

```py
# tasks.py
ffi.compile()
```

这通过生成`.c`文件、`.o`文件和共享库来完成。您刚刚完成的`invoke`任务可以在[命令行](https://realpython.com/python-command-line-arguments/)上运行，以构建 Python 绑定:

```py
$ invoke build-cffi
==================================================
= Building C Library
* Complete
==================================================
= Building CFFI Module
* Complete
```

您已经有了自己的`CFFI` Python 绑定，所以是时候运行这段代码了！

#### 调用您的函数

在完成了配置和运行`CFFI`编译器的所有工作之后，使用生成的 Python 绑定看起来就像使用任何其他 Python 模块一样:

```py
# cffi_test.py
import cffi_example

if __name__ == "__main__":
    # Sample data for your call
    x, y = 6, 2.3

    answer = cffi_example.lib.cmult(x, y)
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
```

你导入新模块，然后就可以直接调用`cmult()`了。要进行测试，请使用`test-cffi`任务:

```py
$ invoke test-cffi
==================================================
= Testing CFFI Module
 In cmult : int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 13.8
```

这将运行您的`cffi_test.py`程序，该程序测试您用`CFFI`创建的新 Python 绑定。这就完成了编写和使用`CFFI` Python 绑定的部分。

[*Remove ads*](/account/join/)

### 优势和劣势

看起来,`ctypes`比您刚刚看到的`CFFI`例子需要更少的工作。虽然这对于这个用例来说是真实的，但是`CFFI`比`ctypes`更好地扩展到更大的项目，因为**自动化了大部分的功能包装。**

也产生了完全不同的用户体验。`ctypes`允许您将预先存在的 C 库直接加载到您的 Python 程序中。另一方面，`CFFI`创建了一个新的 Python 模块，可以像其他 Python 模块一样加载。

更重要的是，使用上面使用的 **out-of-line-API** 方法，创建 Python 绑定的时间代价是在构建时一次性完成的，而不是在每次运行代码时都发生。对于小程序来说，这可能没什么大不了的，但是`CFFI`也可以通过这种方式更好地扩展到更大的项目。

和`ctypes`一样，使用`CFFI`只能让你直接和 C 库接口。C++库需要大量的工作才能使用。在下一节中，您将看到一个侧重于 C++的 Python 绑定工具。

## `PyBind11`

[`PyBind11`](https://pybind11.readthedocs.io/en/stable/) 采用一种完全不同的方法来创建 Python 绑定。除了将重心从 C 转移到 C++，它还**使用 C++来指定和构建模块**，允许它利用 C++中的元编程工具。像`CFFI`一样，从`PyBind11`生成的 Python 绑定是一个完整的 Python 模块，可以直接导入和使用。

`PyBind11`模仿了`Boost::Python`库，有一个相似的接口。然而，它将它的使用限制在 C++11 和更新的版本，这允许它比支持一切的 Boost 更简单和更快。

### 它是如何安装的

`PyBind11`文档的[第一步](https://pybind11.readthedocs.io/en/latest/basics.html)部分将带您了解如何下载和构建`PyBind11`的测试用例。虽然这似乎不是严格要求的，但是完成这些步骤将确保您已经设置了正确的 C++和 Python 工具。

**注意:**`PyBind11`的例子大多使用 [`cmake`](https://cmake.org/) ，这是一个构建 C 和 C++项目的优良工具。然而，对于这个演示，您将继续使用`invoke`工具，它遵循文档的[手动构建](https://pybind11.readthedocs.io/en/latest/compiling.html#building-manually)部分中的说明。

您需要将该工具安装到您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中:

```py
$ python3 -m pip install pybind11
```

是一个全头文件库，类似于 Boost 的大部分内容。这允许`pip`将库的实际 C++源代码直接安装到您的虚拟环境中。

### 调用函数

在开始之前，请注意**您使用的是不同的 C++源文件**，`cppmult.cpp`，而不是您在前面的例子中使用的 C 文件。这两种语言的功能基本相同。

#### 编写绑定

与`CFFI`类似，您需要创建一些代码来告诉工具如何构建您的 Python 绑定。与`CFFI`不同，这段代码将使用 C++而不是 Python。幸运的是，只需要很少的代码:

```py
// pybind11_wrapper.cpp
#include  <pybind11/pybind11.h> #include  <cppmult.hpp> PYBIND11_MODULE(pybind11_example,  m)  { m.doc()  =  "pybind11 example plugin";  // Optional module docstring
  m.def("cpp_function",  &cppmult,  "A function that multiplies two numbers"); }
```

让我们一次看一块，因为`PyBind11`将大量信息打包到几行中。

前两行包括`pybind11.h`文件和 C++库的头文件`cppmult.hpp`。之后，你有了`PYBIND11_MODULE`宏。这扩展成一个 C++代码块，在`PyBind11`源代码中有很好的描述:

> 这个宏创建了一个入口点，当 Python 解释器导入一个扩展模块时，这个入口点将被调用。模块名作为第一个参数给出，不应该用引号括起来。第二个宏参数定义了一个类型为`py::module`的变量，可以用来初始化模块。([来源](https://github.com/pybind/pybind11/blob/1376eb0e518ff2b7b412c84a907dd1cd3f7f2dcd/include/pybind11/detail/common.h#L267))

对于这个例子来说，这意味着你正在创建一个名为`pybind11_example`的模块，其余的代码将使用`m`作为`py::module`对象的名称。在下一行，在您正在定义的 C++函数中，您为该模块创建了一个 [docstring](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings) 。虽然这是可选的，但这是让你的模块更有[python 化](https://realpython.com/learning-paths/writing-pythonic-code/)的好办法。

最后，你有了`m.def()`电话。这将定义一个由您的新 Python 绑定导出的函数，这意味着它将在 Python 中可见。在本例中，您要传递三个参数:

*   **`cpp_function`** 是您将在 Python 中使用的函数的导出名称。如这个例子所示，它不需要匹配 C++函数的名称。
*   **`&cppmult`** 取要导出的函数的地址。
*   **`"A function..."`** 是该函数的可选 docstring。

现在您已经有了 Python 绑定的代码，接下来看看如何将其构建到 Python 模块中。

#### 构建 Python 绑定

您在`PyBind11`中用来构建 Python 绑定的工具是 C++编译器本身。您可能需要修改编译器和操作系统的默认值。

首先，您必须构建要为其创建绑定的 C++库。对于这么小的例子，您可以将`cppmult`库直接构建到 Python 绑定库中。然而，对于大多数真实世界的例子，您将有一个想要包装的预先存在的库，所以您将单独构建`cppmult`库。构建是对编译器构建共享库的标准调用:

```py
# tasks.py
invoke.run(
    "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC cppmult.cpp "
    "-o libcppmult.so "
)
```

用`invoke build-cppmult`运行这个命令会产生`libcppmult.so`:

```py
$ invoke build-cppmult
==================================================
= Building C++ Library
* Complete
```

另一方面，Python 绑定的构建需要一些特殊的细节:

```py
 1# tasks.py
 2invoke.run(
 3    "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
 4    "`python3 -m pybind11 --includes` "
 5    "-I /usr/include/python3.7 -I .  "
 6    "{0} "
 7    "-o {1}`python3.7-config --extension-suffix` "
 8    "-L. -lcppmult -Wl,-rpath,.".format(cpp_name, extension_name)
 9)
```

让我们一行一行地走一遍。第 3 行包含相当标准的 C++编译器标志，表示几个细节，包括您希望所有警告被捕获并被视为错误，您想要一个共享库，以及您正在使用 C++11。

**第四行**是魔法的第一步。它调用`pybind11`模块来为`PyBind11`生成合适的`include`路径。您可以直接在控制台上运行该命令，看看它能做什么:

```py
$ python3 -m pybind11 --includes
-I/home/jima/.virtualenvs/realpython/include/python3.7m
-I/home/jima/.virtualenvs/realpython/include/site/python3.7
```

您的输出应该类似，但显示不同的路径。

在编译调用的**第 5 行**中，您可以看到您还添加了 Python dev `includes`的路径。虽然建议你*不要*链接 Python 库本身，但是源代码需要来自`Python.h`的一些代码来发挥它的魔力。幸运的是，它使用的代码跨 Python 版本相当稳定。

第 5 行还使用`-I .`将当前目录添加到`include`路径列表中。这允许解析包装器代码中的`#include <cppmult.hpp>`行。

**第 6 行**指定了你的源文件的名字，也就是`pybind11_wrapper.cpp`。然后，在**第 7 行**你会看到更多的建造魔法发生。这一行指定了输出文件的名称。Python 对模块命名有一些特殊的想法，包括 Python 版本、机器架构和其他细节。Python 还提供了一个叫做`python3.7-config`的工具来帮助解决这个问题:

```py
$ python3.7-config --extension-suffix
.cpython-37m-x86_64-linux-gnu.so
```

如果您使用的是不同版本的 Python，可能需要修改该命令。如果您使用不同版本的 Python 或在不同的操作系统上，您的结果可能会发生变化。

构建命令的最后一行，**第 8 行**，将链接器指向您之前构建的`libcppmult`库。`rpath`部分告诉链接器将信息添加到共享库中，以帮助操作系统在运行时找到`libcppmult`。最后，您会注意到这个字符串是用`cpp_name`和`extension_name`格式化的。在下一节中，当您使用`Cython`构建 Python 绑定模块时，您将再次使用这个函数。

运行以下命令来构建绑定:

```py
$ invoke build-pybind11
==================================================
= Building C++ Library
* Complete
==================================================
= Building PyBind11 Module
* Complete
```

就是这样！您已经用`PyBind11`构建了您的 Python 绑定。是时候检验一下了！

#### 调用您的函数

类似于上面的`CFFI`示例，一旦您完成了创建 Python 绑定的繁重工作，调用您的函数看起来就像普通的 Python 代码:

```py
# pybind11_test.py
import pybind11_example

if __name__ == "__main__":
    # Sample data for your call
    x, y = 6, 2.3

    answer = pybind11_example.cpp_function(x, y)
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
```

因为您在`PYBIND11_MODULE`宏中使用了`pybind11_example`作为模块的名称，所以这就是您导入的名称。在`m.def()`调用中，你告诉`PyBind11`将`cppmult`函数导出为`cpp_function`，所以这就是你从 Python 中调用它的原因。

你也可以用`invoke`来测试它:

```py
$ invoke test-pybind11
==================================================
= Testing PyBind11 Module
 In cppmul: int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 13.8
```

那就是`PyBind11`的样子。接下来，您将看到`PyBind11`何时以及为什么是这项工作的合适工具。

[*Remove ads*](/account/join/)

### 优势和劣势

`PyBind11`专注于 C++而不是 C，这使得它不同于`ctypes`和`CFFI`。它有几个特性使得它对 C++库很有吸引力:

*   它支持**类**。
*   它处理**多态子类**。
*   它允许您从 Python 和许多其他工具向对象添加动态属性，这在您研究过的基于 C 的工具中是很难做到的。

也就是说，要让`PyBind11`启动并运行，您需要做一些设置和配置工作。获得正确的安装和构建可能有点挑剔，但一旦完成，它似乎相当可靠。另外，`PyBind11`要求您至少使用 C++11 或更高版本。对于大多数项目来说，这不太可能是一个很大的限制，但对于您来说，这可能是一个考虑因素。

最后，创建 Python 绑定所需的额外代码是用 C++而不是 Python 编写的。这对你来说可能是也可能不是问题，但是它*与你在这里看到的其他工具*不同。在下一节中，您将继续讨论`Cython`，它采用了一种完全不同的方法来解决这个问题。

## `Cython`

创建 Python 绑定的方法 [`Cython`](https://cython.org/) 使用**类 Python 语言**来定义绑定，然后生成可以编译到模块中的 C 或 C++代码。用`Cython`构建 Python 绑定有几种方法。最常见的是使用`distutils`中的`setup`。对于这个例子，您将坚持使用`invoke`工具，它将允许您使用正在运行的确切命令。

### 它是如何安装的

`Cython`是一个 Python 模块，可以从 [PyPI](https://realpython.com/courses/how-to-publish-your-own-python-package-pypi/) 安装到您的虚拟环境中:

```py
$ python3 -m pip install cython
```

同样，如果您已经将`requirements.txt`文件安装到您的虚拟环境中，那么它就已经存在了。您可以点击下面的链接获取一份`requirements.txt`:

**获取示例代码:** [单击此处获取示例代码，您将在本教程中使用](https://realpython.com/bonus/python-bindings-code/)来学习 Python 绑定。

这应该让你准备好与`Cython`一起工作！

### 调用函数

为了用`Cython`构建 Python 绑定，您将遵循与用于`CFFI`和`PyBind11`类似的步骤。您将编写绑定，构建它们，然后运行 Python 代码来调用它们。`Cython`可以同时支持 C 和 C++。对于这个例子，您将使用您在上面的`PyBind11`例子中使用的`cppmult`库。

#### 编写绑定

在`Cython`中声明模块最常见的形式是使用一个`.pyx`文件:

```py
 1# cython_example.pyx
 2""" Example cython interface definition """
 3
 4cdef extern from "cppmult.hpp":
 5    float cppmult(int int_param, float float_param)
 6
 7def pymult( int_param, float_param ):
 8    return cppmult( int_param, float_param )
```

这里有两个部分:

1.  **第 3 和第 4 行**告诉`Cython`你正在使用来自`cppmult.hpp`的`cppmult()`。
2.  **第 6 行和第 7 行**创建一个包装器函数`pymult()`，以调用`cppmult()`。

这里使用的语言是 C、C++和 Python 的特殊混合。不过，Python 开发人员会对它相当熟悉，因为目标是使过程更容易。

带有`cdef extern...`的第一部分告诉`Cython`下面的函数声明也可以在`cppmult.hpp`文件中找到。这有助于确保您的 Python 绑定是针对与 C++代码相同的声明构建的。第二部分看起来像一个常规的 Python 函数——因为它就是！本节创建一个 Python 函数，它可以访问 C++函数`cppmult`。

现在您已经定义了 Python 绑定，是时候构建它们了！

#### 构建 Python 绑定

`Cython`的构建过程与您用于`PyBind11`的构建过程有相似之处。您首先在`.pyx`文件上运行`Cython`，生成一个`.cpp`文件。一旦你完成了这些，你就可以用与`PyBind11`相同的函数来编译它:

```py
 1# tasks.py
 2def compile_python_module(cpp_name, extension_name):
 3    invoke.run(
 4        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
 5        "`python3 -m pybind11 --includes` "
 6        "-I /usr/include/python3.7 -I .  "
 7        "{0} "
 8        "-o {1}`python3.7-config --extension-suffix` "
 9        "-L. -lcppmult -Wl,-rpath,.".format(cpp_name, extension_name)
10    )
11
12def build_cython(c):
13    """ Build the cython extension module """
14    print_banner("Building Cython Module")
15    # Run cython on the pyx file to create a .cpp file
16    invoke.run("cython --cplus -3 cython_example.pyx -o cython_wrapper.cpp")
17
18    # Compile and link the cython wrapper library
19    compile_python_module("cython_wrapper.cpp", "cython_example")
20    print("* Complete")
```

首先在您的`.pyx`文件上运行`cython`。您可以在该命令中使用几个选项:

*   **`--cplus`** 告诉编译器生成一个 C++文件而不是 C 文件。
*   **`-3`** 切换`Cython`生成 Python 3 语法，而不是 Python 2。
*   **`-o cython_wrapper.cpp`** 指定要生成的文件的名称。

一旦生成了 C++文件，您就可以使用 C++编译器来生成 Python 绑定，就像您对`PyBind11`所做的一样。注意，使用`pybind11`工具生成额外的`include`路径的调用仍然在那个函数中。这不会伤害任何东西，因为你的源头不需要这些。

在`invoke`中运行该任务会产生以下输出:

```py
$ invoke build-cython
==================================================
= Building C++ Library
* Complete
==================================================
= Building Cython Module
* Complete
```

你可以看到它构建了`cppmult`库，然后构建了`cython`模块来包装它。现在你有了`Cython` Python 绑定。(试着快速说*那个【T4……)是时候测试一下了！*

#### 调用您的函数

调用新 Python 绑定的 Python 代码与您用来测试其他模块的代码非常相似:

```py
 1# cython_test.py
 2import cython_example
 3
 4# Sample data for your call
 5x, y = 6, 2.3
 6
 7answer = cython_example.pymult(x, y)
 8print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
```

第 2 行导入了新的 Python 绑定模块，您在第 7 行调用了`pymult()`。记住，`.pyx`文件提供了一个围绕`cppmult()`的 Python 包装器，并将其重命名为`pymult`。使用 invoke 运行测试会产生以下结果:

```py
$ invoke test-cython
==================================================
= Testing Cython Module
 In cppmul: int: 6 float 2.3 returning  13.8
 In Python: int: 6 float 2.3 return val 13.8
```

你得到的结果和以前一样！

[*Remove ads*](/account/join/)

### 优势和劣势

`Cython`是一个相对复杂的工具，在为 C 或 C++创建 Python 绑定时，它可以为你**提供更深层次的控制**。虽然您在这里没有深入讨论它，但是它提供了一种类似 Python 的方法来编写手动控制 [GIL](https://realpython.com/python-gil/) 的代码，这可以显著地加速某些类型的问题。

然而，这种类似 Python 的语言并不完全是 Python，所以当你要快速弄清楚 C 和 Python 的哪一部分适合哪一部分时，有一个轻微的学习曲线。

## 其他解决方案

在研究本教程时，我遇到了几个不同的工具和选项来创建 Python 绑定。虽然我将这个概述局限于一些更常见的选项，但是我偶然发现了一些其他工具。下面的列表并不全面。如果上面的工具不适合你的项目，这仅仅是其他可能性的一个例子。

### `PyBindGen`

**[`PyBindGen`](https://pybindgen.readthedocs.io/en/latest/tutorial/#supported-python-versions)** 为 C 或 C++生成 Python 绑定，用 Python 编写。它的目标是生成可读的 C 或 C++代码，这将简化调试问题。目前还不清楚最近是否有更新，因为文档将 Python 3.4 列为最新的测试版本。然而，过去几年每年都有发行。

### `Boost.Python`

**[`Boost.Python`](https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html)** 有一个类似于`PyBind11`的界面，你在上面看到了。这不是巧合，因为`PyBind11`就是基于这个库！`Boost.Python`完全用 C++编写，支持大多数平台上的大多数(如果不是全部)C++版本。相比之下，`PyBind11`将自己限制在现代 C++中。

### `SIP`

**[`SIP`](https://www.riverbankcomputing.com/software/sip/intro)** 是为 [PyQt](https://realpython.com/python-pyqt-gui-calculator/) 项目开发的用于生成 Python 绑定的工具集。它也被 [wxPython](https://realpython.com/python-gui-with-wxpython/) 项目用来生成它们的绑定。它有一个代码生成工具和一个额外的 Python 模块，为生成的代码提供支持功能。

### `Cppyy`

**[`cppyy`](https://cppyy.readthedocs.io/en/latest/)** 是一个有趣的工具，它的设计目标与你目前看到的略有不同。用软件包作者的话说:

> “cppyy(可追溯到 2001 年)背后的最初想法是允许生活在 C++世界中的 Python 程序员访问那些 C++包，而不必直接接触 C++(或者等待 C++开发人员来提供绑定)。”([来源](https://news.ycombinator.com/item?id=15098764))

### `Shiboken`

**[`Shiboken`](https://wiki.qt.io/Qt_for_Python/Shiboken)** 是一个用于生成 Python 绑定的工具，它是为与 Qt 项目相关联的 PySide 项目开发的。虽然它是为那个项目设计的工具，但是文档表明它既不是特定于 Qt 的，也不是特定于 PySide 的，并且可用于其他项目。

### `SWIG`

**[`SWIG`](http://swig.org/)** 是一个不同于这里列出的其他工具的工具。这是一个通用工具，用于为许多其他语言[创建 C 和 C++程序的绑定](http://swig.org/compat.html#SupportedLanguages)，而不仅仅是 Python。这种为不同语言生成绑定的能力在一些项目中非常有用。当然，就复杂性而言，这是有代价的。

## 结论

恭喜你。现在，您已经对创建 **Python 绑定**的几种不同选项有了一个概述。您了解了数据编组和创建绑定时需要考虑的问题。您已经看到了使用以下工具从 Python 调用 C 或 C++函数需要什么:

*   **T2`ctypes`**
*   **T2`CFFI`**
*   **T2`PyBind11`**
*   **T2`Cython`**

您现在知道了，虽然`ctypes`允许您直接加载 DLL 或共享库，但是其他三个工具需要额外的步骤，但是仍然创建完整的 Python 模块。另外，您还使用了一点`invoke`工具来运行 Python 中的命令行任务。您可以通过单击下面的链接获得您在本教程中看到的所有代码:

**获取示例代码:** [单击此处获取示例代码，您将在本教程中使用](https://realpython.com/bonus/python-bindings-code/)来学习 Python 绑定。

现在选择您最喜欢的工具，开始构建那些 Python 绑定吧！特别感谢 **Loic Domaigne** 对本教程的额外技术回顾。*******