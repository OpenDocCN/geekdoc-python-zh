# Python 中的 pickle 模块

> 原文：<https://www.askpython.com/python-modules/pickle-module-python>

我们经常遇到需要存储或转移对象的情况。Python 中的`pickle`模块就是这样一个库，它的作用是将 Python 对象作为序列化的字节序列存储到文件中，以便以后检索。让我们看看它在本文中到底做了什么。

## 1.Python Pickle 模块示例

让我们看一些在 Python 中使用 pickle 模块的例子。

### 1.1“酸洗”成文件

由于文件由字节信息组成，我们可以通过`pickle`模块将 Python 对象转换成文件。这叫腌制。让我们通过一个例子来看看如何做到这一点。

为了完成将对象序列化为文件的操作，我们使用了`pickle.dump()`方法。

格式:`pickle.dump(data_to_store, filename, protocol_type)`

*   `data_to_store` - >要序列化(酸洗)的对象
*   `filename` - >存储数据的文件的名称
*   `protocol_type` - >使用的协议类型(Python 3.8 中默认设置为 4)

这里有一个例子来说明这一点。

```py
import pickle

data = ['one', 2, [3, 4, 5]]

with open('data.dat', 'wb') as f:
    pickle.dump(data, f)

```

* * *

### 1.2 从文件中“取消选取”

这与 pickle 正好相反，pickle 从文件中检索对象。该文件包含作为字节序列的对象的序列化信息，现在被反序列化为 Python 对象本身，我们可以恢复原始信息。

为了执行这个操作，我们使用了`pickle.load()`库函数。

格式:`new_object = pickle.load(filename)`

*   `new_object` - >方法存储信息的对象
*   `filename` - >包含序列化信息的文件

```py
import pickle

objdump = None

with open('data.dat', rb') as f:
    # Stores the now deserialized information into objdump
    objdump = pickle.load(f)

```

* * *

## 2.pickle 模块的异常处理

Pickle 模块定义了一些异常，这对程序员或开发人员处理不同的场景并适当地调试它们很有用。

该模块提到以下内容可以腌制:

*   `None`、`True`、`False`
*   整数、浮点、复数
*   字符串、字节、字节数组
*   仅包含**可选择的**对象的元组、列表、集合和字典
*   在模块顶层定义的命名函数
*   在模块顶层定义的类和内置函数

其他任何对象都是不可拾取的，称为**不可拾取的**。

该模块定义了 3 个主要例外，即:

| 异常名 | 这个异常是什么时候出现的？ |
| `pickle.PickleError` | 这只是其他异常的基类。这继承了`Exception` |
| `pickle.PicklingError` | 当遇到不可拆分的对象时引发。 |
| `pickle.UnpicklingError` | 如果有任何问题(如数据损坏、访问冲突等)，则在解除对象检查期间引发 |

这里是一个使用异常处理来处理`pickle.PicklingError`的例子，当试图 pickle 一个不可 pickle 的对象时。

```py
import pickle

# A lambda is unpicklable
data = ['one', 2, [3, 4, 5], lambda l: 1]

with open('data2.dat', 'wb') as f:
    try:
        pickle.dump(data, f)
    except pickle.PicklingError:
        print('Error while reading from object. Object is not picklable')

```

输出

```py
Error while reading from object. Object is not picklable

```

这里是一个使用异常处理来处理`pickle.UnpicklingError`的例子，当尝试解包一个非序列化的文件时。

```py
import pickle

with open('data1.dat', 'wb') as f:
    f.write('This is NOT a pickled file. Trying to unpickle this will cause an exception')

objdump = None
with open('data1.dat', 'rb') as f:
    try:
        objdump = pickle.load(f)
    except pickle.UnpicklingError:
        print('Cannot write into object')

```

输出

```py
Cannot write into object

```

* * *

## 3.酸洗和酸洗面临的问题

*   正如该模块在文档中所述，它为我们提供了一个关于对对象文件进行酸洗和拆洗的严厉警告。除非您绝对信任源代码，否则不要使用此模块来进行拆包，因为任何类型的恶意代码都可能被注入到目标文件中。

*   此外，由于 Python 语言版本之间缺乏兼容性，可能会面临一些问题，因为不同版本的数据结构可能会有所不同，因此`Python 3.0`可能无法从`Python 3.8`中解包一个 pickle 文件。

*   也没有跨语言兼容性，这对于非 Python 数据传输来说可能是一个烦恼。这些信息只是特定于 Python 的。

* * *

## 4.结论

在这里，我们学习了更多关于`pickle`模块的知识，该模块可用于将 Python 对象序列化/反序列化到文件中。它是一种快速简便的传输和存储 Python 对象的方法，帮助程序员方便快捷地存储数据进行数据传输。

* * *

## 5.参考

*   JournalDev 关于泡菜的文章:https://www.journaldev.com/15638/python-pickle-example
*   泡菜模块文档:[https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html)