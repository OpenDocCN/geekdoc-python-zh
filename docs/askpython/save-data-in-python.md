# 用 Python 保存数据的最佳方式

> 原文：<https://www.askpython.com/python/examples/save-data-in-python>

读者你好！在本教程中，我们将讨论如何在 Python 中有效地保存数据。

## 如何在 Python 中保存数据？

当我们开发 Python 应用程序时，我们将直接处理 Python 对象，因为在 Python 中，一切都是一个[对象。让我们看看一些可以轻松存放它们的方法吧！](https://www.askpython.com/python/oops/python-classes-objects)

### 1.使用 Pickle 存储 Python 对象

如果我们想让事情变得简单，我们可以使用 [pickle 模块](https://www.askpython.com/python-modules/pickle-module-python)，它是 Python 中保存数据的标准库的一部分。

我们可以将 Python 对象“pickle”成 pickle 文件，我们可以用它来保存/加载数据。

因此，如果您有一个可能需要存储/检索的自定义对象，您可以使用以下格式:

```py
import pickle

class MyClass():
    def __init__(self, param):
        self.param = param

def save_object(obj):
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

obj = MyClass(10)
save_object(obj)

```

如果您运行这个脚本，您会注意到一个名为`data.pickle`的文件，其中包含保存的数据。

为了再次加载相同的对象，我们可以使用类似的逻辑使用`pickle.load()`。

```py
import pickle

class MyClass():
    def __init__(self, param):
        self.param = param

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

obj = load_object("data.pickle")

print(obj.param)
print(isinstance(obj, MyClass))

```

*输出*

```py
10
True

```

我们刚刚成功检索了我们的旧数据！

### 2.使用 Sqlite3 在 Python 中持久保存数据

如果您想使用持久性数据库来保存 Python 中的数据，您可以使用`sqlite3`库，它为您提供了使用 Sqlite 数据库的 API。

再说一次，这是标准库的一部分，所以不需要 [pip 安装](https://www.askpython.com/python-modules/python-pip)任何东西！

但是，由于这是一个关系数据库，所以不能像在`pickle`中那样直接转储 Python 对象。

您必须将它们序列化和反序列化为适当的数据库类型。

要看一些例子，你可以参考[这篇关于在 Python 中使用 sqlite 的文章](https://www.askpython.com/python-modules/python-sqlite-module)。

### 3.使用 SqliteDict 作为持久缓存

如果你觉得使用`sqlite3`太乏味，有一个更好的解决方案！您可以使用`sqlitedict`来存储持久数据，这在内部使用了一个`sqlite3`数据库来处理存储。

您必须使用 pip 安装这个软件包:

```py
pip install sqlitedict

```

您唯一需要记住的是，您需要使用`key:value`映射来存储/检索数据，就像字典一样！

这里有一个非常简单的使用`MyClass`实例的例子。

```py
from sqlitedict import SqliteDict

class MyClass():
    def __init__(self, param):
        self.param = param

def save(key, value, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value # Using dict[key] to store
            mydict.commit() # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)

def load(key, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key] # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)

obj1 = MyClass(10)
save("MyClass_key", obj1)

obj2 = load("MyClass_key")

print(obj1.param, obj2.param)
print(isinstance(obj1, MyClass), isinstance(obj2, MyClass))

```

*输出*

```py
10 10
True True

```

事实上，我们刚刚成功地加载了 Python 对象！如果你注意到了，`sqlitedict`会自动创建一个数据库`cache.sqlite3`，如果它不存在，然后用它来存储/加载数据。

* * *

## 结论

在本文中，我们研究了如何使用 Python 以不同的方式存储数据。

* * *