# python 101:Pickle 对象序列化简介

> 原文：<https://www.blog.pythonlibrary.org/2013/11/22/python-101-an-intro-to-object-serialization-with-pickle/>

Python 的“包含电池”理念甚至包括一个对象序列化模块。他们称之为**泡菜**模块。有些人用其他名字来称呼序列化，比如编组或扁平化。在 Python 中，它被称为“酸洗”。Pickle 模块还有一个优化的基于 C 的版本，称为 cPickle，运行速度比普通 pickle 快 1000 倍。该文档确实带有警告，但这很重要，所以转载如下:

**警告:pickle 模块并不旨在防止错误或恶意构建的数据。不要从不受信任或未经验证的来源提取数据。**

现在我们已经解决了这个问题，我们可以开始学习如何使用泡菜了！到这篇帖子结束的时候，你可能已经饿了！

### 编写简单的 Pickle 脚本

我们将从编写一个简单的脚本开始，演示如何处理 Python 列表。代码如下:

```py

import pickle

#----------------------------------------------------------------------
def serialize(obj, path):
    """
    Pickle a Python object
    """
    with open(path, "wb") as pfile:
        pickle.dump(obj, pfile)

#----------------------------------------------------------------------
def deserialize(path):
    """
    Extracts a pickled Python object and returns it
    """
    with open(path, "rb") as pfile:
        data = pickle.load(pfile)
    return data

#----------------------------------------------------------------------
if __name__ == "__main__":
    my_list = [i for i in range(10)]
    pkl_path = "data.pkl"
    serialize(my_list, pkl_path)
    saved_list = deserialize(pkl_path)
    print saved_list

```

让我们花几分钟来研究一下这段代码。我们有两个函数，第一个函数用于保存(或酸洗)Python 对象。第二个是用于反序列化(或反挑选)对象。要进行序列化，只需调用 pickle 的 **dump** 方法，并向其传递要 pickle 的对象和一个打开的文件句柄。要反序列化对象，只需调用 pickle 的 **load** 方法。您可以将多个对象 pickle 到一个文件中，但是 pickle 的工作方式类似于 FIFO(先进先出)堆栈。所以你要按照你放东西的顺序把它们拿出来。让我们修改上面的代码来演示这个概念！

```py

import pickle

#----------------------------------------------------------------------
def serialize(objects, path):
    """
    Pickle a Python object
    """
    with open(path, "wb") as pfile:
        for obj in objects:
            pickle.dump(obj, pfile)

#----------------------------------------------------------------------
def deserialize(path):
    """
    Extracts a pickled Python object and returns it
    """
    with open(path, "rb") as pfile:
        lst = pickle.load(pfile)
        dic = pickle.load(pfile)
        string = pickle.load(pfile)
    return lst, dic, string

#----------------------------------------------------------------------
if __name__ == "__main__":
    my_list = [i for i in range(10)]
    my_dict = {"a":1, "b":2}
    my_string = "I'm a string!"

    pkl_path = "data.pkl"
    serialize([my_list, my_dict, my_string], pkl_path)

    data = deserialize(pkl_path)
    print data

```

在这段代码中，我们传入一个包含 3 个 Python 对象的列表:一个列表、一个字典和一个字符串。注意，我们必须调用 pickle 的 dump 方法来存储这些对象。当您反序列化时，您需要调用 pickle 的 load 方法相同的次数。

### 关于酸洗的其他注意事项

你不能腌制所有的东西。例如，您不能处理与 C/C++有关联的 Python 对象，比如 wxPython。如果你试图这样做，你会收到一个 PicklingError。根据[文档](http://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled)，可以对以下类型进行酸洗:

*   无、真和假
*   整数、长整数、浮点数、复数
*   普通字符串和 Unicode 字符串
*   仅包含可选择对象的元组、列表、集合和字典
*   在模块顶层定义的函数
*   在模块顶层定义的内置函数
*   在模块顶层定义的类
*   这些类的实例，它们的 __dict__ 或调用 __getstate__()的结果是可 pickle 的(有关详细信息，请参见 pickle 协议一节)。

还要注意，如果您碰巧使用 cPickle 模块来加速处理时间，那么您不能对它进行子类化。cPickle 模块不支持 Pickler()和 Unpickler()的子类化，因为它们实际上是 cPickle 中的函数。你需要知道这是一个相当狡猾的家伙。

最后，pickle 的输出数据格式使用可打印的 ASCII 表示。让我们看看第二个脚本的输出，只是为了好玩:

```py

(lp0
I0
aI1
aI2
aI3
aI4
aI5
aI6
aI7
aI8
aI9
a.(dp0
S'a'
p1
I1
sS'b'
p2
I2
s.S"I'm a string!"
p0
.

```

现在，我不是这种格式的专家，但你可以看到发生了什么。然而，我不确定如何判断一个部分的结尾是什么。还要注意，默认情况下，pickle 模块使用协议版本 0。有 2 号和 3 号协议。您可以通过将它作为第三个参数传递给 pickle 的 **dump** 方法来指定您想要的协议。

最后，Richard Saunders 在 PyCon 2011 上发布了一个关于 pickle 模块的非常酷的视频。

### 包扎

至此，您应该能够使用 pickle 满足自己的数据序列化需求了。玩得开心！

*   关于 [pickle 模块](http://docs.python.org/2/library/pickle.html)的 Python 文档
*   Python wiki 文章:[使用 Pickle](https://wiki.python.org/moin/UsingPickle)
*   关于[编组模块](http://docs.python.org/library/marshal.html)的 Python 文档
*   Effbot 的[封送页](http://effbot.org/librarybook/marshal.htm)