# Python 201:命名元组

> 原文：<https://www.blog.pythonlibrary.org/2016/03/15/python-201-namedtuple/>

Python 的集合模块有专门的容器数据类型，可用于替换 Python 的通用容器。这里我们要关注的是 **namedtuple** ，它可以用来替换 Python 的 tuple。当然，正如您将很快看到的那样，namedtuple 并不是一个简单的替代品。我见过一些程序员像使用结构一样使用它。如果你还没有使用过包含 struct 的语言，那么这需要一点解释。结构基本上是一种复杂的数据类型，它将一系列变量组合在一个名称下。让我们看一个如何创建命名元组的示例，这样您就可以了解它们是如何工作的:

```py

from collections import namedtuple

Parts = namedtuple('Parts', 'id_num desc cost amount')
auto_parts = Parts(id_num='1234', desc='Ford Engine',
                   cost=1200.00, amount=10)
print auto_parts.id_num

```

这里我们从**集合**模块导入 namedtuple。然后我们调用 namedtuple，它将返回 tuple 的一个新子类，但带有命名字段。所以基本上我们只是创建了一个新的元组类。你会注意到我们有一个奇怪的字符串作为第二个参数。这是我们要创建的以空格分隔的属性列表。

现在我们有了闪亮的新类，让我们创建它的一个实例！正如你在上面看到的，当我们创建 **auto_parts** 对象时，这是我们的下一步。现在我们可以使用点符号来访问 auto_parts 中的各个项目，因为它们现在是我们的 parts 类的属性。

使用 namedtuple 而不是常规 tuple 的好处之一是，您不再需要跟踪每个项目的索引，因为现在每个项目都是通过 class 属性命名和访问的。代码的区别如下:

```py

>>> auto_parts = ('1234', 'Ford Engine', 1200.00, 10)
>>> auto_parts[2]  # access the cost
1200.0
>>> id_num, desc, cost, amount = auto_parts
>>> id_num
'1234'

```

在上面的代码中，我们创建了一个正则元组，并通过告诉 Python 我们想要的适当索引来访问车辆引擎的成本。或者，我们也可以使用多重赋值从元组中提取所有内容。就个人而言，我更喜欢 namedtuple 方法，因为它更容易理解，并且您可以使用 Python 的 **dir()** 方法来检查元组并找出其属性。试试看会发生什么！

有一天，我在寻找一种将 Python 字典转换成对象的方法，我发现了一些代码，它们做了类似这样的事情:

```py

>>> from collections import namedtuple

>>> Parts = {'id_num':'1234', 'desc':'Ford Engine',
             'cost':1200.00, 'amount':10}
>>> parts = namedtuple('Parts', Parts.keys())(**Parts)
>>> parts
Parts(amount=10, cost=1200.0, id_num='1234', desc='Ford Engine')

```

这是一些奇怪的代码，所以让我们一次看一部分。我们导入的第一行和前面一样命名为 duple。接下来，我们创建一个零件字典。到目前为止，一切顺利。现在我们准备好了奇怪的部分。在这里，我们创建了一个名为 duple 的类，并将其命名为“Parts”。第二个参数是字典中的键列表。最后一段就是这段奇怪的代码: **(**Parts)** 。双星号意味着我们使用关键字参数调用我们的类，在本例中是我们的字典。我们可以把这条线分成两部分，让它看起来更清楚一些:

```py

>>> parts = namedtuple('Parts', Parts.keys())
>>> parts
 >>> auto_parts = parts(**Parts)
>>> auto_parts
Parts(amount=10, cost=1200.0, id_num='1234', desc='Ford Engine') 
```

所以这里我们做和以前一样的事情，除了我们首先创建类，然后我们用我们的字典调用类来创建一个对象。我想提到的另一件事是，namedtuple 还接受一个**冗长的**参数和一个**重命名的**参数。verbose 参数是一个标志，如果您将它设置为 True，它将在构建之前打印出类定义。如果要从数据库或其他不受程序控制的系统中创建命名元组，rename 参数非常有用，因为它会自动为您重命名属性。

* * *

### 包扎

现在您已经知道如何使用 Python 方便的 namedtuple。我已经在我的代码库中找到了它的多种用途，我希望它对你的代码也有帮助。编码快乐！

* * *

### 相关阅读

*   [命名元组](https://docs.python.org/3/library/collections.html#collections.namedtuple)的 Python 文档
*   PyMOTW - [命名元组](https://pymotw.com/2/collections/namedtuple.html)
*   StackOverflow - [将 Python 字典转换成对象](http://stackoverflow.com/questions/1305532/convert-python-dict-to-object)
*   将字典转换成对象[要点](https://gist.github.com/href/1319371)