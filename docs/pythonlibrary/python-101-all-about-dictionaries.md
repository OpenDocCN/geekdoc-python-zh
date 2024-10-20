# Python 101:关于字典的一切

> 原文：<https://www.blog.pythonlibrary.org/2017/03/15/python-101-all-about-dictionaries/>

Python 编程语言支持几种内置类型。我最喜欢的东西之一是字典。字典是一个映射对象，它将散列值映射到任意对象。其他语言称字典为“哈希表”。与元组不同，它们是可以随时更改的可变对象。字典的键必须是可散列的或不可变的，这意味着不能使用列表或另一个字典作为键。注意 Python 3.6 之前的词典是不排序的。在 [Python 3.7](https://docs.python.org/3/whatsnew/3.6.html#new-dict-implementation) 中，他们改变了 dict 的实现，所以它现在是有序的。

在这篇文章中，我们将花一些时间来了解一些你可以用字典做的事情。

* * *

### 创建词典

字典是一个**键:值**对。在 Python 中，这些键:值对用大括号括起来，每对之间用逗号隔开。用 Python 创建字典真的很容易。以下是创建词典的三种方法:

```py

>>> my_dict = {}
>>> my_other_dict = dict()
>>> my_other_other_dict = {1: 'one', 2: 'two', 3: 'three'}

```

在第一个例子中，我们通过将变量赋给一对空花括号来创建一个空字典。还可以通过调用 Python 内置的 **dict()** 关键字来创建 dictionary 对象。我看到有人提到调用 dict()比只做赋值操作符要稍微慢一点。最后一个例子展示了如何用一些预定义的键:值对创建一个字典。您可以拥有包含各种类型映射的字典，包括到函数或对象的映射。你也可以在你的字典中嵌套字典和列表！

* * *

### 访问字典值

访问保存在字典中的值非常简单。你所需要做的就是把一个键传递给你的字典，放在方括号内。让我们来看一个例子:

```py

>>> my_other_other_dict = {1: 'one', 2: 'two', 3: 'three'}
>>> my_other_other_dict[1]
'one'

```

如果你在字典里找一个不存在的键会怎么样？你会收到一个**键错误**，就像这样:

```py

>>> my_other_other_dict[4]
Traceback (most recent call last):
  Python Shell, prompt 5, line 1
KeyError: 4

```

这个错误告诉我们，字典中没有名为“4”的键。如果你想避免这个错误，你可以使用 dict 的 **get()** 方法:

```py

>>> my_other_other_dict.get(4, None)
None

```

get()方法将询问字典是否包含指定的键(即 4)，如果不包含，您可以指定返回什么值。在这个例子中，如果键不存在，我们返回 None 类型。

您还可以在操作符中使用 Python 的**来检查字典是否也包含一个键:**

```py

>>> key = 4
>>> if key in my_other_other_dict:
        print('Key ({}) found'.format(key))
    else:
        print('Key ({}) NOT found!'.format(key))

Key (4) NOT found!

```

这将检查键 4 是否在字典中，并打印适当的响应。在 Python 2 中，字典还有一个 **has_key()** 方法，除了使用 In 操作符之外，还可以使用这个方法。但是，在 Python 3 中移除了 has_key()。

* * *

### 更新密钥

您可能已经猜到，更新一个键所指向的值是非常容易的。方法如下:

```py

>>> my_dict = {}
>>> my_dict[1] = 'one'
>>> my_dict[1]
'one'
>>> my_dict[1] = 'something else'
>>> my_dict[1]
'something else'

```

这里我们创建一个空的字典实例，然后向字典中添加一个元素。然后，我们将这个键(在本例中是整数 1)指向另一个字符串值。

* * *

### 拆卸钥匙

有两种方法可以从字典中删除键:值对。我们将涉及第一件事是字典的 **pop()** 方法。Pop 将检查该键是否在字典中，如果在，就删除它。如果密钥不在那里，您将收到一个密钥错误。实际上，您可以通过传入第二个参数(这是默认的返回值)来抑制 KeyError。

让我们来看几个例子:

```py

>>> my_dict = {}
>>> my_dict[1] = 'something else'
>>> my_dict.pop(1, None)
'something else'
>>> my_dict.pop(2)
Traceback (most recent call last):
  Python Shell, prompt 15, line 1
KeyError: 2

```

这里我们创建一个字典并添加一个条目。然后，我们使用 pop()方法删除相同的条目。您会注意到，我们还将缺省值设置为 None，这样，如果键不存在，pop 方法将返回 None。在第一种情况下，键确实存在，所以它返回它移除或弹出的项的值。

第二个例子演示了当您试图对字典中没有的键调用 pop()时会发生什么。

从字典中删除条目的另一种方法是使用 Python 的内置 del:

```py

>>> my_dict = {1: 'one', 2: 'two', 3: 'three'}
>>> del my_dict[1]
>>> my_dict
>>>  {2: 'two', 3: 'three'}

```

这将从字典中删除指定的键:值对。如果这个键不在字典中，您将收到一个 **KeyError** 。这就是为什么我实际上推荐 pop()方法，因为只要你提供一个默认值，你就不需要 **try/except** 包装 pop()。

* * *

### 重复

Python 字典允许程序员使用简单的 **for 循环**来迭代它的键。让我们来看看:

```py

>>> my_dict = {1: 'one', 2: 'two', 3: 'three'}
>>> for key in my_dict:
       print(key)
1
2
3

```

简单提醒一下:Python 字典是无序的，所以当您运行这段代码时，可能不会得到相同的结果。我认为在这一点上需要提及的一件事是，Python 3 在字典方面做了一些改变。在 Python 2 中，您可以调用字典的 keys()和 values()方法来分别返回键和值的 Python 列表:

```py

# Python 2
>>> my_dict = {1: 'one', 2: 'two', 3: 'three'}
>>> my_dict.keys()
[1, 2, 3]
>>> my_dict.values()
['one', 'two', 'three']
>>> my_dict.items()
[(1, 'one'), (2, 'two'), (3, 'three')]

```

但是在 Python 3 中，您将获得返回的视图:

```py

# Python 3
>>> my_dict = {1: 'one', 2: 'two', 3: 'three'}
>>> my_dict.keys()
>>> dict_keys([1, 2, 3])
>>> my_dict.values()
>>> dict_values(['one', 'two', 'three'])
>>> my_dict.items()
dict_items([(1, 'one'), (2, 'two'), (3, 'three')])

```

在任一 Python 版本中，您仍然可以迭代结果:

```py

for item in my_dict.values():
    print(item)

one
two
three

```

原因是列表和视图都是可迭代的。请记住，视图是不可索引的，因此在 Python 3:

```py

>>> my_dict.values()[1]

```

这将引发一个**类型错误**。

Python 有一个可爱的库，叫做 **collections** ，其中包含了字典的一些简洁的子类。我们将在接下来的两节中讨论 defaultdict 和 OrderDict。

* * *

### 默认词典

有一个非常方便的名为 **collections** 的库，里面有一个 **defaultdict** 模块。defaultdict 将接受一个类型作为它的第一个参数，或者默认为 None。我们传入的参数成为一个工厂，用于创建字典的值。让我们看一个简单的例子:

```py

from collections import defaultdict

sentence = "The red for jumped over the fence and ran to the zoo"
words = sentence.split(' ')

d = defaultdict(int)
for word in words:
    d[word] += 1

print(d)

```

在这段代码中，我们向 defaultdict 传递一个 int。这允许我们在这种情况下计算一个句子的字数。下面是上面代码的输出:

```py

defaultdict(, 
            {'and': 1, 
             'fence': 1, 
             'for': 1, 
             'ran': 1, 
             'jumped': 1,
             'over': 1, 
             'zoo': 1, 
             'to': 1, 
             'The': 1, 
             'the': 2, 
             'red': 1}) 
```

如您所见，除了字符串“the”之外，每个单词都只出现一次。您会注意到它是区分大小写的，因为“The”只出现过一次。如果我们将字符串的大小写改为小写，我们可能会使代码变得更好。

* * *

### 有序词典

“收藏库”还允许您创建记住插入顺序的词典。这就是所谓的**有序直接**。让我们来看看我以前的[文章](https://www.blog.pythonlibrary.org/2016/03/24/python-201-ordereddict/)中的一个例子:

```py

>>> from collections import OrderedDict
>>> d = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
>>> new_d = OrderedDict(sorted(d.items()))
>>> new_d
OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
>>> for key in new_d:
...     print (key, new_d[key])
... 
apple 4
banana 3
orange 2
pear 1

```

在这里，我们创建一个常规字典，对其进行排序，并将其传递给我们的 OrderedDict。然后我们迭代 OrderedDict 并打印出来。你会注意到它是按字母顺序打印出来的，因为这是我们插入数据的方式。如果只是迭代原始字典，您可能看不到这一点。

在 collections 模块中还有一个叫做 Counter 的字典子类，我们不会在这里讨论。我鼓励你自己去看看。

* * *

### 包扎

在这篇文章中，我们已经讨论了很多内容。现在，您应该已经基本了解了在 Python 中使用字典的所有知识。您已经学习了几种创建字典、添加字典、更新字典值、删除关键字甚至字典的一些可选子类的方法。我希望您发现这很有用，并且您很快会在自己的代码中发现字典的许多重要用途！

* * *

### 相关阅读

*   Python 字典[文档](https://docs.python.org/2/library/stdtypes.html#mapping-types-dict)
*   收藏模块[文档](https://docs.python.org/3/library/collections.html)
*   Python 201: [OrderedDict](https://www.blog.pythonlibrary.org/2016/03/24/python-201-ordereddict/)
*   Python 201 便捷的默认规则