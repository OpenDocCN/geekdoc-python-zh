# python 201——方便的默认字典

> 原文：<https://www.blog.pythonlibrary.org/2016/03/23/python-201-the-handy-defaultdict/>

collections 模块有另一个方便的工具叫做 **defaultdict** 。defaultdict 是 Python 的 dict 的子类，它接受 default_factory 作为其主要参数。default_factory 通常是 Python 类型，比如 int 或 list，但是也可以使用函数或 lambda。让我们首先创建一个常规的 Python 字典，它计算每个单词在一个句子中的使用次数:

```py

sentence = "The red for jumped over the fence and ran to the zoo for food"
words = sentence.split(' ')

reg_dict = {}
for word in words:
    if word in reg_dict:
        reg_dict[word] += 1
    else:
        reg_dict[word] = 1

print(reg_dict)

```

如果运行此代码，您应该会看到类似于以下内容的输出:

```py

{'The': 1,
 'and': 1,
 'fence': 1,
 'food': 1,
 'for': 2,
 'jumped': 1,
 'over': 1,
 'ran': 1,
 'red': 1,
 'the': 2,
 'to': 1,
 'zoo': 1}

```

现在让我们尝试用 defaultdict 做同样的事情！

```py

from collections import defaultdict

sentence = "The red for jumped over the fence and ran to the zoo for food"
words = sentence.split(' ')

d = defaultdict(int)
for word in words:
    d[word] += 1

print(d)

```

您会马上注意到代码简单多了。defaultdict 会自动将零作为值赋给它还没有的任何键。我们增加一个，这样更有意义，如果这个词在句子中出现多次，它也会增加。

```py

defaultdict(,
            {'The': 1,
             'and': 1,
             'fence': 1,
             'food': 1,
             'for': 2,
             'jumped': 1,
             'over': 1,
             'ran': 1,
             'red': 1,
             'the': 2,
             'to': 1,
             'zoo': 1}) 
```

现在让我们尝试使用 Python 列表类型作为我们的默认工厂。像以前一样，我们先从一本普通的字典开始。

```py

my_list = [(1234, 100.23), (345, 10.45), (1234, 75.00),
           (345, 222.66), (678, 300.25), (1234, 35.67)]

reg_dict = {}
for acct_num, value in my_list:
    if acct_num in reg_dict:
        reg_dict[acct_num].append(value)
    else:
        reg_dict[acct_num] = [value]

print(reg_dict)

```

这个例子基于我几年前写的一些代码。基本上我是一行一行地读一个文件，需要获取账号和支付金额，并跟踪它们。然后在最后，我会总结每个帐户。我们在这里跳过求和部分。如果您运行此代码，您应该会得到类似如下的输出:

```py

{345: [10.45, 222.66], 678: [300.25], 1234: [100.23, 75.0, 35.67]}

```

现在让我们使用 defaultdict 重新实现这段代码:

```py

from collections import defaultdict

my_list = [(1234, 100.23), (345, 10.45), (1234, 75.00),
           (345, 222.66), (678, 300.25), (1234, 35.67)]

d = defaultdict(list)
for acct_num, value in my_list:
    d[acct_num].append(value)

print(d)

```

这又一次去掉了 if/else 条件逻辑，使代码更容易理解。下面是上面代码的输出:

```py

defaultdict(,
            {345: [10.45, 222.66],
             678: [300.25],
             1234: [100.23, 75.0, 35.67]}) 
```

这是一些很酷的东西！让我们继续尝试使用 lambda 作为我们的默认工厂！

```py

>>> from collections import defaultdict
>>> animal = defaultdict(lambda: "Monkey")
>>> animal['Sam'] = 'Tiger'
>>> print animal['Nick']
Monkey
>>> animal
defaultdict( at 0x7f32f26da8c0>, {'Nick': 'Monkey', 'Sam': 'Tiger'}) 
```

这里我们创建一个 defaultdict，它将把“Monkey”作为默认值分配给任何键。第一个键我们设置为“Tiger”，然后下一个键我们根本不设置。如果你打印第二个键，你会看到它被赋值为“Monkey”。如果您还没有注意到，只要您将 default_factory 设置为有意义的值，基本上不可能导致 **KeyError** 发生。文档中确实提到，如果您碰巧将 default_factory 设置为 **None** ，那么您将会收到一个 **KeyError** 。让我们看看它是如何工作的:

```py

>>> from collections import defaultdict
>>> x = defaultdict(None)
>>> x['Mike']
Traceback (most recent call last):
  Python Shell, prompt 41, line 1
KeyError: 'Mike'

```

在这种情况下，我们只是创建了一个非常破的 defaultdict。它不能再为我们的键分配默认值，所以它抛出一个 KeyError。当然，由于它是 dict 的一个子类，我们只需将 key 设置为某个值就可以了。但是这有点违背了默认字典的目的。

* * *

### 包扎

现在，您已经知道如何使用 Python 集合模块中方便的 defaultdict 类型。除了刚才看到的赋值默认值之外，您还可以用它做更多的事情。我希望您能在自己的代码中找到一些有趣的用法。

* * *

### 相关阅读

*   默认字典[文档](https://docs.python.org/2/library/collections.html#collections.defaultdict)
*   本周 Python 模块- [defaultdict](https://pymotw.com/2/collections/defaultdict.html)
*   在 Python 中使用 default dict-[accele brate](https://www.accelebrate.com/blog/using-defaultdict-python/)