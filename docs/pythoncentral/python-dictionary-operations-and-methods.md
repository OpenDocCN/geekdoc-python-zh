# Python 词典

> 原文：<https://www.pythoncentral.io/python-dictionary-operations-and-methods/>

## Python 字典(dict):概述

在内置的 Python 数据类型中，有一种非常通用的类型叫做*字典*。字典类似于列表和元组，因为它们充当您创建的其他对象或变量的存储单元。字典不同于列表和元组，因为它们保存的对象组没有任何特定的顺序，而是每个对象都有自己唯一的*名*，通常称为*键*。

存储在字典中的对象或*值*基本上可以是任何东西(甚至是定义为`None`的 *nothing* 类型)，但是*键*只能是不可变的类型对象。例如*字符串*、*元组*、*整数*等。

**注意**:存储对象的不可变类型像*元组*可以**只有**包含其他不可变类型成为字典*键*。

## 创建 Python 字典

创建字典很容易，下面是一个空字典:

```py

emptydict = {}

```

下面是一个有几个值的例子(注意`{key1: value1, key2: value2, ...}`结构) :

```py

smalldict = {

'dlist': [],

'dstring': 'pystring',

'dtuple': (1, 2, 3)

}

```

也可以使用字典构造函数(注意用这种方法，*键*只能是关键字串):

```py

smalldict = dict(dlist=[], dstring='pystring', dtuple=(1, 2, 3))

```

## 获取 Python 字典的值

要获取其中一个值，只需使用它的键:

```py

>>> smalldict = {

'dlist': [],

'dstring': 'pystring',

'dtuple': (1, 2, 3)

}

>>> smalldict['dstring']

'pystring'

>>> smalldict['dtuple']

(1, 2, 3)

```

## 移除 Python 字典的键

要删除一个*键:值*对，只需用`del`关键字删除它:

```py

>>> del smalldict['dlist']

>>> smalldict

{'dstring': 'pystring', 'dtuple': (1, 2, 3)}

```

## 向 Python 字典添加键

或者分配一个新的*键*，只需使用新的*键*:

```py

>>> smalldict['newkey'] = None

>>> smalldict

{'dstring': 'pystring', 'newkey': None, 'dtuple': (1, 2, 3)}

```

## 覆盖 Python 字典键值对

**注意:**一个字典中所有的键都是唯一的，所以如果你给一个现有的键赋值，它会被覆盖！例如:

```py

>>> smalldict['dstring'] = 'new pystring'

>>> smalldict

{'dstring': 'new pystring', 'newkey': None, 'dtuple': (1, 2, 3)}

```

## 在 Python 字典中交换键和值

如果您想将*键*换成*值*，您可以用*值*创建新的*键*，然后简单地删除旧的*键*:

```py

>>> smalldict['changedkey'] = smalldict['newkey']

>>> del smalldict['newkey']

>>> smalldict

{'dstring': 'new pystring', 'changedkey': None, 'dtuple': (1, 2, 3)}

```

## 通过关键字搜索 Python 字典

要检查字典中是否存在*键*，只需使用 smalldict 中的*键*:

```py

>>> 'dstring' in smalldict

True

```

**注意:**这是**而不是**检查一个*值*是否存在于字典中，只有一个*键*:

```py

>>> 'new pystring' in smalldict

False

```

## 按值搜索 Python 字典

要检查给定的*值*，可以使用字典的标准方法之一`values`。

```py

>>>'new pystring' in smalldict.values()

True

```

对于字典对象，还有许多更有用的方法和配方，但这些都是基本的。

请记住，*key*不一定是字符串，可以是您可以创建的任何不可变对象——值可以是任何东西，包括其他字典。不要害怕发挥创造力，发现你能用它们做什么！