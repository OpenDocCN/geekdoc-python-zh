# Python 101 -了解字典

> 原文：<https://www.blog.pythonlibrary.org/2020/03/31/python-101-learning-about-dictionaries/>

字典是 Python 中的另一种基本数据类型。字典是一个键、值对。一些编程语言将它们称为哈希表。它们被描述为一个*映射*对象，将散列值映射到任意对象。

字典的键必须是不可变的，也就是说，不能改变。从 Python 3.7 开始，字典是有序的。这意味着当你添加一个新的键，值对到字典时，它会记住它们的添加顺序。在 Python 3.7 之前，情况并非如此，您不能依赖插入顺序。

在本章中，您将学习如何执行以下操作:

*   创建词典
*   访问词典
*   字典方法
*   修改词典
*   从字典中删除

让我们从学习创建字典开始吧！

你可以用几种不同的方法创建字典。最常见的方法是将逗号分隔的列表`key: value`对放在花括号内。

让我们看一个例子:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict
{'email': 'jdoe@gmail.com', 'first_name': 'James', 'last_name': 'Doe'}
```

还可以使用 Python 内置的`dict()`函数来创建字典。`dict()`将接受一系列关键字参数(即 1= '一'，2= '二'等)，一个元组列表或另一个字典。

这里有几个例子:

```py
>>> numbers = dict(one=1, two=2, three=3)
>>> numbers
{'one': 1, 'three': 3, 'two': 2}
>>> info_list = [('first_name', 'James'), ('last_name', 'Doe'), ('email', 'jdoes@gmail.com')]
>>> info_dict = dict(info_list)
>>> info_dict
{'email': 'jdoes@gmail.com', 'first_name': 'James', 'last_name': 'Doe'}
```

第一个例子在一系列关键字参数上使用了`dict()`。当你学习函数的时候，你会学到更多。您可以将关键字参数看作是一系列关键字，在它们和它们的值之间有等号。

第二个例子展示了如何创建一个包含 3 个元组的列表。然后您将该列表传递给`dict()`以将其转换为字典。

## 访问词典

词典因其速度快而出名。您可以通过键访问字典中的任何值。如果没有找到密钥，您将收到一个`KeyError`。

让我们来看看如何使用字典:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict['first_name']
'James'
```

要获得`first_name`的值，必须使用以下语法:`dictionary_name[key]`

现在让我们试着得到一个不存在的键:

```py
>>> sample_dict['address']
Traceback (most recent call last):
   Python Shell, prompt 118, line 1
builtins.KeyError: 'address'
```

那没用。你要求字典给你一个字典里没有的值！

您可以使用 Python 的`in`关键字来询问一个键是否在字典中:

```py
>>> 'address' in sample_dict
False
>>> 'first_name' in sample_dict
True
```

您还可以通过使用 Python 的`not`关键字来检查字典中的键是否是**而不是**:

```py
>>> 'first_name' not in sample_dict
False
>>> 'address' not in sample_dict
True
```

访问字典中的键的另一种方法是使用字典方法之一。现在让我们了解更多关于字典方法的知识吧！

## 字典方法

与大多数 Python 数据类型一样，字典有您可以使用的特殊方法。让我们来看看字典的一些方法吧！

### d.get(键[，默认值])

您可以使用`get()`方法来获取一个值。`get()`要求您指定要查找的键。如果找不到键，它允许您返回一个默认值。默认是`None`。让我们来看看:

```py
>>> print(sample_dict.get('address'))
None
>>> print(sample_dict.get('address', 'Not Found'))
Not Found
```

第一个例子向您展示了当您试图在不设置`get`的`default`的情况下`get()`一个不存在的键时会发生什么。在这种情况下，它返回`None`。然后，第二个示例向您展示了如何将默认值设置为字符串“Not Found”。

### d .清除()

方法可以用来从字典中删除所有的条目。

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict
{'email': 'jdoe@gmail.com', 'first_name': 'James', 'last_name': 'Doe'}
>>> sample_dict.clear()
>>> sample_dict
{}
```

### d.copy()

如果您需要创建字典的浅层副本，那么`copy()`方法适合您:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> copied_dict = sample_dict.copy()
>>> copied_dict
{'email': 'jdoe@gmail.com', 'first_name': 'James', 'last_name': 'Doe'}
```

如果你的字典里面有对象或者字典，那么你可能会因为这个方法而陷入逻辑错误，因为改变一个字典会影响到副本。在这种情况下，您应该使用 Python 的`copy`模块，它有一个`deepcopy`函数，可以为您创建一个完全独立的副本。

### d .项目()

`items()`方法将返回字典条目的新视图:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict.items()
dict_items([('first_name', 'James'), ('last_name', 'Doe'), ('email', 'jdoe@gmail.com')])
```

这个视图对象将随着字典对象本身的改变而改变。

### 钥匙()

如果您需要查看字典中的键，那么`keys()`就是适合您的方法。作为一个视图对象，它将为您提供字典键的动态视图。您可以迭代一个视图，也可以检查成员资格视图`in`关键字:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> keys = sample_dict.keys()
>>> keys
dict_keys(['first_name', 'last_name', 'email'])
>>> 'email' in keys
True
>>> len(keys)
3
```

### d.values()

`values()`方法也返回一个视图对象，但是在这种情况下，它是字典值的动态视图:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> values = sample_dict.values()
>>> values
dict_values(['James', 'Doe', 'jdoe@gmail.com'])
>>> 'Doe' in values
True
>>> len(values)
3
```

### d.pop(键[，默认])

你需要从字典中删除一个键吗？那么`pop()`就是适合你的方法。`pop()`方法接受一个键和一个选项默认字符串。如果不设置默认值，并且没有找到密钥，将会出现一个`KeyError`。

以下是一些例子:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict.pop('something')
Traceback (most recent call last):
   Python Shell, prompt 146, line 1
builtins.KeyError: 'something'
>>> sample_dict.pop('something', 'Not found!')
'Not found!'
>>> sample_dict.pop('first_name')
'James'
>>> sample_dict
{'email': 'jdoe@gmail.com', 'last_name': 'Doe'}
```

### d 波普姆()

`popitem()`方法用于从字典中移除和返回一个`(key, value)`对。这些对按照后进先出(LIFO)的顺序返回。如果在一个空字典上被调用，你会收到一个`KeyError`

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict.popitem()
('email', 'jdoe@gmail.com')
>>> sample_dict
{'first_name': 'James', 'last_name': 'Doe'}
```

### d .更新([其他])

用来自*其他*的`(key, value)`对更新字典，覆盖现有的键。返回`None`。

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict.update([('something', 'else')])
>>> sample_dict
{'email': 'jdoe@gmail.com',
'first_name': 'James',
'last_name': 'Doe',
'something': 'else'}
```

## 修改你的字典

你需要不时修改你的字典。假设您需要添加一个新的键，值对:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict['address'] = '123 Dunn St'
>>> sample_dict
{'address': '123 Dunn St',
'email': 'jdoe@gmail.com',
'first_name': 'James',
'last_name': 'Doe'}
```

要向字典中添加一个新条目，可以使用方括号输入一个新键并将其设置为一个值。

如果需要更新预先存在的密钥，可以执行以下操作:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict['email'] = 'jame@doe.com'
>>> sample_dict
{'email': 'jame@doe.com', 'first_name': 'James', 'last_name': 'Doe'}
```

在本例中，您将`sample_dict['email']`设置为`jame@doe.com`。每当您将预先存在的键设置为新值时，都会覆盖以前的值。

## 从字典中删除项目

有时你需要从字典中删除一个键。您可以使用 Python 的`del`关键字来实现:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> del sample_dict['email']
>>> sample_dict
{'first_name': 'James', 'last_name': 'Doe'}
```

在这种情况下，您告诉 Python 从`sample_dict`中删除键“email”。

移除键的另一种方法是使用字典的`pop()`方法，这在上一节中已经提到:

```py
>>> sample_dict = {'first_name': 'James', 'last_name': 'Doe', 'email': 'jdoe@gmail.com'}
>>> sample_dict.pop('email')
'jdoe@gmail.com'
>>> sample_dict
{'first_name': 'James', 'last_name': 'Doe'}
```

当你使用`pop()`时，它将返回被删除的值。

## 包扎

字典数据类型非常有用。你会发现用它来快速查找各种数据很方便。可以将`key: value`对的值设置为 Python 中的任何对象。因此，您可以将列表、元组或对象作为值存储在字典中。

如果你需要一个字典，当你去获取一个不存在的键时，它可以创建一个默认值，你应该看看 Python 的`collections`模块。它有一个`defaultdict`类，就是为这个用例设计的。

## 相关阅读

*   python 101-[了解元组](https://www.blog.pythonlibrary.org/2020/03/26/python-101-learning-about-tuples/)
*   Python 101: [了解列表](https://www.blog.pythonlibrary.org/2020/03/10/python-101-learning-about-lists/)