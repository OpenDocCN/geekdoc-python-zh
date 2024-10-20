# 如何在 Python 中切片定制对象/类

> 原文：<https://www.pythoncentral.io/how-to-slice-custom-objects-classes-in-python/>

关于切片的介绍，请查看文章[如何在 Python](https://www.pythoncentral.io/how-to-slice-listsarrays-and-tuples-in-python/ "How to Slice Lists/Arrays and Tuples in Python") 中对列表/数组和元组进行切片。

## **分割你自己的 Python 对象**

好了，这一切都很酷，现在我们知道如何切片。但是我们如何分割我们自己的物体呢？如果我实现了一个有列表或者自定义数据结构的对象，我如何使它可切片？

首先，我们定义我们的自定义数据结构类。

```py

from collections import Sequence
class my structure(Sequence):
def _ _ init _ _(self):
self . data =[]
def __len__(self): 
返回 len(self.data)
def append(自身，项目):
自身.数据. append(项目)
定义移除(self，item):
self . data . remove(item)
def __repr__(self): 
返回 str(self.data)
def __getitem__(self，sliced): 
返回 self.data[sliced] 

```

这里，我们声明一个带有列表的类作为我们结构的后端。并没有做那么多，但是你可以添加、删除和获取项目，所以从某种意义上来说它还是有用的。这里我们唯一需要特别注意的方法是`__getitem__`方法。每当我们试图从我们的结构中获取一个项目时，就会调用这个方法。当我们调用`structure[0]`时，这实际上是在后台调用`__getitem__`方法，并返回该方法返回的任何内容。这在实现列表样式对象时非常有用。

让我们创建自定义对象，向其添加一个项目，并尝试获取它。

```py

>>> m = MyStructure()

>>> m.append('First element')

>>> print(m[0])

First element

```

是啊！我们做到了！我们分配了我们的元素，并正确地取回了它。很好。现在是切片的部分！

```py

>>> m.append('Second element')

>>> m.append('Third element')

>>> m

['First element', 'Second element', 'Third element']

>>> m[1:3]

['Second element', 'Third element']

```

厉害！因为使用了`list`的功能，我们基本上可以免费获得所有切片。

这真是太神奇了。Python 很神奇。来点更复杂的怎么样？使用字典作为我们的主要数据结构怎么样？

好吧。让我们定义另一个使用字典的类。

```py

class MyDictStructure(Sequence):

def __init__(self):

# New dictionary this time

self.data = {}
def __len__(self): 
返回 len(self.data)
def append(self，item):
self . data[len(self)]= item
def remove(self，item):
if item in self . data . values():
del self . data[item]
def __repr__(self): 
返回 str(self.data)
def __getitem__(self，sliced):
sliced keys = self . data . keys()【sliced】
data = { k:self . data[k]for key in sliced keys }
返回数据

```

好吧，那我们在这里做什么？基本上和我们在`MyStructure`中做的一样，但是这一次我们使用字典作为我们的主要结构，并且`__getitem__`方法定义已经改变。让我解释一下`__getitem__`发生了什么变化。

首先，我们获得传递给`__getitem__`的`slice`对象，并使用它来分割字典的键值。然后，使用字典理解，我们使用我们的分片键将所有的分片键和值放入一个新字典中。这给了我们一个很好的切片字典，我们可以用它来做任何我们喜欢的事情。

这就是它的作用:

```py

>>> m = MyDictStructure()

>>> m.append('First element')

>>> m.append('Second element')

>>> m.append('Third element')

>>> m

{0: 'First element', 1: 'Second element', 2: 'Third element'}

>>> # slicing

>>> m[1:3]

{1: 'Second element', 2: 'Third element'}

```

仅此而已。简单、优雅、强大。

如果你想知道如何分割字符串，在另一篇名为[如何在 Python 中从字符串中获取子字符串——分割字符串](https://www.pythoncentral.io/how-to-get-a-substring-from-a-string-in-python-slicing-strings/ "How to Get a Sub-string From a String in Python – Slicing Strings")的文章中有所涉及。

目前就这些。希望你喜欢学习切片，我希望它会帮助你的追求。

再见。