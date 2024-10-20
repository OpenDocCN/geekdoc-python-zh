# Python 集合模块中的 UserDict 和 UserList–概述！

> 原文：<https://www.askpython.com/python-modules/userdict-and-userlist-in-python>

读者朋友们，你们好！在本文中，我们将关注 Python 集合模块中的 **UserDict 和 UserList。所以，让我们开始吧！🙂**

* * *

## Python 集合模块——快速刷起来！

Python 为我们提供了众多模块来处理不同形式的数据，并推动自动化和可持续发展。Python 集合模块就是这样一个模块。

[集合模块](https://www.askpython.com/python-modules/python-collections)为我们提供了一种简单的方法来在一个屋檐下存储相似类型的数据。顾名思义，集合是一组共享相似特征的实体，集合模块提供的特性也是如此。

在本文过程中，我们将重点关注该模块提供的以下集合——

1.  **用户词典**
2.  **UserList**

在接下来的部分中，让我们来看看它们。

* * *

## 了解 Python UserDict

众所周知，Python 为我们提供了[字典数据结构](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)来处理键值形式的数据。UserDict 为其添加了定制功能。

也就是说，Dictionary 帮助我们创建一个以静态格式保存键值对的数据结构。使用 UserDict，我们可以通过创建一个定制的字典来添加一些修改过的功能。

它的行为类似于字典对象周围的一个[包装类](https://www.askpython.com/python/examples/decorators-in-python)。这样，我们可以很容易地向现有的 dictionary 对象添加新的行为。

UserDict 集合接受现有字典作为参数，并触发一个存储在普通 Dict 对象中的字典结构。

看看下面的语法！

```py
collections.UserDict(data)

```

**举例:**

在下面的例子中，我们使用现有的 dictionary 对象创建了一个 UserDict。在这种情况下，字典现在可以作为属性进行更改。

```py
from collections import UserDict 

data = {'Pune':100, 
	'Satara': 28, 
	'Mumbai': 31} 

user_dict = UserDict(data) 
print(user_dict.data)

```

**输出:**

```py
{'Pune': 100, 'Satara': 28, 'Mumbai': 31}

```

在下面的例子中，我们展示了一个定制类对 UserDict 的定制使用。

在这里，我们已经创建了一个 UserDict，它充当了一个定制列表“mydict”的包装类。

因此，它充当一个包装类，让我们将现有字典的属性添加到 UserDict 中。

这里，我们在字典中添加了一个行为，限制元素的删除。

UserDict 将默认创建的字典包装到其中，并灌输类中声明的定制行为。

**举例:**

```py
from collections import UserDict

class mydata(UserDict):

	def pop(self, s = None):
		raise RuntimeError("Deletion not allowed")

mydict = mydata({'x':10,
    'y': 20})

print(mydict)

#Deliting From Dict
mydict.pop()

```

**输出:**

```py
{'x': 10, 'y': 20}
Traceback (most recent call last):
  File "c:/Users/HP/OneDrive/Desktop/demo.py", line 15, in <module>
    mydict.pop()
  File "c:/Users/HP/OneDrive/Desktop/demo.py", line 7, in pop      
    raise RuntimeError("Deletion not allowed")
RuntimeError: Deletion not allowed

```

* * *

## 了解 Python 用户列表

像 UserDict 一样，UserList 也为我们提供了一种在 Python 中定制[列表的方法，以便灌输到类中。Python List 存储了具有不同数据类型的相似类型的数据。UserList 帮助我们定制列表，并使用它们作为属性来创建用户定义的类。将列表作为实例添加后，它会触发一个列表，该列表保存在常用的列表数据结构中。](https://www.askpython.com/python/difference-between-python-list-vs-array)

**语法:**

```py
collections.UserList(list)

```

**举例:**

在这个例子中，我们利用 UserList 将常规列表作为参数存储在其中。此外，我们可以使用 UserList 集合和属性作为一个列表来创建定制的类。

```py
from collections import UserList 

lst = [1,2,3,4,5]

user_list = UserList(lst) 
print(user_list.data) 

```

**输出:**

```py
[1, 2, 3, 4, 5]

```

在下面的例子中，我们展示了一个定制类对 UserDict 的定制使用。

在这里，我们已经创建了一个用户列表，它作为一个自定义列表“mylist”的包装类。因此，它充当一个包装类，让我们将现有字典的属性添加到用户列表中。这里，我们向列表添加了一个行为，限制元素的删除，我们甚至通过用户列表作为包装类向列表添加/插入值。UserList 将默认创建的字典包装到其中，并灌输类中声明的定制行为。

**举例:**

```py
from collections import UserList

class mydata(UserList):

	def pop(self, s = None):
		raise RuntimeError("Deletion not allowed")

mylist = mydata([10,20,30])

mylist.append(5)
print("Insertion..")
print(mylist)

mylist.pop()

```

**输出:**

```py
After Insertion
[10,20,30,5]
Traceback (most recent call last):
  File "c:/Users/HP/OneDrive/Desktop/demo.py", line 20, in <module>
    L.pop()
  File "c:/Users/HP/OneDrive/Desktop/demo.py", line 7, in pop
    raise RuntimeError("Deletion not allowed")
RuntimeError: Deletion not allowed

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂