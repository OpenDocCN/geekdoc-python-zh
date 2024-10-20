# python chain map–您需要知道的一切！

> 原文：<https://www.askpython.com/python/python-chainmap>

读者朋友们，你们好！在本文中，我们将详细关注 **Python 链图**。

所以，让我们开始吧！！🙂

* * *

## 什么是 Python ChainMap 数据结构？

在深入了解 **ChainMap** 的概念之前，让我们快速回顾一下对 [Python 集合模块](https://www.askpython.com/python-modules/python-collections)的理解。

Python 集合模块为我们提供了各种数据结构来存储和操作数据。通过提供定制的数据存储和操作，它从所有其他默认数据结构中脱颖而出。

该模块表现为存储不同类型的数据对象的容器，并具有用于将其塑造成定制数据结构的定制特征。

集合模块提供的一个这样的容器是 ChainMap！

使用 ChainMap，我们可以将不同类型的键值实体合并并存储到一个地方。ChainMap 容器使我们能够拥有多个字典，然后将它们合并成一个字典。

## 导入 Python 链图

为了实现链图模块的功能，我们需要导入如下所示的模块

```py
from collections import ChainMap

```

做完这些，现在让我们试着从多个字典中创建一个通用的链表结构

**语法**:

```py
ChainMap(dict, dict)

```

**举例**:

在下面的例子中，我们已经创建了两个字典 **A** 和 **B** ，然后我们将它们合并起来作为一个实体。

**输出**:

```py
from collections import ChainMap 

a = {'Age': 1, 'Name': 'X'} 
b = {'Age': 3, 'Name': 'Y'} 
cm = ChainMap(a, b) 

print(cm)

```

**输出—**

```py
ChainMap({'Age': 1, 'Name': 'X'}, {'Age': 3, 'Name': 'Y'})

```

* * *

## 链图模块中需要了解的重要函数

随着 ChainMap container 的出现，出现了一个庞大的函数列表，人们可以用它来操作存储在其中的字典。让我们看看下面的函数。

*   **按键()**
*   **值()**
*   **地图**T2 属性

* * *

### 1.Chainmap.keys()函数

顾名思义，keys()函数使我们能够从每个字典的每个键值对中提取键值。可以一次从多个字典中提取关键字。

**语法—**

```py
ChainMap.keys()

```

**示例—**

```py
from collections import ChainMap 

a = {'Age': 1, 'Name': 'X'} 
b = {'Age': 3, 'Name': 'Y'} 
cm = ChainMap(a, b) 

print(cm)
print ("Keys: ") 
print (list(cm.keys())) 

```

**输出—**

```py
ChainMap({'Age': 1, 'Name': 'X'}, {'Age': 3, 'Name': 'Y'})
Keys:
['Age', 'Name'] 

```

* * *

### 2.ChainMap.maps 属性

为了获得更清晰的输出，maps 属性使我们能够将每个键与字典中的值相关联。因此，它将输出表示为字典的每个键值对。

**语法—**

```py
ChainMap.maps

```

**示例—**

```py
from collections import ChainMap 

a = {'Age': 1, 'Name': 'X'} 
b = {'Age': 3, 'Name': 'Y'} 
cm = ChainMap(a, b) 

print(cm)
print ("Maps: ")
print (list(cm.maps))

```

**输出—**

```py
ChainMap({'Age': 1, 'Name': 'X'}, {'Age': 3, 'Name': 'Y'})
Maps:
[{'Age': 1, 'Name': 'X'}, {'Age': 3, 'Name': 'Y'}]

```

* * *

### 3.Chainmap.values()函数

除了将键和整个字典显示为一个映射之外，values()函数使我们能够独立地提取和表示与特定键相关联的所有值。

**语法—**

```py
ChainMap.values()

```

**示例—**

```py
from collections import ChainMap 

a = {'Age': 1, 'Name': 'X'} 
b = {'Std': 3} 
cm = ChainMap(a, b) 

print(cm)
print ("Values: ") 
print (list(cm.values()))

```

**输出—**

```py
ChainMap({'Age': 1, 'Name': 'X'}, {'Std': 3})
Values: 
[1, 3, 'X']

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂