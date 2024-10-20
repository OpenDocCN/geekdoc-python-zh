# Python 中旋转列表的 4 种简单方法

> 原文：<https://www.askpython.com/python/examples/rotate-lists-in-python>

嘿伙计们！在今天的教程中，我们将学习如何使用 python 编程语言旋转列表。列表旋转是一种简单的方法，对程序员很有帮助。让我们回顾一下实现旋转的各种选择。

* * *

## Python 中的旋转列表

现在让我们来理解如何在 Python 中旋转列表。我们将在下面看看多种不同的方法。

### 方法 1:切片操作

旋转列表的另一种方法是切片。这个 [`len()`方法](https://www.askpython.com/python/list/length-of-a-list-in-python)就是用来做这个的。在下面的示例中，列表被切片。

在这种情况下，`n_splits`的值为 1，表示切片的数量。因此，列表以相同的方式循环。

```py
# Define the list
list_1 = [1,2,3,4,5] 
print("Original list:", list_1)
# Set the number of splits
n_splits = 1
# Rotate The List
list_1 = (list_1[len(list_1) - n_splits:len(list_1)] + list_1[0:len(list_1) - n_splits]) 
print("Rotated list:", list_1)

```

```py
Original list: [1, 2, 3, 4, 5]
Rotated list: [5, 1, 2, 3, 4]

```

* * *

### 方法 2:遍历操作

这是在 Python 中旋转列表最简单的方法。顾名思义，我们逐一查看列表。然后将元素放置在其正确的位置。

下面的示例演示了这种策略。在这种情况下，我们将列表旋转整数 n，即 1。

```py
def ROTATE (lists, n): 
    output_list = [] 
    x= len(lists)
    for item in range(x - n, x): 
        output_list.append(lists[item])        
    for item in range(0, x - n):  
        output_list.append(lists[item]) 
    return output_list 
rotate_num = 1
list_1 = [1, 2, 3, 4, 5] 
print("Original List:", list_1)
print("Rotated list: ",ROTATE(list_1, rotate_num))

```

```py
Original List: [1, 2, 3, 4, 5]
Rotated list:  [5, 1, 2, 3, 4]

```

* * *

### 方法三:列出理解

在这种方法中，我们通过在旋转后给每个元素重新分配一个新的索引来修改列表的索引。在下面的示例中，列表旋转一次，并分配新的索引值。

```py
list_1 = [1, 2, 3, 4, 5] 
print ("Original List : " + str(list_1)) 
list_1 = [list_1[(i + 4) % len(list_1)] for i, x in enumerate(list_1)]
print ("Rotated list : " + str(list_1)) 

```

```py
Original List : [1, 2, 3, 4, 5]
Rotated list : [5, 1, 2, 3, 4]

```

* * *

### 方法 4:使用集合模块

Python 中有一个`collection`模块，它有一个`deque`类。这个类包含一个 rotate()方法。

在下面的例子中，我们利用了内置函数`rotate()`。

```py
from collections import deque 
list_1 = [1, 2, 3, 4, 5]  
print ("Original List : " + str(list_1)) 
list_1 = deque(list_1) 
list_1.rotate(-4) 
list_1 = list(list_1) 
print ("Rotated List: " + str(list_1)) 

```

```py
Original List : [1, 2, 3, 4, 5]
Rotated List: [5, 1, 2, 3, 4]

```

* * *

## 结论

恭喜你！您刚刚学习了如何使用多种方法对列表执行旋转。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [用 Python 将字典转换成列表的 5 种简单方法](https://www.askpython.com/python/dictionary/convert-a-dictionary-to-a-list)
2.  [如何在 Python 中将列表转换成数据帧？](https://www.askpython.com/python-modules/pandas/convert-lists-to-dataframes)
3.  [如何用 Python 把列表转换成字典？](https://www.askpython.com/python/list/convert-list-to-a-dictionary)
4.  [打印 Python 列表的 3 种简单方法](https://www.askpython.com/python/list/print-a-python-list)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *