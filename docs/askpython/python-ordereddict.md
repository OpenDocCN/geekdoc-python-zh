# Python 有序直接

> 原文：<https://www.askpython.com/python-modules/python-ordereddict>

OrderedDict 是 dictionary 的一个子类，它维护向其中添加元素/项目的顺序。

OrderedDict 保留元素的插入顺序。默认的**字典不保存顺序**，并以任意顺序进行迭代。

最初，我们需要导入[集合模块](https://www.askpython.com/python-modules/python-collections)来使用 OrderedDict 库模块。我们还可以从 collections 模块中只导入 OrderedDict 类。

```py
import collections

from collections import OrderedDict

```

* * *

## Python 有序直接功能

*   创建有序的直接对象
*   向 OrderedDict 添加项目
*   替换订单中的项目
*   从订单中删除项目
*   键值改变
*   函数的作用是
*   订购的直接弹出项目
*   反向迭代
*   有序直接相等测试

* * *

### 1.创建有序的直接对象

`**OrderedDict()**`函数用于创建对象。

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
ordered_input = OrderedDict(my_input)
print(ordered_input)

```

**输出:**

```py
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
```

* * *

### 2.向 OrderedDict 添加项目

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
ordered_input = OrderedDict(my_input)
#print(ordered_input)

print("Adding item to OrderedDict....")
ordered_input['Hyderabad'] = 'Karnataka'
print(ordered_input)

```

**输出:**

```py
Adding item to OrderedDict....
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar'), ('Hyderabad', 'Karnataka')])
```

* * *

### 3.替换订单中的项目

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
ordered_input = OrderedDict(my_input)
#print(ordered_input)

print("Replacing item from OrderedDict....")
ordered_input['Pune'] = 'Satara'
print(ordered_input)

```

**输出:**

```py
Adding items to OrderedDict....
OrderedDict([('Pune', 'Satara'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')]) 
```

* * *

### 4.从订单中删除项目

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
ordered_input = OrderedDict(my_input)
#print(ordered_input)

print('Removal of item from OrderedDict....')
ordered_input.pop('Pune')
print(ordered_input)

```

**输出:**

```py
Removal of item from OrderedDict....
OrderedDict([('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
```

* * *

### 5.OrderedDict 中的键值更改

在 OrderedDict 中，如果对应于特定键的值被更改，则该键的位置/索引保持不变。

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
print('Before the change.....')
ordered_input = OrderedDict(my_input)
print(ordered_input)

print('After the change.....')
ordered_input['Pune'] = 'Kiara'
print(ordered_input)

```

**输出:**

```py
Before the change.....
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
After the change.....
OrderedDict([('Pune', 'Kiara'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
```

* * *

### 6.函数的作用是

函数将一个特定的键值对移动到字典的末尾。

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
print('Before using the move_to_end().....')
ordered_input = OrderedDict(my_input)
print(ordered_input)

print('After using the move_to_end().....')
ordered_input.move_to_end('Pune')
print(ordered_input)

```

**输出:**

```py
Before using the move_to_end().....
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
After using the move_to_end().....
OrderedDict([('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar'), ('Pune', 'Maharashtra')]) 
```

* * *

### 7.OrderedDict popitem

这个函数弹出并返回最后一个元素作为输出。

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
print('Original input dict.....')
ordered_input = OrderedDict(my_input)
print(ordered_input)

result = ordered_input.popitem(True)
print('The popped item is: ')
print(result)

print(ordered_input)

```

**输出:**

```py
Original input dict.....
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
The popped item is: 
('Orrisa', 'Bhubhaneshwar')
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat')])
```

* * *

### 8.反向迭代

```py
from collections import OrderedDict

my_input = {'Pune': 'Maharashtra', 'Ahemadnagar': 'Gujarat', 'Orrisa': 'Bhubhaneshwar'}

# creating ordered dict from dict
print('Original input dict.....')
ordered_input = OrderedDict(my_input)
print(ordered_input)

print('Reversed OrderedDict.....')

for elements in reversed(ordered_input):
    print(elements)

```

**输出:**

```py
Original input dict.....
OrderedDict([('Pune', 'Maharashtra'), ('Ahemadnagar', 'Gujarat'), ('Orrisa', 'Bhubhaneshwar')])
Reversed OrderedDict.....
Orrisa
Ahemadnagar
Pune 
```

* * *

### 9.有序直接相等测试

```py
from collections import OrderedDict

# creating regular dict..
my_input1 = {1:'one' , 2:'two'}
my_input2 = {2:'two' , 1:'one'}

#creating ordereddict..
ordered_input1 = OrderedDict({1:'one' , 2:'two'})
ordered_input2 = OrderedDict({2:'two' , 1:'one'})

print(my_input1 == ordered_input1)
print(my_input1 == my_input2)
print(ordered_input1 == ordered_input2)

```

**输出:**

```py
True
True
False
```

* * *

## 结论

因此，在本文中，我们已经理解了常规 Dictionary 和 OrderedDict 之间的区别，并了解了 OrderedDict 提供的功能。

* * *

## 参考

*   Python 有序直接
*   [订购的直接文件](https://docs.python.org/3/library/collections.html#collections.OrderedDict)