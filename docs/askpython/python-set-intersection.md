# Python 集合交集

> 原文：<https://www.askpython.com/python/set/python-set-intersection>

Python [**集合**](https://www.askpython.com/python/set/python-set) 交集基本上是寻找并返回集合间共有的元素。

**语法:**

```py
set1.intersection(set2, set3, ... setN)
```

set1.intersection(set2)等价于 set1 ∩ set2。

**参数**:接受一个或多个集合作为参数。

**返回值**:返回一个集合作为输出，其中包含所有集合共有的元素。

* * *

## 在 Python 中实现集合交集的方法

以下任一方法都可用于执行 Python 集合相交:

*   ***利用交点()法***
*   利用 Python 的***[【按位】&运算符](https://www.askpython.com/python/python-bitwise-operators)***
*   ***利用交集 _ 更新()方法***
*   ***通过使用&=*运算符**

* * *

### Python 使用 Intersection()方法设置交集

*   intersection()方法以一个或多个 iterables 作为参数，即[字符串](https://www.askpython.com/python/string/python-string-functions)、[列表](https://www.askpython.com/python/list/python-list)、[元组](https://www.askpython.com/python/tuple/python-tuple)等。
*   该方法比较并找出传递的 iterables 中的公共元素。
*   最后，创建一个新的集合作为输出，它包含 iterables 共有的元素。

**注意:** *如果集合以外的任何可迭代对象作为参数传递，那么首先将可迭代对象转换为集合对象，然后对其进行交集运算。*

**举例:**

```py
set1 = {10, 20, 30}
set2 = {30, 3, 9}

output = set1.intersection(set2)

print(output)

```

**输出:**

```py
{30}
```

* * *

### Python 使用按位“&”运算符设置交集

*   Python**“&”操作符**也返回两个或更多集合的元素的交集。
*   **&操作符**和**交集()方法**的唯一区别是&操作符只对集合对象进行操作，而交集方法可以对列表、集合等任何可迭代对象进行操作。

**举例:**

```py
set1 = {"Safa", "Aman", "Pooja", "Divya"}

set2 = {"Safa", "Aryan", "Nisha", "Raghav", "Divya"}

Result = set1 & set2

print('Set 1: ',set1)
print('Set 2: ',set2)
print('Set Intersection: ',Result)

```

**输出:**

```py
Set 1:  {'Safa', 'Pooja', 'Divya', 'Aman'}
Set 2:  {'Nisha', 'Aryan', 'Raghav', 'Safa', 'Divya'}
Set Intersection:  {'Safa', 'Divya'}
```

* * *

### Python 使用 intersection_update()方法设置交集

**intersection_update()方法**基本上是返回 iterables 之间的公共元素，并更新执行操作的同一个 set/iterable 对象。

**注意:** *它不会创建一个新的集合作为输出。相反，它用交集操作*的结果更新相同的输入集。*看下面的例子可以更好地理解*

**举例:**

```py
set1 = {"Safa", "Aman", "Pooja", "Divya"}

set2 = {"Safa", "Aryan", "Nisha", "Raghav", "Divya"}

print("Set1 before intersection operation: ", set1)
set1.intersection_update(set2)

print('Set Intersection of set1 and set2: ',set1)
print('Updated Set1: ',set1)

```

**输出:**

```py
Set1 before intersection operation:  {'Aman', 'Pooja', 'Divya', 'Safa'}
Set Intersection of set1 and set2:  {'Divya', 'Safa'}
Updated Set1:  {'Divya', 'Safa'}
```

* * *

### Python 使用“&=”运算符设置交集

**" & = "操作符**也返回集合对象之间的交集。

**注意:***“&=”运算符只对集合对象执行和操作。它不支持任何其他可迭代对象，如列表、字符串等。*

**举例:**

```py
set1 = {"Safa", "Aman", "Pooja", "Divya"}

set2 = {"Safa", "Aryan", "Nisha", "Raghav", "Divya"}

print("Set1 before intersection operation: ",set1)

set1 &= set2

print('Set Intersection of set1 and set2: ',set1)

print("Updated Set1: ", set1)

```

**输出:**

```py
Set1 before intersection operation:  {'Divya', 'Safa', 'Pooja', 'Aman'}
Set Intersection of set1 and set2:  {'Divya', 'Safa'}
Updated Set1:  {'Divya', 'Safa'}
```

* * *

## 结论

因此，在本文中，我们用可能的方法研究并实现了 Python 集合交集。

* * *

## 参考

*   Python 集合交集
*   [Python 集合文档](https://docs.python.org/3.8/library/stdtypes.html#set-types-set-frozenset)