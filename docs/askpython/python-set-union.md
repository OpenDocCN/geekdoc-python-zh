# Python 集合并集

> 原文：<https://www.askpython.com/python/set/python-set-union>

Union 基本上表示相应集合的所有*不同元素*。

[Python 集合](https://www.askpython.com/python/set/python-set) **union()方法**找到集合的并集，并表示一个新集合，该集合包含来自相应输入集合的所有项目。

**注:** *如果一个集合包含一个出现不止一次的元素，那么所表示的输出只包含该特定元素的一次出现。*

**语法:**

```py
set1.union(set2, set3,......, setN)
```

* * *

## 对 Python 集合并集的基本理解

```py
info = {'Safa', 'Aman', 'Divya', 'Elen'}
info1 = {'Safa', 'Aryan'}

print(info.union(info1))

```

**输出:**

```py
{'Aman', 'Safa', 'Divya', 'Aryan', 'Elen'}
```

* * *

## 使用“|”运算符的 Python 集合联合

**"| "操作符**也可以用来寻找输入集合的并集。

```py
info = {'Safa', 'Aman', 'Diya'}
info1 = {'Varun', 'Rashi', 54 }

print(info | info1)

```

**输出:**

```py
{'Diya', 'Aman', 'Safa', 54, 'Varun', 'Rashi'}
```

* * *

## 多个 Python 集的并集

以下任一技术可用于寻找多个集合的并集:

*   *将多个集合作为参数传递给 union()方法*
*   *通过创建 union()方法调用链*

* * *

### 1.使用多个集合作为参数的多个集合的并集

```py
info = {12, 14, 15, 17}
info1 = {35, 545}
info2 = {'Safa','Aman'}

print(info.union(info1, info2))

```

**输出:**

```py
{'Safa', 17, 545, 35, 'Aman', 12, 14, 15}
```

* * *

### 2.通过创建 Union()方法调用链来联合多个集合

```py
info = {12, 14, 15, 17}
info1 = {35, 545}
info2 = {'Safa','Aman'}

print(info.union(info1).union(info2))

```

**输出:**

```py
{'Aman', 17, 545, 35, 'Safa', 12, 14, 15}
```

* * *

## 结论

因此，在本文中，我们已经理解并实现了在 Python 中寻找集合并集的方法。

* * *

## 参考

*   Python 集合
*   [设置文档](https://docs.python.org/3.8/library/stdtypes.html#set-types-set-frozenset)