# Python 集——你必须知道的事情

> 原文：<https://www.askpython.com/python/set/python-set>

Python 集合是一个无序的、无索引的元素集合。

*   每个元素都是独一无二的。
*   该集合包含未排序的元素。
*   不允许重复。
*   集合本身是可变的，即可以从其中添加/删除项目(元素)。
*   与元素按顺序存储的数组不同，集合中元素的顺序没有定义。
*   集合中的元素不是按照它们在集合中出现的顺序存储的。

* * *

## 在 Python 中创建集合

集合可以通过将所有元素放在花括号{}内，用逗号分隔来创建。它们也可以通过使用内置函数`set()`来创建。

元素可以是不同的数据类型，但是集合不支持可变元素。集合是无序的，所以人们不能确定元素出现的顺序。

### 示例:创建集合

```py
Days=set(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
Fruits = {"apple", "banana", "cherry"}
Name=set('Quit')
print(Name)
print(Fruits)
print(Days)

```

**输出**:

{'u '，' Q '，' I '，' t'}
{ '樱桃'，'香蕉'，'苹果' }
{ '太阳'，'周三'，'周一'，'周四'，'周二'，'周六'，' Fri'}

**推荐阅读:**

1.  [Python 中的列表](https://www.askpython.com/python/list/python-list)
2.  [Python 中的数组](https://www.askpython.com/python/array/python-array-examples)
3.  [Python 元组](https://www.askpython.com/python/tuple/python-tuple)

* * *

## 从 Python 集合中访问元素

因为集合是无序的和无索引的，所以不能像数组那样通过引用索引来访问元素。

集合中的元素可以通过以下方式之一进行访问:

1.  使用`for`循环遍历集合项目的循环。
2.  使用`in`关键字检查集合中是否存在指定的值。

### 示例:从集合中访问元素

```py
Fruits = {"apple", "mango", "cherry"}
for a in Fruits:
  print(a)
print("banana" in Fruits)
print("mango" in Fruits)

```

**输出**:

芒果
樱桃
苹果
假
真

* * *

## 向 Python 集添加元素

我们可以使用`add()` 函数向集合中添加元素。如果我们需要添加更多的元素，我们需要使用`update()`方法来完成。

### 示例:向集合中添加元素

```py
Fruits = {"apple", "mango", "cherry"}

Fruits.add("grapes")

print(Fruits)

Fruits.update(["banana", "orange", "strawberry"])

print(Fruits)

```

**输出**:

{ '樱桃'，'苹果'，'芒果'，'葡萄' }
{ '草莓'，'樱桃'，'苹果'，'橘子'，'香蕉'，'芒果'，'葡萄' }

* * *

## 从集合中删除元素

我们可以使用以下任一方法从器械包中删除物品:

1.  通过使用`remove()`方法
2.  通过使用`discard()`方法
3.  通过使用`clear()`方法–从集合中删除所有元素
4.  通过使用`del()`方法–删除整个集合

* * *

### 示例 1:使用 remove()方法

```py
Fruits = {"apple", "grapes", "cherry"}

Fruits.remove("grapes")

print(Fruits)

```

**输出**:

{ '樱桃'，'苹果' }

* * *

### 示例 2:使用 discard()方法

```py
Fruits = {"apple", "grapes", "cherry"}

Fruits.discard("grapes")

print(Fruits)

```

**输出**:

{ '樱桃'，'苹果' }

* * *

### 示例 3:使用 clear()方法

```py
Fruits = {"apple", "grapes", "cherry"}

Fruits.clear()

print(Fruits)

```

**输出**:

集合()

* * *

### 示例 4:使用 del()方法

```py
Fruits = {"apple", "grapes", "cherry"}

del Fruits

print(Fruits)

```

**输出**:

```py
 Traceback (most recent call last):
 File "main.py", line 5, in <module>
 print(Fruits) 
NameError: name 'Fruits' is not defined

```

* * *

## 集中的方法

| 方法 | 描述 |
| 添加() | 将元素添加到集合中 |
| 清除() | 从集合中移除所有元素 |
| 复制() | 返回集合的副本 |
| 差异() | 返回包含两个或多个集合之差的集合 |
| 差异 _ 更新() | 移除此集合中也包含在另一个指定集合中的项目 |
| 丢弃() | 移除指定的项目 |
| 交集() | 返回一个集合，它是另外两个集合的交集 |
| 交集 _ 更新() | 删除此集合中不存在于其他指定集合中的项目 |
| isdisjoint() | 返回两个集合是否有交集 |
| issubset() | 返回另一个集合是否包含这个集合 |
| issuperset() | 返回这个集合是否包含另一个集合 |
| 流行() | 从集合中移除元素 |
| 移除() | 移除指定的元素 |
| 对称 _ 差异() | 返回两个集合的对称差的集合 |
| 对称 _ 差异 _ 更新() | 插入这个集合和另一个集合的对称差异 |
| 联合() | 返回包含集合并集的集合 |
| 更新() | 用这个集合和其他集合的并集更新集合 |

* * *

## Python 中的集合运算

集合用于执行数学功能集合运算，如并、差、交和对称差。

* * *

### 集合并集–包含两个集合中的所有元素。

Union 运算通过以下任一方法执行:

*   通过使用`|`运算符
*   通过使用`union()`方法

#### 示例:集合的并集

```py
X = {1, 2, 3}
Y = {6, 7, 8}

print(X | Y)
print(Y.union(X))

```

**输出**:

{1，2，3，6，7，8}
{1，2，3，6，7，8}

* * *

### 集合交集–包含两个集合共有的元素。

相交操作通过以下任一方法执行:

*   通过使用`&`运算符
*   通过使用`intersection(`方法

#### 示例:集合的交集

```py
X = {1, 2, 3}
Y = {3, 2, 8}

print(X & Y)
print(Y.intersection(X))

```

**输出**:

{2，3}
{2，3}

* * *

### 集合差异–包含任一集合中的元素。

(A–B)包含只在集合 A 中而不在集合 B 中的元素。

(B–A)包含只在集合 B 中而不在集合 A 中的元素。

差分运算通过以下任一方法执行:

*   通过使用`-`运算符
*   通过使用`difference()`方法

#### 示例:集合的差异

```py
X = {1, 2, 3}
Y = {3, 2, 8}

print(X - Y)

print(Y.difference(X))

```

**输出**:

{1}
{8}

* * *

### 集合对称差–包含两个集合中的元素，但集合中的公共元素除外

对称差分运算通过以下任一方法执行:

*   通过使用`^`运算符
*   通过使用`symmetric_difference()`方法

#### 示例:集合的对称差

```py
X = {1, 2, 3, 9, 0}
Y = {3, 2, 8, 7, 5}

print(X ^ Y)

print(Y.symmetric_difference(X))

```

**输出**:

{0，1，5，7，8，9}
{0，1，5，7，8，9}

* * *

## 参考

*   Python 集合
*   [Python 官方文档](https://docs.python.org/3.7/tutorial/datastructures.html?highlight=set#sets)