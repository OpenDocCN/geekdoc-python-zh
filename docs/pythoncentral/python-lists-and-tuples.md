# Python 列表和元组

> 原文：<https://www.pythoncentral.io/python-lists-and-tuples/>

## **Python 列表和元组概述**

Python 中最常用的两种内置数据类型是*列表*和*元组*。

列表和元组是*序列*数据类型组的一部分——换句话说，列表和元组以特定的顺序存储一个或多个对象或*值*。存储在列表或元组中的对象可以是任何类型，包括由关键字`None`定义的 *nothing* 类型。

列表和元组的最大区别是列表是可变的，而元组是不可变的。这意味着一旦创建了元组，就不能添加或删除对象，也不能改变顺序。然而，元组中的某些对象是可以改变的，我们将在后面看到。

### **创建 Python 列表或元组**

创建列表或元组很容易，下面是一些空列表或元组:

```py

# Lists must be surrounded by brackets

>>> emptylst = []

# Tuples may or may not be surrounded by parenthesis

>>> emptytup = ()

```

为了创建非空列表或元组，*值*用逗号分隔:

```py

# Lists must be surrounded by brackets

>>> lst = [1, 2, 3]

>>> lst

[1, 2, 3]

# Tuples may or may not be surrounded by parenthesis

>>> tup = 1, 2, 3

>>> tup

(1, 2, 3)

>>> tup = ('one', 'two', 'three')

>>> tup

('one', 'two', 'three')

```

**注意:**要创建一个只有一个*值*的元组，在值后面添加一个逗号。

```py

# The length of this tuple is 1

>>> tup2 = 0,

>>> tup2

(0,)

```

### **获取 Python 列表或元组值**

与其他序列类型一样，列表和元组中的值由一个*索引*引用。这是一个数字，表示值的序列，第一个值从 0 开始。**比如下面:**

```py

>>> lst[0]

1

>>> tup[1]

'two'

```

使用低于零的索引获得从列表或元组的*末端*开始的值:

```py

>>> lst[-1]

3

>>> tup[-2]

'two'

```

### **切片 Python 列表和元组**

通过对列表或元组进行“切片”，可以引用多个值(引用一个范围[start:stop]):

```py

>>> lst[0:2]

[1, 2]

# Note an out-of-range stopindex translates to the end

>>> tup[1:5]

('two', 'three')

```

### **通过索引分配 Python 列表值**

*列表的值*可以分配索引(只要索引已经存在)，但不能分配给元组:

```py

>>> lst[2] = 'three'

>>> lst

[1, 2, 'three']

>>> tup[2] = 3

Traceback (most recent call last):

File "<pyshell#68>", line 1, in <module>

tup[2] = 3

TypeError: 'tuple' object does not support item assignment

```

### **添加到 Python 列表**

也可以用'+'操作符或标准的`append`方法将值添加到列表中(列表可以合并，或*连接*):

```py

# concatenation, the same as lst = lst + [None]

>>> lst += [None]

>>> lst

[1, 2, 'three', None]

>>> lst.append(5)

>>> lst

[1, 2, 'three', None, 5]

```

### **列表的 Python del 关键字**

可以使用关键字`del`从列表中删除值:

```py

>>> del lst[3]

>>> lst

[1, 2, 'three', 5]

# Slicing deletion; no stop index means through the end

>>> del lst[2:]

>>> lst

[1, 2]

```

列表的赋值和删除方法不适用于元组，但串联有效:

```py

# Note only tuples can be added to tuples

>>> tup += (4,)

>>> tup

('one', 'two', 'three', 4)

```

尽管元组不像列表那样是可变的，但是通过切片技术和一点创造性，元组可以像列表一样被操纵:

```py

# Almost like slice deletion

>>> tup = tup[0:2]

>>> tup

('one', 'two')

>>> tup2 += tup

>>> tup2

# almost like index assignment

>>> tup2 = tup2[0:1] + (1,) + tup2[2:]

>>> tup2

(0, 1, 'two')

```

此外，如果一个元组包含一个列表，则该元组中列表的可变性被保留:

```py

>>> tup3 = 0, 'one', None, []

>>> tup3[3].append('three')

>>> tup3

(0, 'one', None, ['three'])

```

除了这里描述的方法之外，还有更多针对列表和元组的方法，但是这些方法涵盖了最常见的列表和元组操作。而且记住，列表和元组可以**嵌套**(可以包含列表和元组，以及其他存储多组*值*的数据类型，比如字典)，可以同时存储不同的类型，非常有用。