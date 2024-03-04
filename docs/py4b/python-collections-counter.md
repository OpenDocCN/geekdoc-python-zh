# Python 集合计数器

> 原文：<https://www.pythonforbeginners.com/collection/python-collections-counter>

## 收集模块概述

Collections 模块实现了高性能的容器数据类型(超越了
内置的类型 list、dict 和 tuple ),并包含许多有用的数据结构
,您可以使用它们在内存中存储信息。

本文将讨论计数器对象。

## 计数器

计数器是一个跟踪等效值相加的次数的容器。

它可以用来实现其他语言通常使用包或多重集数据结构的算法。

## 导入模块

导入收藏使收藏中的内容可用:
collections.something

```py
import collections 
```

因为我们只打算使用计数器，所以我们可以简单地这样做:

```py
from collections import Counter 
```

## 正在初始化

计数器支持三种形式的初始化。

它的构造函数可以用一个项目序列(iterable)、一个包含键和计数的字典
来调用(映射，或者使用关键字参数将字符串
名称映射到计数(关键字参数)。

```py
import collections

print collections.Counter(['a', 'b', 'c', 'a', 'b', 'b'])

print collections.Counter({'a':2, 'b':3, 'c':1})

print collections.Counter(a=2, b=3, c=1) 
```

所有三种初始化形式的结果都是相同的。

```py
$ python collections_counter_init.py

Counter({'b': 3, 'a': 2, 'c': 1})
Counter({'b': 3, 'a': 2, 'c': 1})
Counter({'b': 3, 'a': 2, 'c': 1}) 
```

## 创建和更新计数器

可以构造一个没有参数的空计数器，并通过
update()方法填充。

```py
import collections

c = collections.Counter()
print 'Initial :', c

c.update('abcdaab')
print 'Sequence:', c

c.update({'a':1, 'd':5})
print 'Dict    :', c 
```

计数值基于新数据而增加，而不是被替换。

在这个例子中，a 的计数从 3 到 4。

```py
$ python collections_counter_update.py

Initial : Counter()

Sequence: Counter({'a': 3, 'b': 2, 'c': 1, 'd': 1})

Dict    : Counter({'d': 6, 'a': 4, 'b': 2, 'c': 1}) 
```

## 访问计数器

一旦填充了计数器，就可以使用字典 API 检索它的值。

```py
import collections

c = collections.Counter('abcdaab')

for letter in 'abcde':
    print '%s : %d' % (letter, c[letter]) 
```

计数器不会引发未知项目的 KeyError。

如果在输入中没有看到一个值(如本例中的 e)，
其计数为 0。

```py
$ python collections_counter_get_values.py

a : 3
b : 2
c : 1
d : 1
e : 0 
```

## 元素

elements()方法返回一个遍历元素的迭代器，每个元素重复的次数与它的计数一样多。

元素以任意顺序返回。

```py
import collections

c = collections.Counter('extremely')

c['z'] = 0

print c

print list(c.elements()) 
```

不保证元素的顺序，计数小于零的项目
不包括在内。

```py
$ python collections_counter_elements.py

Counter({'e': 3, 'm': 1, 'l': 1, 'r': 1, 't': 1, 'y': 1, 'x': 1, 'z': 0})
['e', 'e', 'e', 'm', 'l', 'r', 't', 'y', 'x'] 
```

## 最常见的

使用 most_common()生成 n 个最常出现的
输入值及其各自计数的序列。

```py
import collections

c = collections.Counter()
with open('/usr/share/dict/words', 'rt') as f:
    for line in f:
        c.update(line.rstrip().lower())

print 'Most common:'
for letter, count in c.most_common(3):
    print '%s: %7d' % (letter, count) 
```

本示例对系统
词典中所有单词出现的字母进行计数，以产生一个频率分布，然后打印三个最常见的
字母。

省略 most_common()的参数会产生一个所有条目的列表，按照频率的顺序排列。

```py
$ python collections_counter_most_common.py

Most common:
e:  234803
i:  200613
a:  198938 
```

## 算术

计数器实例支持用于聚合结果的算术和集合运算。

```py
import collections

c1 = collections.Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = collections.Counter('alphabet')

print 'C1:', c1
print 'C2:', c2

print '
Combined counts:'
print c1 + c2

print '
Subtraction:'
print c1 - c2

print '
Intersection (taking positive minimums):'
print c1 & c2

print '
Union (taking maximums):'
print c1 | c2 
```

每次通过操作产生一个新的计数器时，任何零计数或负计数的项目都将被丢弃。

c1 和 c2 中 a 的计数是相同的，所以减法使其为零。

```py
$ python collections_counter_arithmetic.py

C1: Counter({'b': 3, 'a': 2, 'c': 1})
C2: Counter({'a': 2, 'b': 1, 'e': 1, 'h': 1, 'l': 1, 'p': 1, 't': 1})

#Combined counts:
Counter({'a': 4, 'b': 4, 'c': 1, 'e': 1, 'h': 1, 'l': 1, 'p': 1, 't': 1})

#Subtraction:
Counter({'b': 2, 'c': 1})

#Intersection (taking positive minimums):
Counter({'a': 2, 'b': 1})

#Union (taking maximums):
Counter({'b': 3, 'a': 2, 'c': 1, 'e': 1, 'h': 1, 'l': 1, 'p': 1, 't': 1}) 
```

## 计数单词

统计单词在列表中的出现次数。

```py
cnt = Counter()

for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
    cnt[word] += 1

print cnt

Counter({'blue': 3, 'red': 2, 'green': 1}) 
```

计数器接受一个 iterable，也可以这样写:

```py
mywords = ['red', 'blue', 'red', 'green', 'blue', 'blue']

cnt = Counter(mywords)

print cnt

Counter({'blue': 3, 'red': 2, 'green': 1}) 
```

## 找出最常见的单词

找出哈姆雷特中最常见的十个单词

```py
import re

words = re.findall('w+', open('hamlet.txt').read().lower())

print Counter(words).most_common(10)

[('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631),
 ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)] 
```

#### 来源

请不要忘记阅读下面的链接了解更多信息。

[http://www.doughellmann.com/PyMOTW/collections/](http://www.doughellmann.com/PyMOTW/collections/ "pymotw_counters")
http://docs . python . org/2/library/collections . html # collections。柜台