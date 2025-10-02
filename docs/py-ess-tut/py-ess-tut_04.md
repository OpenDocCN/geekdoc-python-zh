# 第四章 字典：当索引不好用时

> 来源：[`www.cnblogs.com/Marlowes/p/5320049.html`](http://www.cnblogs.com/Marlowes/p/5320049.html)
> 
> 作者：Marlowes

我们已经了解到，列表这种数据结构适合于将值组织到一个结构中，并且通过编号对其进行引用。在本章中，你将学到一种通过名字来引用值的数据结构。这种类型的数结构成为*映射*(mapping)。字典是 Python 中唯一內建的映射类型。字典中的值并没有特殊的顺序，但是都存储在一个特定的键(Key)下。键可以是数字、字符串甚至是元组。

## 4.1 字典的使用

*字典*这个名称已经给出了有关这个数据结构功能的一些提示：一方面，对于普通的书来说，都是按照从头到尾的顺序进行阅读。如果愿意，也可以快速翻到某一页，这有点像 Python 的列表。另一方面，构造字典的目的，不管是现实中的字典还是在 Python 中的字典，都是为了可以通过轻松查找某个特定的词语(键)，从而找到它的定义(值)。

某些情况下，字典比列表更加适用，比如：

√ 表示一个游戏棋盘的状态，每个键都是由坐标值组成的元组；

√ 存储文件修改时间，用文件名作为键；

√ 数字电话/地址簿。

假如有一个人名列表如下：

```py
>>> names = ["Alice", "Beth", "Cecil", "Dee-Dee", "Earl"] 
```

如果要创建一个可以存储这些人的电话号码的小型数据库，应该怎么做呢？一种方法是建立一个新的列表。假设只存储四位的分机电话号码，那么可以得到与下面相似的列表：

```py
>>> numbers = ["2341", "9102", "3158", "0142", "5551"] 
```

建立了这些列表后，可以通过如下方式查找 Cecil 的电话号码：

```py
>>> numbers[names.index("Cecil")] 
'3158' 
```

这样做虽然可行，但是并不实用。真正需要的效果应该类似以下面这样：

```py
>>> phonebook["Cecil"] 
'3158' 
```

你猜怎么着？如果`phonebook`是字典，就能像上面那样操作了。

**整数还是数字字符串**

看到这里，读者可能会有疑问：为什么用字符串而不用整数表示电话号码呢？考虑一下 Dee-Dee 的电话号码会怎么样：

```py
>>> 0142
98 
```

这并不是我们想要的结果，是吗？就像第一章曾经简略地提到的那样，八进制数字均以`0`开头。不能像那样表示十进制数字。

```py
>>> 0912 File "<stdin>", line 1 0912
       ^ SyntaxError: invalid token 
```

教训就是：电话号码(以及其他可能以`0`开头的数字)应该表示为数字字符串，而不是整数。

## 4.2 创建和使用字典

字典可以通过下面的方式创建：

```py
>>> phonebook = {"Alice": "2341", "Beth": "9102", "Cecil": "3258"} 
```

字典由多个*键*及与其对应的*值*构成的*键-值*对组成(我们也把键-值对称为*项*)。在上例中，名字是键，电话号码是值。每个键和它的值之间用冒号(`:`)隔开，项之间用逗号(`,`)隔开，而整个字典是由一对大括号括起来。空字典(不包括任何项)由两个大括号组成，像这样：`{}`。

*注：字典中的键是唯一的(其他类型的映射也是如此)，而值并不唯一。*

### 4.2.1 `dict`函数

可以用`dict`函数(`dict`函数根本不是真正的函数，它是个类型，就像`list`、`tuple`和`str`一样)，通过其他映射(比如其他字典)或者(键，值)对的序列建立字典。

```py
>>> items = [("name", "Gumby"), ("age", 42)] 
>>> d = dict(items) >>> d
{'age': 42, 'name': 'Gumby'} 
>>> d["name"] 'Gumby' 
```

`dict`函数也可以通过关键字参数来创建字典，如下例所示：

```py
>>> d = dict(name="Gumby", age=42) 
>>> d
{'age': 42, 'name': 'Gumby'} 
```

尽管这可能是`dict`函数最有用的功能，但是还能以映射作为`dict`函数的参数，以建立其项与映射相同的字典(如果不带任何参数，则`dict`函数返回一个新的空字典，就像`list`、`tuple`以及`str`等函数一样)。如果另一个映射也是字典(毕竟这是唯一內建的映射类型)，也可以使用本章稍后讲到的字典方法`copy`。

### 4.2.2 基本字典操作

字典的基本行为在很多方面与序列(sequence)类似：

√ `len(d)`返回`d`中项(键-值对)的数量；

√ `d[k]`返回关联到键`k`上的值；

√ `d[k]=v`将值`v`关联到键 k 上；

√ `del d[k]`删除键为`k`的项；

√ `k in d`检查`d`中是否有含有键为 k 的项。

尽管字典和列表有很多特性相同，但也有下面一些重要的区别。

√ 键类型：字典的键不一定为整型数据(但也可以是)，键可以是任意的不可变类型，比如浮点型(实型)、字符串或者元组。

√ 自动添加：即使键起初在字典中并不存在，也可以为它赋值，这样字典就会建立新的项。而(在不使用`append`方法或者其他类似操作的情况下)不能将值关联到列表范围之外的索引上。

√ 成员资格：表达式`k in d`(`d`为字典)查找的是*键*，而不是*值*。表达式`v in l`(`l`为列表)则用来查找*值*，而不是*索引*。这样看起来好像有些不太一致，但是当习惯以后就会感觉非常自然了。毕竟，如果字典含有指定的键，查找相应的值也就很容易了。

*注：在字典中检查键的成员资格比在列表中检查值的成员资格更高效，数据结构的规模越大，两者的效率差距越明显。*

第一点——键可以是任意不可变类型——是字典最强大的地方。第二点也很重要。看看下面的区别：

```py
>>> x = []  # 列表
>>> x[42] = "Foobar" Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  IndexError: list assignment index out of range 
>>> x = {}  # 字典
>>> x[42] = "Foobar"
>>> x
{42: 'Foobar'} 
```

首先，程序试图将字符串`"Foobar"`关联到一个空列表的 42 号位置上——这显然是不可能的，因为这个位置根本不存在。为了将其变为可能，我必须用`[None]*43`或者其他方式初始化`x`，而不能仅使用`[]`。但是下一个例子工作得很好。我将`"Foobar"`关联到空字典的键`42`上，没问题！新的项已经添加到字典中，我达到目的了。

代码清单 4-1 所示是电话本例子的代码。

```py
 1 #!/usr/bin/env python
 2 # coding=utf-8
 3 
 4 # 一个简单的数据库
 5 # 字典使用人名作为键。每个人用另一个字典来表示，其键"phone"和"addr"分别表示他们的电话号码和地址。
 6 
 7 people = { 
 8 
 9     "Alice": { 
10         "phone": "2341", 
11         "addr": "Foo drive 23"
12 }, 
13 
14     "Beth": { 
15         "phone": "9102", 
16         "addr": "Bar street 42"
17 }, 
18 
19     "Cecil": { 
20         "phone": "3158", 
21         "addr": "Baz avenue 90"
22 } 
23 } 
24 
25 # 针对电话号码和地址使用的描述性标签，会在打印输出的时候用到
26 labels = { 
27     "phone": "phone number", 
28     "addr": "address"
29 } 
30 
31 name = raw_input("Name: ") 
32 
33 # 查找电话号码还是地址
34 request = raw_input("Phone number (p) or address (a)? ") 
35 
36 # 使用正确的键
37 if request == "p": 
38     key = "phone"
39 if request == "a": 
40     key = "addr"
41 
42 # 如果名字是字典中的有效键才打印信息
43 if name in people: 
44     print "%s's %s is %s." % (name, labels[key], people[name][key]) 
```

Code_Listing 4-1

下面是程序的运行示例：

```py
Name: Beth
Phone number (p) or address (a)? a
Beth's address is Bar street 42. 
```

### 4.2.3 字典的格式化字符串

在第三章，已经见过如何使用字符串格式化功能来格式化元组中所有的值。如果使用的是字典(只以字符串作为键的)而不是元组，会使字符串格式化更酷一些。在每个转换说明符(conversion specifier)中的%字符后面，可以加上键(用圆括号括起来)，后面再跟上其他说明元素。

```py
>>> phonebook
{'Beth': '9102', 'Alice': '2341', 'Cecil': '3258'} 
>>> "Cecil's phone number is %(Cecil)s." % phonebook 
"Cecil's phone number is 3258." 
```

除了增加的字符串键之外，转换说明符还是像以前一样工作。当以这种方式使用字典的时候，只要所有给出的键都能在字典中找到，就可以使用任意数量的转换说明符。这类字符串格式化在模板系统中非常有用(本例中使用 HTML)。

```py
>>> template = """<html>
... <head><title>%(title)s</title></head>
... <body>
... <h1>%(title)s</h1>
... <p>%(text)s</p>
... </body>"""
>>> data = {"title": "My Home Page", "text": "Welcome to my home page!"} 
>>> print template % data 
<html>
<head><title>My Home Page</title></head>
<body>
<h1>My Home Page</h1>
<p>Welcome to my home page!</p>
</body> 
```

*注：`string.Template`类(第三章提到过)对于这类应用也是非常有用的。*

### 4.2.4 字典方法

就像其他內建类型一样，字典也有方法。这些方法非常有用，但是可能不会像列表或者字符串方法那样被频繁地使用。读者最好先简单浏览一下本节，了解有哪些方法可用，然后在需要的时候再回过头来查看特定方法的具体用法。

1\. `clear`

`clear`方法清除字典中所有的项。这是个原地操作(类似于`list.sort`)，所以无返回值(或者说返回`None`)。

```py
>>> d = {} >>> d["name"] = "Gumby"
>>> d["age"] = 42
>>> d
{'age': 42, 'name': 'Gumby'} 
>>> returned_value = d.clear() 
>>> d
{} 
>>> print returned_value
None 
```

为什么这个方法有用呢？考虑以下两种情况。

```py
>>> x = {}  # 第一种情况
>>> y = x 
>>> x["key"] = "value"
>>> y
{'key': 'value'} 
>>> x = {} 
>>> y
{'key': 'value'} 
>>> x = {}  # 第二种情况
>>> y = x 
>>> x["key"] = "value"
>>> y
{'key': 'value'} 
>>> x.clear() >>> y
{} 
```

两种情况中，`x`和`y`最初对应同一个字典。情况 1 中，我通过将`x`关联到一个新的空字典来“清空”它，这对`y`一点影响也没有，它还关联到原先的字典。这可能是所需要的行为，但是如果真的想清空*原始*字典中的所有的元素，必须使用`clear`方法。正如在情况 2 中所看到的，`y`随后也被清空了。

2\. copy

`copy`方法返回一个具有相同键-值对的新字典(这个方法实现的是*浅复制*(shallow copy)，因为值本身就是相同的，而不是副本)。

```py
>>> x = {"username": "admin", "machines": ["foo", "bar", "baz"]} 
>>> y = x.copy() >>> y["username"] = "mlh"
>>> y["machines"].remove("bar") 
>>> y
{'username': 'mlh', 'machines': ['foo', 'baz']} 
>>> x
{'username': 'admin', 'machines': ['foo', 'baz']} 
```

可以看到，当在副本中替换值的时候，原始字典不受影响，但是，如果*修改*了某个值(原地修改，而不是替换)，原始的字典也会改变，因为同样的值也存储在原字典中(就像上面例子中的`machines`列表一样)。

避免这种问题的一种方法就是使用*深复制*(deep copy)，复制其包含的所有值。可以使用`copy`模块的`deepcopy`函数来完成操作：

```py
>>> from copy import deepcopy 
>>> d = {} 
>>> d["names"] = ["Alfred", "Bertrand"] 
>>> c = d.copy() 
>>> dc = deepcopy(d) 
>>> d["names"].append("Clive") 
>>> c
{'names': ['Alfred', 'Bertrand', 'Clive']} 
>>> dc
{'names': ['Alfred', 'Bertrand']} 
```

3\. `fromkeys`

`fromkeys`方法使用给定的键建立新的字典，每个键都对应一个默认的值`None`。

```py
>>> {}.fromkeys(["name", "age"])
{'age': None, 'name': None} 
```

刚才的例子中首先构造了一个空字典，然后调用它的`fromkeys`方法，建立另外一个字典——有些多余。此外，你还可以直接在`dict`上面调用该方法，前面讲过，`dict`是所有字典的类型(关于类型和类的概念在第七章中会深入讨论)。

```py
>>> dict.fromkeys(["name", "age"])
{'age': None, 'name': None} 
```

如果不想使用`None`作为默认值，也可以自己提供默认值。

```py
>>> dict.fromkeys(["name", "age"], "(unknown)")
{'age': '(unknown)', 'name': '(unknown)'} 
```

4\. `get`

`get`方法是个更宽松的访问字典项的方法。一般来说，如果试图访问字典中不存在的项时会出错：

```py
>>> d = {} >>> print d["name"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  KeyError: 'name' 
```

而用`get`就不会：

```py
>>> print d.get("name")
None 
```

可以看到，当使用`get`访问一个不存在的键时，没有任何异常，而得到了`None`值。还可以自定义“默认”值，替换`None`：

```py
>>> d.get("name", "N/A") 'N/A' 
```

如果键存在，`get`使用起来就像普通的字典查询一样：

```py
>>> d["name"] = "Eric"
>>> d.get("name") 'Eric' 
```

代码清单 4-2 演示了一个代码清单 4-1 程序的修改版本，它使用`get`方法访问“数据库”实体。

```py
 1 #!/usr/bin/env python
 2 # coding=utf-8
 3 
 4 # 一个简单的数据库
 5 # 字典使用人名作为键。每个人用另一个字典来表示，其键"phone"和"addr"分别表示他们的电话号码和地址。
 6 
 7 people = { 
 8 
 9     "Alice": { 
10         "phone": "2341", 
11         "addr": "Foo drive 23"
12 }, 
13 
14     "Beth": { 
15         "phone": "9102", 
16         "addr": "Bar street 42"
17 }, 
18 
19     "Cecil": { 
20         "phone": "3158", 
21         "addr": "Baz avenue 90"
22 } 
23 } 
24 
25 # 针对电话号码和地址使用的描述性标签，会在打印输出的时候用到
26 labels = { 
27     "phone": "phone number", 
28     "addr": "address"
29 } 
30 
31 name = raw_input("Name: ") 
32 
33 # 查找电话号码还是地址
34 request = raw_input("Phone number (p) or address (a)? ") 
35 
36 # 使用正确的键
37 key = request  # 如果请求既不是"p"也不是"a"
38 
39 if request == "p": 40     key = "phone"
41 if request == "a": 42     key = "addr"
43 
44 # 使用 get()提供默认值
45 person = people.get(name, {}) 
46 label = labels.get(key, key) 
47 result = person.get(key, "not available") 
48 
49 print "%s's %s is %s." % (name, label, result) 
```

Code_Listing 4-2

以下是程序运行的输出。注意`get`方法带来的灵活性如何使得程序在用户输入我们并未准备的值时也能做出合理的反应。

```py
Name: Gumby
Phone number (p) or address (a)? batting average
Gumby's batting average is not available. 
```

5\. `has_key`

`has_key`方法可以检查字典中是否含有特定的键。表达式`d.has_key(k)`相当于表达式`k in d`。使用哪个方式很大程度上取决于个人的喜好。Python3.0 中不包括这个函数。

下面是一个使用`has_key`方法的例子：

```py
>>> d = {} 
>>> d.has_key("name")
False 
>>> d["name"] = "Eric"
>>> d.has_key("name")
True 
```

6\. `items`和`iteritems`

`items`方法将字典所有的项以列表方式返回，列表中的每一项都表示为(键, 值)对的形式。但是项在返回时并没有遵循特定的次序。

```py
>>> d = {"title": "Python Web Site", "url": "http://www.python.org", "spam": "0"} 
>>> d.items()
[('url', 'http://www.python.org'), ('spam', '0'), ('title', 'Python Web Site')] 
```

`iteritems`方法的作用大致相同，但是会返回一个*迭代器*对象而不是列表：

```py
>>> it = d.iteritems() 
>>> it <dictionary-itemiterator object at 0x00000000029BAEF8>
>>> list(it)  # Convert the iterator to a list
[('url', 'http://www.python.org'), ('spam', '0'), ('title', 'Python Web Site')] 
```

在很多情况下使用`iteritems`会更加高效(尤其是想要迭代结果的情况下)。关于迭代器的更多信息，请参见第九章。

7\. `keys`和`iterkeys`

`keys`方法将字典中的键以列表形式返回，而`iterkeys`则返回针对键的迭代器。

8\. `pop`

`pop`方法用来获得对应于给定键的值，然后将这个键-值对从字典中移除。

```py
>>> d = {"x": 1, "y": 2} 
>>> d.pop("x") 1
>>> d
{'y': 2} 
```

9\. `popitem`

`popitem`方法类似于`list.pop`，后者会弹出列表的最后一个元素。但不同的是，`popitem`弹出随机的项，因为字典并没有“最后的元素”或者其他有关顺序的概念。若想一个接一个地移除并处理项，这个方法就非常有效了(因为不用首先获取键的列表)。

```py
>>> d
{'url': 'http://www.python.org', 'spam': '0', 'title': 'Python Web Site'} 
>>> d.popitem()
('url', 'http://www.python.org') 
>>> d
{'spam': '0', 'title': 'Python Web Site'} 
```

尽管`popitem`和列表的`pop`方法很类似，但字典中没有与`append`等价的方法。因为字典是无序的，类似于`append`的方法是没有任何意义的。

10\. `setdefault`

`setdefault`方法在某种程度上类似于`get`方法，能够获得与给定键相关联的值，除此之外，`setdefault`还能在字典中不含有给定键的情况下设定相应的键值。

```py
>>> d = {} 
>>> d.setdefault("name", "N/A") 'N/A'
>>> d
{'name': 'N/A'} 
>>> d["name"] = "Gumby"
>>> d.setdefault("name", "N/A") 
'Gumby'
>>> d
{'name': 'Gumby'} 
```

可以看到，当键不存在的时候，`setdefault`返回默认值并且相应地更新字典。如果键存在，那么就返回与其对应的值，但不改变字典。默认值是可选的，这点和`get`一样。如果不设定，会默认使用`None`。

```py
>>> d = {} 
>>> print d.setdefault("name")
None 
>>> d
{'name': None} 
```

11\. `update`

`update`方法可以利用一个字典项更新另外一个字典：

```py
>>> d = {
... "title": "Python Web Site",
... "url": "http://www.python.org",
... "changed": "Mar 14 22:09:15 MET 2008" ...     } 
>>> x = {"title": "Python Language Website"} 
>>> d.update(x) 
>>> d
{'url': 'http://www.python.org', 'changed': 'Mar 14 22:09:15 MET 2008', 'title': 'Python Language Website'} 
```

提供的字典中的项会被添加到旧的字典中，若有相同的键则会进行覆盖。

`update`方法可以使用与调用`dict`函数(或者类型构造函数)同样的方式进行调用，这点在本章前面已经讨论。这就意味着`update`可以和映射、拥有(键、值)对的队列(或者其他可迭代的对象)以及关键字参数一起调用。

12\. `values`和`itervalues`

`values`方法以列表的形式返回字典中的值(`itervalues`返回值的迭代器)。与返回键的列表不同的是，返回值的列表中可以包含重复的元素：

```py
>>> d = {} 
>>> d[1] = 1
>>> d[2] = 2
>>> d[3] = 3
>>> d[4] = 1
>>> d.values()
[1, 2, 3, 1] 
```

## 4.3 小结

本章介绍了如下内容。

**映射**：映射可以使用任意不可变对象标识元素。最常用的类型是字符串和元组。Python 唯一的內建映射类型是字典。

**利用字典格式化字符串**：可以通过在格式化说明符中包括名称(键)来对字典应用字符串格式化操作。在当字符串格式化中使用元组时，还需要对元组中每一个元素都设定“格式化说明符”。在使用字典时，所用的说明符可以比在字典中用到的项少。

**字典的方法**：字典有很多方法，调用的方式和调用列表以及字符串方法的方式相同。

### 4.3.1 本章的新函数

本章涉及的新函数如表 4-1 所示。

表 4-1 本章的新函数

```py
dict(seq)                        用(键、值)对(或者映射和关键字参数)建立字典。 
```

### 4.3.2 接下来学什么

到现在为止，已经介绍了很多有关 Python 的基本数据类型的只是，并且讲解了如何使用它们来建立表达式。那么请回想一下第一章的内容，计算机程序还有另外一个重要的组成因素——语句。下一章我们会对语句进行详细的讨论。