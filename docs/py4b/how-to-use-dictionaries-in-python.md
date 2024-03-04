# Python 字典——如何用 Python 创建字典

> 原文：<https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python>

Python 中的字典是无序的条目列表，可以通过使用内置方法进行更改。字典用于创建唯一键到值的映射。

### 关于 Python 中的字典

要创建字典，请使用{}花括号来构造字典，并使用[]方括号来索引它。

在每对之间，用冒号:和逗号分隔键和值。

必须引用关键字，例如:“title”:“如何在 Python 中使用字典”

与列表一样，我们可以通过打印词典的参考来打印出词典。

字典将一组对象(键)映射到另一组对象(值)，因此您可以创建一个无序的对象列表。

字典是可变的，这意味着它们可以被改变。

键指向的值可以是任何 Python 值。

字典是无序的，所以添加键的顺序不一定反映它们被报告回来的顺序。因此，您可以通过值的键名来引用它。

### 创建新词典

#为了构造一个字典，你可以从一个空的开始。

```py
>>> mydict={}
```

#这将创建一个字典，它最初有六个键-值对，其中 iphone*是键，年份是值

```py
released = {
		"iphone" : 2007,
		"iphone 3G" : 2008,
		"iphone 3GS" : 2009,
		"iphone 4" : 2010,
		"iphone 4S" : 2011,
		"iphone 5" : 2012
	}
print released 
```

```py
 >>Output
{'iphone 3G': 2008, 'iphone 4S': 2011, 'iphone 3GS': 2009, '
	iphone': 2007, 'iphone 5': 2012, 'iphone 4': 2010} 
```

### 向字典中添加值

您可以在下面的示例中向字典添加一个值。此外，我们将继续更改该值，以展示字典与 Python 集的不同之处。

```py
#the syntax is: mydict[key] = "value"
released["iphone 5S"] = 2013
print released
>>Output
{'iphone 5S': 2013, 'iphone 3G': 2008, 'iphone 4S': 2011, 'iphone 3GS': 2009,
'iphone': 2007, 'iphone 5': 2012, 'iphone 4': 2010} 
```

### 删除一个键及其值

您可以使用 del 运算符删除元素

```py
del released["iphone"]
print released
>>output
{'iphone 3G': 2008, 'iphone 4S': 2011, 'iphone 3GS': 2009, 'iphone 5': 2012,
'iphone 4': 2010} 
```

### 检查长度

len()函数给出字典中的对数。换句话说，字典里有多少个条目。

```py
print len(released) 
```

### 测试字典

使用 in 运算符检查给定字典中是否存在某个键，如下所示:

```py
>>> my_dict = {'a' : 'one', 'b' : 'two'}
>>> 'a' in my_dict
True
>>> 'b' in my_dict
True
>>> 'c' in my_dict
False 
```

或者像这样在 for 循环中

```py
for item in released:
    if "iphone 5" in released:
        print "Key found"
        break
    else:
        print "No keys found"
>>output
Key found 
```

### 获取指定键的值

```py
print released.get("iphone 3G", "none") 
```

### 用 for 循环打印所有键

```py
print "-" * 10
print "iphone releases so far: "
print "-" * 10
for release in released:
    print release

>>output
----------
iphone releases so far: 
----------
iphone 3G
iphone 4S
iphone 3GS
iphone
iphone 5
iphone 4 
```

### 打印所有键和值

```py
for key,val in released.items():
    print key, "=>", val

>>output
iphone 3G => 2008
iphone 4S => 2011
iphone 3GS => 2009
iphone => 2007
iphone 5 => 2012
iphone 4 => 2010 
```

### 仅从字典中获取键

```py
phones = released.keys()
print phones 
```

#或者像这样打印出来:

```py
print "This dictionary contains these keys: ", " ".join(released)
>>iphone 3G iphone 4S iphone 3GS iphone iphone 5 iphone 4 
```

#或者像这样:

```py
print "This dictionary contains these keys: ", " ", released.keys()
>>['iphone 3G', 'iphone 4S', 'iphone 3GS', 'iphone', 'iphone 5', 'iphone 4'] 
```

### 打印值

元素可以通过方括号引用，例如:
print released[“iphone”]

```py
print "Values:
",
for year in released:
    releases= released[year]
    print releases

>>output:
Values:
2008
2011
2009
2007
2012
2010 
```

### 用 pprint 打印

```py
pprint.pprint(released) 
```

### 整理字典

除了打印之外，让我们对数据进行排序，以便得到预期的输出。

```py
for key, value in sorted(released.items()):
    print key, value
>>output:
('iphone', 2007)
('iphone 3G', 2008)
('iphone 3GS', 2009)
('iphone 4', 2010)
('iphone 4S', 2011)
('iphone 5', 2012)
""" 
```

```py
for phones in sorted(released, key=len):
    print phones, released[phones]

>>output:
iphone 2007
iphone 5 2012
iphone 4 2010
iphone 3G 2008
iphone 4S 2011
iphone 3GS 2009 
```

### 包括…在内

最后，让我们展示一下在字典中使用计数器来显示条目数量的价值。

```py
count = {}
for element in released:
    count[element] = count.get(element, 0) + 1
print count

>>output:
{'iphone 3G': 1, 'iphone 4S': 1, 'iphone 3GS': 1, 'iphone': 1, 
'iphone 5': 1, 'iphone 4': 1} 
```