# Python 中的字典是什么？

> 原文：<https://www.pythonforbeginners.com/basics/dictionary>

Dictionary 是 Python 中的另一种数据类型。

字典是具有“键”和“值”的条目的集合。

Python 字典也称为关联数组或哈希表。

它们就像列表一样，除了没有指定的索引号，而是由你自己编索引。

字典是无序的，所以添加键的顺序并不一定反映它们被报告回来的顺序。

使用{}花括号来构造字典。

使用[]查找与键相关联的值

提供一个键和值。

冒号放在键和值之间(键:值)

每个键必须是唯一的，并且每个键在字典中只能出现一次。

## 如何创建字典？

要创建字典，请提供键和值，并确保每一对用逗号分隔

```py
# This is a list
mylist = ["first","second","third"]

# This is a dictionary
mydictionary = {0:"first",1:"second",2:"third"} 
```

**让我们创建一些随机字典:**

```py
# Empty declaration + assignment of key-value pair
emptyDict = {}
emptyDict['key4']=’value4?

# Create a three items dictionary
x = {"one":1,"two":2,"three":3}

#The name of the dictionary can be anything you like
dict1 = {'abc': 456};
dict2 = {'abc':123,98.6:37}; 
```

## 访问/获取值

要访问字典元素，可以使用方括号和键
来获取它的值。

```py
data = {'Name':'Zara','Age':7,'Class':'First'};

# Get all keys
data.keys()

# Get all values
data.values()

# Print key1
print data['Name']

# Prints 7
print data['Age']

# Prints name and age
print 'Name', data['Name'];
print 'Age', data['Age']; 
```

## 在字典中循环

默认情况下，字典上的 for 循环遍历它的键。

这些键将以任意顺序出现。

方法 dict.keys()和 dict.values()显式返回键或值
的列表。

还有一个 items()返回(键，值)元组的列表，这是检查字典中所有键值数据的最有效的方法。

所有这些列表都可以传递给 sorted()函数。

[基本语法](https://www.pythonforbeginners.com/basics/python-syntax-basics)循环遍历字典(键和值)
输入数据:
打印数据[key]

**假设你有一本名为“数据”的字典**

```py
data = {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3'
        }

for key, value in data.items():
    print key,value

Looping their values directory (not in order)
for value in data.values():
    print value 
```

## 更新字典

如果字典中没有该关键字，则创建一个新条目。

如果密钥已经存在，则覆盖以前的值。

您可以通过以下方式更新词典:

添加新的条目或项目(即，键值对)

修改现有条目

删除现有条目

**让我们看看它是如何工作的:**

```py
data = {'Name':'Zara','Age':7,'Class':'First'};

data['Age'] = 8;                    # update existing entry
data['School'] = "DPS School";      # Add new entry

print "data['Age']: ", data['Age'];
print "data['School']: ", data['School']; 
```

**我们再举一个例子:**

```py
birthday = {}
birthday['Darwin'] = 1809
birthday['Newton'] = 1942  # oops
birthday['Newton'] = 1642
print birthday 
```

## 删除键/值

“del”运算符执行删除操作。

在最简单的情况下，它可以删除变量的定义，就像变量
没有被定义一样。

Del 也可以用在列表元素或片上，以删除列表的这一部分，并从字典中删除条目。

```py
data = {'a':1,'b':2,'c':3}
del dict['b']   ## Delete 'b' entry
print dict      ## {'a':1, 'c':3} 
```

## 来自谷歌课堂的例子

让我们看看这个来自 https://developers.google.com/edu/python/dict-files 的例子

```py
## Can build up a dict by starting with the the empty dict {}
## and storing key/value pairs into the dict like this:
## dict[key] = value-for-that-key

dict = {}
dict['a'] = 'alpha'
dict['g'] = 'gamma'
dict['o'] = 'omega'

print dict          ## {'a': 'alpha', 'o': 'omega', 'g': 'gamma'}

print dict['a']     ## Simple lookup, returns 'alpha'

dict['a'] = 6       ## Put new key/value into dict

'a' in dict         ## True

## print dict['z']                  ## Throws KeyError

if 'z' in dict: print dict['z']     ## Avoid KeyError
print dict.get('z')                 ## None (instead of KeyError) 
```

默认情况下，字典上的 for 循环遍历它的键。

这些键将以任意顺序出现。

方法 dict.keys()和 dict.values()显式返回键或值
的列表。

还有一个 items()返回(键，值)元组的列表，这是检查字典中所有键值数据的最有效的方法。

所有这些列表都可以传递给 sorted()函数。

```py
## Note that the keys are in a random order.
for key in dict: print key
## prints a g o

## Exactly the same as above
for key in dict.keys(): print key

## Get the .keys() list:
print dict.keys()  ## ['a', 'o', 'g']

## Likewise, there's a .values() list of values
print dict.values()  ## ['alpha', 'omega', 'gamma']

## Common case -- loop over the keys in sorted order,
## accessing each key/value
for key in sorted(dict.keys()):
    print key, dict[key]

## .items() is the dict expressed as (key, value) tuples
print dict.items()  ##  [('a', 'alpha'), ('o', 'omega'), ('g', 'gamma')]

## This loop syntax accesses the whole dict by looping
## over the .items() tuple list, accessing one (key, value)
## pair on each iteration.
for k, v in dict.items(): print k, '>', v
## a > alpha    o > omega     g > gamma 
```

## 字典格式

%操作符可以方便地将字典中的值按名称替换成字符串

```py
hash = {}
hash['word'] = 'garfield'
hash['count'] = 42
s = 'I want %(count)d copies of %(word)s' % hash  # %d for int, %s for string

# Will give you:
>> 'I want 42 copies of garfield' 
```

## 常见字典操作

```py
 # create an empty dictionary
x = {}

# create a three items dictionary
x = {"one":1, "two":2, "three":3}

# get a list of all the keys
x.keys()

# get a list of all the values
x.values()

# add an entry
x["four"]=4

# change an entry
x["one"] = "uno"

# delete an entry
del x["four"]

# make a copy
y = x.copy()

# remove all items
x.clear()

#number of items
z = len(x)

# test if has key
z = x.has_key("one")

# looping over keys
for item in x.keys(): print item

# looping over values
for item in x.values(): print item

# using the if statement to get the values
if "one" in x:
    print x['one']

if "two" not in x:
    print "Two not found"

if "three" in x:
    del x['three'] 
```

##### 来源

[https://developers.google.com/edu/python/dict-files](https://developers.google.com/edu/python/dict-files "google") http://docs.python.org/2/tutorial/datastructures.htmlhttp://software-carpentry.org/3_0/py04.html
http://yuji.wordpress.comT12)http://www.tutorialspoint.com/python/python_dictionary.htm