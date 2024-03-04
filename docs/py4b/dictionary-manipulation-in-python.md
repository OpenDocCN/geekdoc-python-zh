# Python 中的字典操作

> 原文：<https://www.pythonforbeginners.com/dictionary/dictionary-manipulation-in-python>

## 概观

字典是键值对的集合。

字典是一组键:值对。

字典中的所有键必须是唯一的。

在字典中，键和值由冒号分隔。

键、值对用逗号分隔。

键和值对列在大括号“{ }”之间

我们使用方括号“[ ]”查询字典

## 词典操作

每当你需要将你想要的条目链接在一起时，字典就很有用，例如存储结果以便快速查找。

创建一个空字典

```py
months = {} 
```

用一些对子创建一个字典

#注意:每个键必须是唯一的

```py
months = { 1 : "January", 
     	2 : "February", 
    	3 : "March", 
        4 : "April", 
     	5 : "May", 
     	6 : "June", 
    	7 : "July",
        8 : "August",
     	9 : "September", 
    	10 : "October", 
        11 : "November",
    	12 : "December" } 
```

月份[1-12]是键，“1 月-12 月”是值

打印所有密钥

```py
print "The dictionary contains the following keys: ", months.keys() 
```

输出:

该字典包含以下关键字:[1，2，3，4，5，6，7，8，9，10，
11，12]

**访问**

要从字典中获取一个值，您必须提供它的键，您不能提供值并获取键

```py
whichMonth = months[1]
print whichMonth 
```

产出:1 月

要从字典中删除元素，请使用 del

```py
del(months[5])
print months.keys() 
```

输出:
【1，2，3，4，6，7，8，9，10，11，12】

要向字典中添加新元素，请为新键赋值

```py
months[5] = "May"
print months.keys() 
```

输出:
【1，2，3，4，5，6，7，8，9，10，11，12】

要更新字典中的元素，请为其键分配一个新值

```py
months[1] = "Jan"
print months 
```

输出:
{1:'一月'，2:'二月'，3:'三月'，4:'四月'，5… }

整理

```py
sortedkeys = months.keys()
print sortedkeys 
```

输出:
【1，2，3，4，5，6，7，8，9，10，11，12】

字典和循环

遍历键

```py
for key in months:
    print key, months[key] 
```

输出:
1 月 1 日
2 月 2 日
3 月
4 月
5 月
6 月
7 月
8 月
9 月
10 月
11 月
12 月

迭代(键，值)对

```py
for key, value in months.iteritems():
    print key, value

print "The entries in the dictionary are:"
for item in months.keys():
    print "months[ ", item, " ] = ", months[ item ] 
```

结合列表和字典

字典列表示例

```py
customers = [{"uid":1,"name":"John"},
    {"uid":2,"name":"Smith"},
           {"uid":3,"name":"Andersson"},
            ]
print customers 
```

Output:
[{'uid': 1，' name': 'John'}，{'uid': 2，' name': 'Smith'}，{'uid': 3，' name ':【T1]' Andersson ' }]

打印每个客户的 uid 和姓名

```py
for x in customer:
    print x["uid"], x["name"] 
```

输出:
1 约翰
2 史密斯
3 安德森

修改条目

这将把客户 2 的名字从 Smith 改为 Charlie

```py
customers[2]["name"]="charlie"
print customers 
```

输出:
[{'uid': 1，' name': 'John'}，{'uid': 2，' name': 'Smith'}，{'uid': 3，' name ':【T1]' Charlie ' }]

向每个条目添加一个新字段

```py
for x in customers:
    x["password"]="123456" # any initial value

print customers 
```

输出:
[{'password': '123456 '，' uid': 1，' name': 'John'}，{'password': '123456 '，' uid':
2，' name': 'Smith'}，{'password': '123456 '，' uid': 3，' name': 'Andersson'}]

删除字段

```py
del customers[1]
print customers 
```

Output:
[{'uid': 1，' name': 'John'}，{'uid': 3，' name': 'Andersson'}]

删除所有字段

```py
# This will delete id field of each entry.
for x in customers:
    del x["id"] 
```

Output:
[{'name': 'John'}，{'name': 'Smith'}，{'name': 'Andersson'}]

有关字典的更多信息，请参见这篇文章。