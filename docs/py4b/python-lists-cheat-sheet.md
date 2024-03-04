# 列表

> 原文：<https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet>

## 什么是列表？

Python 中最简单的数据结构，用于存储值列表。

列表是项目的集合(字符串、整数甚至其他列表)。

列表中的每个项目都有一个指定的索引值。

列表括在[ ]中

列表中的每一项都用逗号分隔

与字符串不同，列表是可变的，这意味着它们可以被改变。

## 列表创建

列表是使用由方括号
包围的逗号分隔的值列表创建的。

列表保存一个值序列(就像字符串可以保存一个字符序列
)。

列表很容易创建，这是制作列表的一些方法

```py
emptyList = [ ]  

list1 = ['one, two, three, four, five']

numlist = [1, 3, 5, 7, 9]

mixlist = ['yellow', 'red', 'blue', 'green', 'black']

#An empty list is created using just square brackets:
list = []
```

## 列表长度

使用 length 函数，我们可以得到一个列表的长度

```py
list = ["1", "hello", 2, "world"]
len(list)
>>4
```

## 列表追加

列表追加将在末尾添加项目。

如果想在开头添加，可以使用插入功能(见下文)

```py
list.insert(0, "Files")

list = ["Movies", "Music", "Pictures"]

list.append(x) #will add an element to the end of the list
list.append("Files")

print list
['Movies', 'Music', 'Pictures', 'Files’]
```

## 列表插入

语法是:

```py
list.insert(x, y) 	#will add element y on the place before x

list = ["Movies", "Music", "Pictures"] 

list.insert(2,"Documents")

print list
['Movies', 'Music', 'Documents', 'Pictures', 'Files']

#You can insert a value anywhere in the list

list = ["Movies", "Music", "Pictures"] 
list.insert(3, "Apps”)
```

## 列表删除

要删除列表中的第一个元素，只需使用 list.remove

语法是:

```py
list.remove(x)

List = ['Movies', 'Music', 'Files', 'Documents', 'Pictures']

list.remove("Files")

print list
['Movies', 'Music', 'Documents', 'Pictures']

a = [1, 2, 3, 4]
a.remove(2)
print a
[1, 3, 4]
```

## 列表扩展

语法是:

```py
list.extend(x) 	#will join the list with list x

list2 = ["Music2", "Movies2"]
list1.extend(list2)

print list1
['Movies', 'Music', 'Documents', 'Pictures', 'Music2', 'Movies2’]
```

## 列表删除

使用 del 删除基于索引位置的项目。

```py
list = ["Matthew", "Mark", "Luke", "John"]
del list[1]

print list
>>>Matthew, Luke, John
```

## 列出关键词

关键字“in”可用于测试一个项目是否在列表中。

```py
list = ["red", "orange", "green", "blue"]
if "red" in list:
    do_something()

#Keyword "not" can be combined with "in".

list = ["red", "orange", "green", "blue"]
if "purple" not in list:
    do_something()
```

## 反向列表

reverse 方法反转整个列表的顺序。

```py
L1 = ["One", "two", "three", "four", "five"]

#To print the list as it is, simply do:
print L1

#To print a reverse list, do:
for i in L1[::-1]:
    print i

#OR

L = [0, 10, 20, 40]
L.reverse()

print L
[40, 20, 10, 0]
```

## 列表排序

对列表排序最简单的方法是使用 sorted(list)函数。

它接受一个列表并返回一个新列表，其中的元素按排序顺序排列。

原始列表不变。

sorted()函数可以通过可选参数定制。

sorted()可选参数 reverse=True，例如 sorted(list，reverse=True)，
使其向后排序。

```py
#create a list with some numbers in it
numbers = [5, 1, 4, 3, 2, 6, 7, 9]

#prints the numbers sorted
print sorted(numbers)

#the original list of numbers are not changed
print numbers
my_string = ['aa', 'BB', 'zz', 'CC', 'dd', "EE"]

#if no argument is used, it will use the default (case sensitive)
print sorted(my_string)

#using the reverse argument, will print the list reversed
print sorted(strs, reverse=True)   ## ['zz', 'aa', 'CC', 'BB']

This will not return a value, it will modify the list
list.sort()
```

## 列表拆分

拆分列表中的每个元素。

```py
mylist = ['one', 'two', 'three', 'four', 'five']
newlist = mylist.split(',')

print newlist
['one', ' two', ' three', ' four', 'five’]
```

## 列表索引

列表中的每个项目都有一个从 0 开始的指定索引值。

访问列表中的元素称为索引。

```py
list 	= ["first", "second", "third"]
list[0] == "first"
list[1] == "second"
list[2] == "third”
```

## 列表切片

访问部分数据段称为切片。

通过使用[ ]操作符，可以像访问字符串一样访问列表。

要记住的关键点是:end 值代表第一个值，即
不在所选切片中。

因此，结束和开始之间的差异是所选元素的数量
(如果步长为 1，则为默认值)。

让我们创建一个包含一些值的列表

```py
colors = ['yellow', 'red', 'blue', 'green', 'black']

print colors[0]
>>> yellow

print colors [1:]
>>> red, blue, green, black
```

让我们来看看这个取自[这篇](https://stackoverflow.com/questions/509211/the-python-slice-notation "so_post") stackoverflow 帖子的例子。

```py
a[start:end] 			# items start through end-1
a[start:]    			# items start through the rest of the array
a[:end]      			# items from the beginning through end-1
a[:]         			# a copy of the whole array
```

还有一个步长值，可以与上述任何一个一起使用

```py
a[start:end:step] 		# start through not past end, by step
```

另一个特性是 start 或 end 可能是负数，这意味着它从数组的结尾而不是开头开始计数
。

```py
a[-1]    			# last item in the array
a[-2:]   			# last two items in the array
a[:-2]   			# everything except the last two items
```

## 列表循环

在编程中使用循环时，有时需要存储
循环的结果。

在 Python 中做到这一点的一种方法是使用列表。

这一小段将展示如何使用一个 [Python for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)来迭代列表并处理
列表项。

```py
#It can look something like this:
matching = []
for term in mylist:
    do something

#For example, you can add an if statement in the loop, and add the item to the (empty) list
if it's matching.
matching = []    #creates an empty list using empty square brackets []
for term in mylist:
    if test(term):
        matching.append(term)

#If you already have items in a list, you can easily loop through them like this:
items = [ 1, 2, 3, 4, 5 ]
for i in items:
    print i
```

## 列出方法

对列表方法的调用使它们所操作的列表出现在方法名之前。

该方法完成工作所需的任何其他值都以正常方式提供，作为圆括号内的一个额外参数。

```py
s = ['h','e','l','l','o']	#create a list
s.append('d')         		#append to end of list
len(s)                		#number of items in list
s.sort()               		#sorting the list
s.reverse()           		#reversing the list
s.extend(['w','o'])    		#grow list
s.insert(1,2)         		#insert into list
s.remove('d')           	#remove first item in list with value e
s.pop()               		#remove last item in the list
s.pop(1)              		#remove indexed value from list
s.count('o')            	#search list and return number of instances found
s = range(0,10)          	#create a list over range 
s = range(0,10,2)        	#same as above, with start index and increment
```

## 列举例子

让我们以展示一些列表示例来结束这篇文章:首先，创建一个只包含数字的列表。

```py
list = [1,2,3,5,8,2,5.2]	#creates a list containing the values 1,2,3,5,8,2,5.2
i = 0
while i < len(list):		#The while loop will print each element in the list
    print list[i]		 #Each element is reached by the index (the letter in the square bracket)
    i = i + 1			 #Increase the variable i with 1 for every time the while loop runs.
```

下一个例子将计算列表中元素的平均值。

```py
list = [1,2,3,5,8,2,5.2]
total = 0
i = 0
while i < len(list):
    total = total + list[i]
    i = i + 1

average = total / len(list)
print average
```

## 相关职位

[用 Python 列出理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)

[Python 中的列表操作](https://www.pythonforbeginners.com/basics/python-list-manipulation)

[反转列表和字符串](https://www.pythonforbeginners.com/code-snippets-source-code/reverse-loop-on-a-list)