# Python:列举例子

> 原文：<https://www.pythonforbeginners.com/basics/python-list-examples>

### 什么是列表？

请记住，列表是用方括号[ ]创建的，元素必须在方括号内。

列表中的元素不必是同一类型(可以是数字、字母、字符串)。

在这个例子中，我们将创建一个只包含数字的列表。

### 列表示例

创建列表

```py
myList = [1,2,3,5,8,2,5.2] 
i = 0
while i < len(myList):
    print myList[i]
    i = i + 1 
```

### 这是做什么的？

这个脚本将创建一个包含值 1，2，3，5，8，2，5.2 的列表(list1)

while 循环将打印列表中的每个元素。

list1 中的每个元素都是通过索引(方括号中的字母)到达的。

len 函数用于获取列表的长度。

然后，每当 while 循环运行一次，我们就将变量 I 增加 1。

##### 输出

```py
 Output >> 1 2 3 5 8 2 5.2 
```

例子

下一个例子将计算列表中元素的平均值。

```py
list1 = [1,2,3,5,8,2,5.2]
total = 0
i = 0
while i < len(list1):
    total = total + list1[i]
    i = i + 1

average = total / len(list1)
print average 
```

```py
#Output >> 3.74285714286
```