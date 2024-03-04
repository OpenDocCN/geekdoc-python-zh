# Python 范围函数

> 原文：<https://www.pythonforbeginners.com/modules-in-python/python-range-function>

## 范围函数

Python 中内置的 range 函数对于生成列表形式的
数字序列非常有用。

给定的端点决不是生成列表的一部分；

range(10)生成 10 个值的列表，长度为 10 的
序列的合法索引。

可以让范围从另一个数字开始，或者指定一个
不同的增量(甚至是负数；

有时这被称为“步骤”):

## 产品系列示例

```py
>>> range(1,10)
[1, 2, 3, 4, 5, 6, 7, 8, 9]

# You can use range() wherever you would use a list. 

a = range(1, 10) 
for i in a: 
    print i 

for a in range(21,-1,-2):
   print a,

#output>> 21 19 17 15 13 11 9 7 5 3 1

# We can use any size of step (here 2)
>>> range(0,20,2)
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

>>> range(20,0,-2)
[20, 18, 16, 14, 12, 10, 8, 6, 4, 2]

# The sequence will start at 0 by default. 
#If we only give one number for a range this replaces the end of range value.
>>> range(10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# If we give floats these will first be reduced to integers. 
>>> range(-3.5,9.8)
[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8] 
```

##### 更多阅读

[http://www.python.org](https://www.python.org "python_org")