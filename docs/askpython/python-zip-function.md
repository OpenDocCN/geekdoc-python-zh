# Python zip()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-zip-function>

**Python** zip()函数在里面存储数据。该函数接受 iterable 元素作为输入，并返回 iterable 作为输出。

如果没有向 python zip 函数提供可迭代元素，它将返回一个空迭代器。

因此，它从 iterables 中聚合元素，并返回元组的 iterables。

**Python Zip()函数语法:**

```py
zip(*iterators)
```

**Python** **zip()函数参数:**

它可以是容器/iterables ( [列表](https://www.askpython.com/python/list/python-list)、[字符串](https://www.askpython.com/python/string/python-string-functions)等)

**zip()函数返回的值:**

这个函数从相应的容器中返回一个 iterable 对象映射值。

* * *

### **举例:**对 Python zip()函数的基本了解

```py
# initializing the input list 
city = [ "Pune", "Ajanta", "Aundh", "Kochi" ] 
code = [ 124875, 74528, 452657, 142563 ] 

# zip() to map values 
result = zip(city, code) 

result = set(result) 

print ("The zipped outcome is : ",end="") 
print (result) 

```

**输出:**

```py
The zipped outcome is : {('Ajanta', 74528), ('Kochi', 142563), ('Aundh', 452657), ('Pune', 124875)}
```

* * *

### Python zip()函数具有多个可迭代项

如果用户将多个 iterable 传递给 python zip()函数，该函数将返回一个包含与 iterable 相对应的元素的[元组](https://www.askpython.com/python/tuple/python-tuple)的 iterable。

**举例:**

```py
numbers = [23,33,43]
input_list = ['five', 'six', 'seven']
# No iterables being passed to zip() function
outcome = zip()

result = list(outcome)
print(result)
# Two iterables being passed to zip() function
outcome1 = zip(numbers, input_list)

result1 = set(outcome1)
print(result1)

```

**输出:**

```py
[]
{(23, 'five'), (33, 'six'), (43, 'seven')}
```

* * *

### Python zip()函数具有长度不等的可迭代元素

```py
numbers = [23, 33, 43]
input_list = ['one', 'two']
input_tuple = ('YES', 'NO', 'RIGHT', 'LEFT')
# the size of numbers and input_tuple is different
outcome = zip(numbers, input_tuple)

result = set(outcome)
print(result)
result1 = zip(numbers, input_list, input_tuple)
outcome1 = set(result1)
print(outcome1)

```

**输出:**

```py
{(33, 'NO'), (43, 'RIGHT'), (23, 'YES')}
{(23, 'one', 'YES'), (33, 'two', 'NO')}
```

* * *

### zip()函数来解压缩这些值

`**"*" operator**`用于解压缩值，即将元素转换回独立的值。

```py
alphabets = ['a', 'c', 'e']
number = [1, 7, 9]
result = zip(alphabets, number)
outcome = list(result)
print(outcome)
test, train =  zip(*outcome)
print('test =', test)
print('train =', train)

```

**输出:**

```py
[('a', 1), ('c', 7), ('e', 9)]
test = ('a', 'c', 'e')
train = (1, 7, 9)
```

* * *

## 结论

在本文中，我们已经了解了 Python 的 zip()函数的工作原理。

* * *

## 参考

*   Python zip()函数
*   [zip()函数文档](https://docs.python.org/3.3/library/functions.html#zip)