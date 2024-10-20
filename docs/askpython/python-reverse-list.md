# Python 反向列表

> 原文：<https://www.askpython.com/python/list/python-reverse-list>

Python 提供了多种方法来反转[列表](https://www.askpython.com/python/list/python-list)中的元素。

## Python 反向列表元素

以下技术可用于反转 Python 列表:

*   *通过使用反转的()函数*
*   *通过使用 reverse()函数*
*   *利用切片技术*
*   *通过使用 for 循环和 range()函数*

* * *

### 1.反向()函数

`**reversed()**`方法创建一个反向迭代器，以逆序遍历列表。

```py
def reverse_list(input): 
	return [x for x in reversed(input)] 

input = [0, 22, 78, 1, 45, 9] 
print(reverse_list(input)) 

```

**输出:**

```py
[9, 45, 1, 78, 22, 0]
```

* * *

### 2.反向()函数

`**reverse()**`函数提供的功能是**反转元素并将它们存储在同一个列表**中，而不是将元素复制到另一个列表中然后反转。

```py
def reverse_list(input): 
    input.reverse() 
    return input 

input = [0, 22, 78, 1, 45, 9]
print(reverse_list(input)) 

```

**输出:**

```py
[9, 45, 1, 78, 22, 0]
```

* * *

### 3.切片技术

`**slicing technique**`提供了反转列表的功能。

```py
def reverse_list(input): 
	output = input[::-1] 
	return output 

input = [0, 22, 78, 1, 45, 9]
print(reverse_list(input)) 

```

**输出:**

```py
[9, 45, 1, 78, 22, 0]
```

* * *

### 4.通过使用 for 循环和 range()函数

```py
input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get list length
list_len = len(input)

# i goes from 0 to the middle
for x in range(int(list_len/2)):

    n = input[x]
    input[x] = input[list_len-x-1]
    input[list_len-x-1] = n

# At this point the list should be reversed
print(input)

```

**输出:**

```py
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

* * *

## 结论

因此，在本文中，我们已经理解并实现了用 Python 反转列表的各种技术。

* * *

## 参考

*   Python 反向列表
*   [反向列表文件](https://docs.python.org/3/tutorial/datastructures.html)