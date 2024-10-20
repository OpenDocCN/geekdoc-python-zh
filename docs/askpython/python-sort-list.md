# Python 排序列表

> 原文：<https://www.askpython.com/python/list/python-sort-list>

Python 的 List `**sort()**`方法按照升序/降序/用户定义的顺序对元素进行排序。

## Python 排序列表

以下是对元素进行排序的各种技术:

*   **按升序排列列表**
*   **按降序排列列表**
*   使用用户定义的顺序对列表进行排序
*   **对对象列表进行排序**
*   **使用键对列表进行排序**

* * *

### 1.按升序对列表元素排序

`**sort()**`函数用于对列表中的元素进行升序排序。

```py
input = [1.2, 221, 0.025, 0.124, 1.2]

print(f'Before sorting of elements: {input}')

input.sort()

print(f'After sorting of elements: {input}')

```

**输出:**

```py
Before sorting of elements: [1.2, 221, 0.025, 0.124, 1.2]
After sorting of elements: [0.025, 0.124, 1.2, 1.2, 221]
```

* * *

### 2.按降序对列表元素进行排序

`**reverse**`参数用于按降序对列表元素进行排序。

**语法:** `**list**-**name.sort(reverse=True)**`

```py
input = [8, 1, 12, 0]

input.sort(reverse = True)

print(input)

```

**输出:**

```py
[12, 8, 1, 0]
```

* * *

### 3.使用关键字函数的 Python 排序列表

Python 使用一个键函数作为参数来提供列表元素的排序。基于 key 函数的输出，列表将被排序。

```py
# takes third element for sort
def third_element(x):
    return x[2]

input = [(2, 2, 1), (3, 4, 9), (4, 1, 0), (1, 3, 7)]

# sort list with key
input.sort(key=third_element)

# prints sorted list
print('Sorted list:', input)

```

**输出:**

```py
Sorted list: [(4, 1, 0), (2, 2, 1), (1, 3, 7), (3, 4, 9)]
```

* * *

### 4.使用用户定义的顺序对列表排序

```py
# takes third element for sort
def third_element(x):
    return x[2]

input = [(2, 2, 1), (3, 4, 9), (4, 1, 0), (1, 3, 7)]

# sorts list with key in ascending order
input.sort(key=third_element)

# prints sorted list
print('Sorted list in ascending order:', input)

# sorts list with key in descending order
input.sort(key=third_element, reverse=True)

print('Sorted list in descending order:', input)

```

**输出:**

```py
Sorted list in ascending order: [(4, 1, 0), (2, 2, 1), (1, 3, 7), (3, 4, 9)]
Sorted list in descending order: [(3, 4, 9), (1, 3, 7), (2, 2, 1), (4, 1, 0)]
```

* * *

### 5.对对象列表进行排序

为了使用 sort()函数对自定义对象列表进行排序，我们需要指定 key 函数来指定对象的字段，以达到同样的目的。

```py
class Details:

    def __init__(self, name, num):
        self.name = name
        self.num = num

    def __str__(self):
        return f'Details[{self.name}:{self.num}]'

    __repr__ = __str__

D1 = Details('Safa', 12)
D2 = Details('Aman', 1)
D3 = Details('Shalini', 45)
D4 = Details('Ruh', 30)

input_list = [D1, D2, D3, D4]

print(f'Before Sorting: {input_list}')

def sort_by_num(details):
    return details.num

input_list.sort(key=sort_by_num)
print(f'After Sorting By Number: {input_list}')

```

**输出:**

```py
Before Sorting: [Details[Safa:12], Details[Aman:1], Details[Shalini:45], Details[Ruh:30]]
After Sorting By Number: [Details[Aman:1], Details[Safa:12], Details[Ruh:30], Details[Shalini:45]]
```

* * *

## 结论

因此，我们已经了解了对列表中的元素进行排序的各种技术。

* * *

## 参考

*   Python 排序列表
*   [整理文件](https://docs.python.org/3.3/howto/sorting.html)