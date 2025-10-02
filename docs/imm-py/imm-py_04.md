# Map 和 Filter

Map 和 Filter 这两个函数能为函数式编程提供便利。我们会通过实例一个一个讨论并理解它们。

# Map

# `Map`

`Map`会将一个函数映射到一个输入列表的所有元素上。这是它的规范：

**规范**

```py
map(function_to_apply, list_of_inputs) 
```

大多数时候，我们要把列表中所有元素一个个地传递给一个函数，并收集输出。比方说：

```py
items = [1, 2, 3, 4, 5]
squared = []
for i in items:
    squared.append(i**2) 
```

`Map`可以让我们用一种简单而漂亮得多的方式来实现。就是这样：

```py
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items)) 
```

大多数时候，我们使用匿名函数(lambdas)来配合`map`, 所以我在上面也是这么做的。 不仅用于一列表的输入， 我们甚至可以用于一列表的函数！

```py
def multiply(x):
        return (x*x)
def add(x):
        return (x+x)

funcs = [multiply, add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))
    print(value)

# Output:
# [0, 0]
# [1, 2]
# [4, 4]
# [9, 6]
# [16, 8] 
```

# Filter

# `Filter`

顾名思义，`filter`能创建一个列表，其中每个元素都是对一个函数能返回`True`. 这里是一个简短的例子：

```py
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

# Output: [-5, -4, -3, -2, -1] 
```

这个`filter`类似于一个`for`循环，但它是一个内置函数，并且更快。

注意：如果`map`和`filter`对你来说看起来并不优雅的话，那么你可以看看另外一章：列表/字典/元组推导式。

> 译者注：大部分情况下推导式的可读性更好