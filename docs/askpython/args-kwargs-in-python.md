# * Python 中的 args 和**kwargs

> 原文：<https://www.askpython.com/python/args-kwargs-in-python>

Python 提供了一些方便的方法，通过这些方法我们可以让函数接受可变数量的参数。`*args`和`**kwargs`就是这么做的。

`*args` - >代表一个 ***列表*/**/*元组*** 的**位置**参数要传递给任何函数**

`**kwargs` - >代表一个 ***字典*** 中的**关键字**参数要传递给任何函数

* * *

## *参数的用途

在定义函数时，如果您不确定要作为函数参数传递的参数的数量，通常会使用`*args`。所以本质上，这种类型的语法允许我们向函数传递任意数量的参数，确切的数量在运行时确定。

有两种情况下`*`(星号)运算符的含义不同。

### 案例 1:在函数定义中

这里，`*`操作符用于将参数打包到传递给函数的元组/列表(包含所有位置)中。因此，我们在定义中使用`*args`来表示传递给函数的所有位置参数都被打包到一个名为`args`的列表/元组中(可以给出任何其他名称，但是通常的做法是编写`*args`来表示使用了参数打包)

```py
def find_average(*args):
    total = 0
    print('Packed Argument Tuple ->', args)
    for i in args:
        total += i
    return total / len(args)

print('Average ->', find_average(1, 2, 3, 4, 5))

```

输出

```py
Packed Argument Tuple -> (1, 2, 3, 4, 5)
Average -> 3.0

```

* * *

### 案例 2:在函数调用中

这里，`*`操作符用于解包传递给它的对应列表/元组，甚至是一个生成器。

```py
a = [1, 2, 3]
print(*a)

```

输出

```py
1 2 3

```

如果您希望一个 iterable 只被相应的函数调用，这可能是有用的。

* * *

### 组合案例 1 和案例 2 以使用*args

这里有一个例子，使用 **Case1** 和 **Case2** 来计算一个被解包并传递到一个函数中的列表的最大值，这个函数采用可变数量的参数。

```py
def compute_maximum(*args):
    maximum = 0
    for i in args:
        if i > maximum:
           maximum = i
    return maximum

a = [4, 5, 10, 14, 3]
print('Maximum ->', compute_maximum(*a))

```

输出

```py
Maximum -> 14

```

* * *

## **kwargs 的目的

这里，`**`操作符的使用方式与前一种情况类似，但是它专门用于将传递给函数的关键字参数打包到字典中。`**kwargs`习语只适用于函数定义，与`*args`不同，它在函数调用中没有任何特殊意义。

这里有一个例子来说明`**kwargs`的用途

```py
def find_average(**kwargs):
    total = 0
    print('Keyword Argument Dictionary ->', kwargs)
    for key, value in kwargs.items():
        total += value
    return total / len(kwargs.items())

print('Average ->', find_average(first=1, second=2, third=3, fourth=4, fifth=5))

```

输出

```py
Keyword Argument Dictionary -> {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5}
Average -> 3.0

```

这里可以使用`*`操作符来解包`**kwargs`，并将所有的`keys` / `values`传递给关键字字典

```py
>>> print(*kwargs)
first second third fourth fifth
>>> print(*kwargs.values())
1 2 3 4 5

```

* * *

## 结论

这篇文章帮助我们更深入地理解了如何在函数定义中使用`*args`和`**kwargs`来获得可变数量的位置/关键字参数并操作它们，以及程序员如何在编写易于使用的函数的常见实践中使用它。

## 参考

stack overflow:[https://stack overflow . com/questions/3394835/use-of-args-and-kwargs](https://stackoverflow.com/questions/3394835/use-of-args-and-kwargs)