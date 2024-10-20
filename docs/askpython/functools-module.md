# Python 中的 functools 模块

> 原文：<https://www.askpython.com/python-modules/functools-module>

在本文中，我们将使用 **functools** **模块**来看看 Python 中使用的一个重要函数模式。在编程中，我们经常使用高阶函数。它们是接受另一个函数作为参数并对其进行操作或返回另一个函数的函数。在 Python 中通常被称为 decorators，它们非常有用，因为它们允许对现有函数进行扩展，而无需对原始函数源代码进行任何修改。它们增强了我们的功能，并扩展它们以获得额外的功能。

***functools*** 模块用于使用 Python 内置的高阶函数。在 Python 中，任何作为可调用对象的函数都可以被认为是使用 functools 模块的函数。

## 涵盖的 functools 函数列表

1.  `partial()`
2.  `partialmethod()`
3.  `reduce()`
4.  `wraps()`
5.  `lru_cache()`
6.  `cache()`
7.  `cached_property()`
8.  `total_ordering()`
9.  `singledispatch()`

## 解释和用法

现在让我们开始进一步使用 functools 模块，并实际理解每个函数。

### 1.部分()

**partial** 是以另一个函数为自变量的函数。它接受一组位置参数和关键字参数，这些输入被锁定在函数的参数中。

然后 **partial** 返回所谓的 *partial 对象*，其行为类似于已经定义了那些参数的原始函数。

它用于将多参数函数转换为单参数函数。它们可读性更强、更简单、更容易键入，并且提供了高效的代码补全。

例如，我们将尝试找出 0-10 范围内的数字的平方，首先使用常规函数，然后使用 functools 的 partial()得到相同的结果。这将有助于我们理解它的用法。

*   使用传统功能

```py
def squared(num):
    return pow(num, 2) 
print(list(map(squared, range(0, 10))))
# Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

```

*   使用 functools 中的`partial()`

```py
from functools import partial

print(list(map(partial(pow, exp=2), range(0, 10))))
# Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

```

### 2.partialmethod()

partialmethod 函数类似于 partial 函数，用作类方法的分部函数。它被设计成一个方法定义，而不是可直接调用的。简单地说，它是包含在我们自定义类中的方法。它可用于创建方便的方法，为该方法预定义一些设定值。让我们看一个使用这种方法的例子。

```py
from functools import partialmethod

class Character:
    def __init__(self):
        self.has_magic = False
    @property
    def magic(self):
        return self.has_magic

    def set_magic(self, magic):
        self.has_magic = bool(magic)

    set_has_magic = partialmethod(set_magic, True)

# Instantiating
witcher = Character()
# Check for Magical Powers
print(witcher.magic)  # False
# Providing Magical Powers to our Witcher
witcher.set_has_magic()
print(witcher.magic)  # True

```

### 3.减少()

这种方法经常用于输出某种累计值，由某种预定义的函数计算。它将一个函数作为第一个参数，将一个 iterable 作为第二个参数。它还有一个初始值设定项，如果函数中没有指定初始值设定项的任何特定值，默认为 0，还有一个迭代器来遍历所提供的 iterable 的每一项。

**附加内容:**[Python 中的`reduce()`函数](https://www.askpython.com/python/reduce-function)

*   使用`reduce()`

```py
from functools import reduce

# acc - accumulated/initial value (default_initial_value = 0)
# eachItem - update value from the iterable
add_numbers = reduce(lambda acc, eachItem: acc + eachItem, [10, 20, 30])
print(add_numbers)  # 60

```

### 4.换行()

**包装器**接受它正在包装的函数，并在定义包装器函数时充当函数装饰器。它更新包装函数的属性以匹配被包装的函数。如果没有使用 **`wraps()`** 将此更新传递给包装器函数，则包装器函数的元数据将被返回，而不是原始函数的元数据或属性，它们应该是整个函数的实际元数据。为了避免这些类型的错误程序，包装()非常有用。

还有一个相关的函数是 **`update_wrapper()`** 函数，与 wraps()相同。*wraps()函数是 update_wrapper()函数的语法糖，也是调用 update_wrapper()的方便函数。*

让我们看一个上面的例子来更好地理解这个函数。

*   在不调用包装并尝试访问元数据的情况下定义装饰函数

```py
def my_decorator(func):
    def wrapper(*args, **kwargs):
        """
        Wrapper Docstring
        """
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def original_func():
    """
    Original Function Doctstring
    """
    return "Something"

print(original_func())
print(original_func.__name__)
print(original_func.__doc__)

# Output
'''
Something
wrapper

        Wrapper Docstring
'''

```

*   定义相同的装饰函数，调用`wraps()`并尝试访问元数据

```py
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper Docstring
        """
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def original_func():
    """
    Original Function Doctstring
    """
    return "Something"
print(original_func())
print(original_func.__name__)
print(original_func.__doc__)

# Output
"""
Something
original_func

    Original Function Doctstring
"""

```

### 5.lru_cache( *maxsize=128* ， *typed=False* )

它是一个装饰器，用一个记忆化的可调用函数包装一个函数，这简单地意味着当相同的参数被用于那个非常特殊的昂贵函数时，它在执行昂贵的 I/O 函数操作的同时使函数调用变得高效。它实际上使用最近 maxsize 调用的缓存结果。 **LRU** 在 **lru_cache()** 中是 **L** 东 **R** 最近 **U** sed 的缩写。

它有一个默认的 ***maxsize*** 为 128，它设置了最近调用的数量，这些调用将被缓存或保存，以便在以后的某个时间点用于相同的操作。 **typed=False** 将不同类型的函数参数值缓存在一起，简单来说，这意味着如果我们编写 **type=True** ，那么一个 **integer 10** 和一个 **float 10.0** 的缓存值将被分别缓存。

lru_cache 函数还集成了其他三个函数来执行一些额外的操作，即

*   **cache _ parameters()**–它返回一个新的**字典**，显示 maxsize 和 typed 的值。这只是为了提供信息，数值突变不会影响它。
*   **cache _ info()**–用于测量缓存在需要时重新调优 maxsize 的效率。它输出一个命名元组，显示被包装函数的命中、未命中、最大大小和当前大小。
*   **cache _ clear()**–用于清除之前缓存的结果。

在使用函数 **lru_cache()** 时，还有一些要点需要记住。

*   由于使用 lru_cache()时结果的底层存储是一个字典，** args 和**kwargs 必须是可散列的。*
*   因为只有当一个函数无论执行多少次都返回相同的结果时，缓存才起作用，所以它应该是一个在执行后不会产生副作用的纯函数。

为了便于理解，我们将看到打印斐波那契数列的经典例子。我们知道，对于这个计算，迭代是使用相同的值一次又一次地完成的，以在找到 Fibonacci 数时输出该数。*嗯，缓存那些重复数字的值似乎是 **lru_cache()** 的一个很好的用例。*让我们看看它的代码。

**代码:**

```py
from functools import lru_cache
@lru_cache(maxsize=32)
def fibonacci(n):
    if n < 2:
        return n
    print(f"Running fibonacci for {n}")
    return fibonacci(n - 1) + fibonacci(n - 2)

print([fibonacci(n) for n in range(15)])
print(fibonacci.cache_parameters())
print(fibonacci.cache_info())

```

**输出:**

```py
Running fibonacci for 2
Running fibonacci for 3 
Running fibonacci for 4 
Running fibonacci for 5 
Running fibonacci for 6 
Running fibonacci for 7 
Running fibonacci for 8 
Running fibonacci for 9 
Running fibonacci for 10
Running fibonacci for 11
Running fibonacci for 12
Running fibonacci for 13
Running fibonacci for 14
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
{'maxsize': 32, 'typed': False}
CacheInfo(hits=26, misses=15, maxsize=32, currsize=15)

```

**说明**:

我们可以看到，对于 15 范围内的每个不同值，print 语句只提供了一次输出。虽然迭代已经做了很多，但是通过使用 **lru_cache()** ，我们只能看到唯一的结果被打印，重复的结果被缓存。在结果旁边是上述函数执行后的信息。

### 6.缓存()

这个函数是 lru_cache()本身的一个较小的轻量级形式。但是缓存值的数量没有固定的界限。因此，我们不需要为它指定 maxsize。和使用 **lru_cache(maxsize=None)是一样的。**

因为使用这个函数不会忘记正在缓存的值，所以它使得 cache()比有大小限制的 lru_cache()快得多。这是 Python 3.9 的一个新的补充

使用此函数时要记住的一件事是，在使用大量输入的函数实现它时，我们可能会得到非常大的缓存大小。因此，应该谨慎使用，并事先考虑清楚。

### 7.cached _ property()

**cached_property()** 类似于 Python 中的 **property()** ，允许我们将类属性作为内置函数转换为属性或托管属性。*它额外提供了**缓存功能*** 并在 **Python 3.8** 中引入

计算值计算一次，然后保存为该实例生命周期的普通属性。与 property()函数不同，cache_property()还允许在没有定义 setter 的情况下进行写入。

Cached_property 仅运行:

*   如果属性还不存在
*   如果它必须执行查找

通常，如果该属性已经存在于函数中，它会像普通属性一样执行读写操作。为了清除缓存的值，需要删除属性，这实际上允许再次调用 cached_property()函数。

*   不使用 catched_property()，查看输出

```py
class Calculator:
    def __init__(self, *args):
        self.args = args

    @property
    def addition(self):
        print("Getting added result")
        return sum(self.args)

    @property
    def average(self):
        print("Getting average")
        return (self.addition) / len(self.args)

my_instance = Calculator(10, 20, 30, 40, 50)

print(my_instance.addition)
print(my_instance.average)

"""
Output

Getting added result
150
Getting average
Getting added result
30.0
"""

```

*   使用 catched_property()，查看输出

```py
from functools import cached_property

class Calculator:
    def __init__(self, *args):
        self.args = args

    @cached_property
    def addition(self):
        print("Getting added result")
        return sum(self.args)

    @property
    def average(self):
        print("Getting average")
        return (self.addition) / len(self.args)

my_instance = Calculator(10, 20, 30, 40, 50)

print(my_instance.addition)
print(my_instance.average)

"""
Output

Getting added result
150
Getting average
30.0
"""

```

**说明:**

我们可以看到，在计算平均值时，得到相加结果的**已经打印了两次*，因为对于第一个函数，它的结果还没有被缓存。但是当我们使用**@ cached _ property decorator**时，加法的结果已经被缓存，因此直接从内存中使用来获得平均值。***

### ***8.total_ordering()***

***functools 中的这个高阶函数，当用作类装饰器时，假设我们的类包含一个或多个丰富的比较排序方法，它提供了其余的方法，而没有在我们的类中显式定义。***

***这意味着当我们在我们的类中使用比较 dunder/magic 方法 `**__gt__()**, **__lt__(), __ge__(), __le__()**`时，如果我们使用 define **`__eq__()`** 和*只是其他四个方法*中的一个，其余的将由 functools 模块中的 **`total_ordering()`** 自动定义。***

***需要注意的一件重要事情是，它确实降低了代码执行的速度，并且为我们的类中没有明确定义的比较方法创建了更复杂的堆栈跟踪。***

***让我们看一个简单的例子。***

*****代码:*****

```py
*@functools.total_ordering
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def __eq__(self, other):
        return self.marks == other.marks

    def __gt__(self, other):
        return self.marks > other.marks

student_one = Student("John", 50)
student_two = Student("Peter", 70)

print(student_one == student_two)
print(student_one > student_two)

print(student_one >= student_two)
print(student_one < student_two)
print(student_one <= student_two)

"""
Output:

False
False
False
True
True
"""* 
```

*****说明:*****

****在上面的代码中，我们用不同的语法导入了* **`total_ordering()`** *。这与从前面所有示例中使用的 functools 导入是一样的。****

***我们创建的类只包含两个比较方法。但是通过使用 total_ordering() class decorator，我们使我们的类实例能够自己派生其余的比较方法。***

### ***9.**单一调度**()***

***当我们定义一个函数时，它对不同输入类型的参数执行相同的操作。但是，如果我们希望一个函数在参数的输入类型不同时有不同的行为呢？***

***我们发送一个列表或一个字符串或一些其他类型，我们想要不同的输出取决于我们发送的数据。如何才能实现这一点？***

***functools 模块有 **`singledispatch()`** 装饰器，帮助我们编写这样的函数。实现基于被传递的参数的类型。***

***泛型函数的 **`register()`** 属性和 `**singledispatch()**`方法用于修饰重载的实现。如果实现像静态类型语言一样用类型注释，装饰器会自动推断传递的参数的类型，否则类型本身就是装饰器的参数。***

***同样，它可以通过使用 **`singledispatchmethod()`** 实现为类方法，以达到相同的结果。***

***让我们看一个例子来更好地理解它。***

```py
*from functools import singledispatch

@singledispatch
def default_function(args):
    return f"Default function arguments: {args}"

@default_function.register
def _(args: int) -> int:
    return f"Passed arg is an integer: {args}"

@default_function.register
def _(args: str) -> str:
    return f"Passed arg is a string: {args}"

@default_function.register
def _(args: dict) -> dict:
    return f"Passed arg is a dict: {args}"

print(default_function(55))
print(default_function("hello there"))
print(default_function({"name": "John", "age": 30}))

print(default_function([1, 3, 4, 5, 6]))
print(default_function(("apple", "orange")))

"""
Output:

Passed arg is an integer: 55
Passed arg is a string: hello there
Passed arg is a dict: {'name': 'John', 'age': 30}

Default function arguments: [1, 3, 4, 5, 6]
Default function arguments: ('apple', 'orange') 

"""* 
```

*****说明:*****

***在上面的例子中，我们可以看到函数实现是基于传递的参数类型完成的。与其他类型不同，未定义的类型由默认函数执行。***

## ***结论***

***在本文中，我们介绍了 Python 中 functools 模块提供的大多数函数。这些高阶函数提供了一些很好的方法来优化我们的代码，从而产生干净、高效、易于维护和读者友好的程序。***

## ***参考***

***[Python functools 文档](https://docs.python.org/3/library/functools.html)***