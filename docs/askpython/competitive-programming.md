# Python 中的竞争性编程:你需要知道什么？

> 原文：<https://www.askpython.com/python/competitive-programming>

你好，这里是编码器！我很确定你熟悉什么是竞争性编程。但是在用 python 编码时，需要记住一些重要的事情。这些小事情会给你的代码带来巨大的不同。

## Python 中的竞争性编程

让我们逐一研究其中的几个。

### 1.发电机的使用

使用[生成器](https://www.askpython.com/python/examples/generators-in-python)可以减少空间和时间的复杂性，并且比使用函数要好。下面显示了一个生成器函数的图示。

同时一个接一个地返回多个值也很有帮助。

```py
def FirstGen():
    yield 1
    yield 2
    yield 3
for i in FirstGen():
    print(i,end=" ")

```

### 2.内置函数的使用

使用[内置函数](https://www.askpython.com/python/python-functions)和库是比普通方法更好的方法。让我们看下面一个简单的程序，它有一个新的列表，包含第一个列表的元素的平方。

为了更好地说明差异，我们将在 time `time`模块的帮助下计算程序的执行时间。

```py
import time
start_time = time.time()

def get_square(x):
    return x**2
l1 = [i for i in range(100000)]
l2 = []
for i in l1:
    l2.append(get_square(i))
print(len(l2))

print("Time taken by the code: %s seconds."% (time.time() - start_time))

```

上面的方法显示了在`0.06881594657897949`秒内的正确输出，这无疑是相当不错的。

现在让我们使用内置函数`map`来尝试相同的程序，并将声明的函数直接应用于列表。

```py
import time
start_time = time.time()

def get_square(x):
    return x**2
l1 = [i for i in range(100000)]
l2 = list(map(get_square,l1))
print(len(l2))

print("Time taken by the code: %s seconds."% (time.time() - start_time))

```

在这里，我们看到相同列表所用的时间是`0.048911094665527344`秒，这看起来是一个非常小的差异，但是对于更大的数据，这个差异可能会变得更大。

### 3.使用 itertools

这个模块对于解决一些复杂的问题非常有帮助。例如，看看下面给出的程序，找出一个列表的所有排列。

```py
import itertools
x = list(itertools.permutations([1,2,3]))
print(x)

```

同样的事情也可以通过创建你自己的逻辑和函数来完成，但是那样会太复杂，时间复杂度也更高。

### 4.使用地图功能

每当我们需要在由空格分隔的一行中输入一个整数数组的所有元素时， [map 函数](https://www.askpython.com/python-modules/mmap-function)是实现这一点的最佳方法。

```py
l1 = list(map(int,input("Enter all the elements: ").split()))
print(l1)

```

使用`map`函数简化了在一行中输入多个值的复杂性。

### 5.串并置

要将多个字符串连接在一起，我们可以使用两种方法:将字符串添加到字符串或使用 join 函数。

建议使用`join`函数，因为它在一行中执行整个连接过程，如果字符串数量很大，可以降低复杂性。

让我们看看第一种方法:对字符串使用加法运算。下面给出的程序最后的执行时间是`0.00498509407043457`秒。

```py
import time
start_time = time.time()
l = [str(i) for i in range(10000)]
st=""
for i in l:
    st+=i
print(len(st))
print("Time taken by the code: %s seconds."% (time.time() - start_time))

```

然而，第二种方法:使用 join 操作给出的时间复杂度只有`0.002988576889038086`秒，这显然要小得多。

```py
import time
start_time = time.time()
l = [str(i) for i in range(10000)]
st = ""
st.join(l)
print(len(st))
print("Time taken by the code: %s seconds."% (time.time() - start_time))

```

## 结论

恭喜你！今天，您学习了一些非常基本但重要的事情，在使用 python 编程语言进行竞争性编程时，请记住这些事情。

这些技巧肯定能在很大程度上帮助您提高我们解决方案的效率和准确性。

自己去试试吧！编码快乐！