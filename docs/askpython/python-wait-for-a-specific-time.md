# 如何在 Python 中等待一个特定的时间？

> 原文：<https://www.askpython.com/python/examples/python-wait-for-a-specific-time>

大家好！在这篇文章中，我们将看看如何在 Python 中等待一个特定的时间。

当您将某些事件或任务安排在特定时间段之后时，这一点尤为重要。

Python 为我们提供了不同的方法来做到这一点。因此，让我们看看我们可以使用的所有方法:在单线程和多线程环境中！

* * *

## python–在单线程环境中等待特定的时间

如果你的主程序只包含一个单独的线程/程序，那么 Python 让这变得非常容易。

在 Python 中让程序等待特定时间的一种可能方法是使用 **[时间](https://www.askpython.com/python-modules/python-time-module)** 模块。

### 使用 time.sleep()等待

我们可以用`time.sleep(n)`来等待`n`秒。当然，我们可以将`n`设为十进制，这样我们的区间会更精确！

下面是一个简单的例子，它将两个函数调用安排在 3 秒钟之内。

```py
import time

def fun1(a):
    return 2 * a

def fun2(a):
    return a * a

if __name__ == '__main__':
    inp = 10
    print(f"Input = {inp}")
    print(f"Result of fun1 call: {fun1(inp)}")
    time.sleep(3) # Wait for 3 seconds
    print(f"After 3 milliseconds, Result of fun2 call: {fun2(inp)}")

```

**输出**

```py
Input = 10
Result of fun1 call: 20
After 3 seconds, Result of fun2 call: 100

```

很简单，对吧？

* * *

## 在多线程环境中等待

如果你的程序中有多个线程，你仍然可以通过`time.sleep()`使用相同的逻辑让一个特定的线程在 Python 中等待特定的时间

下面是一个例子，它产生了 3 个线程，让它们交替休眠一秒钟，而其他线程继续打印从 1 开始的数字。

我们将使用`concurrent.futures`模块并使用`ThreadPoolExecutor`来共享和执行线程，同时使用`as_completed()`来获取结果。

使用线程生成和获取的基本结构如下:

```py
from concurrent.futures import ThreadPoolExecutor, as_completed

# The threads will call this function
def callback():
    pass

with ThreadPoolExecutor() as thread_executor:
    # Await all results
    await_results = [thread_executor.submit(callback) for i in range(1, tid+1)]
    # Fetch them!
    for f in as_completed([future for future in await_results]):
        print(f.result())

```

现在，让我们编写主程序的代码。

```py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Global Variable for the Thread ID Number
tid = 0
# Spawn 4 threads
NUM_THREADS = 4

def func(arg):
    time.sleep(1)
    return arg * arg

if __name__ == '__main__':
    with ThreadPoolExecutor() as thread_executor:
        start_time = time.time()
        # Going to spawn threads
        tid += NUM_THREADS
        # Await all results
        await_results = [thread_executor.submit(func, arg=i) for i in range(1, tid+1)]
        for f in as_completed([future for future in await_results]):
            print(f.result())
        end_time = time.time()
        print(f"Total Time taken for {NUM_THREADS} threads: {end_time - start_time}")

```

**输出**

```py
1
4
9
16
Total Time taken for 4 threads: 1.0037879943847656

```

如您所见，我们产生了 4 个线程，它们在给出函数结果之前都等待了 1 秒钟。这非常接近 1 秒，所以我们的输出是有意义的！

### 使用线程。Timer()来调度函数调用

然而，如果你想让一个特定的函数在 Python 中等待特定的时间，我们可以使用来自`threading`模块的`threading.Timer()`方法。

我们将展示一个简单的例子，它每 5 秒调度一次函数调用。

```py
from threading import Timer
import time

def func(a, b):
    print("Called function")
    return a * b

# Schedule a timer for 5 seconds
# We pass arguments 3 and 4
t = Timer(5.0, func, [3, 4])

start_time = time.time()

# Start the timer
t.start()

end_time = time.time()

if end_time - start_time < 5.0:
    print("Timer will wait for sometime before calling the function")
else:
    print("5 seconds already passed. Timer finished calling func()")

```

**输出**

```py
Timer will wait for sometime before calling the function
Called function

```

这里，主程序在 5 秒钟之前到达了最后一行，所以 Timer 让程序等待，直到它调用`func()`。

在调用`func(a, b)`之后，它终止程序，因为没有其他东西可以运行。

还要注意，函数的返回值不能被主程序使用。

希望这给了你更多关于安排和等待间隔的想法。

* * *

## 结论

在本文中，我们学习了使用`time.sleep()`在 Python 中等待一段特定的时间。我们还看到了如何在多线程环境中使用它，以及调度函数调用。

## 参考

*   线程计时器上的 [Python 文档](https://docs.python.org/3/library/threading.html#timer-objects)
*   [StackOverflow 问题](https://stackoverflow.com/questions/510348/how-do-i-make-a-time-delay)关于创建时间延迟

* * *