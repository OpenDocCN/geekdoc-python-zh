# Python 中的新特性:异步理解/生成器

> 原文：<https://www.blog.pythonlibrary.org/2017/02/14/whats-new-in-python-asynchronous-comprehensions-generators/>

Python 3.6 增加了创建异步理解和异步生成器的能力。你可以在 [PEP 530](https://www.python.org/dev/peps/pep-0530/) 中阅读异步理解，而异步发电机在 [PEP 525](https://www.python.org/dev/peps/pep-0525/) 中描述。该文档指出，您现在可以创建异步列表、集合和字典理解以及生成器表达式。他们的[示例](https://docs.python.org/3.6/whatsnew/3.6.html#pep-530-asynchronous-comprehensions)如下所示:

```py

result = [i async for i in aiter() if i % 2]

```

基本上你只需要在你的表达式中添加 Python 新的 **async** 关键字，并调用一个实现了 **__aiter__** 的 callable。但是，尝试遵循这种语法实际上会导致语法错误:

```py

>>> result = [i async for i in range(100) if i % 2]
  File "", line 1
    result = [i async for i in range(100) if i % 2]
                    ^
SyntaxError: invalid syntax 
```

这实际上是 be 的定义。如果你在 PEP 530 中查找，你会看到它声明如下:*异步理解只允许在异步定义函数中使用。*当然，你也不能把 Python 的**wait**放在理解中，因为这个关键字只能在 **async def** 函数体内使用。只是为了好玩，我尝试定义一个异步 def 函数，看看我的想法是否可行:

```py

import asyncio

async def test(): 
    return [i async for i in range(100) if i % 2]

loop = asyncio.get_event_loop()
loop.run_until_complete(test())

```

如果运行此代码，将会得到一个**类型错误:“async for”需要一个具有 __aiter__ 方法的对象，得到的范围为**。你真正想做的是调用另一个异步 def 函数，而不是直接调用 range。这里有一个例子:

```py

import asyncio

async def numbers(numbers):
    for i in range(numbers):
        yield i
        await asyncio.sleep(0.5)

async def main():
    odd_numbers = [i async for i in numbers(10) if i % 2]
    print(odd_numbers)

if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(main())
    finally:
        event_loop.close()

```

从技术上讲， **numbers** 函数是一个异步生成器，它为我们的异步列表理解生成值。

* * *

### 包扎

创建异步列表理解与创建常规列表理解有很大不同。如您所见，要让它工作需要更多的代码。根据 [PEP 525](https://www.python.org/dev/peps/pep-0525/) 的说法，除了增加开箱即用的异步能力，新的异步生成器实际上应该比实现为异步迭代器的等效生成器快*2 倍。虽然我目前还没有这个新功能的用例，但看看我现在能做什么真的很好，我将把这些新概念归档，以备将来应用。*

* * *

### 相关阅读

*   Python 3.6 的新特性:[异步理解](https://docs.python.org/3.6/whatsnew/3.6.html#pep-530-asynchronous-comprehensions)
*   Python 3.6 的新特性:[异步生成器](https://docs.python.org/3.6/whatsnew/3.6.html#pep-525-asynchronous-generators)
*   人教版 530 -异步理解
*   [PEP 525](https://www.python.org/dev/peps/pep-0525/) -异步发电机
*   InfoWorld: [Python 3.6 充满了优点](http://www.infoworld.com/article/3149782/application-development/python-36-is-packed-with-goodness.html)