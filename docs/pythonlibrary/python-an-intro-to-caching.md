# Python:缓存介绍

> 原文：<https://www.blog.pythonlibrary.org/2016/02/25/python-an-intro-to-caching/>

缓存是一种存储有限数据量的方式，以便将来可以更快地检索对所述数据的请求。在本文中，我们将看一个简单的例子，它为我们的缓存使用了一个字典。然后我们将继续使用 Python 标准库的 **functools** 模块来创建缓存。让我们首先创建一个将构造我们的缓存字典的类，然后我们将根据需要扩展它。代码如下:

```py

########################################################################
class MyCache:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.cache = {}
        self.max_cache_size = 10

```

这个类的例子没有什么特别的地方。我们只是创建一个简单的类，并设置两个类变量或属性， **cache** 和 **max_cache_size** 。缓存只是一个空字典，而另一个是不言自明的。让我们充实这段代码，让它真正做点什么:

```py

import datetime
import random

########################################################################
class MyCache:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.cache = {}
        self.max_cache_size = 10

    #----------------------------------------------------------------------
    def __contains__(self, key):
        """
        Returns True or False depending on whether or not the key is in the 
        cache
        """
        return key in self.cache

    #----------------------------------------------------------------------
    def update(self, key, value):
        """
        Update the cache dictionary and optionally remove the oldest item
        """
        if key not in self.cache and len(self.cache) >= self.max_cache_size:
            self.remove_oldest()

        self.cache[key] = {'date_accessed': datetime.datetime.now(),
                           'value': value}

    #----------------------------------------------------------------------
    def remove_oldest(self):
        """
        Remove the entry that has the oldest accessed date
        """
        oldest_entry = None
        for key in self.cache:
            if oldest_entry is None:
                oldest_entry = key
            elif self.cache[key]['date_accessed'] < self.cache[oldest_entry][
                'date_accessed']:
                oldest_entry = key
        self.cache.pop(oldest_entry)

    #----------------------------------------------------------------------
    @property
    def size(self):
        """
        Return the size of the cache
        """
        return len(self.cache)

```

这里我们导入了**日期时间**和**随机**模块，然后我们看到了我们之前创建的类。这一次，我们添加了一些方法。其中一种方法是一种叫做**_ _ 包含 __** 的魔法方法。我有点滥用它，但基本思想是，它将允许我们检查类实例，看看它是否包含我们正在寻找的键。 **update** 方法将使用新的键/值对更新我们的缓存字典。如果达到或超过最大缓存值，它还会删除最旧的条目。 **remove_oldest** 方法实际上删除了字典中最老的条目，在这种情况下，这意味着具有最老访问日期的条目。最后，我们有一个名为 **size** 的属性，它返回我们缓存的大小。

如果您添加以下代码，我们可以测试缓存是否按预期工作:

```py

if __name__ == '__main__':
    # Test the cache
    keys = ['test', 'red', 'fox', 'fence', 'junk',
            'other', 'alpha', 'bravo', 'cal', 'devo',
            'ele']
    s = 'abcdefghijklmnop'
    cache = MyCache()
    for i, key in enumerate(keys):
        if key in cache:
            continue
        else:
            value = ''.join([random.choice(s) for i in range(20)])
            cache.update(key, value)
        print("#%s iterations, #%s cached entries" % (i+1, cache.size))
    print

```

在这个例子中，我们设置了一组预定义的键，并对它们进行循环。如果键不存在，我们就把它们添加到缓存中。缺少的部分是更新访问日期的方法，但是我将把它留给读者作为练习。如果您运行这段代码，您会注意到当缓存填满时，它开始适当地删除旧的条目。

现在让我们继续，看看使用 Python 内置的 **functools** 模块创建缓存的另一种方式！

* * *

### 使用 functools.lru_cache

functools 模块提供了一个方便的装饰器，叫做 [lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache) 。注意是在 3.2 中添加的。根据文档，它将“用一个记忆化的可调用函数来包装一个函数，这个函数可以保存最大大小的最近调用”。让我们根据文档中的例子编写一个快速函数，它将抓取各种网页。在这种情况下，我们将从 Python 文档站点获取页面。

```py

import urllib.error
import urllib.request

from functools import lru_cache

@lru_cache(maxsize=24)
def get_webpage(module):
    """
    Gets the specified Python module web page
    """    
    webpage = "https://docs.python.org/3/library/{}.html".format(module)
    try:
        with urllib.request.urlopen(webpage) as request:
            return request.read()
    except urllib.error.HTTPError:
        return None

if __name__ == '__main__':
    modules = ['functools', 'collections', 'os', 'sys']
    for module in modules:
        page = get_webpage(module)
        if page:
            print("{} module page found".format(module))

```

在上面的代码中，我们用 **lru_cache** 来装饰我们的**get _ 网页**函数，并将其最大大小设置为 24 个调用。然后，我们设置一个网页字符串变量，并传入我们希望函数获取的模块。我发现，如果你在 Python 解释器中运行它，效果最好，比如 IDLE。这就要求你对函数运行几次循环。您将很快看到，它第一次运行代码时，输出相对较慢。但是如果您在同一个会话中再次运行它，您会看到它会立即打印出来，这表明 lru_cache 已经正确地缓存了调用。在您自己的解释器实例中尝试一下，亲自看看结果。

还有一个**类型的**参数，我们可以将它传递给装饰器。它是一个布尔值，告诉装饰器如果 typed 设置为 True，就分别缓存不同类型的参数。

* * *

### 包扎

现在，您已经对使用 Python 编写自己的缓存有所了解。如果您正在进行大量昂贵的 I/O 调用，或者如果您想要缓存诸如登录凭证之类的东西，这是一个有趣的工具，非常有用。玩得开心！