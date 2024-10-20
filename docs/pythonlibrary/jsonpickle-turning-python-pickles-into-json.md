# jsonpickle:将 Python pickles 变成 JSON

> 原文：<https://www.blog.pythonlibrary.org/2014/08/13/jsonpickle-turning-python-pickles-into-json/>

前几天，我在 StackOverflow 上看到一个有趣的问题，作者问是否有办法将 Python 字典序列化为人类可读的格式。给出的答案是使用一个名为 [jsonpickle](http://jsonpickle.github.io/) 的包，它将复杂的 Python 对象序列化到 JSON 和从 JSON 序列化。本文将向您简要介绍如何使用这个项目。

* * *

### 入门指南

要正确开始，您需要下载并安装 jsonpickle。通常，您可以使用 pip 来完成这项任务:

```py

pip install jsonpickle

```

Python 2.6 或更高版本没有依赖性。对于旧版本的 Python，您需要安装一个 JSON 包，比如 simplejson 或 demjson。

* * *

### 使用 jsonpickle

让我们从创建一个简单的基于汽车的类开始。然后我们将使用 jsonpickle 序列化该类的一个实例，并对其进行反序列化。

```py

import jsonpickle

########################################################################
class Car(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.wheels = 4
        self.doors = 5

    #----------------------------------------------------------------------
    def drive(self):
        """"""
        print "Driving the speed limit"

if __name__ == "__main__":
    my_car = Car()
    serialized = jsonpickle.encode(my_car)
    print serialized

    my_car_obj = jsonpickle.decode(serialized)
    print my_car_obj.drive()

```

如果您运行此代码，您应该会看到类似下面的输出:

```py

{"py/object": "__main__.Car", "wheels": 4, "doors": 5}
Driving the speed limit

```

这非常有效。序列化的对象在打印出来时非常容易阅读。重构序列化对象也非常简单。

* * *

### 包扎

jsonpickle 包允许开发人员通过其 **load_backend** 和 **set_preferred_backend** 方法选择他们想要使用的 JSON 后端来编码和解码 JSON。如果愿意，您还可以自定义序列化处理程序。总的来说，我相信对于需要能够容易地阅读他们的序列化输出的开发人员来说，这可能是一个方便的项目。

* * *

### 相关阅读

*   jsonpickle [API 引用](http://jsonpickle.github.io/api.html)