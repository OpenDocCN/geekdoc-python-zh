# python——如何判断函数是否被调用

> 原文：<https://www.blog.pythonlibrary.org/2017/03/10/python-how-to-tell-if-a-function-has-been-called/>

去年，我遇到了一种情况，我需要知道一个函数是否被调用了。基本上，我们试图防止关闭一个扭曲的事件循环两次或启动其中两次。总之，在我的研究中，我无意中发现了一个有趣的帖子，展示了几个这样做的方法。

第一种利用了 Python 中的一切都是对象的事实，包括函数本身。让我们看一个简单的例子:

```py

def self_aware_function(a, b):
    self_aware_function.has_been_called = True
    return a + b

if __name__ == '__main__':
    self_aware_function.has_been_called = False

    for i in range(2):
        if self_aware_function.has_been_called:
            print('function already called')
        else:
            print('function not called')

        self_aware_function(1, 2)

```

在这个例子中，我们在被命名为**的函数上创建了一个属性。当函数被调用时，我们将其设置为 True。当你启动你的程序时，你会想要初始化这个属性为 False，就像上面我们做的那样。然后我们用一个 for 循环循环两次。第一次通过它将检查函数是否被调用。既然没有，你会看到它落到 else 语句。现在我们调用了函数，第二次通过循环执行 if 语句的第一部分。**

StackOverflow 的那篇文章还提到了一种使用装饰器跟踪函数调用的好方法。下面是我写的一个例子:

```py

import functools

def calltracker(func):
    @functools.wraps(func)
    def wrapper(*args):
        wrapper.has_been_called = True
        return func(*args)
    wrapper.has_been_called = False
    return wrapper

@calltracker
def doubler(number):
    return number * 2

if __name__ == '__main__':
    if not doubler.has_been_called:
        print("You haven't called this function yet")
        doubler(2)

    if doubler.has_been_called:
        print('doubler has been called!')

```

在这个例子中，我导入了 **functools** 并创建了一个装饰器，我称之为 **calltracker** 。在这个函数中，我们设置了与上一个例子相同的属性，但是在这个例子中，我们将它附加到我们的包装器(即装饰器)上。然后我们修饰一个函数，并尝试一下我们的代码。第一个 **if 语句**检查函数是否已经被调用。它没有，所以我们继续调用它。然后，我们确认在第二个 if 语句中调用了该函数。

### 包扎

虽然这些东西在运行时确实有用，但是您也可以使用 Python 的 [trace](https://docs.python.org/2/library/trace.html) 模块来跟踪代码的执行，从而做类似的事情。这类事情也是通过覆盖工具来完成的。你也会在 [Python 模拟对象](http://stackoverflow.com/questions/3829742/assert-that-a-method-was-called-in-a-python-unit-test)中发现这种功能，因为模拟可以告诉你它何时被调用。

无论如何，希望你会发现这个练习和我一样有趣。虽然我已经知道 Python 中的一切都是对象，但我还没有想过使用这种功能给函数添加属性。