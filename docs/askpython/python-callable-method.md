# 如何使用 Python callable()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-callable-method>

## 介绍

在本教程中，我们将讨论**Python callable()方法**及其用法和工作原理。

基本上，当一个对象或实例有一个已定义的 `__call__()`函数时，它被称为**可调用的**。那样的话，我们可以用更简单的方式引用 a.__call__(arg1，arg2，…)，a(arg1，arg2，…)。因此，它变得可调用。

## Python callable()方法

此外，Python 中的`callable()`方法使得用户更容易识别可调用和不可调用的对象和函数。这是一个单参数函数，如果传递的对象是可调用的，则返回 **true** ，否则返回 **false** 。

下面给出了该方法的**语法**,

```py
callable(obj)

```

这里的`obj`是用户想要检查是否可调用的实例或对象。

## Python callable()方法的工作原理

让我们看一些例子来清楚地理解 Python 中的`callable()`方法。

## 当 Python callable()返回 True 时

如前所述，当传递的对象可调用时，该方法返回 **true** 。让我们看看它是在什么条件下这样做的。

```py
#true
def demo():
    print("demo() called!")

#object created
demo_obj = demo

class demo_class:
    def __call__(self, *args, **kwargs): #__call__() is defined here
        print("__call__() defined!")

demo_class_obj = demo_class()

print("demo_obj is callable? ",callable(demo_obj))
print("demo_class is callable? ",callable(demo_class)) #classes are always callable
print("demo_class_obj is callable? ",callable(demo_class_obj))

demo_obj() #calling demo()'s object
demo_class_obj() #calling the demo_class object

```

**输出**:

```py
demo_obj is callable?  True
demo_class is callable?  True
demo_class_obj is callable?  True
demo() called!
__call__() defined!

```

这里，

*   我们定义`demo()`函数，创建它的新实例 **demo_obj** ，
*   然后用`__call__()`函数定义一个新的类 **demo_class** ，
*   并创建名为 **demo_class_obj** 的类 demo_class 的对象，
*   最后，检查创建的对象和类是否是可调用的。我们可以从输出中看到，是可调用的。
*   最后，我们调用函数`demo()`和`demo_class_obj()`。在演示类的对象调用中，执行了 **__call__()** 方法，我们可以从输出中看到这一点。

**注意:**所有的类都是可调用的，所以对于任何一个类，callable()方法都返回 true。这在上面的例子中很明显，我们试图检查 demo_class 的`callable()`输出。

## 当 Python callable()返回 False 时

同样，当传递的对象不可调用时，`callable()`返回 **false** 。让我们看看它是在什么条件下这样做的。

```py
n = 10

class demo_class:
    def print_demo(self):
        print("demo")

demo_class_obj = demo_class()

print("n is callable? ",callable(n))

print("demo_class_obj is callable? ",callable(demo_class_obj))

```

**输出**:

```py
n is callable?  False
demo_class_obj is callable?  False

```

在上面的代码中，

*   我们将整数 n 初始化为 **n = 10** ，
*   然后用成员函数`print_demo()`定义一个类 **demo_class** ，
*   之后，我们创建一个名为 **demo_class_obj** 的 demo_class 对象，
*   最后，检查 **n** 和 **demo_class_obj** 是否可调用，从上面的输出可以看出，它们是不可调用的。

**n** 是一个整数，显然不能调用。而在 demo_class_obj 的情况下，类(demo_class)没有定义好的
`__call__()`方法。因此不可调用。

## 结论

因此，在本教程中，我们学习了 Python callable()方法及其工作原理。该方法广泛用于程序的防错。

在实际调用之前检查对象或函数是否可调用有助于避免**类型错误**。

希望你对题目有清晰的认识。如有任何问题，欢迎在下面评论。

## 参考

*   Python 可调用()和 _ _ call _ _()–Journal Dev Post，
*   [什么是“可赎回”？](https://stackoverflow.com/questions/111234/what-is-a-callable)–stack overflow 问题，
*   [Python 文档](https://docs.python.org/3/library/functions.html#callable)关于 callable()。