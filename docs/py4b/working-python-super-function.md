# 使用 Python 超级函数

> 原文：<https://www.pythonforbeginners.com/super/working-python-super-function>

Python 2.2 引入了一个名为“super”的内置函数，该函数返回一个代理对象，将方法调用委托给一个类——本质上可以是父类或兄弟类。

除非您有使用 Python 的经验，否则这种描述可能没有意义，所以我们将对其进行分解。

本质上，super 函数可以用来访问继承的方法——从父类或兄弟类——在类对象中被覆盖。

或者，正如官方 Python 文档所说的那样:

"[Super 用于] *返回一个代理对象，该对象将方法调用委托给类型的父类或同级类。这对于访问在类中被重写的继承方法很有用。搜索顺序与 getattr()使用的顺序相同，只是跳过了类型本身。*

## 超级功能是怎么用的？

超级函数有点多才多艺，可以以多种方式使用。

用例 1:可以在单个继承中调用 Super，以便引用父类或多个类，而无需显式命名它们。这在某种程度上是一种捷径，但更重要的是，它有助于在可预见的未来保持代码的可维护性。

用例 2:可以在动态执行环境中调用 Super 来实现多重或协作继承。这种用法被认为是 Python 独有的，因为对于只支持单一继承或静态编译的语言来说是不可能的。

当超级功能被引入时，它引发了一些争议。许多开发人员发现文档不清楚，而且函数本身很难实现。它甚至获得了有害的名声。但是重要的是要记住，Python 自 2.2 以来已经有了很大的发展，其中许多问题不再适用。

super 的伟大之处在于它可以用来增强任何模块方法。另外，不需要知道被用作扩展器的基类的细节。超级函数为您处理所有这些。

因此，对于所有意图和目的来说，super 是访问基类的快捷方式，而不必知道它的类型或名称。

在 Python 3 和更高版本中，super 的语法是:

```py
super().methoName(args)
```

而调用 super 的正常方式(在 Python 的旧版本中)是:

```py
super(subClass, instance).method(args)
```

如您所见，Python 的新版本使语法稍微简单了一些。

## 如何在 Python 2 和 Python 3 中调用 Super？

首先，我们将获取一个常规的类定义，并通过添加超级函数来修改它。最初的代码看起来会像这样:

```py
class MyParentClass(object):
def __init__(self):
pass

class SubClass(MyParentClass):
def __init__(self):
MyParentClass.__init__(self) 
```

如您所见，这是一个通常用于单一继承的设置。我们可以看到有一个基类或父类(有时也称为超类)，以及一个指定的子类。

但是我们仍然需要在子类中初始化父类。为了使这个过程更容易，Python 的核心开发团队创建了 super 函数。目标是为初始化类提供一个更加抽象和可移植的解决方案。

如果我们使用的是 Python 2，我们会像这样编写子类(使用 super 函数):

```py
class SubClass(MyParentClass):
def __init__(self):
super(SubClass, self).__init__() 
```

然而，用 Python 3 编写时，相同的代码略有不同。

```py
class MyParentClass():
def __init__(self):
pass

class SubClass(MyParentClass):
def __init__(self):
super() 
```

注意父类不再直接基于对象基类了吗？此外，由于这个超级函数，我们不需要从父类传递任何东西给它。你不认为这要容易得多吗？

现在，记住大多数类也会有参数传递给它们。当这种情况发生时，超能力会发生更大的变化。

它将如下所示:

```py
class MyParentClass():
def __init__(self, x, y):
pass

class SubClass(MyParentClass):
def __init__(self, x, y):
super().__init__(x, y) 
```

同样，这个过程比传统方法简单得多。在这种情况下，我们必须调用超级函数的 __init__ 方法来传递我们的参数。

## **超能力又是干什么用的？**

当你关心向前兼容性时，超级函数是非常有用的。通过将它添加到您的代码中，您可以确保您的工作在未来保持可操作性，只需进行一些全面的更改。

最终，它消除了声明一个类的某些特征的需要，只要你正确地使用它。

为了正确使用该功能，必须满足以下条件:

*   被 *super()* 调用的方法必须存在
*   调用者和被调用者函数都需要有匹配的参数签名
*   使用该方法后，该方法的每次出现都必须包含 *super()*

您可能从单个继承类开始，但是后来，如果您决定添加另一个基类——或者更多——这个过程会顺利得多。你只需要做一些改变，而不是很多。

一直在谈论使用 super 函数进行依赖注入，但是我们还没有看到这方面的任何可靠的例子——至少没有实际的例子。目前，我们只是坚持我们给出的描述。

不管怎样，现在你会明白 super 并不像其他开发者声称的那样糟糕。

## **了解更多关于超级功能的信息**

我们知道很多关于超级函数和 Python 语言的知识，但是我们并不知道所有的事情！你可以去很多其他地方了解更多！

我们建议查看的网站有:

[官方 Python 文档](https://docs.python.org/2/library/functions.html#super)

[Raymond Hettinger 对 Super 和 Python 的探索](https://rhettinger.wordpress.com/2011/05/26/super-considered-super/)

[如何超级有效地使用——Python 2.7 版秘方](https://code.activestate.com/recipes/577721-how-to-use-super-effectively-python-27-version/)

[艰难地学习 Python–练习 44:继承与组合](https://learnpythonthehardway.org/book/ex44.html)

[Python 2.3 方法解析顺序](https://www.python.org/download/releases/2.3/mro/)

[六十北:Python 的超级没你想的那么简单](http://sixty-north.com/blog/pythons-super-not-as-simple-as-you-thought)