# Python 201:什么是超级？

> 原文：<https://www.blog.pythonlibrary.org/2014/01/21/python-201-what-is-super/>

Python 编程语言在 2.2 版本中添加了 **super()** 类型。出于某种原因，这仍然是一个很多初学者不理解的话题。我的一个读者最近问我关于它的问题，由于我并不真正使用它，我决定做一些研究，希望自己能理解它的用法，这样我就能解释什么是 super 以及为什么你会使用它。我们将花一些时间来看看不同的人对超级的定义，然后看一些例子来试图弄清楚这一点。

### 什么是超级？

下面是官方的 [Python 文档](http://docs.python.org/2/library/functions.html#super)对**超级** : *返回一个代理对象，该对象将方法调用委托给父类或兄弟类。这对于访问在类中被重写的继承方法很有用。搜索顺序与 getattr()使用的顺序相同，只是跳过了类型本身。*

几年前，Cody Precord 发布了一本 wxPython 食谱，其中他一直使用 super。这在 wxPython 邮件列表上引起了几个星期对 super 的讨论。让我们来看看其中的一些[线程](https://groups.google.com/forum/?hl=en&fromgroups=#!topic/wxpython-users/-YYLRZ9NvFs)。

wxPython 的创建者 Robin Dunn 陈述如下:*在大多数情况下没有任何区别。在有多重继承的情况下，super()有助于确保在继承树上移动时遵循正确的
方法解析顺序(MRO)。*然后他链接到[斯塔克韦尔弗洛](http://stackoverflow.com/questions/576169/understanding-python-super)。

在另一个[主题](http://wxpython-users.1045709.n5.nabble.com/Super-object-usage-explanation-td3408498.html)中，Tim Roberts(一个非常有经验的 Python 程序员)说:

这是一条“捷径”,允许你访问一个派生类的基类，而不必知道或键入基类名。比如:

```py

class This_is_a_very_long_class_name(object):
    def __init__(self):
        pass

class Derived(This_is_a_very_long_class_name):
    def __init__(self):
        super(Derived,self).__init__()   #1
        This_is_a_very_long_class_name.__init__(self)    #2

```

最后两行是同一事物的两种拼法。除了拼写之外，这还允许您更改基类，而不必遍历所有代码并替换基类名称。C++程序员经常在他们的派生类中使用 typedef 来实现这一点。

所以我从这两个陈述中得到的是，当你做多重继承时， **super** 是很重要的，但是除此之外，你是否使用它并不重要。如果你向下滚动第二个线程，你会看到 Robin Dunn 和 Tim Roberts 的几个例子，它们有助于说明使用 **super** 是有帮助的。让我们看看邓恩先生的例子。

*注意:下面的代码使用 Python 2.x 语法进行 super。在 Python 3 中，不需要传递类名，甚至不需要传递 self！*

```py

class A(object):
    def foo(self):
        print 'A'

class B(A):
    def foo(self):
        print 'B'
        super(B, self).foo()

class C(A):
    def foo(self):
        print 'C'
        super(C, self).foo()

class D(B,C):
    def foo(self):
        print 'D'
        super(D, self).foo()

d = D()
d.foo() 

```

如果您运行这段代码，您将得到字母“DBCA”(每行一个字母)作为输出。正如 Robin Dunn 在帖子中指出的，“A”只被打印一次，即使从它派生出两个类。这是因为 Python 的新样式类中固有的方法解析顺序(MRO)。你可以通过在类 D 的 foo 函数中的超级调用下面添加 **print D.__mro__** 来签出 MRO(注意:这只在基类是从**对象**派生的情况下才有效)。这个[栈溢出条目](http://stackoverflow.com/a/1848647/393194)对此进行了更详细的解释。如果你想好好读一读关于 MRO 的主题，我会推荐 Python.org 上的以下摘要:[http://www.python.org/download/releases/2.3/mro/](http://www.python.org/download/releases/2.3/mro/)。

现在你可能认为这个例子不仅抽象，而且没什么用，你可能是对的。这就是为什么值得在网上做一些额外的挖掘来寻找其他的例子。幸运的是，我记得几年前在 super 上看过 Raymond Hettinger 的一篇文章。他是 Python 的核心开发人员，经常在 PyCon 上发言。总之，他的文章给出了几个使用 super 向子类添加新特性的真实例子。我强烈推荐看看他的文章，因为它对以一种适用的方式解释**超级**大有帮助。甚至 super()的 Python 文档也链接到了他的文章！

这里有一个关于这个话题的很好的列表:

*   关于 super 的官方 Python [文档](http://docs.python.org/2/library/functions.html#super)
*   雷蒙德·赫廷格[对超级](http://rhettinger.wordpress.com/2011/05/26/super-considered-super/)的沉思
*   [Python 的 Super 很俏皮，但是不能用](http://docs.python.org/2/library/functions.html)
*   StackOverflow: [了解 Python 超级](http://stackoverflow.com/questions/576169/understanding-python-super)
*   StackOverflow: [如何在 Python 中使用 super](http://stackoverflow.com/questions/222877/how-to-use-super-in-python)
*   [了解 Python 的超级](http://blog.timvalenta.com/2009/02/understanding-pythons-super/)
*   [如何有效使用 super()](http://code.activestate.com/recipes/577721-how-to-use-super-effectively-python-27-version/)-Python 2.7 版本(Python 菜谱)
*   Zed Shaw 在他的在线书籍中对 super()的评论
*   方法解析顺序(MRO) [摘要](http://www.python.org/download/releases/2.3/mro/)