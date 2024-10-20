# Python 201:修饰主函数

> 原文：<https://www.blog.pythonlibrary.org/2012/05/31/python-201-decorating-the-main-function/>

上周，我读了 Brett Cannon 的博客,他谈到了函数签名和修饰主函数。我没有完全听懂他讲的内容，但我认为这个概念非常有趣。下面的代码是基于 Cannon 先生提到的[食谱](http://code.activestate.com/recipes/577791/)的一个例子。我认为这说明了他在说什么，但基本上提供了一种消除标准的方法

```py

if __name__ == "__main__":
   doSomething()

```

总之，这是代码。

```py

#----------------------------------------------------------------------
def doSomething(name="Mike"):
    """"""
    print "Welcome to the program, " + name

#----------------------------------------------------------------------
def main(f):
    """"""
    if f.__module__ == '__main__':
        f()
    return f

main(doSomething)

```

这一点的好处是 main 函数可以在代码中的任何地方。如果你喜欢使用 decorators，那么你可以重写如下:

```py

#----------------------------------------------------------------------
def main(f):
    """"""
    if f.__module__ == '__main__':
        f()
    return f

#----------------------------------------------------------------------
@main
def doSomething(name="Mike"):
    """"""
    print "Welcome to the program, " + name

```

请注意，在您可以修饰 **doSomething** 函数之前，必须先有 **main** 函数。我不确定我会如何或者是否会使用这种技巧，但是我想在某个时候尝试一下可能会很有趣。