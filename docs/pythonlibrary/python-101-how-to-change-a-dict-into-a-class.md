# Python 101:如何将字典变成类

> 原文：<https://www.blog.pythonlibrary.org/2014/02/14/python-101-how-to-change-a-dict-into-a-class/>

我在工作中接触了很多字典。有时字典会变得非常复杂，因为其中嵌入了许多嵌套的数据结构。最近，我有点厌倦了试图记住我的字典中的所有键，所以我决定将我的一个字典改为一个类，这样我就可以将键作为实例变量/属性来访问。如果你曾经生病

这里有一个简单的方法:

```py

########################################################################
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

#----------------------------------------------------------------------
if __name__ == "__main__":
    ball_dict = {"color":"blue",
                 "size":"8 inches",
                 "material":"rubber"}
    ball = Dict2Obj(ball_dict)

```

这段代码使用 **setattr** 将每个键作为属性添加到类中。下面显示了它如何工作的一些例子:

```py

>>> ball.color
'blue'
>>> ball.size
'8 inches'
>>> print ball
<__main__.dict2obj object="" at="">

```

当我们打印球对象时，我们从类中得到一个无用的字符串。让我们覆盖我们类的 **__repr__** 方法，让它打印出更有用的东西:

```py

########################################################################
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        return "" % self.__dict__

#----------------------------------------------------------------------
if __name__ == "__main__":
    ball_dict = {"color":"blue",
                 "size":"8 inches",
                 "material":"rubber"}
    ball = Dict2Obj(ball_dict) 
```

现在，如果我们打印出球对象，我们会得到以下结果:

```py

>>> print ball

```

这有点不直观，因为它使用类的内部 **__dict__** 打印出一个字典，而不仅仅是属性名。这与其说是一个问题，不如说是一个品味问题，但是让我们试着只得到方法名:

```py

########################################################################
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        attrs = str([x for x in self.__dict__])
        return "" % attrs

#----------------------------------------------------------------------
if __name__ == "__main__":
    ball_dict = {"color":"blue",
                 "size":"8 inches",
                 "material":"rubber"}
    ball = Dict2Obj(ball_dict) 
```

在这里，我们只是循环遍历 **__dict__** 的内容，并返回一个只包含与属性名称匹配的键列表的字符串。你也可以这样做:

```py

attrs = str([x for x in dir(self) if "__" not in x])

```

我相信还有很多其他的方法来完成这类事情。不管怎样，我发现这段代码对我的一些工作很有帮助。希望你也会发现它很有用。