# Python 2 和 3 中元类在技术上是如何工作的

> 原文：<https://www.pythoncentral.io/how-metaclasses-work-technically-in-python-2-and-3/>

`metaclass`是定义其他类的类型/类的类/对象。在 Python 中，`metaclass`可以是`class`、函数或任何支持调用接口的对象。这是因为要创建一个`class`对象；它的`metaclass`是用`class`名称、基类和属性(方法)调用的。当没有定义`metaclass`时(通常是这样)，使用默认的`metaclass` `type`。

例如:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
# Here __metaclass__ points to the metaclass object.
class ExampleClass(metaclass=type):
pass
[/python]

*   [Python 2.x](#)

[python]
# Here __metaclass__ points to the metaclass object.
class ExampleClass(object):
__metaclass__ = type
pass
[/python]

创建`class`时，解释器:

1.  获取`class`的名称。
2.  获取`class`的基类。
3.  获取`class`的`metaclass`。如果它被定义，它将首先使用这个。否则，它将检查`metaclass`的基类。如果在`base class`中找不到`metaclass`，则使用`type` `object`代替。
4.  获取`class`中的变量/属性，并将它们存储为一个字典。
5.  将该信息作为`metaclass(name_of_class, base_classes, attributes_dictionary)`传递给`metaclass`，并返回一个`class`对象。

例如:

```py

# type(name, base, attrs)

# name is the name of the class

# base is a tuple of base classes (all methods/attributes are inherited

# from these) attrs is a dictionary filled with the class attributes

classObject = type('ExampleClass', (object,) ,{})

```

当 type 被调用时，它的`__call__`方法被调用。这个方法依次调用`__new__`和`__init__`方法。`__new__`方法创建一个新对象，而`__init__`方法初始化它。我们可以轻松地玩方法。这是一个工作示例:

*   [Python 3.x](#custom-tab-1-python-3-x)
*   [Python 2.x](#custom-tab-1-python-2-x)

*   [Python 3.x](#)

[python]
class a:
def __init__(self, data):
self.data = data

def getd3(self):
返回 self.data * 3

class my meta(type):
def _ _ New _ _(meta name，classname，baseclasses，attrs):
print(' New called with ')
print(' meta name '，metaname)
print('classname '，class name)
print(' base classes '，baseclasses)
print('attrs '，attrs)
attrs[' get data ']= a . _ _ dict _ _[' get D3 ']
# attrs[' get data ']= a . get D3__new__(元名，类名，基类，属性)

def __init__(classobject，classname，baseclasses，attrs):
print(' init called with ')
print(' class object '，class object)
print(' class name '，class name)
print(' base classes '，baseclasses)
print('attrs '，attrs)

Kls 类(元类=MyMeta):
def __init__(self，data):
self.data = data

def printd(self):
打印(self.data)

我= kls(‘arun’)【t0 I . printtd()【print(我. getdata())

运行代码时，我们得到:

```py

New called with

metaname <class '__main__.MyMeta'>

classname Kls

baseclasses ()

attrs {'__module__': '__main__', 'printd': <function printd at 0x7f3ebca86958>, '__init__': <function __init__ at 0x7f3ebca868d0>}

init called with

classobject <class '__main__.Kls'>

classname Kls

baseclasses ()

attrs {'__module__': '__main__', 'getdata': <function getd3 at 0x7f3ebca86408>, 'printd': <function printd at 0x7f3ebca86958>, '__init__': <function __init__ at 0x7f3ebca868d0>}

arun

arunarunarun

[/shell]

```

*   [Python 2.x](#)

[python]
class a(object):
def __init__(self, data):
self.data = data

def getd3(self):
返回 self.data * 3

class my meta(type):
def _ _ New _ _(meta name，classname，baseclasses，attrs):
print ' New called with '
print ' meta name '，metaname
print 'classname '，classname
print 'baseclasses '，baseclasses
print 'attrs '，attrs
attrs[' get data ']= a . _ _ dict _ _[' get D3 ']
# attrs[' get data ']= a . get D3
返回类型。__new__(元名，类名，基类，属性)

def __init__(classobject，classname，baseclasses，attrs):
print ' init called with '
print ' class object '，classobject
print 'classname '，classname
print 'baseclasses '，baseclasses
print 'attrs '，attrs

Kls 类(object):
_ _ 元类 __ = MyMeta

def __init__(self，data):
self.data = data

def printd(self):
打印自我数据

我= kls(‘arun’)【t0 我. printtd()
我. getdata()

运行代码时，我们得到:

```py

New called with

metaname <class '__main__.MyMeta'>

classname Kls

baseclasses (<type 'object'>,)

attrs {'__module__': '__main__', '__metaclass__': <class '__main__.MyMeta'>, 'printd': <function printd at 0x7fbdab0176e0>, '__init__': <function __init__ at 0x7fbdab017668>}

init called with

classobject <class '__main__.Kls'>

classname Kls

baseclasses (<type 'object'>,)

attrs {'__module__': '__main__', 'getdata': <function getd3 at 0x7fbdab017500>, '__metaclass__': <class '__main__.MyMeta'>, 'printd': <function printd at 0x7fbdab0176e0>, '__init__': <function __init__ at 0x7fbdab017668>}

arun

arunarunarun

[/shell]

```

通常我们只需要覆盖一个方法`__new__`或`__init__`。我们也可以用`function`来代替`class`。这里有一个例子:

*   [Python 3.x](#custom-tab-2-python-3-x)
*   [Python 2.x](#custom-tab-2-python-2-x)

*   [Python 3.x](#)

[python]
def meta_func(name, bases, attrs):
print('meta function called with', name, bases, attrs)
nattrs = {'mod' + key:attrs[key] for key in attrs}
return type(name, bases, nattrs)

MyMeta = meta_func

Kls 类(元类=MyMeta):
def setd(self，data):
self.data = data

def getd(self):
返回 self.data

k = Kls()
k . modsetd(' arun ')
print(k . modgetd())

为我们提供了以下输出:

```py

meta function called with Kls () {'setd': <function setd at 0x7f3bafe7cd10>, '__module__': '__main__', 'getd': <function getd at 0x7f3bafe7cd98>}

arun

[/shell]

```

*   [Python 2.x](#)

[python]
def meta_func(name, bases, attrs):
print 'meta function called with', name, bases, attrs
nattrs = {'mod' + key:attrs[key] for key in attrs}
return type(name, bases, nattrs)

MyMeta = meta_func

Kls 类(object):
_ _ 元类 __ = MyMeta

def setd(self，data):
self.data = data

def getd(self):
返回 self.data

k = Kls()
k . modsetd(' arun ')
打印 k.modgetd()

为我们提供了以下输出:

```py

meta function called with Kls (<type 'object'>,) {'setd': <function setd at 0x88b21ec>, 'getd': <function getd at 0x88b22cc>, '__module__': '__main__', '__metaclass__': <function meta_func at 0xb72341b4>}

arun

[/shell]

```

除了修改基类和要创建的类的方法，元类还可以修改实例创建过程。这是因为当我们创建一个实例(`ik = Kls()`)时，这就像调用`class Kls`。需要注意的一点是，无论何时我们调用一个对象，它的类型的`__call__`方法都会被调用。所以在这种情况下，`class`类型是`metaclass`，因此它的`__call__`方法将被调用。我们可以这样检查:

*   [Python 3.x](#custom-tab-3-python-3-x)
*   [Python 2.x](#custom-tab-3-python-2-x)

*   [Python 3.x](#)

[python]
class MyMeta(type):
def __call__(clsname, *args):
print('MyMeta called with')
print('clsname:', clsname)
print('args:', args)
instance = object.__new__(clsname)
instance.__init__(*args)
return instance

Kls 类(元类=MyMeta):
def __init__(self，data):
self.data = data

def printd(self):
打印(self.data)

ik = Kls(' arun ')
ik . printd()
[/python]

*   [Python 2.x](#)

[python]
class MyMeta(type):
def __call__(clsname, *args):
print 'MyMeta called with'
print 'clsname:', clsname
print 'args:' ,args
instance = object.__new__(clsname)
instance.__init__(*args)
return instance

Kls 类(object):
_ _ 元类 __ = MyMeta

def __init__(self，data):
self.data = data

def printd(self):
打印自我数据

ik = Kls(' arun ')
ik . printd()
[/python]

输出如下所示:

```py

MyMeta called with

clsname: <class '__main__.Kls'>

args: ('arun',)

arun

```

有了这些信息，如果我们开始讨论`class`的创建过程，它以调用`metaclass`对象结束，该对象提供了一个`class`对象。事情是这样的:

```py

Kls = MetaClass(name, bases, attrs)

```

因此这个调用应该调用`metaclass`的类型。`metaclass`型是`metaclass`的`metaclass`！我们可以这样检查:

*   [Python 3.x](#custom-tab-4-python-3-x)
*   [Python 2.x](#custom-tab-4-python-2-x)

*   [Python 3.x](#)

[python]
class SuperMeta(type):
def __call__(metaname, clsname, baseclasses, attrs):
print('SuperMeta Called')
clsob = type.__new__(metaname, clsname, baseclasses, attrs)
type.__init__(clsob, clsname, baseclasses, attrs)
return clsob

class MyMeta(type，metaclass = super meta):
def _ _ call _ _(cls，*args，* * kwargs):
print(' MyMeta called '，cls，args，kwargs)
ob = object。__new__(cls，*args)
ob。__init__(*args)
返回 ob

打印(“创建类”)

Kls 类(元类=MyMeta):
def __init__(self，data):
self.data = data

def printd(self):
打印(self.data)

print(' class created ')
ik = Kls(' arun ')
ik . printd()
ik2 = Kls(' avni ')
ik2 . printd()
[/python]

*   [Python 2.x](#)

[python]
class SuperMeta(type):
def __call__(metaname, clsname, baseclasses, attrs):
print 'SuperMeta Called'
clsob = type.__new__(metaname, clsname, baseclasses, attrs)
type.__init__(clsob, clsname, baseclasses, attrs)
return clsob

class MyMeta(type):
_ _ metaclass _ _ = super meta
def _ _ call _ _(cls，*args，* * kwargs):
print ' MyMeta called '，cls，args，kwargs
ob = object。__new__(cls，*args)
ob。__init__(*args)
返回 ob

打印“创建类”

Kls 类(object):
_ _ 元类 __ = MyMeta

def __init__(self，data):
self.data = data

def printd(self):
打印自我数据

打印“类别已创建”

我= kls(‘arun’)【t0 I . printd()
ik2 = kls(名称)
ik2 . print TD()
[/python]

为我们提供了以下输出:

```py

create class

SuperMeta Called

class created

MyMeta called class '__main__.Kls' ('arun',) {}

arun

MyMeta called &lt;class '__main__.Kls' ('avni',) {}

avni

```