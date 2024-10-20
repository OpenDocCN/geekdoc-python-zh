# Python 检查模块

> 原文：<https://www.askpython.com/python-modules/python-inspect-module>

Python 的`**inspect module**`提供了活体对象的自省和相同的源代码。

它还提供了整个程序中使用的[类](https://www.askpython.com/python/oops/python-classes-objects)和[函数](https://www.askpython.com/python/python-functions)的自省。

inspect 模块为用户提供了利用其中的函数/方法来获取相同的源代码、提取和解析所需的库文档的功能。

这个模块用于获取关于特定用户程序中的函数、类和模块的信息。这是通过对模块的属性、模块的类/方法等进行检查来实现的。

* * *

## Python 检查模块功能

*   `**Introspection of a module**`
*   `**Introspection of classes in a module**`
*   `**Introspection of methods/functions in a class**`
*   `**Introspection of objects of a class**`
*   `**Retrieval of source of a class**`
*   `**Retrieval of source of a method/function**`
*   `**Fetching the method signature**`
*   `**Documentation of Strings for a class**`
*   `**Introspecting Class Hierarchies**`
*   `**Introspection of Frames and Stacks in the run-time environment**`

* * *

### 1.模块的自省

最初，我们需要创建一个样本模块以及用于自省的类/函数，即 ***test.py***

**test.py**

```py
def funct(arg1, arg2 = 'default', *args):
    #module-level function
    x = arg1 * 1
    return x

class P(object):

    def __init__(self, city):
        self.city = city

    def get_city(self):

        return self.city

obj_p = P('sample_instance')

class Q(P):

    def country(self):
       return ("Hello")
    def get_country(self):

        return 'Q(' + self.city + ')'

```

然后我们将使用 inspect.getmembers()函数对创建的模块进行自省。

**注意:**包含样本模块的 python 文件和包含自检代码的文件要放在同一个目录下。

**模块自检代码:**

```py
import inspect
import test

for city, data in inspect.getmembers(test):
    if city.startswith('__'):
        continue
    print('{} : {!r}'.format(city, data))

```

**输出:**

```py
P : <class 'test.P'>
Q : <class 'test.Q'>
funct : <function funct at 0x000001ED62A2EAF0>
obj_p : <test.P object at 0x000001ED62A6A640>
```

* * *

### 2.模块中类的自省

`**getmembers()**`函数和`**isclass**`属性标识符用于检查模块中的类。

```py
import inspect
import test

for key, data in inspect.getmembers(test, inspect.isclass):
    print('{} : {!r}'.format(key, data))

```

**输出:**

```py
P : <class 'test.P'>
Q : <class 'test.Q'>
```

* * *

### 3.类中方法/函数的自省

`**getmembers()**`函数和`**isfunction**`属性标识符用于检查模块中的类。

```py
import inspect
from pprint import pprint
import test

pprint(inspect.getmembers(test.P, inspect.isfunction))

```

**输出:**

```py
[('__init__', <function P.__init__ at 0x000001D519CA7CA0>),
 ('get_city', <function P.get_city at 0x000001D519CA7D30>)]
```

* * *

### 4.类对象的内省

```py
import inspect
from pprint import pprint
import test

result = test.P(city='inspect_getmembers')
pprint(inspect.getmembers(result, inspect.ismethod))

```

**输出:**

```py
[('__init__',
  <bound method P.__init__ of <test.P object at 0x00000175A62C5250>>),
 ('get_city',
  <bound method P.get_city of <test.P object at 0x00000175A62C5250>>)]
```

* * *

### 5.检索一个类的源代码

`**getsource()**`函数返回特定模块/类的源代码。

```py
import inspect
import test

print(inspect.getsource(test.Q))

```

**输出:**

```py
class Q(P):

    def country(self):
       return ("Hello")
    def get_country(self):

        return 'Q(' + self.city + ')'
```

* * *

### 6.检索方法/函数的源代码

```py
import inspect
import test

print(inspect.getsource(test.Q.get_city))

```

**输出:**

```py
def get_city(self):

        return self.city
```

* * *

### 7.获取方法签名

`**inspect.signature()**` 方法返回该方法的签名，从而使用户容易理解传递给该方法的参数类型。

```py
import inspect
import test

print(inspect.signature(test.funct))

```

**输出:**

```py
(arg1, arg2='default', *args) 
```

* * *

### 8.类的字符串文档

inspect 模块的`**getdoc()**`函数提取一个特定的类及其函数来表示给最终用户。

```py
import inspect
import test

print('P.__doc__:')
print(test.P.__doc__)
print()
print('Documentation String(P):')
print(inspect.getdoc(test.P))

```

**输出:**

```py
P.__doc__:
Implementation of class P

Documentation String(P):
Implementation of class P
```

* * *

### 9.反思类层次结构

`**getclasstree()**`方法返回类的层次结构及其依赖关系。它使用来自给定类的元组和列表创建一个树形结构。

```py
import inspect
import test

class S(test.Q):
    pass

class T(S, test.P):
    pass

def print_class_tree(tree, indent=-1):
    if isinstance(tree, list):
        for node in tree:
            print_class_tree(node, indent+1)
    else:
        print( '  ' * indent, tree[0].__name__ )

if __name__ == '__main__':
    print( 'P, Q, S, T:')
    print_class_tree(inspect.getclasstree([test.P, test.Q, S, T]))

```

**输出:**

```py
P, Q, S, T:
 object
   P
     T
     Q
       S
         T
```

* * *

### 10.**运行时环境中帧和堆栈的自检**

inspect 模块还在函数执行期间检查程序的动态环境。这些函数主要处理调用堆栈和调用框架。

`**currentframe()**`描述当前正在执行的函数的堆栈顶部的帧。`**getargvalues()**`结果是一个元组，其中包含参数的名称，以及来自帧的局部值的字典。

```py
import inspect

def recurse(threshold):
    x = '.' * threshold
    print (threshold, inspect.getargvalues(inspect.currentframe()))
    if threshold <= 0:
        return
    recurse(threshold - 1)
    return

if __name__ == '__main__':
    recurse(4)

```

**输出:**

```py
4 ArgInfo(args=['threshold'], varargs=None, keywords=None, locals={'threshold': 4, 'x': '....'})
3 ArgInfo(args=['threshold'], varargs=None, keywords=None, locals={'threshold': 3, 'x': '...'})
2 ArgInfo(args=['threshold'], varargs=None, keywords=None, locals={'threshold': 2, 'x': '..'})
1 ArgInfo(args=['threshold'], varargs=None, keywords=None, locals={'threshold': 1, 'x': '.'})
0 ArgInfo(args=['threshold'], varargs=None, keywords=None, locals={'threshold': 0, 'x': ''})
```

* * *

## 结论

因此，在本文中，我们已经理解了 Python 的 inspect 模块所提供的功能。

* * *

## 参考

*   Python 检查模块
*   [检查模块文件](https://docs.python.org/3/library/inspect.html)