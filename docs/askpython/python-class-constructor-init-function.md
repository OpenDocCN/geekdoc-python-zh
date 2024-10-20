# Python 类构造函数–Python _ _ init _ _()函数

> 原文：<https://www.askpython.com/python/oops/python-class-constructor-init-function>

Python 类构造函数的作用是初始化类的实例。Python __init__()是 Python 中[类的构造函数。](https://www.askpython.com/python/oops/python-classes-objects)

* * *

## Python __init__()函数语法

__init__()函数的语法是:

```py
def __init__(self, [arguments])

```

*   def 关键字用于定义它，因为它是一个[函数](https://www.askpython.com/python/python-functions)。
*   第一个参数引用当前对象。它将实例绑定到 init()方法。通常按照命名约定命名为“self”。你可以在 [Python 自变量](https://www.askpython.com/python/python-self-variable)了解更多。
*   init()方法参数是可选的。我们可以用任意数量的参数定义一个构造函数。

* * *

## Python 类构造函数示例

让我们看一些不同场景下的构造函数的例子。

### 1.没有构造函数的类

我们可以创建一个没有任何构造函数定义的类。在这种情况下，调用超类构造函数来初始化该类的实例。`object`类是 Python 中所有类的基础。

```py
class Data:
    pass

d = Data()
print(type(d))  # <class '__main__.Data'>

```

下面是另一个例子，确认超类构造函数被调用来初始化子类的实例。

```py
class BaseData:

    def __init__(self, i):
        print(f'BaseData Constructor with argument {i}')
        self.id = i

class Data(BaseData):
    pass

d = Data(10)
print(type(d))

```

输出:

```py
BaseData Constructor with argument 10
<class '__main__.Data'>

```

* * *

### 2.不带参数的简单构造函数

我们可以创建一个不带任何参数的构造函数。这对于记录日志很有用，比如记录类的实例数。

```py
class Data1:
    count = 0

    def __init__(self):
        print('Data1 Constructor')
        Data1.count += 1

d1 = Data1()
d2 = Data1()
print("Data1 Object Count =", Data1.count)

```

输出:

```py
Data1 Constructor
Data1 Constructor
Data1 Object Count = 2

```

* * *

### 3.带参数的类构造函数

大多数时候，你会发现带有一些参数的构造函数。这些参数通常用于初始化实例变量。

```py
class Data2:

    def __init__(self, i, n):
        print('Data2 Constructor')
        self.id = i
        self.name = n

d2 = Data2(10, 'Secret')
print(f'Data ID is {d2.id} and Name is {d2.name}')

```

输出:

```py
Data2 Constructor
Data ID is 10 and Name is Secret

```

* * *

### 4.具有继承的类构造函数

```py
class Person:

    def __init__(self, n):
        print('Person Constructor')
        self.name = n

class Employee(Person):

    def __init__(self, i, n):
        print('Employee Constructor')
        super().__init__(n)  # same as Person.__init__(self, n)
        self.id = i

emp = Employee(99, 'Pankaj')
print(f'Employee ID is {emp.id} and Name is {emp.name}')

```

输出:

```py
Employee Constructor
Person Constructor
Employee ID is 99 and Name is Pankaj

```

*   调用超类构造函数是我们的责任。
*   我们可以使用 super()函数来调用超类构造函数。
*   我们也可以使用超类名来调用它的 init()方法。

* * *

### 5.具有多级继承的构造函数链接

```py
class A:

    def __init__(self, a):
        print('A Constructor')
        self.var_a = a

class B(A):

    def __init__(self, a, b):
        super().__init__(a)
        print('B Constructor')
        self.var_b = b

class C(B):

    def __init__(self, a, b, c):
        super().__init__(a, b)
        print('C Constructor')
        self.var_c = c

c_obj = C(1, 2, 3)
print(f'c_obj var_a={c_obj.var_a}, var_b={c_obj.var_b}, var_c={c_obj.var_c}')

```

输出:

```py
A Constructor
B Constructor
C Constructor
c_obj var_a=1, var_b=2, var_c=3

```

* * *

### 6.多重继承构造函数

在多重继承的情况下，我们不能使用 super()来访问所有的超类。更好的方法是使用超类的类名调用它们的构造函数。

```py
class A1:
    def __init__(self, a1):
        print('A1 Constructor')
        self.var_a1 = a1

class B1:
    def __init__(self, b1):
        print('B1 Constructor')
        self.var_b1 = b1

class C1(A1, B1):
    def __init__(self, a1, b1, c1):
        print('C1 Constructor')
        A1.__init__(self, a1)
        B1.__init__(self, b1)
        self.var_c1 = c1

c_obj = C1(1, 2, 3)
print(f'c_obj var_a={c_obj.var_a1}, var_b={c_obj.var_b1}, var_c={c_obj.var_c1}')

```

输出:

```py
C1 Constructor
A1 Constructor
B1 Constructor
c_obj var_a=1, var_b=2, var_c=3

```

* * *

## Python 不支持多个构造函数

Python 不支持多个构造函数，不像 Java 等其他流行的面向对象编程语言。

我们可以定义多个 __init__()方法，但最后一个将覆盖前面的定义。

```py
class D:

    def __init__(self, x):
        print(f'Constructor 1 with argument {x}')

    # this will overwrite the above constructor definition
    def __init__(self, x, y):
        print(f'Constructor 1 with arguments {x}, {y}')

d1 = D(10, 20) # Constructor 1 with arguments 10, 20

```

* * *

## Python __init__()函数能返回什么吗？

如果我们试图从 __init__()函数返回一个非 None 值，它将引发 TypeError。

```py
class Data:

    def __init__(self, i):
        self.id = i
        return True

d = Data(10)

```

输出:

```py
TypeError: __init__() should return None, not 'bool'

```

如果我们将 return 语句改为 **`return None`** ，那么代码将无一例外地工作。

* * *

## 参考资料:

*   [object __init__()函数文档](https://docs.python.org/3/reference/datamodel.html#object.__init__)