# Python 类型()函数的变体

> 原文：<https://www.askpython.com/python/built-in-methods/python-type-function>

嘿，伙计们！在本文中，我们将从*调试* — **Python type()函数**的角度来看一下 Python 的一个重要内置函数。

* * *

## Python type()函数入门

Python type()函数的作用是调试整个程序。type()函数可用于调试代码中各种类和数据变量的数据类型。

type()函数可以用两种变体来表示

*   **带一个参数的 type()函数**
*   **带三个参数的 type()函数**

在下一节中，我们将了解 Python 中 type()函数的两种变体的功能。

* * *

## 1.带有一个参数的 Python 类型()

当将单个参数传递给 type()函数时，它将分别返回给定类/对象的数据类型。

**语法:**

```py
type(object)

```

*   它只接受一个**单参数**。
*   带有单参数的 type()函数返回传递给它的对象的**类类型。**

**举例:**

```py
dict_map = {"Python":'A',"Java":'B',"Kotlin":'C',"Ruby":'D'}
print("The variable dict_map is of type:",type(dict_map))

list_inp = [10,20,30,40]
print("The variable list_inp is of type:",type(list_inp))

str_inp = "Python with JournalDev"
print("The variable str_inp is of type:",type(str_inp))

tup_inp = ('Bajaj', 'Tata','Royce')
print("The variable tup_inp is of type:",type(tup_inp))

```

在上面的例子中，我们已经创建了不同数据结构的数据对象，如 dict、list 等。此外，我们将它传递给 type()函数来调试对象的类型。

**输出:**

```py
The variable dict_map is of type: <class 'dict'>
The variable list_inp is of type: <class 'list'>
The variable str_inp is of type: <class 'str'>
The variable tup_inp is of type: <class 'tuple'>

```

* * *

## 2.具有三个参数的 Python 类型()

当三个参数被传递给`type() function`时，它创建并返回一个新类型的对象作为函数的输出。

**语法:**

```py
type(name, bases, dict)

```

这三个参数如下

*   `name`:是一个**字符串**，基本上代表了这个类的**名称。**
*   `bases`:指定主类的**基类的**元组**。**
*   `dict`:是一个“**字典**，用于**创建指定的**类的主体。

因此，带有上述三个参数的 type()函数用于在运行时动态创建类。

**举例:**

```py
var1 = type('ob', (object,), dict(A='Python', B='Cpp'))

print(type(var1))
print(vars(var1))

class apply:
  A = 'Python'
  B = 'Cpp'

var2 = type('oc', (apply,), dict(A = 'Python', B = 'Kotlin'))
print(type(var2))
print(vars(var2))

```

在上面的例子中，我们已经在动态运行时创建了类，一个有一个对象类，另一个有“应用”基类。 `vars() function`表示一个类/模块的 __dict__ 参数。

**输出:**

```py
<class 'type'>
{'A': 'Python', 'B': 'Cpp', '__module__': '__main__', '__dict__': <attribute '__dict__' of 'ob' objects>, '__weakref__': <attribute '__weakref__' of 'ob' objects>, '__doc__': None}
<class 'type'>
{'A': 'Python', 'B': 'Kotlin', '__module__': '__main__', '__doc__': None}

```

* * *

## 摘要

*   带有单个参数的 type()函数返回参数的类类型，广泛用于代码调试。
*   type()函数和三个参数用于动态创建类，即在运行时创建。

* * *

## 结论

因此，在本文中，我们已经分别理解了 Python type()在不同参数下的工作方式。

* * *

## 参考

*   Python type()函数— JournalDev