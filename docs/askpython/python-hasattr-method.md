# 如何使用 Python hasattr()方法？

> 原文：<https://www.askpython.com/python/built-in-methods/python-hasattr-method>

嘿，读者们！在本文中，我们将详细介绍 Python hasattr()方法的行为。

* * *

## 需要 Python 的 hasattr()方法

在[面向对象编程](https://www.askpython.com/python/oops/object-oriented-programming-python)的世界中，我们处理现实生活场景到类和对象的表示或映射。对象可以被认为是一个描述属性及其行为的类的蓝图。

有时，我们可能会遇到这样的情况，我们需要检查某个类是否存在某个属性。同样可以通过 Python hasattr()方法实现。它有助于检查类中属性的存在。

既然我们已经理解了 Python hasattr()方法的必要性和起源，那么让我们来理解它的工作原理。

* * *

## Python hasattr()方法的工作原理

Python 类通过对象表示属性及其行为。

`hasattr() method`用于检查一个类中属性的存在。

```py
hasattr(Class, attribute)

```

hasattr()方法**返回一个布尔值**，即根据类中属性的存在，要么是**真**要么是**假**。

**例 1:**

```py
class Info:
  name = "JournalDev"
  lang = "Python"
  site = "Google"
print(hasattr(Info, 'lang'))

```

在上面的例子中，属性“lang”包含在类“Info”中。因此，hasattr()函数返回 **True** 。

**输出:**

```py
True

```

**例 2:**

```py
class Info:
  name = "JournalDev"
  lang = "Python"
  site = "Google"
print(hasattr(Info, 'date'))

```

如上例所示，hasattr()函数返回 False，因为属性“date”未在该类中定义。

**输出:**

```py
False

```

* * *

## Python 2 的 hasattr()方法 v/s Python 3 的 hasattr()方法

在 **Python 2** 中，hasattr()压倒了所有的异常，并为某个条件返回 False。

例如，如果一个给定的属性“A”包含在一个类中，但是被一些异常占据。此时，hasattr()将忽略所有的异常，并返回 False，即使属性“A”恰好存在于类中。

另一方面，在 **Python 3** 中，如果属性涉及一些异常标准，hasattr()会引发异常。

**例子:** Python 2 带有 hasattr()函数

```py
class Info(object):
     @property
     def hey(self):
         raise SyntaxError
     def say(self):
         raise SyntaxError
obj = Info()

print(hasattr(obj,'hey'))
print(hasattr(obj,'say'))

```

在上面的代码中，尽管由于 decorator 导致了语法错误，hasattr()方法没有引发任何错误，忽略了异常并返回 False，即使该类恰好包含该特定属性。

**输出:**

```py
False
True

```

**例子:** Python 3 带有 hasattr()函数

在下面的代码中，hasattr()函数使用属性“hey”引发语法错误原因的异常错误。

```py
class Info(object):
     @property
     def hey(self):
         raise SyntaxError
     def say(self):
         raise SyntaxError
obj = Info()

print(hasattr(obj,'hey'))
print(hasattr(obj,'say'))

```

**输出:**

```py
Traceback (most recent call last):

  File "c:\users\hp\appdata\local\programs\python\python36\lib\site-packages\IPython\core\interactiveshell.py", line 3319, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)

  File "<ipython-input-20-e14f6e57b66e>", line 9, in <module>
    print(hasattr(obj,'hey'))

  File "<ipython-input-20-e14f6e57b66e>", line 4, in hey
    raise SyntaxError

  File "<string>", line unknown
SyntaxError

```

* * *

## 结论

因此，在本文中，我们已经了解了 Python hasattr()在 Python 版本 2 和 3 中的工作方式。

* * *

## 参考

*   Python 的 hasattr()方法— JournalDev