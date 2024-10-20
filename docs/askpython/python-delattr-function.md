# 如何使用 Python delattr()函数？

> 原文：<https://www.askpython.com/python/built-in-methods/python-delattr-function>

嘿，读者们！在本文中，我们将重点介绍 ***Python delattr()函数*** 。

* * *

## Python delattr()方法入门

在**面向对象编程**中，`Class` 是一个实体，它包装了**属性**和**行为**，并用`Object`来表示它们。

一个[类](https://www.askpython.com/python/oops/python-classes-objects)的特性确实存在于 Python 中，因为它是一种面向对象的语言。在创建属性和定义行为时，我们可能会遇到想要删除 Python 类的某些属性的情况。这就是 Python delattr()出现的原因。

`The delattr() function`用于**删除与特定类相关的属性**。

**语法:**

```py
delattr(object, attribute)

```

**举例:**

```py
class Info:
  name = "AskPython"
  lang = "Python"
  site = "Google"
obj = Info()
print(obj.name)
delattr(Info, 'lang')

```

**输出:**

```py
AskPython

```

在上面的例子中，我们创建了一个具有如下属性的类信息:

*   name = AskPython
*   lang = Python
*   网站=谷歌

此外，我们使用下面的命令创建了一个类别为'**信息**的**对象**:

```py
obj = <classname>()

```

创建对象后，我们使用 delattr()函数删除了属性–'**lang**'。

* * *

## delattr()的错误和异常

删除一个属性后，如果我们试图访问那个特定的对象，编译器将抛出`AttributeError`。

**举例:**

```py
class Info:
  name = "AskPython"
  lang = "Python"
  site = "Google"
obj = Info()
delattr(Info, 'lang')
print(obj.lang)

```

**输出:**

```py
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-5-699f50a1e7c7> in <module>
      5 obj = Info()
      6 delattr(Info, 'lang')
----> 7 print(obj.lang)

AttributeError: 'Info' object has no attribute 'lang'

```

* * *

## 使用 Python del 运算符删除属性

`Python del operator`也可以用来直接删除类的属性，不需要通过对象访问。

**语法:**

```py
del Class-name.attribute-name

```

**举例:**

```py
class Info:
  name = "AskPython"
  lang = "Python"
  site = "Google"
print("Attribute before deletion:\n",Info.lang)
del Info.lang
print("Attribute after deletion:\n",Info.lang)

```

**输出:**

```py
Attribute before deletion:
 Python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-16-9d8ed21690ff> in <module>
      5 print("Attribute before deletion:\n",Info.lang)
      6 del Info.lang
----> 7 print("Attribute after deletion:\n",Info.lang)

AttributeError: type object 'Info' has no attribute 'lang'

```

* * *

## Python delattr()方法 v/s Python del 运算符

在动态删除属性方面，Python delattr()方法比 Python del 运算符更有效。

另一方面，与 Python delattr()方法相比，Python del 运算符执行操作的速度更快。

* * *

## 结论

因此，在本文中，我们已经通过 Python delattr()方法和 del 运算符了解了属性的删除。

* * *

## 参考

*   [Python delattr()方法—文档](https://docs.python.org/3/library/functions.html#delattr)
*   Python deattr() v/s Python del 运算符— JournalDev