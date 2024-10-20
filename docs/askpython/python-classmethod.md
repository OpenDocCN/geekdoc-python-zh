# Python 类方法()

> 原文：<https://www.askpython.com/python/oops/python-classmethod>

Python classmethod()是 Python 标准库的内置函数！

Python 中有三种类型的方法:

*   实例方法
*   **类方法**
*   [静态方法](https://www.askpython.com/python/staticmethod-in-python)

在本文中，我们将重点讨论 Python 的 class 方法。所以让我们开始吧。

* * *

## 什么是 Python classmethod()？

一个 **Python classmethod()** 可以被一个类调用，也可以被一个对象调用。为给定函数返回类方法的方法称为 class method()。

它是所有对象共享的方法。它是 Python 编程语言中的内置函数。

classmethod()只接受一个名为 function 的参数。因此，python 中 classmethod()方法的语法如下:

```py
Syntax : classmethod(function)

```

## 使用 Python 类方法()

让我们在这里看一个 Python classmethod()函数的例子。在下面的例子中，我们使用 classmethod()函数打印在类中声明的学生的分数，而不是将变量传递给函数。

```py
class Student:
    marks = 50

    def printMarks(cls):
        print("Marks obtained by the Student is:",cls.marks)

Student.printMarks=classmethod(Student.printMarks)
Student.printMarks()

```

输出:

```py
============= RESTART: C:/Users/Admin/Student.py =============
Marks obtained by the Student is: 50

```

**在上面的代码中，我们观察到:**

*   我们有一个班级学生，其成员变量**标记为**。
*   我们有一个函数 **printMarks** ，它接受参数 **cls** 。
*   我们已经传递了总是附加到一个类的 classmethod()。
*   这里，第一个参数总是类本身，这就是为什么我们在定义类时使用' **cls'** 。
*   然后我们将方法 **Student.printMarks** 作为参数传递给 **classmethod( )** 。最后的 **printMarks** 被调用，它打印类变量**的标记**。

* * *

## 使用 Decorator @ class 方法

使用 classmethod()函数的另一种方法是使用 **@classmethod 装饰器。**

通过使用 decorator 方法，我们可以调用类名而不是对象。它可以应用于该类的任何方法。

**@classmethod Decorator** 方法的语法如下:它接受一个参数和多个实参。

```py
class c(object):
    @classmethod
    def function(cls, argument1, argument2, ...):
    /*remaining code*/

```

让我们看看如何使用 decoartor classmethod 运行一个类似的示例，而不是像前面的示例那样使用函数。

```py
class Student:
    course = 'Python'

    @classmethod
    def printCourse(cls):
        print("Course opted by the Student is:",cls.course)

Student.printCourse()

```

输出:

```py
============= RESTART: C:/Users/Admin/Student.py =============
Course opted by the Student is: Python

```

在上面提到的例子中， **@classmethod 装饰器**已经应用于 **printCourse( )** 方法。它有一个参数 **cls** 。然后，该方法自动选取该类的成员变量，而不是要求用户将一个成员变量传递给函数。

* * *

## 结论

简单来说，这是关于使用 Python classmethod()函数的两种方法。敬请关注更多关于 Python 的文章！