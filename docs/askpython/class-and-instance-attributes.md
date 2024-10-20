# Python 类属性和实例属性

> 原文：<https://www.askpython.com/python/oops/class-and-instance-attributes>

在本文中，我们将重点关注 **Python 类属性和实例属性**。

属性是编程语言的关键。它们负责保存重要的数据值，也有助于数据操作。

让我们现在就开始吧！

* * *

## 了解 Python 类属性

`Python Class Attribute`是包含在类中的属性/变量。也就是说，它的范围位于 [Python 类](https://www.askpython.com/python/oops/python-classes-objects)内。

类属性**仅创建其自身的单个副本**，并且该单个副本由该特定类中的所有函数和对象共享和利用。

**语法:**

```py
class Class-name:
     variable = value

```

现在让我们通过下面的例子来理解相同的实现。

* * *

### 用示例实现 Class 属性

```py
class class_attribute: 
	val = 1

	def product(self): 
		class_attribute.val *= 10
		print(class_attribute.val)

obj1 = class_attribute() 
obj1.product()		 

obj2 = class_attribute() 
obj2.product()		 

```

在这个例子中，我们创建了一个类变量‘val’并将其初始化为 1。

此外，我们访问函数 product()中的变量“val ”,并通过将该值乘以 10 来处理它。

可以清楚地看到，创建的两个对象使用了变量‘val’的同一个副本。因此，起初，val = 1。

当对象 obj1 调用该函数时，使用相同的“val”副本(该值不会重置)，因此它变为 val=10。在被 obj2 调用时，val 变为 val*10，即 10*10 = 100。

**输出:**

```py
10
100

```

* * *

## 了解 Python 实例属性

`Python Instance attribute`是一个局部属性/变量，其范围位于使用该属性的特定函数内。因此，它被一个特定的函数所包围。

每当实例属性**被一个[函数](https://www.askpython.com/python/python-functions)/对象调用时，它就会创建一个自身**的新副本。也就是说，每次对象或函数试图访问该变量时，都会使用该变量的一个不同副本。

**语法:**

```py
def function-name():
    variable = value

```

现在让我们借助一个例子来实现局部属性。

* * *

### 用示例实现实例属性

```py
class instance_attribute: 

	def product(self): 
	   val = 20
	   val *= 10
	   print(val)

obj1 = instance_attribute() 
obj1.product()		 

obj2 = instance_attribute() 
obj2.product()

```

在这个例子中，我们将一个实例属性声明并初始化为 val = 20。

此外，当 obj1 试图通过函数访问变量时，它会创建自己的新副本，将默认值重置为初始化值，然后提供对它的访问。

当 obj2 试图访问实例变量‘val’时，同样的场景会重复出现。

**输出:**

```py
200
200

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

* * *

## 参考

*   [Python 属性—文档](https://docs.python.org/3/tutorial/classes.html)