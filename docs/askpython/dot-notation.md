# Python 中的点符号

> 原文：<https://www.askpython.com/python/built-in-methods/dot-notation>

今天我们来讨论一下 Python 中的点符号。如果你有一点用 Python 编程的经验，或者如果你一直在关注我们的 AskPython 博客，你应该会遇到术语[面向对象编程](https://www.askpython.com/python/oops/object-oriented-programming-python)。

它是一种基于现实世界对象概念的编程范式。每个对象都有描述其状态的特定属性和使它们执行特定任务的方法(相当于执行一个函数)。Python 就是这样一种语言。

在 Python 中，几乎每个实体都作为一个对象进行交易。了解这一点是理解点的重要性的基础。)批注。

## 什么是点符号？

简单来说，就是点(。)符号是访问不同对象类实例的每个方法的属性和方法的一种方式。

它通常位于对象实例之前，而点符号的右端包含属性和方法。

让我们用多个方法创建一个类，然后使用(。)符号来访问这些方法。

**创建你的[类和对象](https://www.askpython.com/python/oops/python-classes-objects) :**

```py
class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sayHello(self):
        print( "Hello, World" )

    def sayName(self):
        print( f"My name is {self.name}")

"""
First, we create a Person Class that takes two parameters: name, and age. This is an object. 
The object currently contains two methods: sayHello() and sayName().
Now, we'll see how we can access those attributes and methods using dot notation from an instance of the class. 
"""

```

现在我们的类已经准备好了，我们需要创建一个实例对象。

```py
#We create an instance of our Person Class to create an object. 
randomPerson = Person( "Marshall Mathers", 49)

#Checking attributes through dot notation
print( "Name of the person: " + randomPerson.name) 
print( "Age of the person: " + str(randomPerson.age) + "years" )

#Accessing the attributes through dot notation
randomPerson.sayHello()
randomPerson.sayName()

```

在最后两行中，我们使用格式为`<object name>`和`<method name>`的类的对象来访问类中的方法。

**输出:**

```py
Name of the person: Marshall Mathers
Age of the person: 49 years

Hello, World
My name is Marshall Mathers

```

希望上面的例子能消除你对 Python 中使用点符号的疑虑。

## 我们还在哪里使用点符号？

任何使用过 Python 的开发人员都遇到过(。)符号。这里有一些你过去一定遇到过的例子。

### 1.列表的索引

```py
#A simple list called array with 3 elements
words = ['godzilla', 'darkness', 'leaving heaven']

#Getting the index of the list
words.index()

```

### 2.拆分字符串

```py
#A random string
pun = "The movie Speed didn't have a director...Because if Speed had direction, it would have been called Velocity."

#We use the split method to separate the string into two sections on either side of "..."
print(pun.split("..."))

```

这是一些日常使用的点符号的例子。

## 结论

点符号不仅仅是一种访问内部方法的方式。这是一项复杂的技术，在确保完整功能的同时，保持代码的整洁和最小化。