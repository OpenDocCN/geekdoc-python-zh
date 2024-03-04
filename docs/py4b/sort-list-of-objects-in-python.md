# Python 中对象的排序列表

> 原文：<https://www.pythonforbeginners.com/basics/sort-list-of-objects-in-python>

我们可以简单地使用`sort()`方法或`sorted()`函数对数字列表进行排序。但是，我们不能对使用自定义类创建的对象列表这样做。在本文中，我们将讨论如何使用 python 中的`sort()`方法和`sorted()`函数对对象列表进行排序。

## 如何在 Python 中对对象列表进行排序？

通常，当我们有一个数字列表时，我们可以使用如下所示的`sort()`方法进行排序。

```py
myList = [1, 2, 9, 3, 6, 17, 8, 12, 10]
print("Original list is:", myList)
myList.sort()
print("The sorted list is:", myList)
```

输出:

```py
Original list is: [1, 2, 9, 3, 6, 17, 8, 12, 10]
The sorted list is: [1, 2, 3, 6, 8, 9, 10, 12, 17]
```

现在，让我们尝试对对象列表进行排序。为此，我们将创建一个属性为`name`和`age`的`Person`类。之后，我们将创建不同的 person 对象，并制作一个包含这些对象的列表。当我们试图对列表进行排序时，[程序会遇到如下所示的类型错误异常](https://www.pythonforbeginners.com/error-handling/exception-handling-in-python-increasing-robustness-of-your-python-program)。

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return self.name

person1 = Person("Sam", 12)
person2 = Person("Harry", 23)
person3 = Person("Tom", 17)
person4 = Person("John", 30)
person5 = Person("Kite", 40)
person6 = Person("Emily", 23)

myList = [person1, person2, person3, person4, person5, person6]
print("Original list is:")
for person in myList:
    print(person, end=",")
print("\n")
myList.sort()
print("The sorted list is:", myList)
```

输出:

```py
Original list is:
Sam,Harry,Tom,John,Kite,Emily,

/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 22, in <module>
    myList.sort()
TypeError: '<' not supported between instances of 'Person' and 'Person' 
```

这里发生错误是因为我们不能使用<操作符来比较两个`Person`对象。因此，我们必须指定一个`Person`类的属性，用于比较两个不同的 Person 对象。为此，我们在`sort()`方法中使用了`key`参数。

我们将首先创建一个函数`getAge()`，它接受一个`Person`对象作为输入参数，并返回`age`。

之后，我们将把`getAge()`函数赋给`key`参数。执行后，sort() [方法将对 Person 对象的列表](https://www.pythonforbeginners.com/basics/lists-methods)进行排序，如下例所示。

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return self.name

def getAge(person):
    return person.age

person1 = Person("Sam", 12)
person2 = Person("Harry", 23)
person3 = Person("Tom", 17)
person4 = Person("John", 30)
person5 = Person("Kite", 40)
person6 = Person("Emily", 23)

myList = [person1, person2, person3, person4, person5, person6]
print("Original list is:")
for person in myList:
    print(person, end=",")
print("\n")
myList.sort(key=getAge)
print("The sorted list is:")
for person in myList:
    print(person, end=",")
```

输出:

```py
Original list is:
Sam,Harry,Tom,John,Kite,Emily,

The sorted list is:
Sam,Tom,Harry,Emily,John,Kite,
```

如果不允许修改输入列表，可以使用`sorted()`功能对对象列表进行排序。这里，我们将传递对象列表作为第一个输入参数，传递函数`getAge()` 作为第二个输入参数。执行后，它将返回一个对象的排序列表，如下所示。

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return self.name

def getAge(person):
    return person.age

person1 = Person("Sam", 12)
person2 = Person("Harry", 23)
person3 = Person("Tom", 17)
person4 = Person("John", 30)
person5 = Person("Kite", 40)
person6 = Person("Emily", 23)

myList = [person1, person2, person3, person4, person5, person6]
print("Original list is:")
for person in myList:
    print(person, end=",")
print("\n")
newList = sorted(myList, key=getAge)
print("The sorted list is:")
for person in newList:
    print(person, end=",")
```

输出:

```py
Original list is:
Sam,Harry,Tom,John,Kite,Emily,

The sorted list is:
Sam,Tom,Harry,Emily,John,Kite,
```

## 使用 Lambda 函数对对象列表进行排序

我们可以使用 lambda 函数对对象列表进行排序，而不是定义一个不同的函数，比如`getAge()`。这里，我们将创建一个 lambda 函数，它接受一个对象并返回其属性。然后，我们将把 lambda 函数赋给`sort()` 方法中的参数键。在执行了`sort()`方法之后，列表将被排序，如下例所示。

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return self.name

person1 = Person("Sam", 12)
person2 = Person("Harry", 23)
person3 = Person("Tom", 17)
person4 = Person("John", 30)
person5 = Person("Kite", 40)
person6 = Person("Emily", 23)

myList = [person1, person2, person3, person4, person5, person6]
print("Original list is:")
for person in myList:
    print(person, end=",")
print("\n")
myList.sort(key=lambda p: p.age)
print("The sorted list is:")
for person in myList:
    print(person, end=",") 
```

输出:

```py
Original list is:
Sam,Harry,Tom,John,Kite,Emily,

The sorted list is:
Sam,Tom,Harry,Emily,John,Kite,
```

除了使用`sort()`方法，您还可以使用带有 lambda 函数的`sorted()`函数对对象列表进行排序，如下例所示。

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return self.name

person1 = Person("Sam", 12)
person2 = Person("Harry", 23)
person3 = Person("Tom", 17)
person4 = Person("John", 30)
person5 = Person("Kite", 40)
person6 = Person("Emily", 23)

myList = [person1, person2, person3, person4, person5, person6]
print("Original list is:")
for person in myList:
    print(person, end=",")
print("\n")
newList = sorted(myList, key=lambda p: p.age)
print("The sorted list is:")
for person in newList:
    print(person, end=",")
```

输出:

```py
Original list is:
Sam,Harry,Tom,John,Kite,Emily,

The sorted list is:
Sam,Tom,Harry,Emily,John,Kite,
```

## 结论

在本文中，我们讨论了如何在 python 中对一列[对象进行排序。要了解更多关于 python 中的列表，你可以阅读这篇关于 python 中的](https://www.pythonforbeginners.com/basics/callable-objects-in-python)[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)