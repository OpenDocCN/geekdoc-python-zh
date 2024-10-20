# Python 中的神奇方法

> 原文：<https://www.askpython.com/python/oops/magic-methods>

Python 编程语言中的神奇方法是专门针对面向对象设计的。我们创建的每个类都有自己的神奇方法。Python 的标准解释器将这些分配给我们在其中创建的每个类。因此，在本文中，我们将详细了解如何调用和使用魔法方法来获得更好的编程方法。让编码的乐趣开始吧！

## 复习面向对象的知识

在进入正题之前，我们先来了解和打磨一下 OOP 概念的知识。我们将只看到基本的东西。因此，面向对象编程是一种将数据成员和成员函数封装到用户定义的实体中的方法，该实体被称为**类**。

类是保存特定数据项的东西，这些数据项相互关联并以特定的方式进行通信。我们使用**对象**访问属性和成员函数。对象是一个类的实例。在任何编程语言中，当我们创建一个类的时候，内存是不会被分配的，但是当我们创建它的实例，也就是对象的时候，内存就被分配了。

**举例:**

**动物**是类的一种类型。在这方面，我们包括居住在地球上的所有生物。所以，每个人都有自己的生活方式，食物和住所。动物只是定义了所有这些的蓝图。例如，**猫是动物类的对象。它有四条腿，吃老鼠，住在房子或灌木丛里。同样，老虎有四条腿，但它杀死和吃掉许多动物，所以我们说老虎吃肉，它生活在森林里。**

**Python 代码示例:**

```py
class Animal:
    def __init__(self, legs, food, shelter):
        self.legs = legs
        self.food = food
        self.shelter = shelter

    def showAnimal(self):
        print("The animal has {} legs: ".format(self.legs))
        print("The animal eats: {}".format(self.food))
        print("The animal lives in: {}".format(self.shelter))

cat = Animal(4, "Mouse", "House")
tiger = Animal(4, "Meat", "Forest")
cat.showAnimal()
tiger.showAnimal()

```

**输出:**

```py
The animal has 4 legs: 
The animal eats: Mouse
The animal lives in: House
The animal has 4 legs: 
The animal eats: Meat
The animal lives in: Forest

```

**说明:**

1.  animal 类包含**条腿、食物和住所作为属性。**
2.  当我们创建一个实例并在构造函数中插入值时，它们的行为差异就很明显了。
3.  **所以，同一类的对象可以根据价值观的行为而有所不同。**

## 面向对象的神奇方法

所以，在上面的例子中，我们有一个动物类。Python 有一组方法，即 **Dunder methods** ，负责保存类的属性、数据成员和成员函数。

***定义:当我们创建一个对象时，Python 解释器在代码执行的后端调用特殊函数。他们被称为魔术方法或邓德方法。***

为什么我们说邓德？因为他们的名字位于**双下划线**之间。当我们创建一个类的对象时，它们执行一些像魔术一样的计算。那么，我们如何检查它们是什么，在标准类中有多少？使用以下步骤找到它们:

1.  **创建一个样本类。**
2.  **创建其对象。**
3.  使用 **dir()** 函数并将对象插入其中。
4.  这个函数打印了所有魔术方法的列表，以及分配给这个类的数据成员和成员函数。

**代码:**

```py
print(dir(cat))

```

**输出:**

```py
__class__
__delattr__
__dict__
__dir__
__doc__
__eq__
__format__
__ge__
__getattribute__
__gt__
__hash__
__init__
__init_subclass__
__le__
__lt__
__module__
__ne__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__
food
legs
shelter
showAnimal

```

**你在双下划线中看到的名字都是神奇的方法。**其余属性由用户定义。正如我们所见,`__init__()`是 Python 中任何类的构造函数，也是一个神奇的方法。让我们逐一看看它们的用途。**要理解它们的功能，总是要尝试覆盖这些功能。**

需要注意的一点是，对于用户定义的任何类，都有一些默认的魔法方法。

### 一些神奇方法的使用和实现

在本节中，我们将看到一些神奇方法的使用、实现和使用，以编写更好的 [OOP 设计](https://www.askpython.com/python/oops/object-oriented-programming-python)。

#### 1.__new__():

此方法帮助构造函数 __init__()方法为类创建对象。因此，当我们创建一个类的实例时，Python 解释器首先调用 __new__()方法，然后调用 __init__()方法。他们彼此携手合作。

1.  当程序员选择创建一个对象时，会调用 __new__()来接受该对象的名称。
2.  然后 __init__()被调用，其中包括 **self** 的参数被插入到对象中，这反过来帮助我们修改类属性。

**代码:**

```py
class Sample:
    def __new__(self, parameter):
        print("new invoked", parameter)
        return super().__new__(self)

    def __init__(self, parameter):
        print("init invoked", parameter)

obj = Sample("a")

```

**输出:**

```py
new invoked a
init invoked a

```

**说明:**

1.  首先，我们创建一个类作为样本。
2.  然后通过创建 __new__()方法来覆盖它。然后，像往常一样，self 参数来了，之后给出一个简单的参数。
3.  使用带有 self 参数的`__new__()`函数返回一个 super()函数，以访问我们对该方法进行的定制。
4.  然后，用同样的方法调用带有参数的`__init__()`函数。
5.  然后创建一个样本类的对象。
6.  **现在，当我们运行代码时，解释器首先调用 __new__()，然后它调用 __init__()方法。**

#### 2.__init__():

Python 是一种面向对象的编程语言。所以，这个类必须有一个构造函数。使用 __init__()方法可以满足这一要求。当我们创建一个类并想给它一些初始参数时。初始化器方法为我们执行这个任务。

**代码:**

```py
class Sample:        
    def __init__(self, parameter):
        print("init invoked", parameter)

obj = Sample("a")

```

**输出:**

```py
init invoked a

```

**说明:**

1.  创建/覆盖 __init__()方法。插入 self 参数来通知解释器这是一个类方法。
2.  插入必需的参数。
3.  然后使用 print()函数打印该参数。
4.  之后，创建一个对象。
5.  当我们运行代码时，我们得到的输出是“init invoked a”，这表明**解释器调用 init()并打印该参数。**

#### 3.__str__():

这个方法帮助我们根据我们的需求显示对象。假设我们创建一个对象并试图打印它。函数的作用是:显示对象的存储位置。如果我们想修改，我们可以这样做。__str__()函数很好地展示了对象。

**代码(在使用 __str__())之前):**

```py
class Student:
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

stud_1 = Student("Suresh", 1)
print(stud_1) 

```

**输出:**

```py
<__main__.Student object at 0x0000023E2CF37CA0>

```

**代码(使用 __str__())后):**

```py
class Student:
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

    def __str__(self):
        return ("{} {}".format(self.name, self.roll_no))

stud_1 = Student("Suresh", 1)
print(stud_1) 

```

**输出:**

```py
Suresh 1

```

酷吧！现在我们也可以用类似的方法。我们可以根据需要设置对象的格式。

#### 4.__repr__():

类似于 __str__()，我们可以使用 __repr__ 函数对对象进行修饰。代码类似于 __str__()实现。

```py
class Student:
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

    def __repr__(self):
        print("repr invoked")
        return ("{} {}".format(self.name, self.roll_no))

stud_1 = Student("Suresh", 1)
print(stud_1) 

```

**输出:**

```py
repr invoked
Suresh 1

```

#### 5.__sizeof__():

当我们创建一个类时，解释器从来不会给它分配内存。它将内存分配给对象。如果我们想知道分配给该对象的内存，那么我们可以调用或覆盖 __sizeof__()函数并传递我们的对象。这也返回 list =，tuple，dictionary 对象的大小。

**代码:**

```py
class Student:
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no

stud_1 = Student("Suresh", 1)
print("Size of student class object: ", stud_1.__sizeof__()) 

list_1 = [1, 2, 3, 4]
tup_1 = (1, 2, 3, 4, 5)
dict_1 = {"a":1, "b":2, "c":3, "d":4}
print("Size of list: ", list_1.__sizeof__())
print("Size of tuple: ", tup_1.__sizeof__())
print("Size of dictionary: ", dict_1.__sizeof__())

```

**输出:**

```py
Size of student class object:  32
Size of list object:  104
Size of tuple object:  64
Size of dictionary object:  216

```

#### 6.__ 添加 _ _():

这种神奇的方法与其名字特别相似。它[增加了两个变量](https://www.askpython.com/python/examples/addition-in-python)。对于整数，它返回总和，对于字符串，它返回它们的连接结果。

**代码:**

```py
class Numbers:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __add__(self):
        print("__add__ invoked")
        return self.a + self.b

num = Numbers(3, 4)
num_2 = Numbers("a", "b")
print(num.__add__())
print(num_2.__add__())

```

**输出:**

```py
__add__ invoked
7
__add__ invoked
ab

```

#### 7.__reduce__():

这个神奇的方法以 **key: value** 格式返回一个类的所有参数及其值的集合或字典。这可以使用带有点运算符的对象名来直接调用。所以，当我们[创建一个类](https://www.askpython.com/python/oops/python-classes-objects)并用一些值实例化它。该函数将使用在类声明过程中给定的参数名返回它。

**代码:**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

emp = Employee("Shrinivas", 150000)
print(emp.__reduce__())

```

**输出:**

```py
(<function _reconstructor at 0x0000023E22892EE0>, (<class '__main__.Employee'>, <class 'object'>, None), {'name': 'Shrinivas', 'sal': 150000})

```

**代码(覆盖 __reduce__())后):**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

    def __reduce__(self):
        return self.name, self.sal

emp = Employee("Shrinivas", 150000)
print(emp.__reduce__())

```

**输出:**

```py
{"Shrinivas", 150000}

```

**说明:**

当我们重写并试图返回参数时，我们只能在一个集合中获得它们的值。

#### 8.__hash__():

__hash__()函数返回存储在[堆内存](https://www.askpython.com/python/examples/min-heap)中的对象的特定哈希值。我们既可以覆盖它，也可以使用对象名调用它。[散列法](https://www.askpython.com/python-modules/oshash-module)对于获取计算机中任意随机元素的内存地址非常有用。为了简单和内存分配，所有编程语言都使用散列。

**代码:**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

    def __hash__(self):
        return super().__hash__()

emp = Employee("Shrinivas", 150000)
print(emp.__hash__())

```

**输出:**

```py
154129100057

```

**代码:**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

emp = Employee("Shrinivas", 150000)
print(emp.__hash__())

```

**输出:**

```py
154129054082

```

#### 9\. __getattribute__(name):

这个函数返回一个类的属性值，如果它存在的话。我们需要调用函数并传递我们使用关键字 **self** 分配给类参数的属性。比如，如果我们将**薪水**的值赋给 **self.sal** ，我们需要在 __getattribute__()函数中调用 **sal** 。

**代码:**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

    def __getattribute__(self, name):
        return super().__getattribute__(name)

emp = Employee("Ravi", 500000)
print(emp.__getattribute__("sal"))

```

**输出:**

```py
50000

```

**说明:**

在该函数中，将**“self . sal”**赋给雇员类的**薪金**参数。该函数将其值作为存在于类中的属性返回。如果不存在，该函数将返回一条错误消息。

#### 10.__setattr__(名称，值):

顾名思义，这个神奇的方法帮助我们在定义对象时改变属性值。不需要重写 **__getattribute__()和 __setattr__()** 函数。只需使用创建的对象调用它们。

**代码:**

```py
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.sal = salary

emp = Employee("Ravi", 500000)
emp.__setattr__("name", "Suresh")
emp.__setattr__("sal":600000)
print("The changed name of the employee is: ", emp.__getattribute__("name"))
print("The changed salary of the employee is: ", emp.__getattribute__("sal"))

```

**输出:**

```py
The changed name of the employee is: Suresh
The changed salary of the employee is: 600000

```

**说明:**

1.  **setattr _ _()接受两个参数。**
    1.  **属性名称**
    2.  **它的新值**
2.  然后，它将该特定值赋给该属性。
3.  之后，为了检查分配给它的值，使用 employee 对象和点运算符调用 __getattrbute__()函数。电磁脉冲。__getattribute("name ")。

**点注:这两个函数代替了 Python 中一个类的 getter 和 setter 方法。**

## 结论

所以，我们看到了 Python 中一些神奇方法的深入实现。我希望这有所帮助，并将使编程更容易。它们被证明是有助于快速实现和使用的代码。快乐的 python 编程🐍🐍😎。