# Python 中类的属性

> 原文：<https://www.askpython.com/python/oops/class-instance-attributes>

类是 Python 语言的基础部分。它们提供了一种将相关功能组合在一起的方法，并且在面向对象编程中起着核心作用。在本文中，我们将看看 Python 中类的属性。

1.  ***[继承](https://www.askpython.com/python/oops/inheritance-in-python):将父类的属性引入子类。***
2.  ***[多态](https://www.askpython.com/python/oops/polymorphism-in-python):从一个形态创造出多个形态。***
3.  ******[抽象](https://www.askpython.com/python/oops/abstraction-in-python)*** :显示必要的数据，隐藏不必要的数据。***
4.  ***[封装](https://www.askpython.com/python/oops/encapsulation-in-python):保护类的信息。***

## 关于一般类的更多信息

1.  这些类只是一个接口，其中包含变量和函数。这些分别被称为**数据成员和成员函数**。
2.  要访问它们，我们需要创建一个该类的对象。对象是我们可以编辑其属性的实例。
3.  为了给类本身提供一些参数，有一个特殊的方法叫做**构造函数**。**当我们形成对象时，方法在运行时调用。**
4.  我们可以只使用**对象**来访问所有的数据成员和成员函数。

### 在 Python 中创建类

Python 中的类是用关键字 class 创建的，后跟类名。类属性是在类名之后定义的，由该类的所有实例共享。单个实例属性是在类属性之后定义的，它们对于每个实例都是唯一的。方法定义也放在类定义之后。方法是与类相关联的函数，它们用于处理或操作存储在类实例中的数据。

现在让我们定义一个类来更好地理解这一点。

**代码:**

```py
class student:
    def __init__(self, name, std, roll_no):
        self.nm = name
        self.std = std
        self.rl_no = roll_no

    def getData(self):
        print("Student name: ", self.nm)
        print("Standard: ", self.std)
        print("Roll number: ", self.rl_no)

    def setData(self, name, std, roll_no):
        self.nm = name
        self.std = std
        self.rl_no = roll_no

stud = student("Om", "4th", 9)
stud.getData()
print() # to print a new line in between
stud_1 = student("Hari", "5th", 14) 
stud_1.getData()

```

**输出:**

```py
Student name:  Om
Standard:  4th
Roll number:  9

Student name:  Hari
Standard:  5th
Roll number:  14

```

**说明:**

1.  使用 class 关键字声明一个类。然后在它后面添加 class_name，并给出一个冒号来开始数据插入。
2.  然后调用 **"__init__()"** 方法。这是 Python 中任何类的构造函数方法。
3.  我们创建一个**学生**类，然后给它一些属性，比如**名字、标准和编号**。
4.  然后我们使用 **self** **关键字来确保属性被正确地绑定到类。如果我们不使用 self 关键字，就没有使用类声明。**
5.  该类中有两个方法。
    1.  第一个“getData()”检索实例属性。
    2.  第二个“setData()”允许更改这些属性的值。
6.  现在我们为这个类创建两个对象。第一个有不同的参数。这是两个学生的名字和信息。
7.  这些被称为**实例变量或实例属性。**它们对每个对象都是唯一的。

### 访问类变量实例属性

您可以使用点运算符(.).例如，如果要访问 myClass 的属性 x，可以使用表达式 myClass.x。如果要调用 myClass 的方法 myMethod，可以使用表达式 myClass.myMethod()。

在这个演示中，让我们在类中定义一些实例属性。

**访问实例属性的语法:**

```py
object = class_name(parameter1 = value_!, parameter2 = value_2, .., parameter_N = value_N)
object.parameter_1
object.parameter_2
.
.
object.parameter_N

```

**代码:**

```py
class Rectangle:
    def __init__(self,  length, width):
        self.side_1 = length
        self.side_2 = width

    def area(self):
        a = self.side_1*self.side_2 
        print("Area of the rectangle is: ", a)

rect = Rectangle(45.0, 56.98)

# printing the type of object
print(type(rect)) 

 # accessing the values through object
print(rect.side_1)
print(rect.side_2)

```

**输出:**

```py
<class '__main__.Rectangle'>
45.0
56.98

```

因此，通过这种方式，我们可以访问它们。

### 访问类方法和其他实例属性

**语法:**

```py
class_name.variable_1
class_name.variable_2
.
.
class_name.variable_N

```

这个概念的简单变化就是我们所说的类属性就是类变量。这些变量只能通过使用 **class_name** 来访问。它们也被称为静态变量。内存不会清除它，而是在每次成功运行代码之后。它**更新**新值，保留以前的值。

例如，我们取同一个**学生类**，并为其创建**类属性**:

```py
class student:
    school = "Universal Public School"
    batch = "2020-2021"

    def __init__(self, name, std, roll_no):
        self.nm = name
        self.std = std
        self.rl_no = roll_no

    def getData(self):
        print("Student name: ", self.nm)
        print("Standard: ", self.std)
        print("Roll number: ", self.rl_no)

    def setData(self, name, std, roll_no):
        self.nm = name
        self.std = std
        self.rl_no = roll_no

print("The school name is: ", student.school) 
print("The batch year is: ", student.batch, "\n")

stud = student("Om", "4th", 9)
stud.getData()
print() # to print a new line in between
stud_1 = student("Hari", "5th", 14) 
stud_1.getData()

```

**输出:**

```py
The school name is:  Universal Public School
The batch year is:  2020-2021

Student name:  Om
Standard:  4th
Roll number:  9

Student name:  Hari
Standard:  5th
Roll number:  14

```

**说明:**

1.  学生类一开始只包含两个新东西。它包含了**学校**和**批次**变量。
2.  下一个代码与第一个代码相同。其他的只是`getter() and setter()`方法。
3.  现在在第 21 和 22 行代码中，我们调用这些变量。
4.  注意区别:
    1.  我们只使用类名**调用它们，而不是创建一个对象。**
    2.  然后使用点运算符**.”**访问被占用。
5.  另外，请注意，我们可以在运行时使用等号“=”操作符来改变它们的值，也可以调用它们。

**示例(运行期间):**

```py
class Employee:

    # class attributes
    COMPANY = ""
    BRANCH = ""

    def __init__(self, name, designation, ID):
        self.name = name
        self.designation = designation
        self.id = ID

    def getData(self):
        print(self.name)
        print(self.designation)
        print(self.id)
        print()

    def setData(self, name, desig, ID):
        self.name = name
        self.designation = desig
        self.id = ID

def main():
    Employee.COMPANY = input("Enter the company name: ")
    Employee.BRANCH = input("Enter the branch: ")
    print()

    print("...The employee details are...")
    print("The company name is: ", Employee.COMPANY)
    print("The company branch is at: ", Employee.BRANCH)

    emp_1 = Employee("Varun", "Tirpathi", 1001)
    emp_2 = Employee("Dhanush", "Reddy", 1002)
    emp_3 = Employee("Neha", "Singh", 1003)

    emp_1.getData()
    emp_2.getData()
    emp_3.getData()

main()

```

**输出:**

```py
Enter the company name: Microsoft
Enter the branch: Bengaluru

...The employee details are...      
The company name is:  Microsoft     
The company branch is at:  Bengaluru
Varun
Tirpathi
1001

Dhanush
Reddy
1002

Neha
Singh
1003

```

**说明:**

1.  这里我们有一个简单的雇员类。构造函数包含雇员的姓名、职务和 ID 等参数。
2.  接下来的方法是名字中的`getData() and setData().` ,我们可以理解为第一个方法用于检索数据，第二个方法用于编辑数据。
3.  这个类有两个属性:
    1.  公司。
    2.  分支。
4.  这个函数接受这两个类属性的输入。
5.  在最后六行中，我们有三个 Employee 类的对象。
6.  然后为了检索数据，我们调用 getData()方法。

## 结束了

所以，这样，我们可以说一个类的属性也叫做**类变量**。我希望这将有助于学习与 OOP 和 Python 中的类相关的新概念。更多新的话题。在那之前，继续学习和进步。