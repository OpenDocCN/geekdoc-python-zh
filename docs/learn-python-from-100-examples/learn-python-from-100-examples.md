

Kinjalk Gupta

## 通过100多个示例学习Python

版权所有 © 2021 Kinjalk Gupta

保留所有权利。未经版权所有者书面许可，不得以任何方式复制或使用本书的任何部分，但书评中引用的使用除外。更多信息，请联系：kinjalkg01@gmail.com

第一版

## 第1章 入门

### 1. 什么是Python？

Python是一种解释型、高级、通用的编程语言。它可用于多种任务，例如人工智能/机器学习、构建桌面、移动和Web应用程序、编写自动化脚本等等。

有两个版本 -

- Python 2 - 2020年后将过时
- Python 3 - 将用于未来

### 2. 安装Python

前往python.org并安装Python

确保勾选“Add Python 3.x.x to Path”复选框

### 3. Python解释器

打开CMD并输入Python(Windows)或Python3(Linux)，然后按回车键

表达式是产生值的一段代码。

### 4. 代码编辑器

我们可以使用编辑器或IDE

IDE具有额外的功能：

- 1. 自动补全
- 2. 调试
- 3. 测试

编辑器 - VSCode, Atom, Sublime

IDE - Pycharm

### 5. Hello World

创建一个扩展名为“.py”的文件。
每当我们调用一个函数时，我们写()
写print("Hello World")
按ctrl + `
这将打开命令行/终端，现在输入python name_of_the_file.py并按回车键

### 6. Python扩展

这将把VSCode转换为一个强大的IDE，并添加以下功能

- 1. 代码检查
- 2. 自动补全
- 3. 单元测试
- 4. 调试
- 5. 代码格式化
- 6. 代码片段

### 7. 代码检查

查找潜在错误

### 8. 格式化代码

PEP - Python增强提案

最流行的是pep8
按cmd+shift+p并搜索Format，然后安装autopep 8

### 9. 运行python代码

打开扩展面板并搜索“code runner”

### 10. Python实现

默认的python实现 - Cpython
其他实现
- Jython - 用Java编写
- IronPython - 用C编写
- PyPy - Python的子集

### 11. Python代码如何执行

C编译器是一个知道如何将c代码编译成机器码（0和1）的程序
机器码特定于编译它的C程序类型，在windows上编译的程序在mac上无法运行
Java将其转换为不依赖于机器的java字节码
JVM将此字节码转换为机器码

C#和python采用了相同的路线

## 第2章 基本类型

### 1. 变量

```
students_count = 1000  # int

rating = 4.99  # float

is_published = False  # boolean (can be true or false)

course_name = "Python Programming"  # string

print(students_count)
```

### 2. 变量名

变量名应具有描述性

使用小写字母命名变量

使用下划线分隔多个单词

在等号周围留空格

### 3. 字符串

我们可以根据自己的喜好使用单引号或双引号。

要编写多行字符串，请使用三引号

```
course = "Python Programming"

message ="""

Hi John,

Blah blah blah

"""

course = "Python Programming"

print(len(course))  # 18

print(course[0])  # P

print(course[-1])  # g
```

```
#### 末尾索引处的字符不包含在内 (Pyt) 切片字符串
print(course[0:3])
print(course[:])
```

### 4. 转义序列

\ 是转义字符
" 是转义序列

```
course = "Python \nProgramming"
print(course)
```

### 5. 格式化字符串

```
first = 'Kinjalk'
last = 'Gupta'

full = f"{len(first)} {last}"
print(full)
```

### 6. 字符串方法

```
course = "Python Programming"

print(course.upper())  # PYTHON PROGRAMMING

print(course.lower())  # python programming

print(course.title())  # Python Programming

print(course.strip())

#### 如果未找到字符串，它将返回 -1，返回一个索引

print(course.find("P"))

print(course.replace("P", "j"))

#### 所有这些都创建了新的字符串

print(course)

print("Pro" in course)  # 返回布尔值
```

### 7. 数字

```
#### 增强运算符

x = 10

x = x+10

x += 10
```

### 8. 处理数字

```
print(round(2.9))

print(abs(-2))

print(math.ceil(2.3))

print(math.floor(2.3))
```

### 9. 类型转换

```
x = int(input("x: "))

print(type(x))

y = x+1
```

假值

- ""
- 0
- none

真值

其他任何值都将为真

## 第3章 控制流

### 1. 比较运算符

- >
- <
- >=
- <=

### 2. 条件语句

```
temp = 15

if temp >= 30:
    print("it's warm")

elif temp >= 20:
    print("it's nice")

else:
    print("it's cold")

print("Done")
```

### 3. 三元运算符

```
age = 17

if age >= 18:
    message = "Eligible"

else:
    message = "Not Eligible"

print(message)
```

```
message = "Eligible" if age >= 18 else "Not eligible"
print(message)
```

### 4. 逻辑运算符

```
high_income = True
good_credit = False
student = True

if (high_income or good_credit) and not student:
    print("eligible")
else:
    print("Not eligible")
```

### 5. 短路求值

### 6. 链式比较运算符

```
age = 22

if age >= 18 and age <= 65:
    print("Eligible")

if 18 <= age <= 65:
    print("Eligible")
```

### 7. For循环

```
for number in range(3):
    print("Attempt", number+1, (number+1)*".")

for number in range(1, 4):
    print("Attempt", number, (number)*".")
```

### 8. For Else循环

```
successful = False

for number in range(3):
    print("Attempt")

    if successful:
        print("Successful")
        break
else:
    print("Attempted 3 times and failed")
```

### 9. 嵌套循环

```
for x in range(5):
    for y in range(3):
        print(f"{x}, {y}")
```

### 10. 可迭代对象

```
print(type(range(5)))
#### range对象是可迭代的
for x in "Python":
    print(x)

for x in [2, 4, 5, 6]:
    print(x)
```

### 11. While循环

```
number = 100

while number > 0:
    print(number)
    number //= 2

command = ''

while command.lower() != "quit":
    command = input(">")
    print("ECHO", command)
```

### 12. 无限循环

```
while True:
    command = input(">")
    if command.lower() == "quit":
        break
    print("ECHO", command)
```

## 第4章 函数

### 1. 函数

函数是一个可重用的代码块，我们可以根据需要多次调用它。

```
def greet():
    print("Hi there")
    print("Welcome aboard")

greet()
```

### 2. 参数

参数是我们在程序中定义的输入。
实参是我们传递的实际值。

```
def greet(first_name, last_name):
    print(f"Hi there {first_name} {last_name}")
    print("Welcome aboard")

greet("Kinjalk", "G")
```

### 3. 函数类型

- A. 执行任务的函数
- B. 返回值的函数

```
def get_greeting(first_name, last_name):
    return f"Hi there {first_name} {last_name}"

msg = get_greeting("k", "g")
print(msg)

def greet(first_name, last_name):
    print(f"Hi there {first_name} {last_name}")

print(greet("Kinjalk", "G"))
```

### 4. 关键字参数

```
def increment(number, by):
    return number + by

print(increment(2, by=5))
```

### 5. 默认参数

```
def increment(number, by=1):
    return number + by

print(increment(2, 3))
```

### 6. Xargs

```
def multiply(*numbers):
    # 这里numbers是一个元组
    product = 1
    for n in numbers:
        product *= n
    return product

print(multiply(3, 5, 6, 5))
```

### 7. Xxargs

```
def save_user(**user):
    # 返回一个字典
    print(user["name"])

save_user(id=1, name="john", age=22)
```

### 8. 作用域

```
def greet(name):
    global message  # 全局变量是邪恶的
    message = "b"
    print(message)

greet("kinjalk")
print(message)
```

## 第5章 数据结构

### 1. 列表

```
letters = ["a", "b"]

matrix = [
    [0, 1],
    [2, 3],
    [3, 5] ]

zeros = [0] * 5

combined = letters + zeros

numbers = list(range(20))

chars = list("Hello World")

print(letters)
print(matrix)
print(zeros)
print(combined)
print(numbers)
print(chars)
print(len(chars))
```

### 2. 访问元素

```
letters = ["a", "b", "c"]
print(letters[0])
print(letters[-1])
letters[0] = "g"
print(letters)

letters = ["a", "b", "c", 'd', 'e']
print(letters[0:3])
print(letters[::2])

numbers = list(range(20))
print(numbers[::2])

numbers = list(range(20))
print(numbers[::-1])
```

### 3. 解包列表

```
numbers = [5, 2, 3, 5, 6, 6, 7, 8, 9]
first, second, *others = numbers
print(first)
print(others)
```

### 4. 遍历列表

```python
letters = ["a", "b", "c"]

for index, item in enumerate(letters):
    print(index, item)

#### enumerate 函数返回一个可迭代的 enumerate 对象。在每次迭代中，它返回一个元组，我们可以在 for 循环中解包它
```

### 5. 添加或删除元素

```python
#### 当一个函数是类的一部分时，我们称它为方法

letters = ["a", "b", "c"]

#### 添加
letters.append("d")  # 在列表末尾添加
print(letters)
letters.insert(0, '-')  # 在指定索引处添加
print(letters)

#### 删除
letters.pop(0)  # 指定索引删除
print(letters)
letters.remove('b')
print(letters)
letters.clear()
```

### 6. 列表排序

```python
letters = ["a", "b", "c"]
print(letters.index("d"))
print(letters.count('a'))
```

### 7. 列表排序

```python
numbers = [3, 51, 2, 8, 4]
numbers.sort()
print(numbers)
```

```python
numbers = [3, 51, 2, 8, 4]
numbers.sort(reverse=True)
print(numbers)
```

```python
n = sorted(numbers) # 返回一个新列表
print(n)
```

```python
items = [
    ("Product1", 10),
    ("Product2", 9),
    ("product3", 13),
    ("product3", 28),
    ("product3", 3)
]
```

```python
def sort_item(item):
    return item[1]
```

```python
items.sort(key=sort_item)
print(items)
```

### 8. Lambda 函数

```python
items = [
    ("Product1", 10),
    ("Product2", 9),
    ("product3", 13),
    ("product3", 28),
    ("product3", 3)
]

items.sort(key=lambda item: item[1])
print(items)
```

### 9. Map 函数

```python
items=[
    ("Product1", 10),
    ("Product2", 9),
    ("product3", 13),
    ("product3", 28),
    ("product3", 3)
]

#### 将 lambda 函数应用于列表中的所有元素
prices=list(map(lambda item: item[1], items))
print(prices)
```

### 10. Filter 函数

```python
items=[
    ("Product1", 10),
    ("Product2", 9),
    ("product3", 13),
    ("product3", 28),
    ("product3", 3)
]

x=list(filter(lambda item: item[1] >= 10, items))
print(x)
```

### 11. 列表推导式

```python
items=[
    ("Product1", 10),
    ("Product2", 9),
    ("product3", 13),
    ("product3", 28),
    ("product3", 3)
]

x=[item[1] for item in items]
print(x)

y=[item for item in items if item[1] >= 10]
print(y)
```

### 12. Zip 函数

```python
list1=[1, 2, 3, 4]
list2=[10, 20, 40]

print(list(zip(list1, list2)))
```

### 13. 栈

```python
#### LIFO - 后进先出

browsing_session=[]

browsing_session.append(1)

browsing_session.append(2)

browsing_session.append(3)

print(browsing_session)



browsing_session.pop()

browsing_session.pop()

browsing_session.pop()

if not browsing_session:
    print(browsing_session[-1])
```

### 14. 队列

```python
#### FIFO - 先进先出

from collections import deque



queue=deque([])

queue.append(1)

queue.append(2)

queue.popleft()

queue.popleft()



print(queue)
```

### 15. 元组

```python
point=(1,)

print(point)

point1=(1, 2) * 3

print(point1)

print(point1[0:2])
```

### 16. 交换变量

```python
x=10

y=11

z=x

x=y

y=z

print(x)

print(y)

x, y=y, x  # 我们定义一个元组并解包它

#### x, y = (y, x)
```

### 17. 数组

```python
from array import array

#### 接受一个类型码

numbers=array('i', [1, 2, 3])

numbers[0]=5
```

### 18. 集合

```python
numbers=[1, 1, 2, 3, 5]

first=set(numbers)

second={15, 7, 5}

#### print(first)
#### print(second)

print(first | second)  # 在第一个或第二个集合中的所有元素
print(first & second)  # 同时存在于两个集合中的元素
#### 这个第一个集合包含第二个集合中没有的额外数字
print(first - second)
print(first ^ second)  # 存在于第一个或第二个集合中，但不同时存在于两个集合中的元素
```

### 19. 字典

```python
point={"x": 1, "y": 2}
point1=dict(x=1, y=2)
print(point1["y"])
point1["z"]=20
#### print(point1["a"])
print(point1.get("a", "doesn't exist"))
del point['x']
for key in point1:
    print(key, point1[key])

for key, value in point1.items():
    print(key, value)
```

### 20. 字典推导式

```python
l={x*2 for x in range(5)}  # 集合
print(l)

l={x: x*2 for x in range(5)}  # 字典

print(l)

l=(x*2 for x in range(5))  # 元组，返回一个生成器对象

print(l)
```

### 21. 生成器对象

```python
from sys import getsizeof

l=(x*2 for x in range(10000))
print(getsizeof(l))

l=[x*2 for x in range(10000)]
print(getsizeof(l))
```

### 22. 解包运算符

```python
values=[*range(5)]
print(values)

string=[*"Hello"]
print(string)

combined=[values, string]  # [[0, 1, 2, 3, 4], ['H', 'e', 'l', 'l', 'o']]
combined=[*values, *string]  # [0, 1, 2, 3, 4, 'H', 'e', 'l', 'l', 'o']
print(combined)

first={"x": 1}
second={"x": 10, "y": 2}
combined={**first, **second}
print(combined)
```

### 23. 练习

编写一些代码，找出句子中出现次数最多的字母。

```python
from pprint import pprint

sentence="This is a common interview question"

count={}

for char in sentence:
    count[char]=count.get(char, 0) + 1

pprint(count, width=1)

frequency_sorted=sorted(count.items(),
key=lambda kv: kv[1],
reverse=True)

print(frequency_sorted[0])
```

## 第6章 异常

### 1. 异常

```python
try:
    age=int(input("Age: "))
except ValueError as error:
    print(error)
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")
    print("Execution continues")
```

### 2. 处理多个异常

```python
try:
    age=int(input("Age: "))
    xfactor=10/age
except ValueError as error:
    print(error)
    print("You didn't enter a valid age")
except ZeroDivisionError:
    print("Age can not be zero")
else:
    print("No exceptions were thrown")
print("Execution continues")
#--------------------------------------------------
try:
    age=int(input("Age: "))
    xfactor=10/age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")
print("Execution continues")
```

### 3. 清理资源

```python
try:
    file=open("app.py")
    age=int(input("Age: "))
    xfactor=10/age

except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")
finally:
    file.close()
```

### 4. With 语句

```python
try:
    # 自动释放外部资源
    with open("app.py") as file, open("another.txt") as target:
        print("File opened")
    age=int(input("Age: "))
    xfactor=10/age

except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")
finally:
    file.close()

#### 如果一个对象有 __exit__ 和 __enter__ 方法，它就支持上下文管理协议
```

### 5. 抛出异常

```python
def calculate_xfactor(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less")
    return 10/age

print(calculate_xfactor(1))
```

### 6. 抛出异常的代价

```python
from timeit import timeit

code1="""
def calculate_xfactor(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less")
    return 10/age

try:
    print(calculate_xfactor(-1))
except ValueError as error:
    pass
"""

code2="""
def calculate_xfactor(age):
    if age <= 0:
        return None
    return 10/age

xfactor = calculate_xfactor(-1)

if xfactor == None:
    pass
"""

print(timeit(code1, number=10000))
print(timeit(code2, number=10000))
```

## 第7章 类

### 1. 类

类用于定义新的类型或

类是创建新对象的蓝图

对象是类的一个实例

### 2. 创建类

```python
class Point:  # 帕斯卡命名约定
    def draw(self):
        print("draw")

point1=Point()
print(type(Point()))  # <class '__main__.Point'>
point1.draw()
print(isinstance(point1, int))
```

### 3. 构造函数

```python
class Point:  # 帕斯卡命名约定
    def __init__(self, x, y):  # 魔术方法或双下划线方法，self 是对当前对象的引用
        self.x = x
        self.y = y

    def draw(self):
        print(f"Point ({self.x},{self.y})")

point1 = Point(5, 8)
point1.draw()
print(point1.x)
```

### 4. 类属性与实例属性

```python
class Point:  # 帕斯卡命名约定
    default_color = "red"

    def __init__(self, x, y):  # 魔术方法或双下划线方法，self 是对当前对象的引用
        self.x = x
        self.y = y

    def draw(self):
        print(f"Point ({self.x},{self.y})")

Point.default_color = "yellow"

point1 = Point(5, 8)
print(point1.default_color)
print(Point.default_color)
```

point1.draw()

point2 = Point(45, 58)

print(point2.default_color)

point2.draw()

### 5. 类与实例方法

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        print(f"Point ({self.x},{self.y})")

    @classmethod
    def zero(cls):
        return cls(0, 0)

point1 = Point(5, 8)
point1.draw()
print(point1.x)

zero = Point.zero()
zero.draw()
```

### 6. 魔术方法

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def draw(self):
        print(f"(Point {self.x}, {self.y})")

point = Point(1, 2)
print(point)
print(str(point))
```

### 7. 对象比较

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

point = Point(1, 2)
other = Point(2, 4)

print(point == other)
print(point < other)
```

### 8. 执行算术运算

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

point = Point(1, 2)
other = Point(2, 4)
print(point == other)
print(point < other)
sum = point + other
print(sum.y)
```

### 9. 创建自定义容器

```python
class TagCloud:
    def __init__(self):
        self.tags = {}

    def add(self, tag):
        self.tags[tag.lower()] = self.tags.get(tag.lower(), 0) + 1

    def __getitem__(self, tag):
        return self.tags.get(tag.lower(), 0)

    def __setitem__(self, tag, count):
        self.tags[tag.lower()] = count

    def __len__(self):
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)

cloud = TagCloud()
cloud.add("python")
cloud.add("python")
cloud.add("pytHon")
cloud["Programming"] = 10
print(len(cloud))
print(cloud["Python"])
print(cloud.tags)
for item in cloud:
    print(item)
```

### 10. 私有成员

```python
class TagCloud:
    def __init__(self):
        self.__tags = {}  # 私有成员

    def add(self, tag):
        self.__tags[tag.lower()] = self.__tags.get(tag.lower(), 0) + 1

    def __getitem__(self, tag):
        return self.__tags.get(tag.lower(), 0)

    def __setitem__(self, tag, count):
        self.__tags[tag.lower()] = count

    def __len__(self):
        return len(self.__tags)

    def __iter__(self):
        return iter(self.__tags)

cloud = TagCloud()
cloud.add("python")
cloud.add("python")
cloud.add("pytHon")
cloud["Bubbles"] = 1000000
print(len(cloud))
print(cloud["Python"])
#### print(cloud.__tags)
for item in cloud:
    print(item)
print(cloud.__dict__)

print(cloud._TagCloud__tags)
```

### 11. 属性

```python
class Product:
    def __init__(self, price):
        self.set_price(price)

    def get_price(self):
        return self.__price

    def set_price(self, value):
        if value < 0:
            raise ValueError("Price cannot be negative")
        self.__price = value

product = Product(50)
print(product.get_price())
#### 非Python风格的代码

class Product:
    def __init__(self, price):
        self.price = price

    @property
    def price(self):
        return self.__price

    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("Price cannot be negative")
        self.__price = value

product = Product(50)
product.price = 23
print(product.price)
#### 属性在外部是一个属性，但内部有两个方法，一个getter和一个setter
```

### 12. 继承

```python
class Animal:
    def __init__(self):
        self.age = 1

    def eat(self):
        print("Eat")

#### Animal: 父类，基类
#### Mammal: 子类，派生类

class Mammal(Animal):
    def walk(self):
        print("walk")

class Fish(Animal):
    def swim(self):
        print("walk")

m = Mammal()

m.eat()

m.walk()

print(m.age)
```

### 13. 对象类

### 14. 方法重写

```python
class Animal:
    def __init__(self):
        print("Animal Constructor")
        self.age = 1

    def eat(self):
        print("Eat")

class Mammal(Animal):
    def __init__(self):
        super().__init__()
        print("Mammal Constructor")
        self.weight = 2

    def walk(self):
        print("walk")

m = Mammal()

print(m.age)

print(m.weight)
```

### 15. 多级继承

```python
class Animal:
    def eat(self):
        print("eat")

class Bird(Animal):
    def fly(self):
        print("fly")

class Chicken(Bird):
    pass
```

### 16. 多重继承

```python
class Employee:
    def greet(self):
        print("Employee greet")

class Person:
    def greet(self):
        print("Person greet")

class Manager(Employee, Person):
    pass

manager = Manager()

manager.greet()

class Flyer:
    def fly(self):
        pass

class Swimmer:
    def swim(self):
        pass

class FlyingFish(Flyer, Swimmer):
    pass
```

### 17. 继承的一个好例子

```python
class InvalidOperationError(Exception):
    pass

class Stream:
    def __init__(self):
        self.opened = False

    def open(self):
        if self.opened:
            raise InvalidOperationError("Stream is already opened")
        self.opened = True

    def close(self):
        if not self.opened:
            raise InvalidOperationError("Stream is already closed")
        self.opened = False

class FileStream(Stream):
    def read(self):
        print("Reading data from a file")

class NetworkStream(Stream):
    def read(self):
        print("Reading data from a network")
```

### 18. 抽象基类

```python
from abc import ABC, abstractmethod

class InvalidOperationError(Exception):
    pass

class Stream(ABC):
    def __init__(self):
        self.opened = False

    def open(self):
        if self.opened:
            raise InvalidOperationError("Stream is already opened")
        self.opened = True

    def close(self):
        if not self.opened:
            raise InvalidOperationError("Stream is already closed")
        self.opened = False

    @abstractmethod
    def read(self):
        pass

class FileStream(Stream):
    def read(self):
        print("Reading data from a file")

class NetworkStream(Stream):
    def read(self):
        print("Reading data from a network")

class VideoStream(Stream):
    pass

stream = Stream()
n = VideoStream()
```

### 19. 多态

```python
from abc import ABC, abstractmethod

class UIControl(ABC):
    @abstractmethod
    def draw(self):
        pass

class DropDownList(UIControl):
    def draw(self):
        print("DropDownlist")

class TextBox(UIControl):
    def draw(self):
        print("TextBox")

def draw(*controls):
    for control in controls:
        control.draw()

ddl = DropDownList()
textbox = TextBox()
draw(ddl, textbox)
```

### 20. 鸭子类型

### 21. 扩展内置类型

```python
class Text(str):
    def duplicate(self):
        return self + self

text = Text("Python")
print(text.duplicate())

class TrackableList(list):
    def append(self, object):
        print("Append called")
        super().append(object)

list = TrackableList()
list.append("I")
```

### 22. 数据类型

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p1 = Point(x=1, y=2)
p2 = Point(x=5, y=6)
print(p2 == p1)
#### 它们是不可变的
```

## 第8章 模块

### 1. 模块

```python
from sales import calc_shipping, calc_tax  # 导入特定函数
import sales  # 将整个模块作为对象导入

sales.calc_shipping()
sales.calc_tax()
calc_shipping()
calc_tax()
```

### 2. 包

```python
import ecommerce.sales

from ecommerce.sales import calc_shipping

from ecommerce import sales

ecommerce.sales.calc_tax()
```

### 3. 包内引用

```python
from ecommerce.customer import contact  # 绝对引用

from ..customer import contact  # 相对引用

def calc_tax():
    pass
```

### 4. Dir函数

```python
from ecommerce.shopping import sales

print(dir(sales))

print(sales.__name__)  # ecommerce.shopping.sales

print(sales.__package__)  # ecommerce.shopping

# c:\Users\kimjlk\Desktop\Hello World\ecommerce\shopping\sales.py

print(sales.__file__)
```

### 5. 将模块作为脚本执行

if __name__ == "__main__":
    print("Sales started")