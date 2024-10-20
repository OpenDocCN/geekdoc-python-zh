# Python 中的运算符重载

> 原文：<https://www.askpython.com/python/operator-overloading-in-python>

操作符过载是一种现象，它赋予操作符执行的操作超出其预定义操作功能的替代/不同含义。运算符重载也称为**运算符特别多态性**。

Python 运算符适用于内置类。但是同一个运算符用不同的类型表示是不同的。例如，`+`运算符将对两个数字执行算术加法，合并两个列表并连接两个字符串。Python 允许同一个操作符根据引用的上下文有不同的含义。

* * *

## 示例:描述基本算术运算符的不同用法

```py
# Program to show use of 
# + operator for different purposes. 

print(5 + 5) 

# concatenate two strings 

print("Safa"+"Mulani")  

# Product two numbers 
print(10 * 10) 

# Repeat the String 
print("Safa"*4) 

```

**输出:**

```py
10
SafaMulani
100
SafaSafaSafaSafa
```

* * *

## 如何在 python 中霸王一个运算符？

为了执行操作符重载，Python 提供了一些特殊的函数或*神奇函数*，当它与特定的操作符相关联时会被自动调用。例如，当我们使用+操作符时，神奇的方法`__add__`被自动调用，其中定义了+操作符的操作。

* * *

## Python 中的特殊函数

以双下划线 __ 开头的全局函数在 Python 中被称为特殊函数。是因为他们不一般。我们通常定义的类似于构造函数的 __init__()函数就是其中之一。每次我们创建该类的新对象时都会调用它。

* * *

### Python 中二元运算符的神奇方法

| 操作员 | 魔法方法 |
| **+** | __ 添加 _ _(自己，其他) |
| **–** | __sub__(自己，其他) |
| ***** | __mul__(自己，其他) |
| **/** | __truediv__(自己，其他) |
| **//** | __floordiv__(自己，其他) |
| **%** | __mod__(自己，其他) |
| ****** | __pow__(自己，其他) |

* * *

### Python 中比较运算符的神奇方法

| 操作员 | 魔法方法 |
| **<** | __lt__(自己，其他) |
| **>** | __gt__(自己，其他) |
| **< =** | __le__(自己，其他) |
| **> =** | __ge__(自己，其他) |
| **==** | __eq__(自己，他人) |
| **！=** | __ne__(自己，其他) |

* * *

### Python 中赋值运算符的神奇方法

| 操作员 | 魔法方法 |
| **-=** | __isub__(自己，其他) |
| **+=** | __iadd__(自己，其他) |
| ***=** | __imul__(自己，其他) |
| **/=** | __idiv__(自己，其他) |
| **//=** | __ifloordiv__(自己，其他) |
| **%=** | __imod__(自己，其他) |
| ****=** | __ipow__(自己，其他) |

* * *

### 一元运算符的神奇方法

| 操作员 | 魔法方法 |
| **–** | __neg__(自己，其他) |
| **+** | __pos__(自己，其他) |
| **~** | __invert__(自身，其他) |

* * *

## 示例:在 Python 中重载二进制+运算符

当我们使用+操作符时，神奇的方法`__add__`被自动调用，其中定义了+操作符的操作。因此，通过改变魔术方法的代码，我们可以赋予+运算符另一种含义。

```py

# Program to overload an binary + operator 

class X: 
    def __init__(self, x): 
        self.x = x 

    # adding two objects  
    def __add__(self, y): 
        return self.x + y.x 
ob1 = X(5) 
ob2 = X(5) 
ob3 = X("Safa") 
ob4 = X("Mulani") 

print(ob1 + ob2) # simple addition of objects
print(ob3 + ob4) # concatenation of strings through object addition

```

**输出**:

```py
10
SafaMulani
```

* * *

## 示例:在 Python 中重载比较运算符

```py
class X: 
    def __init__(self, x): 
        self.x = x 
    def __lt__(self, other): # Overloading < operator
        if(self.x<other.x): 
            return "ob1 is less than ob2"
        else: 
            return "ob2 is less than ob1"
    def __eq__(self, other): 
        if(self.x == other.x): # Overloading == operator
            return "Both are equal"
        else: 
            return "Not equal"

ob1 = X(2) 
ob2 = X(3) 
print(ob1 < ob2) 

ob3 = X(4) 
ob4 = X(4) 
print(ob1 == ob2) 

```

**输出**:

```py
ob1 is less than ob2
Not equal
```

* * *

## 示例:运算符重载程序示例

```py
class Animal:

    def __init__(self, age):
        self.__age = age

    def setage(self, age):
        self.__age = age

    def getage(self):
        return self.__age

    def __add__(self, predict):
        return Animal( self.__age + predict.__age )

    def __gt__(self, predict):
        return self.__age > predict.__age

    def __lt__(self, predict):
        return self.__age < predict.__age

    def __str__(self):
        return "Animal with original age " + str(self.__age)

c1 = Animal(5)
print(c1.getage())

c2 = Animal(5)
print(c2.getage())

c3 = c1 + c2
print(c3.getage())

print( c3 > c2) 

print( c1 < c2) 

print(c3) 

```

**输出**:

```py
5                                                                                                                                             
5                                                                                                                                             
10                                                                                                                                            
True                                                                                                                                          
False                                                                                                                                         
Animal with original age 10      
```

* * *

## 参考

*   Python 运算符重载
*   Python 比较运算符