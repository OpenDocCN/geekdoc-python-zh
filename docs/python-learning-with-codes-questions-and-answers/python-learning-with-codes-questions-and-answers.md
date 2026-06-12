

## 技术专业人士

第一版
2023

## 带代码的Python学习：问题与解答

## 带代码的Python学习

问题与解答

第一版 2023

作者：技术专业人士

当while循环执行完毕后，变量i的值将是多少？

```
i=0
while i != 0:
    i=i-1
else:
    i=i+1
```

- A. 1
- B. 0
- C. 2
- D. 变量变得不可用

选择关于组合的正确陈述。

- A. 组合通过添加新组件和修改现有组件来扩展类的功能
- B. 组合允许一个类被设计为不同类的容器
- C. 组合是促进代码重用的概念，而继承促进封装
- D. 组合基于“has a”关系，因此不能与继承一起使用

**正确答案：** *B*

分析以下代码片段并选择最能描述它的陈述。

```
class OwnMath:
    pass

def calculate_value(numerator, denominator):
    try:
        value = numerator / denominator
    except ZeroDivisionError as e:
        raise OwnMath from e
    return value

calculate_value(4, 0)
```

- A. 该代码是隐式链接异常的示例。
- B. 该代码有错误，因为OwnMath类没有继承任何异常类型类
- C. 该代码没有问题，脚本执行不会被任何异常中断。
- D. 该代码是显式链接异常的示例

**正确答案：** A

分析以下代码片段并选择最能描述它的陈述。

```
class Sword:
    var1 = 'weapon'

    def __init__(self):
        self.name = 'Excalibur'
```

- A. self.name是类变量的名称
- B. var1是全局变量的名称
- C. Excalibur是传递给实例变量的值
- D. weapon是传递给实例变量的值

**正确答案：** C

以下代码片段代表了面向对象编程的支柱之一。是哪一个？

```
class A:
    def run(self):
        print("A is running")

class B:
    def fly(self):
        print("B is flying")

class C:
    def run(self):
        print("C is running")

for element in A(), B(), C():
    element.run()
```

- A. 序列化
- B. 继承
- C. 封装
- D. 多态

**正确答案：** D

分析以下函数并选择最能描述它的陈述。

```
def my_decorator(coating):
    def level1_wrapper(my_function):
        def level2_wrapper(*args):
            our_function(*args)
        return level2_wrapper
    return level1_wrapper
```

- A. 这是一个接受自身参数的装饰器示例。
- B. 这是一个装饰器堆叠的示例。
- C. 这是一个可能触发无限递归的装饰器示例
- D. 该函数有错误。

**正确答案：** A

分析以下代码片段并选择最能描述它的陈述。

```
def fl(*arg, **args):
    pass
```

- A. 尽管函数参数的名称不符合命名约定，但该代码在语法上是正确的
- B. *arg参数保存一个未命名参数的列表。
- C. 该代码缺少未命名参数的占位符。
- D. 该代码在语法上不正确 - 函数应定义为 def f1 (*args, **kwargs):

**正确答案：** D

分析以下代码片段并判断代码是否正确和/或哪个方法应被标识为类方法。

```python
class Crossword:
    number_of_Crosswords = 0

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.progress = 0

    @staticmethod
    def isElementCorrect(word):
        if self.isSolved():
            print('The crossword is already solved')
            return True
        result = True
        for char in word:
            if char.isdigit():
                result = False
                break
        return result

    def isSolved(self):
        if self.progress == 100:
            return True

    def getNumberOfCrosswords(cls):
        return cls.number_of_Crosswords
```

- A. 只有一个初始化器，所以不需要类方法
- B. getNumberOfCrosswords()方法应该用@classmethod装饰
- C. 代码有错误
- D. getNumberOfCrosswords()和isSolved方法应该用@classmethod装饰

**正确答案：** *B*

分析代码并选择最能描述它的陈述。

```
class Item:
    def __init__(self, initial_value):
        self.value = initial_value

    def __ne__(self, other):
        ...
```

- A. __ne__()不是内置的特殊方法。
- B. 代码有错误
- C. 该代码负责支持否定运算符，例如 a = - a
- D. 该代码负责支持不等运算符，即 !=

**正确答案：** *D*

能够执行位移的运算符编码为（选择两个。）

- A. --
- B. ++
- C. <<
- D. >>

**正确答案：** *CD*

当以下循环执行完毕后，变量i的值将是多少？

```
for i in range(10):
    pass
```

- A. 10
- B. 变量变得不可用
- C. 11
- D. 9

**正确答案：** *B*

以下表达式 -

```
1+-2
```

- A. 等于1
- B. 无效
- C. 等于2
- D. 等于-1

**正确答案：** *D*

编译器是设计用来（选择两个。）的程序。

- A. 重新排列源代码使其更清晰
- B. 检查源代码以查看其是否正确
- C. 执行源代码
- D. 将源代码翻译成机器码

**正确答案：** *BD*

以下代码片段的输出是什么？

```
a= 'ant'
b= "bat"
c= 'camel'
print(a, b, c, sep="")
```

- A. ant' bat' camel
- B. antλ€batλ€ camel
- C. antbatcamel
- D. print (a, b, c, sep= ' λ€ ')

**正确答案：** B

以下代码片段的预期输出是什么？

```
i=5
while i>0:
    i=i //2
    if i % 2==0:
        break
else:
    i+=1
print(i)
```

- A. 代码有错误
- B. 3
- C. 7
- D. 15

**正确答案：** *A*

以下代码片段输出多少行？

```
for i in range(1, 3):
    print("*", end="")
else:
    print("*")
```

- A. 三行
- B. 一行
- C. 两行
- D. 四行

**正确答案：** B

以下哪些字面量反映了给定的值34.23？（选择两个。）

- A. .3423e2
- B. 3423e-2
- C. .3423e-2
- D. 3423e2

**正确答案：** *AB*

以下代码片段的预期输出是什么？

```
a=2
if a>0:
    a+=1
else:
    a-=1
print(a)
```

- A. 3
- B. 1
- C. 2
- D. 代码有错误

**正确答案：** A

假设以下代码片段已成功执行，以下哪些等式为真？（选择两个。）

```
a= [1]
b=a
a[0] = 0
```

- A. len(a) == len(b)
- B. b[0] +1 ==a[0]
- C. a[0] == b[0]
- D. a[0] + 1 ==b[0]

**正确答案：** *AC*

假设以下代码片段已成功执行，以下哪些等式为假？（选择两个。）

- A. len(a)== len(b)
- B. a[0]-1 ==b[0]
- C. a[0]== b[0]
- D. b[0] - 1 ==a[0]

**正确答案：** *AB*

以下哪些陈述是正确的？（选择两个。）

- A. Python字符串实际上是列表
- B. Python字符串可以连接
- C. Python字符串可以像列表一样切片
- D. Python字符串是可变的

**正确答案：** *BC*

以下哪些句子是正确的？（选择两个。）

- A. 列表不能存储在元组内
- B. 元组可以存储在列表内
- C. 元组不能存储在元组内
- D. 列表可以存储在列表内

**正确答案：** *BD*

假设字符串长度为六个或更多字母，以下切片

```
string[1:-2]
```

比原始字符串短：

- A. 四个字符
- B. 三个字符
- C. 一个字符
- D. 两个字符

**正确答案：** *B*

以下代码片段的预期输出是什么？

```
lst = [1,2,3,4]
lst = lst[-3:-2]
lst = lst[-1]
print(lst)
```

- A. 1
- B. 4
- C. 2
- D. 3

**正确答案：C**

以下哪些陈述是正确的？（选择两项。）

- A. Python字符串实际上是列表
- B. Python字符串可以连接
- C. Python字符串可以像列表一样切片
- D. Python字符串是可变的

**正确答案：BC**

以下哪些句子是正确的？（选择两项。）

- A. 列表不能存储在元组内
- B. 元组可以存储在列表内
- C. 元组不能存储在元组内
- D. 列表可以存储在列表内

**正确答案：BD**

以下代码片段的预期输出是什么？

```
s = 'abc'
for i in len(s):
    s[i] = s[i].upper()
print(s)
```

- A. abc
- B. 代码将导致运行时异常
- C. ABC
- D. 123

**正确答案：B**

执行以下代码片段后，list2列表将包含多少个元素？

```
list1 = [False for i in range(1, 10)]
list2 = list1[-1:1:-1]
```

- A. 零个
- B. 五个
- C. 七个
- D. 三个

**正确答案：C**

如果你想检查一个名为dict的字典中是否存在某个'key'，你会用什么代替XXX？（选择两项。）

```
if XXX:
    print('Key exists')
```

- A. 'key' in dict
- B. dict['key'] != None
- C. dict.exists('key')
- D. 'key' in dict.keys()

**正确答案：BD**

你需要一个可以充当简单电话簿的数据。你可以使用以下子句来获取它（选择两项。）（假设之前没有创建其他项目）

- A. dir={'Mom': 5551234567, 'Dad': 5557654321}
- B. dir={'Mom': '5551234567', 'Dad': '5557654321'}
- C. dir={Mom: 5551234567, Dad: 5557654321}
- D. dir={Mom: '5551234567', Dad: '5557654321'}

**正确答案：CD**

如果你不喜欢像下面这样长的包路径，你可以用什么代替？

```
import alpha.beta.gamma.delta.epsilon.zeta
```

- A. 你可以使用alias关键字为名称创建别名
- B. 无能为力，你需要接受它
- C. 你可以将其缩短为alpha.zeta，Python会找到正确的连接
- D. 你可以使用as关键字为名称创建别名

**正确答案：D**

以下代码的预期输出是什么？

```
str = 'abcdef'
def fun(s):
    del s[2]
    return s

print(fun(str))
```

- A. abcef
- B. 程序将导致运行时异常/错误
- C. acdef
- D. abdef

**正确答案：B**

以下代码的预期输出是什么？

```
def f(n):
    if n == 1:
        return '1'
    return str(n) + f(n-1)

print(f(2))
```

- A. 21
- B. 2
- C. 3
- D. 12

**正确答案：A**

以下代码片段的预期行为是什么？

```
def x():           # 第01行
    return 2       # 第02行

x = 1 + x()        # 第03行
print(x)           # 第04行
```

它将：

- A. 在第02行导致运行时异常
- B. 在第01行导致运行时异常
- C. 在第03行导致运行时异常
- D. 打印3

**正确答案：D**

以下代码的预期行为是什么？

```
def f(n):
    for i in range(1, n+1):
        yield i

print(f(2))
```

它将：

- A. 打印4321
- B. 打印 <generator object f at (some hex digits)>
- C. 导致运行时异常
- D. 打印1234

**正确答案：B**

如果你需要一个什么都不做的函数，你会用什么代替XXX？（选择两项。）

```
def idler():
    XXX
```

- A. pass
- B. return
- C. exit
- D. None

**正确答案：AD**

是否可以安全地检查一个类/对象是否具有某个属性？

- A. 可以，使用hasattr属性
- B. 可以，使用hasattr()方法
- C. 可以，使用hassattr()函数
- D. 不可以，这是不可能的

**正确答案：B**

每个方法的第一个参数：

- A. 持有对当前处理对象的引用
- B. 总是设置为None
- C. 设置为一个唯一的随机值
- D. 由第一个参数的值设置

**正确答案：D**

Python中最简单的类定义可以表示为：

- A. class X:
- B. class X: pass
- C. class X: return
- D. class X: { }

**正确答案：A**

如果你想访问异常对象的组件并将它们存储在一个名为e的对象中，你必须使用以下形式的异常语句：

- A. except Exception(e):
- B. except e = Exception:
- C. except Exception as e:
- D. 在Python中不可能执行这样的操作

**正确答案：C**

在每个对象中单独存储的变量称为：

- A. 没有这样的变量，所有变量都在对象之间共享
- B. 类变量
- C. 对象变量
- D. 实例变量

**正确答案：D**

有一个名为s的流已打开用于写入。你将选择哪个选项向流中写入一行？

- A. s.write('Hello\n')
- B. write(s, 'Hello\n')
- C. s.writeln('Hello\n')
- D. s.writeline('Hello\n')

**正确答案：A**

你打算从一个名为s的流中读取一个字符。你会使用哪个语句？

- A. ch = read(s, 1)
- B. ch = s.input(1)
- C. ch = input(s, 1)
- D. ch = s.read(1)

**正确答案：D**

从以下语句中你可以推断出什么？（选择两项。）

```
str = open('file.txt', 'rt')
```

- A. str是从名为file.txt的文件中读入的字符串
- B. 在读取期间将执行换行符转换
- C. 如果file.txt不存在，它将被创建
- D. 打开的文件不能使用str变量进行写入

**正确答案：AD**

给定以下类层次结构。代码的预期输出是什么？

```
class A:
    def a(self):
        print("A", end=' ')
    def b(self):
        self.a()

class B(A):
    def a(self):
        print("B", end=' ')
    def do(self):
        self.b()

class C(A):
    def a(self):
        print("C", end=' ')
    def do(self):
        self.b()

B().do()
C().do()
```

- A. BB
- B. CC
- C. AA
- D. BC

**正确答案：D**

Python的内置函数open()尝试打开一个文件并返回：

- A. 一个标识已打开文件的整数值
- B. 一个错误代码（0表示成功）
- C. 一个流对象
- D. 总是None

**正确答案：C**

以下哪些单词可以用作变量名？（选择两项。）

- A. for
- B. True
- C. true
- D. For

**正确答案：CD**

Python字符串可以使用运算符`粘合`在一起：

- A. .
- B. &
- C. _
- D. +

**正确答案：D**

一个关键字（选择两项。）

- A. 可以用作标识符
- B. 由Python的词法定义
- C. 也称为保留字
- D. 不能在用户的代码中使用

**正确答案：BC**

代码片段打印了多少个星号(*)？

```
s = '*****'
s = s - s[2]
print(s)
```

- A. 代码有误
- B. 五个
- C. 四个
- D. 两个

**正确答案：A**

哪一行可以代替注释，使代码片段产生以下预期输出？（选择两项。）
预期输出：
1 2 3
代码：

```
c, b, a = 1, 3, 2
# 在此处放置一行
print(a, b, c)
```

- A. c, b, a = b, a, c
- B. c, b, a = a, c, b
- C. a, b, c = c, a, b
- D. a, b, c = a, b, c

**正确答案：AC**

假设V变量持有整数值2，以下哪个运算符应该代替OPER，使表达式等于1？

V OPER 1 -

- A. <<<
- B. >>>
- C. >>
- D. <<

**正确答案：C**

以下代码片段打印了多少个星号(*)？

```
i = 3
while i > 0:
    i -= 1
    print("*")
else:
    print("*")
```

- A. 代码有误
- B. 五个
- C. 三个
- D. 四个

**正确答案：D**

UNICODE是：

- A. 操作系统的名称
- B. 用于编码和处理文本的标准
- C. 编程语言的名称
- D. 文本处理器的名称

**正确答案：B**

以下代码片段的预期输出是什么？

```
s = '* - *'
s = 2 * s + s * 2
print(s)
```

## 什么是静态方法？

- A. 作用于类本身的方法
- B. 使用 `@method` 特性装饰的方法
- C. 一个不需要引用类本身作为参数的方法
- D. 作用于已实例化的类对象的方法

**正确答案：C**

在面向对象编程的意义上，关于类型，以下哪项是正确的？

- A. 它是任何对象都可以继承的最底层类型。
- B. 它是一种内置方法，允许枚举复合对象。
- C. 它是任何类都可以继承的最顶层类型。
- D. 它是用于实例化类的对象。

**正确答案：C**

术语“反序列化”是什么意思？（选择最佳答案。）

- A. 它是基于字节序列创建 Python 对象的过程。
- B. 它是为每个新创建的 Python 对象分配唯一标识符的过程。
- C. 它是数据传输过程的另一个名称。
- D. 它是将对象结构转换为字节流的过程。

**正确答案：D**

什么是 `__traceback__`？（选择两项。）

- A. 每个异常对象都拥有的一个属性。
- B. 由 traceback 模块提供的特殊方法，用于检索描述回溯的完整字符串列表。
- C. 当导入 traceback 模块时，添加到每个对象的一个属性。
- D. 一个保存有用信息的属性，当程序员希望将异常细节存储在其他对象中时特别有用。

**正确答案：AD**

以下哪些使用换行和不同缩进方法的示例符合 PEP 8 的建议？（选择两项。）

A.

spam = my_function(arg_one, arg_two,
                  arg_three, arg_four)

B.

eggs = (1, 2, 3,
        4, 5, 6)

C.

my_list = [
    1, 2, 3,
    4, 5, 6,
]

D.

foo = my_function(
    arg_one, arg_two,
    arg_three, arg_four
)

## 正确答案：CD

查看以下 Python 中注释和文档字符串的示例。选择那些有用且符合 PEP 8 建议的示例。（选择两个。）

A.

def area_price(area, price=1.25):
    """Calculate the area in square meters.
    Keyword arguments:
    area -- the land area of the slot
    price -- price per sq/m (default 1.25)"""
    ...

B.

def area_price(area, price=2.25):
    """Calculate the area in square meters.

    Keyword arguments:
    area -- the land area of the slot
    price -- price per sq/m (default 2.25)"""
    ...

# 该示例展示了如何创建
# 一个包含两个元素的列表，并将
# 列表内容打印到屏幕上。

my_list = [a, b]
print(my_list)

- D. price = price + 1 # 将价格减少一以补偿损失。

**正确答案：AB**

选择与 PEP 257 相关的正确陈述。

- A. 紧接在另一个文档字符串之后出现的字符串字面量称为属性文档字符串
- B. 属性文档字符串和附加文档字符串是两种可以被软件工具提取的额外文档字符串类型
- C. 出现在模块、函数或类定义的第一个语句之外的字符串字面量可以充当文档。它们被 Python 字节码编译器识别，并且可以作为运行时对象属性访问。
- D. 紧接在模块顶层简单赋值之后出现的字符串字面量称为补充文档字符串

**正确答案：C**

选择与 PEP 8 代码编写编程建议相关的正确陈述。（选择两个。）

- A. 你应该使用 `not ... is` 运算符（例如 `if not spam is None:`），而不是 `is not` 运算符（例如 `if spam is not None:`），以提高可读性。
- B. 你应该使用 `isinstance()` 方法（例如 `if isinstance(obj, int):`）进行对象类型比较，而不是直接比较类型（例如 `if type(obj) is type(1)`）。
- C. 你应该以有利于 CPython 实现而非 PyPy、Cython 和 Jython 的方式编写代码。
- D. 你不应该编写依赖于重要尾随空格的字符串字面量，因为它们在视觉上可能难以区分，并且某些编辑器可能会修剪它们。

**正确答案：BD**