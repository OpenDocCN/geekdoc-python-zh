

# Python 解决方案

第5版

Yashavant Kanetkar
Aditya Kanetkar

www.bpbonline.com

第一版 2020
第五次修订与更新版 2023
版权所有 © BPB Publications, India
ISBN: 978-93-5551-184-3

保留所有权利。未经出版商事先书面许可，不得以任何形式或任何方式将本出版物的任何部分存储在检索系统中或进行复制。

## 责任限制与免责声明

本书的作者和出版商已尽最大努力确保书中描述的程序、过程和函数是正确的。但是，作者和出版商对这些程序或书中包含的文档不作任何明示或暗示的保证。在任何情况下，作者和出版商均不对因提供、执行或使用这些程序、过程和函数而引起的或与之相关的任何附带或间接损害负责。提及的产品名称仅用于识别目的，可能是其各自公司的商标。

书中提及的所有商标均被确认为其各自所有者的财产。

## 分销商：

**BPB PUBLICATIONS**
20, Ansari Road, Darya Ganj
New Delhi-110002
电话: 23254990/ 23254991

**DECCAN AGENCIES**
4-3-329, Bank Street,
Hyderabad-500195
电话: 24756967 / 24756400

**MICRO MEDIA**
Shop No. 5, Mahendra Chambers,
150 DN Rd. Next to Capital Cinema,
V.T. (C.S.T.) Station, MUMBAI-400 001
电话: 22078296 / 22078297

**BPB BOOK CENTRE**
376 Old Lajpat Rai Market,
Delhi-110006
电话: 23861747

由 Manish Jain 为 BPB Publications 出版，地址：20 Ansari Road, Darya Ganj, New Delhi-110002，印刷于 Akash Press, New Delhi
www.bpbonline.com

谨以此书献给
Nalinee & Prabhakar Kanetkar...

## 关于 Yashavant Kanetkar

通过他的书籍和关于 C、C++、数据结构、VC++、.NET 等的在线 Quest 视频课程，Yashavant Kanetkar 在过去的二十五年里创造、塑造和培养了数以万计的 IT 职业生涯。Yashavant 的书籍和在线课程为印度和海外顶尖 IT 人才的培养做出了重大贡献。

Yashavant 的书籍在全球范围内得到认可，数百万学生/专业人士从中受益。他的书籍已被翻译成印地语、古吉拉特语、日语、韩语和中文。他的许多书籍在印度、美国、日本、新加坡、韩国和中国出版。

Yashavant 是 IT 领域备受追捧的演讲者，曾在 TedEx、IITs、NITs、IIITs 和全球软件公司举办研讨会/讲习班。

Yashavant 因其在创业、专业和学术方面的卓越成就，被印度理工学院坎普尔分校授予享有盛誉的“杰出校友奖”。该奖项授予了印度理工学院坎普尔分校过去50年中对其专业和社会进步做出重大贡献的50名校友。

为表彰他对印度 IT 教育的巨大贡献，他连续5年被微软授予“最佳 .NET 技术贡献者”和“最有价值专业人士”奖项。

Yashavant 持有孟买 VJTI 的工程学士学位和印度理工学院坎普尔分校的工程硕士学位。他目前的职务包括 KICIT Pvt. Ltd. 的董事和班加罗尔 IIIT 的兼职教员。可以通过 kanetkar@kicit.com 或 http://www.kicit.com 联系他。

## 关于 Aditya Kanetkar

Aditya Kanetkar 目前在微软印度开发中心（位于班加罗尔）担任软件工程师。他在软件开发行业拥有6年的工作经验。他目前的兴趣是任何与 Python、机器学习、分布式系统、云计算、容器和 C# 相关的事物。

此前，他曾在位于华盛顿州雷德蒙德的微软总部和位于加利福尼亚州红木城的甲骨文总部工作。Aditya 持有印度理工学院古瓦哈提分校的计算机科学与工程学士学位和佐治亚理工学院（亚特兰大）的计算机科学硕士学位。

可以通过 http://www.kicit.com 联系 Aditya。

## 目录

1. Python 简介
2. 入门
3. Python 基础
4. 字符串
5. 判断控制指令
6. 循环控制指令
7. 控制台输入/输出
8. 列表
9. 元组
10. 集合
11. 字典
12. 推导式
13. 函数
14. 递归
15. 函数式编程
16. 模块和包
17. 命名空间
18. 类和对象
19. 类和对象的复杂性
20. 容器和继承
21. 迭代器和生成器
22. 异常处理
23. 文件输入/输出
24. 杂项
25. 并发与并行
26. 同步
27. Numpy 库
28. 定期测试

## 1 Python 简介

### [A] 回答以下问题：

(a) 列举 Python 广泛应用的5个领域。

答案

- 系统编程
- 游戏编程
- 机器人编程
- 快速原型开发
- 互联网脚本

(b) 事件驱动编程主要应用于哪里？

答案

事件驱动编程主要用于创建包含窗口、复选框、按钮、组合框、滚动条、菜单等元素的 GUI 应用程序。当我们通过鼠标/键盘/触摸与这些 GUI 元素交互时，会发生一个事件，并调用一个函数来处理该事件。

(c) 为什么 Python 被称为可移植语言？

答案

我们可以在一个平台上创建和测试代码，然后在任何其他平台上运行它。这使得 Python 成为一种可移植的语言。

(d) 本章讨论的不同编程模型中，最重要的单一特征是什么？

答案

函数式编程模型 - 它将问题分解为一组函数。

过程式编程模型 - 它通过一次实现一个语句（过程）来解决问题。因此，它包含按特定顺序执行的明确步骤。它也使用函数，但这些不是函数式编程中使用的数学函数。函数式编程侧重于表达式，而过程式编程侧重于语句。

面向对象的编程模型 - 它通过在计算机内部创建一个对象的微型世界来模拟现实世界。

事件驱动编程模型 - 当我们与不同的 GUI 元素（如窗口、复选框、按钮、组合框、滚动条、菜单等）交互时，它会生成事件。每个事件都通过调用事件处理函数来处理。

(e) 以下哪项不是 Python 的特性？

- 静态类型
- 使用前声明变量
- 通过错误号码进行运行时错误处理
- 对列表、字典、元组等容器的库支持

答案

- 静态类型
- 使用前声明变量
- 通过错误号码进行运行时错误处理

(f) 为以下每种编程模型给出一个示例应用：

- 函数式模型
- 过程式模型
- 面向对象模型
- 事件驱动模型

答案

- 函数式模型：求数字的阶乘值。
- 过程式模型：对一组数字进行排序的逐步过程。
- 面向对象模型：客户、产品、订单等对象之间的交互。
- 事件驱动模型：一个 GUI 应用程序，鼠标左键点击时显示 "Hi"，右键点击时显示 "Hello"。

### [B] 判断以下陈述是对还是错：

(a) Python 可以免费使用和分发。

答案

对

(b) 相同的 Python 程序可以在不同的操作系统-微处理器组合上运行。

答案

对

(c) 可以在 Python 程序中使用 C++ 或 Java 库。

答案

对

(d) 在 Python 中，变量的类型是根据其使用方式决定的。

答案

对

(e) Python 不能用于构建 GUI 应用程序。

答案

错

(f) Python 支持函数式、过程式、面向对象和事件驱动编程模型。

答案

对

(g) GUI 应用程序基于事件驱动编程模型。

答案

对

(h) 函数式编程模型由多个对象的交互组成。

答案

错

### [C] 将以下内容匹配：

a. 函数式编程
b. 事件驱动编程
c. 过程式编程

1. 基于 GUI 元素的交互
2. 对象的交互
3. 语句

## 第一章：Python 简介

d. 面向对象编程

4. 类数学函数

答案

- 函数式编程 - 类数学函数
- 事件驱动编程 - 基于图形用户界面元素的交互
- 过程式编程 - 语句
- 面向对象编程 - 对象之间的交互

[D] 填空：

(a) 函数式编程范式也称为<u>声明式</u>编程模型。
(b) 过程式编程范式也称为<u>命令式</u>编程模型。
(c) Python 由<u>吉多·范罗苏姆</u>创建。
(d) Python 程序员通常被称为<u>Pythonists 或 Pythonistas</u>。

## 第二章：入门

![](img/ac61a02854da7ea89272a34be93604dc_14_0.png)

[A] 回答以下问题：

(a) 提示符 C:\>、$ 和 >>> 分别表示什么？

答案

>>> 表示 Python 交互式解释器提示符。

(b) IDLE 可以在哪两种模式下使用？

答案

交互模式和脚本模式是 IDLE 中使用的两种模式。

(c) IDLE 提供的两种编程模式的目的是什么？

答案

交互模式用于探索 Python 语法、寻求帮助和调试短程序。脚本模式用于编写完整的 Python 程序。

(d) 如何在 Python 程序中使用第三方库？

答案

2

[B] 将以下配对项匹配起来：

- a. pip
- b. Jupyter
- c. Spyder
- d. PyPI
- e. NumPy
- f. SciPy
- g. Pandas
- h. MatPlotLib
- i. OpenCV

- 1. 高级数学运算
- 2. 科学计算
- 3. 操作数值表格
- 4. 可视化
- 5. 计算机视觉
- 6. 包安装工具
- 7. 构建和记录应用程序
- 8. 科学库
- 9. Python 包索引

答案

- a. pip - 6. 包安装工具
- b. Jupyter - 7. 构建和记录应用程序
- c. Spyder - 3. 高级数学运算
- d. PyPI - 9. Python 包索引
- e. NumPy - 8. 科学库
- f. SciPy - 2. 科学计算
- g. Pandas - 3. 操作数值表格
- h. MatPlotLib - 4. 可视化
- i. OpenCV - 5. 计算机视觉

[C] 判断以下陈述的正误：

(a) Python 是一个规范，可以通过 Python、C#、Java 等语言实现。

答案

正确

(b) CPython 是用 C 语言编写的 Python 规范的实现。

答案

正确

(c) Python 程序首先被编译成字节码，然后被解释执行。

答案

正确

(d) 大多数 Linux 发行版已经包含了 Python。

答案

错误

(e) Windows 系统不包含 Python，需要单独安装。

答案

正确

(f) Python 程序可以使用 IDLE、NetBeans、PyCharm 和 Visual Studio Code 构建。

答案

正确

(g) 第三方 Python 包通过 PyPI 分发。

答案

正确

## 第三章：Python 基础

![](img/ac61a02854da7ea89272a34be93604dc_18_0.png)

### 12 Let Us Python 解答

[A] 回答以下问题：

(a) 编写一个程序，交换变量 **a** 和 **b** 的值。不允许使用第三个变量。不允许对 **a** 和 **b** 进行算术运算。

程序

```
#### 交换两个变量的值
a = 5
b = 10
a, b = b, a
print('a =', a)
print('b =', b)
```

输出

a = 10
b = 5

(b) 编写一个程序，使用 math 模块中可用的三角函数。

程序

```
#### 使用三角函数
import math
a = math.pi / 6
print('The value of sine of pi / 6 is', end = ' ')
print(math.sin(a))
print('The value of cosine of pi / 6 is', end = ' ')
print(math.cos(a))
```

输出

The value of sine of pi / 6 is 0.49999999999999994
The value of cosine of pi / 6 is 0.8660254037844387

(c) 编写一个程序，在 10 到 50 的范围内生成 5 个随机数。使用种子值 6。通过将其与执行时间关联，提供每次执行程序时更改此种子值的功能。

程序

```
#### 生成随机数
import random
import time

random.seed(6)
for i in range(5) :
    print(random.randint(10, 50))

print( )
t = int(time.time( ))
random.seed(t)
for i in range(5) :
    print(random.randint(10, 50))
```

输出

46
15
41
26
12

39
36
21
13
18

(d) 对数字 -2.8、-0.5、0.2、1.5 和 2.9 使用 **trunc( )**、**floor( )** 和 **ceil( )**，以清楚地理解这些函数之间的区别。

程序

```
#### 使用 trunc( )、ceil( ) 函数
import math
print(math.floor(-2.8))
print(math.trunc(-2.8))
print(math.ceil(-2.8))
print(math.floor(-0.5))
print(math.trunc(-0.5))
print(math.ceil(-0.5))
print(math.floor(0.2))
print(math.trunc(0.2))
print(math.ceil(0.2))
print(math.floor(1.5))
print(math.trunc(1.5))
print(math.ceil(1.5))
print(math.floor(2.9))
print(math.trunc(2.9))
print(math.ceil(2.9))
```

输出

-3
-2
-2
-1
0
0
0
0
1
1
1
2
2
2
3

(e) 假设一个城市温度的华氏度值。编写一个程序将此温度转换为摄氏度，并打印两种温度。

程序

```
farh = 212
cen = ((farh - 32) * 5 / 9)
print(farh, cen)
```

输出

212 100.0

(f) 给定三角形的三条边 a、b、c，编写一个程序以获取并打印三个角的值，四舍五入到下一个整数。使用以下公式：

a² = b² + c² - 2bc cos A, b² = a² + c² - 2ac cos B, c² = a² + b² - 2ab cos C

程序

```
import math
a, b, c = 3, 4, 5
angleA = (math.acos((b * b + c * c - a * a ) / ( 2 * b * c )) * 180) / 3.14
print(angleA)
angleB = (math.acos((a * a + c * c - b * b ) / ( 2 * a * c )) * 180) / 3.14
print(angleB)
angleC = (math.acos((a * a + b * b - c * c ) / ( 2 * a * b )) * 180) / 3.14
print(angleC)
```

输出

36.88859859324559
53.157050713468216
90.04564930671381

[B] 你将如何执行以下操作？

(a) 打印 2 + 3j 的虚部

答案

```
print(a.imag)
```

(b) 获取 4 + 2j 的共轭复数

答案

```
a = 4 + 2j
b = a.conjugate( )
```

(c) 打印二进制 '1100001110' 的十进制等价值

答案

```
print(int('1100001110', 2))
```

(d) 将浮点值 4.33 转换为数字字符串

答案

```
a = str(4.33)
```

(e) 计算 29 除以 5 的整数商和余数

答案

```
divmod(29, 5)
```

(f) 获取十进制数 34567 的十六进制等价值

答案

```
hex(34567)
```

(g) 将 45.6782 四舍五入到小数点后第二位

答案

```
a = round(45.6782, 2)
```

(h) 从 3.556 获取 4

答案

```
a = round(3.556)
```

(i) 从 16.7844 获取 17

答案

```
a = round(16.7844)
```

(j) 计算 3.45 除以 1.22 的余数

答案

```
a = 3.45 % 1.22
```

[C] 以下哪个是无效的变量名，为什么？

- BASICSALARY - 有效
- _basic - 有效
- basic-hra - 无效。不能包含特殊字符 -
- #MEAN - 无效。不能以 # 开头
- group. - 无效。不能以 . 结尾
- 422 - 无效。不能以数字开头
- pop in 2020 - 无效。不能包含空格
- over - 有效
- timemindovermatter - 有效
- SINGLE - 有效
- hELLO - 有效
- queue. - 无效。不能以 . 结尾
- team'svictory - 无效。不能包含特殊字符 '
- Plot # 3 - 无效。不能包含空格和特殊字符 #
- 2015_DDay - 无效。不能以数字开头

[D] 计算以下表达式的值：

(a) 2 ** 6 // 8 % 2

答案
= 64 // 8 % 2
= 8 % 2
= 0

(b) 9 ** 2 // 5 - 3

答案
= 81 // 5 - 3
= 16 - 3
= 13

(c) 10 + 6 - 2 % 3 + 7 - 2

答案
= 10 + 6 - 2 + 7 - 2
= 16 - 5 + 7 - 2
= 14 + 7 - 2
= 21 - 2
= 19

(d) 5 % 10 + 10 - 23 * 4 // 3

答案
= 5 + 10 - 23 * 4 // 3
= 5 + 10 - 92 // 3
= 5 + 10 - 30
= 15 - 30
= -15

(e) 5 + 5 // 5 - 5 * 5 ** 5 % 5

答案
= 5 + 5 // 5 - 5 * 3125 % 5
= 5 + 1 - 5 * 3125 % 5
= 5 + 1 - 15625 % 5
= 5 + 1 - 0
= 6

(f) 7 % 7 + 7 // 7 - 7 * 7

答案
= 0 + 7 // 7 - 7 * 7
= 0 + 1 - 7 * 7
= 0 + 1 - 49
= 1 - 49
= -48

[E] 计算以下表达式的值：

(a) min(2, 6, 8, 5)

答案
2

(b) bin(46)

答案
0b101110

(c) round(10.544336, 2)

答案
10.54

(d) math.hypot(6, 8)

答案
10

(e) math.modf(3.1415)

答案
0.14150000000000018, 3.0

[F] 将以下配对项匹配起来：

- a. complex
- b. 转义特殊字符
- c. Tuple
- d. 自然对数
- e. 常用对数

- 1. \n
- 2. 容器类型
- 3. 基本类型
- 4. log( )
- 5. log10( )

答案

- complex - 基本类型
- 转义特殊字符 - \n
- Tuple - 容器类型
- 自然对数 - log( )
- 常用对数 - log10( )

## 4 字符串

![](img/ac61a02854da7ea89272a34be93604dc_28_0.png)

**[A]** 回答以下问题：

(a) 编写一个程序，从字符串 'Shenanigan' 生成以下输出。

S h
a n
enanigan
Shenan
Shenan
Shenan
Shenan
Shenanigan
Seaia
Snin
Saa
ShenaniganType
ShenanWabbite

程序

```
#### 提取字符串子部分
s = 'Shenanigan'
print(s[0], s[1])
print(s[4], s[5])
print(s[2:])
print(s[:6])
print(s[:-4])
print(s[-10:-4])
print(s[0:6])
print(s[:])
print(s[0:10:2])
print(s[0:10:3])
print(s[0:10:4])
s = 'Shenanigan'
g = 'Type'
a = s + g
print(a)
s = 'Shenanigan'
t = 'Wabbite'
b = s[:6] + t
print(b)
```

输出

S h
a n
enanigan
Shenan
Shenan
Shenan
Shenan
Shenanigan
Seaia
Snin
Saa
ShenaniganType
ShenanWabbite

(b) 编写一个程序，将以下字符串

'Visit ykanetkar.com for online courses in programming'

转换为

'Visit Ykanetkar.com For Online Courses In Programming'

程序

```
#### 将字符串的每个单词首字母大写
s = 'Visit ykanetkar.com for online courses in programming'
t = ''
for w in s.split() :
    t = t + w.capitalize() + ' '
print(t)
```

输出

Visit Ykanetkar.com For Online Courses In Programming

(c) 编写一个程序，将以下字符串

'Light travels faster than sound. This is why some people appear bright until you hear them speak.'

转换为

'LIGHT travels faster than SOUND. This is why some people appear bright until you hear them speak.'

程序

```
#### 在字符串中搜索并替换
msg = 'Light travels faster than sound. This is why some people appear bright until you hear them speak.'
newmsg = msg.replace('Light', 'LIGHT').replace('sound', 'SOUND')
print(newmsg)
```

输出

LIGHT travels faster than SOUND. This is why some people appear bright until you hear them speak.

(d) 以下程序的输出是什么？

```
s = 'HumptyDumpty'
print('s = ', s)
print(s.isalpha())
print(s.isdigit())
print(s.isalnum())
print(s.islower())
print(s.isupper())
print(s.startswith('Hump'))
print(s.endswith('Dump'))
```

输出

s = HumptyDumpty
True
False
True
False
False
True
False

(e) 原始字符串（raw string）的目的是什么？

回答

Python 原始字符串通过在字符串字面量前加上 'r' 或 'R' 来创建。Python 原始字符串将反斜杠 (\) 视为普通字符。当我们想要一个包含反斜杠且不希望它被当作转义字符处理的字符串时，这很有用。

(f) 如果我们希望处理以下字符串中的单个单词，你将如何将它们分离出来：

'The difference between stupidity and genius is that genius has its limits'

程序

```
msg = 'The difference between stupidity and genius is that genius has its limits'
for word in msg.split( ) :
    print(word)
```

输出

The
difference
between
stupidity
and
genius
is
that
genius
has
its
limits

(g) 提及两种存储字符串的方法：He said, "Let Us Python"。

回答

```
s1 = "He said, \"Let Us Python\""
s2 = r'He said, "Let Us Python"'
```

(h) 以下代码片段的输出是什么？

```
print(id('Imaginary'))
print(type('Imaginary'))
```

回答

```
36339048
<class 'str'>
```

(i) 以下代码片段的输出是什么？

```
s3 = 'C:\Users\Kanetkar\Documents'
print(s3.split('\\'))
print(s3.partition('\\'))
```

回答

```
['C:', 'Users', 'Kanetkar', 'Documents']
('C:', '\\', 'Users\\Kanetkar\\Documents')
```

(j) Python 中的字符串是可迭代的、可切片的且不可变的。（正确/错误）

回答

正确

(k) 你将如何从字符串 'ThreadProperties' 中提取 'TraPoete'？

回答

```
s = 'ThreadProperties'
print(s[::2])
```

(l) 你将如何消除字符串两侧的空格

' Flanked by spaces on either side '？

回答

```
s = ' Flanked by spaces on either side '
print(s.strip())
```

(m) 以下代码片段的输出是什么？

```
s1 = s2 = s3 = "Hello"
print(id(s1), id(s2), id(s3))
```

回答

36330016 36330016 36330016

(n) 在以下代码片段中，**ch** 将存储什么值：

```
msg = 'Aeroplane'
ch = msg[-0]
```

回答

A

**[B]** 假设 msg = 'Keep yourself warm'，将以下内容匹配：

- a. msg.partition(' ')
- b. msg.split(' ')
- c. msg.startswith('Keep')
- d. msg.endswith('Keep')
- e. msg.swapcase( )
- f. msg.capitalize( )
- g. msg.count('e')
- h. len(msg)
- i. msg[0]
- j. msg[-1]
- k. msg[1:1:1]
- l. msg[-1:3]
- m. msg[:-3]
- n. msg[-3:]
- o. msg[0:-2]

- 1. 18
- 2. kEEP YOURSELF WARM
- 3. Keep yourself warm
- 4. 3
- 5. True
- 6. False
- 7. ['Keep', 'yourself', 'warm']
- 8. ('Keep', ' ', 'yourself warm')
- 9. Keep yourself w
- 10. keep yourself wa
- 11. K
- 12. 空字符串
- 13. m
- 14. arm
- 15. 空字符串

回答

msg.partition(' ') - ('Keep', ' ', 'yourself warm')
msg.split(' ') - ['Keep', 'yourself', 'warm']
msg.startswith('Keep') - True
msg.endswith('Keep') - False
msg.swapcase( ) - kEEP YOURSELF WARM
msg.capitalize( ) - Keep yourself warm
msg.count('e') - 3
len(msg) - 18
msg[0] - K
msg[-1] - m
msg[1:1:1] - 空字符串
msg[-1:3] - 空字符串
msg[:-3] - Keep yourself w
msg[-3:] - arm
msg[0:-2] - Keep yourself wa

**[C]** 给出一个示例字符串，使其与以下正则表达式匹配：

```
\w+
\d{2}
\w{1,}
\w{2,4}
A*B
\d+?
```

回答

01
smiling
1234
AAAAB
1 in 12345

## 5 判断控制指令

![](img/ac61a02854da7ea89272a34be93604dc_36_0.png)

**[A]** 回答以下问题：

(a) 为以下情况编写条件表达式：

- 如果 a < 10 则 b = 20，否则 b = 30
- 如果 time < 12 则打印 'Morning'，否则打印 'Afternoon'
- 如果 marks >= 70，则将 remarks 设置为 True，否则为 False

回答

```
b = 20 if a < 10 else 30
print('Morning') if time < 12 else print('Afternoon')
remarks = 'True' if marks >= 70 else 'False'
```

(b) 将以下代码片段重写为一行：

```
x = 3
y = 3.0
if x == y :
    print('x and y are equal')
else :
    print('x and y are not equal')
```

回答

```
x, y = 3, 3.0
print('x and y are equal') if x == y else print('x and y are not equal')
```

输出

x and y are equal

(c) 当执行 **pass** 语句时会发生什么？

回答

pass 语句是一个空操作指令，执行时不会发生任何事情。

**[B]** 以下程序的输出是什么？

(a) i, j, k = 4, -1, 0

```
w = i or j or k
x = i and j and k
y = i or j and k
z = i and j or k
print(w, x, y, z)
```

输出

4 0 4 -1

(b) a = 10

```
a = not not a
print(a)
```

输出

True

(c) x, y, z = 20, 40, 45

```
if x > y and x > z :
    print('biggest = ' + str(x))
elif y > x and y > z :
    print('biggest = ' + str(y))
elif z > x and z > y :
    print('biggest = ' + str(z))
```

输出

biggest = 45

(d) num = 30

```
k = 100 if num <= 10 else 500
print(k)
```

输出

500

(e) a = 10

```
b = 60
if a and b > 20 :
    print('Hello')
else :
    print('Hi')
```

输出

Hello

(f) a = 10

```
b = 60
if a > 20 and b > 20 :
    print('Hello')
else :
    print('Hi')
```

输出

Hi

(g) a = 10

```
if a = 30 or 40 or 60 :
    print('Hello')
else :
    print('Hi')
```

输出

错误

(h) a = 10

```
if a = 30 or a == 40 or a == 60 :
    print('Hello')
else :
    print('Hi')
```

输出

错误

(i) a = 10

```
if a in (30, 40, 50) :
    print('Hello')
else :
    print('Hi')
```

输出

Hi

**[C]** 指出以下程序中的错误（如果有的话）：

(a) a = 12.25

```
b = 12.52
if a = b :
    print('a and b are equal!')
```

回答

错误：语法无效。请使用 a == b

(b)

```
if ord('X') < ord('x')
    print('Unicode value of X is smaller than that of x')
```

回答

错误：语法无效。请在 if 语句末尾使用 :，如下所示：
if ord('X') < ord('x') :

(c) x = 10

```
if x >= 2 then
    print('x')
```

回答

错误：语法无效。请在 if 语句末尾使用 :，如下所示：
if x >= 2 :

(d) x = 10 ; y = 15

```
if x % 2 = y % 3
    print('Carpathians\n')
```

回答

错误：语法无效。在比较时请使用 == 代替 =

(e) x, y = 30, 40

```
if x == y :
    print('x is equal to y')
elseif x > y :
    print('x is greater than y')
elseif x < y :
    print('x is less than y')
```

回答

错误：语法无效。请使用 **elif** 代替 **elseif**

**[D]** 如果 a = 10, b = 12, c = 0，求以下表达式的值：

a != 6 and b > 5
a == 9 or b < 3

## 第五章：决策控制指令

not ( a < 10 )
not ( a > 5 and c )
5 and c != 8 or c

答案
True
False
True
True
True

### [E] 尝试以下练习：

(a) 通过键盘输入任意整数。编写一个程序来判断它是奇数还是偶数。

程序

```
#### 判断数字是奇数还是偶数
x = int(input('Enter any number: '))
j = 2
if x % j == 0 :
    print('Even Number')
else :
    print('Odd Number')
```

输出
Enter any number: 48
Even Number

(b) 通过键盘输入任意年份。编写一个程序来判断该年份是否是闰年。

程序

```
#### 判断年份是否是闰年
year = int(input('Enter a year: '))
if year % 4 == 0 :
    if year % 100 == 0 :
        if year % 400 == 0 :
            print(year, 'is a Leap Year')
        else :
            print(year, 'is not a Leap Year')
    else :
        print(year, 'is a Leap Year')
else :
    print(year, 'is not a Leap Year')
```

输出

Enter a year: 1996
1996 is a Leap Year

Enter a year: 2000
2000 is a Leap Year

Enter a year: 1900
1900 is not a Leap Year

(c) 如果通过键盘输入了Ram、Shyam和Ajay的年龄，编写一个程序来确定三人中谁最年轻。

程序

```
#### 确定三人中最年轻的人
ram_age = int(input('Enter Ram\'s age: '))
shyam_age = int(input('Enter Shyam\'s age: '))
ajay_age = int(input('Enter Ajay\'s age: '))
if ram_age < shyam_age and ram_age < ajay_age :
    print('Youngest is Ram')
elif shyam_age < ram_age and shyam_age < ajay_age :
    print('Youngest is Shyam')
elif ajay_age < ram_age and ajay_age < shyam_age :
    print('Youngest is Ajay')
```

输出

Enter Ram's age: 23
Enter Shyam's age: 45
Enter Ajay's age: 34
Youngest is Ram

(d) 当通过键盘输入三角形的三个角度时，编写一个程序来检查三角形是否有效。如果三个角度之和等于180度，则三角形有效。

程序

```
#### 判断三角形是否有效
x = int(input('Enter angle no. 1: '))
y = int(input('Enter angle no. 2: '))
z = int(input('Enter angle no. 3: '))
sum_of_angles = x + y + z
if sum_of_angles == 180 :
    print('Valid Triangle')
else :
    print('Is not a Valid Triangle')
```

输出

Enter angle no. 1: 45
Enter angle no. 2: 45
Enter angle no. 3: 90
Valid Triangle

(e) 编写一个程序，求通过键盘输入的数字的绝对值。

程序

```
#### 获取数字的绝对值
x = int(input('Enter any number: '))
if x < 0 :
    y = x * (-1)
else :
    y = x
print('Absolute value of', x, 'is', y)
```

输出

Enter any number: -20
Absolute value of -20 is 20

Enter any number: 23
Absolute value of 23 is 23

(f) 给定矩形的长和宽，编写一个程序来判断矩形的面积是否大于其周长。例如，长=5，宽=4的矩形，其面积大于周长。

程序

```
#### 判断矩形的面积是否大于其周长
length = int(input('Enter length of rectangle: '))
breadth = int(input('Enter breadth of rectangle: '))

area = length * breadth
perimeter = 2 * (length + breadth)
print('Area =', area, ' Perimeter =', perimeter)
if area > perimeter :
    print('Area of Rectangle is greater than perimeter')
else :
    print('Perimeter of Rectangle is greater than area')
```

输出

```
Enter length of rectangle: 4
Enter breadth of rectangle: 5
Area = 20 Perimeter = 18
Area of Rectangle is greater than perimeter

Enter length of rectangle: 2
Enter breadth of rectangle: 1
Area = 2 Perimeter = 6
Perimeter of Rectangle is greater than area
```

(g) 给定三个点 (x1, y1), (x2, y2) 和 (x3, y3)，编写一个程序来检查这三个点是否在同一条直线上。

程序

```
#### 判断三个点是否共线
x1 = int(input('Enter the co-ordinate of x1: '))
y1 = int(input('Enter the co-ordinate of y1: '))
x2 = int(input('Enter the co-ordinate of x2: '))
y2 = int(input('Enter the co-ordinate of y2: '))
x3 = int(input('Enter the co-ordinate of x3: '))
y3 = int(input('Enter the co-ordinate of y3: '))

if x1 == x2 and x2 == x3 :
    print('Collinear')
elif x1 != x2 and x2 != x3 and x3 != x1 :
    # 计算每对点之间直线的斜率
    s1 = (float(abs(y2 - y1))) / (float(abs(x2 - x1)))
    s2 = (float(abs(y3 - y2))) / (float(abs(x3 - x2)))
    s3 = (float(abs(y3 - y1))) / (float(abs(x3 - x1)))

    if s1 == s2 and s2 == s3 :
        print('Collinear')
    else :
        print('Non Collinear')
```

输出

Enter the co-ordinate of x1: 4
Enter the co-ordinate of y1: 4
Enter the co-ordinate of x2: 5
Enter the co-ordinate of y2: 5
Enter the co-ordinate of x3: 6
Enter the co-ordinate of y3: 6
All the 3 points lies on the one straight line

(h) 给定圆心的坐标 (x, y) 和半径，编写一个程序来判断一个点是在圆内、圆上还是圆外。（提示：使用 sqrt( ) 函数）

程序

```
#### 判断点在圆内、圆外还是圆上
import math

centerX = int(input('Enter X coord. of center of circle: ' ))
centerY = int(input('Enter Y coord. of center of circle: ' ))

radius = int(input('Enter radius of circle: '))

print('Enter coordinates of point:')
pointX = int(input('Enter X coord. of point: '))
pointY = int(input('Enter Y coord. of point: '))

xDiff = centerX - pointX ;
yDiff = centerY - pointY ;

distance = math.sqrt((xDiff * xDiff) + (yDiff * yDiff))

if distance == radius :
    print('Point is on the circle')
elif distance < radius :
    print('Point lies inside the circle')
else :
    print('Point lies outside the circle')
```

输出

Enter X coord. of center of circle: 0
Enter Y coord. of center of circle: 0
Enter radius of circle: 5
Enter coordinates of point:
Enter X coord. of point: 5
Enter Y coord. of point: 0
Point is on the circle

(i) 给定一个点 (x, y)，编写一个程序来判断它是在X轴上、Y轴上还是在原点。

程序

```
#### 判断点在坐标系中的位置
x = int(input('Enter X Coord of the point:'))
y = int(input('Enter Y coord of the point:'))

if x == 0 and y == 0 :
    print('Point is the origin')
elif x == 0 and y != 0 :
    print('Point lies on the Y axis')
elif x != 0 and y == 0 :
    print('Point lies on the X axis')
else :
    if x > 0 and y > 0 :
        print('Point lies in the First Quadrant')
    elif x < 0 and y > 0 :
        print('Point lies in the Second Quadrant')
    elif x < 0 and y < 0 :
        print('Point lies in the Third Quadrant')
    else :
        print('Point lies in the Fourth Quadrant')
```

输出

Enter X Coord of the point:0
Enter Y coord of the point:0
Point is the origin

Enter X Coord of the point:-10
Enter Y coord of the point:-20
Point lies in the Third Quadrant

(j) 通过键盘输入一个年份，编写一个程序来判断该年份是否是闰年。使用逻辑运算符 **and** 和 **or**。

程序

```
#### 判断年份是否是闰年
year = int(input('Enter a year: '))
if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 :
    print(year, 'is a leap year')
else :
    print(year, 'is not a leap year')
```

输出

Enter a year: 2016
2016 is a Leap Year

(k) 如果通过键盘输入三角形的三条边，编写一个程序来检查三角形是否有效。如果两条边之和大于三条边中最大的那条边，则三角形有效。

程序

```
#### 判断三角形是否有效
s1 = int(input('Enter the 1st side of triangle: '))
s2 = int(input('Enter the 2nd side of triangle: '))
s3 = int(input('Enter the 3rd side of triangle: '))
if s1 + s2 <= s3 or s2 + s3 <= s1 or s1 + s3 <= s2 :
    print('Invalid Triangle')
else :
    print('Valid Triangle')
```

输出

## 第五章：决策控制指令

输入三角形的第1条边：6
输入三角形的第2条边：7
输入三角形的第3条边：10
有效三角形

输入三角形的第1条边：5
输入三角形的第2条边：3
输入三角形的第3条边：12
无效三角形

(I) 如果通过键盘输入三角形的三条边，编写一个程序来检查该三角形是等腰三角形、等边三角形、不等边三角形还是直角三角形。

## 程序

```
#### 判断三角形类型
s1 = int(input('输入三角形的第1条边：'))
s2 = int(input('输入三角形的第2条边：'))
s3 = int(input('输入三角形的第3条边：'))

if s1 + s2 <= s3 or s2 + s3 <= s1 or s1 + s3 <= s2 :
    print('这些边无法构成三角形')
else :
    if s1 != s2 and s2 != s3 and s3 != s1 :
        print('不等边三角形')

    if s1 == s2 and s2 != s3 :
        print('等腰三角形')

    if s2 == s3 and s3 != s1 :
        print('等腰三角形')

    if s1 == s3 and c3 != s2 :
        print('等腰三角形')

    if s1 == s2 and s2 == s3 :
        print('等边三角形')

a = ( s1 * s1) == ( s2 * s2) + ( s3 * s3)
b = ( s2 * s2) == ( s1 * s1) + ( s3 * s3)
c = ( s3 * s3) == ( s1 * s1) + ( s2 * s2)

if a or b or c :
    print('直角三角形')
```

#### 输出

输入三角形的第1条边：6
输入三角形的第2条边：8
输入三角形的第3条边：10
不等边三角形
直角三角形

输入三角形的第1条边：3
输入三角形的第2条边：3
输入三角形的第3条边：3
等边三角形

输入三角形的第1条边：5
输入三角形的第2条边：3
输入三角形的第3条边：12
这些边无法构成三角形

## 6 重复控制指令

![](img/ac61a02854da7ea89272a34be93604dc_50_0.png)

[A] 回答以下问题：

(a) **while** 循环的 **else** 代码块何时开始工作？

答案

Else 代码块是可选的。如果存在，当条件失败时执行。

(b) **range( )** 函数能否用于生成从 0.1 到 1.0，步长为 0.1 的数字？

答案

不能。**range( )** 函数无法生成浮点数。

(c) **while** 循环能否嵌套在 **for** 循环中，反之亦然？

答案

可以，**while** 循环可以嵌套在 **for** 循环中，反之亦然。

(d) **while/for** 循环能否在 **if/else** 中使用，反之亦然？

答案

可以，**while/for** 循环可以在 **if/else** 中使用，反之亦然。

(e) do-while 循环能否用于重复一组语句？

答案

Python 中没有 **do-while** 循环。

(f) 你将如何为以下代码编写等效的 **for** 循环？

```
count = 1
while count <= 10 :
    print(count)
    count = count + 1
```

答案

```
for count in range(1, 11) :
    print(count)
```

(g) 以下代码片段的输出是什么？

```
for index in range(20, 10, -3) :
    print(index, end = ' ')
```

答案

20 17 14 11

(h) 为什么 **break** 和 **continue** 应始终与嵌入在 **while** 或 **for** 循环中的 **if** 一起使用？

答案

如果在没有 **if** 的情况下使用，**break** 会在第一次迭代时就终止循环。
如果在没有 **if** 的情况下使用 **continue**，其下方的语句将永远不会被执行。

**[B]** 指出以下程序中的错误（如果有的话）：

(a)
```
j = 1
while j <= 10 :
    print(j)
    j++
```

答案

错误。我们不能使用 j++ 来递增 j，因为 Python 中没有自增运算符。应使用 j = j + 1 或 j += 1。

(b)
```
while true :
    print('Infinite loop')
```

答案

错误。应使用 **True** 而不是 **true**。

(c)
```
lst = [10, 20, 30, 40, 50]
for count = 1 to 5 :
    print(lst[ i ])
```

答案

错误。**for** 循环使用错误。应使用：
```
for num in lst :
    print(num)
```

(d)
```
i = 15
not while i < 10 :
    print(i)
    i -= 1
```

答案

错误。**while** 使用不当。应使用：
```
while i >= 10 :
```

(e)
```
#### 打印从 A 到 Z 的字母
for alpha in range(65, 91) :
    print(ord(alpha), end=' ')
```

答案

错误。应使用 **chr( )** 函数来打印与 ASCII 值对应的字符，而不是 **ord( )**。

(f)
```
for i in range(0.1, 1.0, 0.25) :
    print(i)
```

答案

错误。**range( )** 函数不能用于生成浮点数序列。

(g)
```
i = 1
while i <= 10 :
    j = 1
    while j <= 5 :
        print(i, j )
        j += 1
        break
    print(i, j)
    i += 1
```

答案

无错误。

[C] 将以下各对与 **range( )** 函数在 **for** 循环中使用时将生成的值进行匹配。

a. range(5) 1. 1, 2, 3, 4
b. range(1, 10, 3) 2. 0, 1, 2, 3, 4
c. range(10, 1, -2) 3. 无
d. range(1, 5) 4. 10, 8, 6, 4, 2
e. range(-2) 5. 1, 4, 7

答案

range(5) - 0, 1, 2, 3, 4
range(1, 10, 3) - 1, 4, 7
range(10, 1, -2) - 10, 8, 6, 4, 2
range(1, 5) - 1, 2, 3, 4, 5
range(-2) - 无

[D] 尝试完成以下任务：

(a) 编写一个程序，使用 range( ) 打印前 25 个奇数。

## 程序

```
#### 生成前 25 个奇数
j = 1
print('前 25 个奇数：')
for i in range(50) :
    if i % j == 1 :
        print(i, end = ', ')
    i += 1
```

#### 输出

前 25 个奇数：
1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
41, 43, 45, 47, 49,

(b) 使用 for 循环重写以下程序。

```
lst = ['desert', 'dessert', 'to', 'too', 'lose', 'loose']
s = 'Mumbai'
i = 0
while i < len(lst) :
    if i > 3 :
        break
    else :
        print(i, lst[i], s[i])
        i += 1
```

## 程序

```
#### 使用 for 循环重写
lst = ['desert','dessert','to','too','lose','loose']
s = 'Mumbai'
i = 0
for i, ele in enumerate(lst) :
    if i > 3 :
        break
    print(i, ele, s[i])
```

#### 输出

0 desert M
1 dessert u
2 to m
3 too b

(c) 编写一个程序，统计字符串 'Nagpur-440010' 中字母和数字的数量。

## 程序

```
#### 统计字符串中的字母、数字和特殊符号
string = 'Nagpur-440010'
alphabets = digits = special = 0

for i in range(len(string)) :
    if(string[i].isalpha( )) :
        alphabets = alphabets + 1
    elif(string[i].isdigit( )) :
        digits = digits + 1
    else :
        special = special + 1

print('字母数量 =', alphabets)
print('数字数量 =', digits)
print('特殊字符数量 =', special)
```

#### 输出

字母数量 = 6
数字数量 = 6
特殊字符数量 = 1

(d) 通过键盘输入一个五位数。编写一个程序来获得反转后的数字，并确定原始数字和反转后的数字是否相等。

## 程序

```
#### 反转一个 5 位数并与原始数字比较
num = int(input('输入一个 5 位数：'))
orinum = num
revnum = 0
while(num > 0) :
    rem = num % 10
    revnum = (revnum * 10) + rem
    num = num // 10

print('原始数字 =', orinum)
print('反转后的数字 = ', revnum)
if orinum == revnum :
    print('原始数字和反转后的数字相同')
else :
    print('原始数字和反转后的数字不同')
```

#### 输出

输入一个 5 位数：12345
原始数字 = 12345
反转后的数字 = 54321
原始数字和反转后的数字不同

输入一个 5 位数：12221
原始数字 = 12221
反转后的数字 = 12221
原始数字和反转后的数字相同

(e) 编写一个程序，计算通过键盘输入的任意数字的阶乘值。

## 程序

```
number = int(input('输入一个数字：'))
fact = 1
for i in range(1, number + 1) :
    fact = fact * i
print('阶乘值 = ', fact)
```

#### 输出

```
输入一个数字：5
阶乘值 = 120
```

(f) 编写一个程序，打印 1 到 500 之间所有的阿姆斯特朗数。如果一个数的每个数字的立方和等于该数本身，则该数被称为阿姆斯特朗数。例如，153 = ( 1 * 1 * 1 ) + ( 5 * 5 * 5 ) + ( 3 * 3 * 3 )。

## 程序

```
print('1 到 500 之间的阿姆斯特朗数是：')
for num in range(1, 501) :
    n = num
    d3 = n % 10
    n = int(n / 10)
    d2 = n % 10
    n = int(n / 10)
    d1 = n % 10
    if d1 * d1 * d1 + d2 * d2 * d2 + d3 * d3 * d3 == num :
        print(num)
```

#### 输出

```
1 到 500 之间的阿姆斯特朗数是：
1
153
370
371
407
```

(g) 编写一个程序，打印 1 到 300 之间所有的质数。

## 程序

```
lower = 1
upper = 300
print('1 到 300 之间所有的质数：')
```

## 第六章：重复控制指令

```python
for num in range(lower, upper + 1) :
    for n in range(2, num) :
        if (num % n) == 0 :
            break
    else :
        print(num, end = ',')
```

输出

1到300之间的所有质数：
1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,

(h) 编写一个程序，打印用户输入数字的乘法表。表格应按以下形式显示：

29 * 1 = 29
29 * 2 = 58
...

程序

```python
num = int(input('Enter any number: '))
lower = 1
upper = 10
for i in range(lower, upper + 1) :
    print(num, 'x', i , '=', num * i)
```

输出

Enter any number: 29
29 x 1 = 29
29 x 2 = 58
29 x 3 = 87
29 x 4 = 116
29 x 5 = 145
29 x 6 = 174
29 x 7 = 203
29 x 8 = 232
29 x 9 = 261
29 x 10 = 290

(i) 当利息以年利率 **r** % 每年复利 **q** 次，持续 **n** 年时，本金 **p** 将按以下公式复利计算为金额 **a**：

a = p ( 1 + r / q ) ^nq

编写一个程序，读取10组 **p**、**r**、**n** 和 **q** 的值，并计算相应的 a。

程序

```python
for i in range(10) :
    p = float(input('Enter value of p: '))
    r = float(input('Enter value of r: '))
    n = float(input('Enter value of n: '))
    q = float(input('Enter value of q: '))
    a = p * ((1 + r / 100 / q) ** (n * q))
    print('Compound Interest = Rs.', a)
```

输出

```
Enter value of p: 1000
Enter value of r: 5
Enter value of n: 3
Enter value of q: 4
Compound Interest = Rs. 1160.7545177229981
...
```

(j) 编写一个程序，生成所有边长小于或等于30的毕达哥拉斯三元组。

程序

```python
n = 31
for i in range(1, n) :
    for j in range((i + 1), (n + 1), 1) :
        t = (i * i) + (j * j)
        for k in range((i + 2), (n + 1), 1) :
            if (t == k * k) :
                print(i , j, k)
```

输出

3 4 5
5 12 13
6 8 10
7 24 25
8 15 17
9 12 15
10 24 26
12 16 20
15 20 25
18 24 30
20 21 29

(k) 某城镇今天的人口为100,000。在过去10年中，人口以每年10%的速率稳定增长。编写一个程序，确定过去十年中每年年底的人口数量。

程序

```python
population = 100000
for i in range(10) :
    population += int(population * 10 / 100)
    print('Year', i + 1, ':', population)
```

输出

Year 1 : 110000
Year 2 : 121000
Year 3 : 133100
Year 4 : 146410
Year 5 : 161051
Year 6 : 177156
Year 7 : 194871
Year 8 : 214358
Year 9 : 235793
Year 10 : 259372

(l) 拉马努金数是可以用两种不同方式表示为两个立方数之和的最小数字。编写一个程序，打印所有不超过合理上限的此类数字。

程序

```python
print('Ramanujan Numbers:')
for i in range(1, 31):
    for j in range(1, 31):
        for k in range(1, 31):
            for l in range(1, 31):
                if (i != j and i != k and i != l) and (j != k and j != l) and (k != l ) :
                    if i * i * i + j * j * j == k * k * k + l * l * l:
                        print(i, j, k, l)
```

输出

Ramanujan Numbers:
1    12    9    10
1    12    10    9
2    16    9    15
.. .. .. .. .. ..

(m) 编写一个程序，打印一天24小时的时间，并带有合适的后缀，如 AM、PM、Noon 和 Midnight。

程序

```python
for hour in range(24) :
    if hour == 0 :
        print('12 Midnight')
        continue
    if hour < 12 :
        print(hour, 'AM')
    if hour == 12 :
        print('12 Noon')
    if hour > 12 :
        print(hour % 12, 'PM')
```

输出

12 Midnight
1 AM
2 AM
... ...

## 第七章：控制台输入/输出

![](img/ac61a02854da7ea89272a34be93604dc_62_0.png)

[A] 尝试以下操作：

(a) 你将如何使以下代码更紧凑？

```python
print('Enter ages of 3 persons')
age1 = input( )
age2 = input( )
age3 = input( )
```

答案

```python
age1, age2, age3 = input('Enter 3 values: ').split(',')
```

(b) 你将如何在一行中打印 "Rendezvous" 并将光标保留在输出打印的同一行中？

答案

```python
print('Rendezvous', end = "")
```

(c) 以下代码片段的输出是什么？

```python
l, b = 1.5678, 10.5
print('length = {l} breadth = {b}')
```

输出

```
length = {l} breadth = {b}
```

(d) 在以下语句中，`> 5`、`> 7` 和 `> 8` 分别表示什么？

```python
print(f'{n:>5}{n ** 2:>7}{n ** 3:>8}')
```

输出

`n > 5` 表示 `n` 的值将在5列中右对齐打印。类似地，`n ** 2:>7` 表示 `n ** 2` 的值将在7列中右对齐打印。

(e) 以下代码段的输出是什么？

```python
name = 'Sanjay'
cellno = 9823017892
print(f'{name:15} : {cellno:10}')
```

输出

Sanjay : 9823017892

(f) 你将如何使用 f-string 打印以下代码段的输出？

```python
x, y, z =10, 20, 40
print('{0:<5}{1:<7}{2:<8}'.format(x, y, z))
```

输出

```python
print(f'{x:<5}{y:<7}{z:<8}')
```

(g) 你将如何从键盘接收任意数量的浮点数？

输出

```python
numbers = [float(x) for x in input('Enter values: ').split( )]
for n in numbers :
    print(n + 10)
```

(h) 应对以下代码进行哪些更改：

```python
print(f'\nx = :4}{x:>10}{\ny = :4}{y:>10}')
```

以产生以下输出：

```
x =     14.99
y =    114.39
```

输出

```python
print(f'{x = :>10}\n{y = :>10}')
```

(i) 你将如何接收布尔值作为输入？

输出

```python
b = bool(input('Enter boolean value: '))
```

(j) 你将如何接收复数作为输入？

输出

```python
c = complex(input('Enter complex value: '))
```

(k) 你将如何在10列中显示 **price**，并保留小数点后4位？假设 price 的值为 1.5567894。

输出

```python
price = 1.5567894
print(f'{price =:10.4}')
```

(l) 编写一个程序，使用一个 **input( )** 语句接收任意数量的浮点数。计算接收到的浮点数的平均值。

程序

```python
#### 接收任意数量的浮点数
num = int(input('How many numbers do you wish to input: '))
totalsum = 0
number = [float(x) for x in input('Enter all numbers: ').split( )]
for n in range(len(number)):
    totalsum = totalsum + number[n]
avg = totalsum / num
print('Average of', num, 'numbers is:', avg)
```

输出

```
How many numbers do you wish to input: 5
Enter all numbers: 10 20 30 40 50
Average of 5 numbers is: 30.0
```

(m) 编写一个程序，使用一个 **input( )** 语句接收以下信息：

- 人的姓名
- 服务年限
- 收到的排灯节奖金

根据以下公式计算并打印协议扣除额：

```
deduction = 2 * years of service + bonus * 5.5 / 100
```

程序

```python
data = input('Name, year of service, diwali bonus: ').split(',')
name = data[0]
yos = int(data[1])
bonus = float(data[2])
deduction = float((2 * yos + bonus * 5.5 ) / 100)
print('Deduction = Rs.', deduction)
```

输出

Name, year of service, diwali bonus: Ramesh, 3, 9500
Deduction = Rs. 522.56

(n) 应添加哪个导入语句才能使用内置函数 **input( )** 和 **print( )**？

答案

使用 **input( )** 和 **print( )** 不需要导入语句，因为它们是全局函数，随处可用。

(o) 以下语句是否正确？

```python
print('Result = ' + 4 > 3)
```

答案

不正确。字符串不能与布尔值连接。正确的形式应为：

```python
print('Result = ' + str(4 > 3))
```

(p) 编写一个程序，打印以下值：

a = 12.34, b = 234.39, c = 444.34, d = 1.23, e = 34.67

如下所示：

a = 12.34
b = 234.39
c = 444.34
d = 1.23
e = 34.67

程序

```python
a, b, c, d, e = 12.34, 234.39, 444.34, 1.23, 34.67
print(f'a = {a:>10}')
print(f'b = {b:>10}')
print(f'c = {c:>10}')
```

## 8 列表

![](img/ac61a02854da7ea89272a34be93604dc_68_0.png)

### [A] 以下程序的输出会是什么？

```python
(a) msg = list('www.kicit.com')
    ch = msg[-1]
    print(ch)
```

答案

m

```python
(b) msg = list('kanlabs.teachable.com')
    s = msg[4:6]
    print(s)
```

答案

['a', 'b']

```python
(c) msg = 'Online Courses - KanLabs'
    s = list(msg[:3])
    print(s)
```

答案

['O', 'n', 'l']

```python
(d) msg = 'Rahate Colony'
    s = list(msg[-5:-2])
    print(s)
```

答案

['o', 'l', 'o']

```python
(e) s = list('KanLabs')
    t = s[::-1]
    print(t)
```

答案

['s', 'b', 'a', 'L', 'n', 'a', 'K']

```python
(f) num1 = [40, 42, 35, 28]
    num2 = num1
    print(num1 is num2)
```

答案

```
40423528
<class 'list'>
True
True
```

```python
(g) num = [10, 20, 30, 40, 50]
    num[2:4] = [ ]
    print(num)
```

答案

```
[10, 20, 50]
```

```python
(h) num1 = [10, 20, 30, 40, 50]
    num2 = [60, 70, 80]
    num1.append(num2)
    print(num1)
```

答案

```
[10, 20, 30, 40, 50, [60, 70, 80]]
```

```python
(i) lst = [10, 25, 4, 12, 3, 8]
    sorted(lst)
    print(lst)
```

答案

```
[10, 25, 4, 12, 3, 8]
```

```python
(j) a = [1, 2, 3, 4]
    b = [1, 2, 5]
    print(a < b)
```

答案

```
True
```

### [B] 尝试回答以下问题：

(a) 以下哪一个是有效的列表？

```
['List']    {"List"}    ("List")    "List"
```

答案

['List']

(b) 执行以下代码片段会发生什么？

```python
s = list('Hello')
s[1] = 'M'
```

答案

列表 s 中的元素 'e' 将被 'M' 替换。

(c) 以下代码片段从列表中删除了元素 30 和 40：

```python
num = [10, 20, 30, 40, 50]
del(num[2:4])
```

还可以用哪种其他方式获得相同的效果？

答案

num[2:4] = [ ]

(d) 以下哪一个是不正确的列表？

```python
a = [0, 1, 2, 3, [10, 20, 30]]
a = [10, 'Suraj', 34555.50]
a = [[10, 20, 30], [40, 50, 60]]
```

答案

无。所有列表都是正确的。列表可以包含不同类型的元素。

(e) 从下面给出的列表

```python
num1 = [10, 20, 30, 40, 50]
```

你将如何创建包含以下内容的列表 **num2**：
['A', 'B', 'C', 10, 20, 30, 40, 50, 'Y', 'Z']

答案

```python
num1 = [10, 20, 30, 40, 50]
num2 = ['A', 'B', 'C', *num1, 'Y', 'Z']
```

(f) 给定一个列表
lst = [10, 25, 4, 12, 3, 8]
你将如何按降序对其进行排序？

答案

lst.sort(reverse = True)

(g) 给定一个列表
lst = [10, 25, 4, 12, 3, 8]
你将如何检查 30 是否存在于列表中？

答案

print(30 in lst)

(h) 给定一个列表
lst = [10, 25, 4, 12, 3, 8]
你将如何在 25 和 4 之间插入 30？

答案

lst.insert(2, 30)

(i) 给定一个字符串
s = 'Hello'
你将如何从中获得列表 ['H', 'e', 'l', 'l', 'o']？

答案

lst = list(s)

### [C] 回答以下问题：

(a) 编写一个程序来创建一个包含 5 个奇数的列表。将第三个元素替换为一个包含 4 个偶数的列表。展平、排序并打印该列表。

程序

```python
#### 修改、展平并排序列表
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8]
x[2] = y
print(x)
x = x[:2] + [*y] + x[3:]
print(x)
x.sort()
print(x)
```

输出

```
[1, 3, [2, 4, 6, 8], 7, 9]
[1, 3, 2, 4, 6, 8, 7, 9]
[1, 2, 3, 4, 6, 7, 8, 9]
```

(b) 假设一个列表包含 20 个随机生成的整数。从键盘接收一个数字，并报告该数字在列表中所有出现的位置。

程序

```python
#### 报告列表中某个数字的出现次数
import random
lst = [ ]
for k in range(20) :
    n = random.randint(0, 50)
    lst.append(n)
print(lst)

num = int(input('Enter number: '))
for i in range(len(lst)) :
    if lst[i] == num:
        print('Number found at position:', i)
```

输出

```
[44, 4, 22, 11, 36, 29, 38, 32, 14, 34, 48, 49, 4, 14, 23, 5, 28, 43, 49, 3]
Enter number: 14
Number found at position: 8
Number found at position: 13
```

(c) 假设一个列表有 20 个数字。编写一个程序来删除该列表中的所有重复项。

程序

```python
#### 删除列表中的重复项
lst = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 10, 20, 68, 8, 40, 45, 1, 5, 53, 45, 17]
print('Original list: ', lst)
final_lst=[ ]
for num in lst :
    if num not in final_lst :
        final_lst.append(num)

lst = final_lst
print('List after removing duplicates:', lst)
```

输出

Original list: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 10, 20, 68, 8, 40, 45, 1, 5, 53, 45, 17]
List after removing duplicates: [1, 2, 3, 4, 5, 10, 20, 68, 8, 40, 45, 53, 17]

(d) 假设一个列表包含正数和负数。编写一个程序来创建两个列表——一个包含正数，另一个包含负数。

程序

```python
#### 将正数和负数分离到两个列表中
lst1 = [1, -9, -6, -45, -78,-1, 2, 3, 4, 5]
lst2 = [ ]
lst3 = [ ]
count, ncount = 0, 0
for num in lst1 :
    if num >= 0 :
        lst2.append(num)
    else :
        lst3.append(num)

print('Original list:', lst1)
print('Positive numbers list:', lst2)
print('Negative numbers list:', lst3)
```

输出

Original list: [1, -9, -6, -45, -78, -1, 2, 3, 4, 5]
Positive numbers list: [1, 2, 3, 4, 5]
Negative numbers list: [-9, -6, -45, -78, -1]

(e) 假设一个列表包含 5 个字符串。编写一个程序将所有这些字符串转换为大写。

程序

```python
#### 将列表中的字符串转换为大写
lst = ['abc', 'def', 'ghi', 'jkl', 'lmn']
for i, item in enumerate(lst) :
    lst[i] = item.upper( )

print(lst)
```

输出

['ABC', 'DEF', 'GHI', 'JKL', 'LMN']

(f) 编写一个程序，将华氏温度列表转换为等效的摄氏温度。

程序

```python
#### 将列表中的华氏温度转换为摄氏度
fahr = [212, 120, 100, 93, 37]
for i, f in enumerate(fahr) :
    c = int(5 / 9 * (f - 32))
    fahr[i] = c
    print(f, c)
print(fahr)
```

输出

212 100
120 48
100 37
93 33
37 2
[100, 48, 37, 33, 2]

(g) 编写一个程序来获取数字列表的中位数，而不打乱列表中数字的顺序。

程序

```python
#### 获取列表的中位数
num = [1, 2, 3, 4, 5, 6, 7]
n = len(num)
if n % 2 == 0 :
    i = int(n / 2 - 1)
    j = int(n / 2)
    median = (num[i] + num[j]) / 2
else :
    i = int(n / 2)
    median = num[i]

print('Median value =', median)
```

输出

Median value = 4

(h) 一个列表只包含正整数和负整数。编写一个程序来获取列表中负数的数量，而不使用循环。

程序

```python
#### 不使用循环计算列表中的负数个数
lst = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
if 0 not in lst :
    lst.append(0)
lst = sorted(lst)
pos = lst.index(0)
print('Number of negative numbers in the list =', pos)
```

输出

Number of negative numbers in the list = 5

(i) 假设一个列表包含几个单词。编写一个程序来创建另一个列表，该列表包含第一个列表中每个单词的首字母。

程序

```python
msg = ['Dialogue', 'is', 'dead', 'Chatalogue', 'is', 'in']
abbrmsg = [ ]
for word in msg :
    abbrmsg.append(word[0])
print(abbrmsg)
```

输出

['D', 'i', 'd', 'C', 'i', 'i']

(j) 一个列表包含 10 个数字。编写一个程序来消除列表中的所有重复项。

程序

```python
#### 消除列表中的所有重复项
lst1= [1, 2, 1, 3, 4, 5, 6, 5, 2, 4]
lst2 = [ ]
print('Original list =', lst1)
for i in lst1 :
    if i not in lst2 :
        lst2.append(i)
print('List after eliminating duplicates =', lst2)
```

输出

[1, 2, 3, 4, 5, 6]

(k) 编写一个程序来查找 10 个数字列表的平均值、中位数和众数。

程序

```python
#### 查找 10 个数字列表的平均值、中位数、众数
lst = [10, 20, 30, 40, 30, 60, 70, 30, 80, 30]

#### 平均值
n = len(lst)
```

## 9 元组

![](img/ac61a02854da7ea89272a34be93604dc_80_0.png)

### [A] 以下哪些属性适用于字符串、列表和元组？

- 可迭代
- 可切片
- 可索引
- 不可变
- 序列
- 可以为空
- 有序集合
- 有序集合
- 无序集合
- 可以通过元素在集合中的位置来访问元素

**答案**

- 可迭代 - 字符串、列表和元组
- 可切片 - 字符串、列表和元组
- 可索引 - 字符串、列表和元组
- 不可变 - 字符串、元组
- 序列 - 字符串、列表和元组
- 可以为空 - 字符串、列表和元组
- 有序集合 - 无
- 有序集合 - 字符串、列表和元组
- 无序集合 - 集合
- 可以通过元素在集合中的位置来访问元素 - 字符串、列表和元组

### [B] 以下哪些操作可以对字符串、列表和元组执行？

- a = b + c
- a += b
- 在末尾追加一个新元素
- 删除第0个位置的元素
- 修改最后一个元素
- 原地反转

**答案**

- a = b + c - 字符串、列表和元组
- a += b - 字符串、列表和元组
- 在末尾追加一个新元素 - 列表
- 删除第0个位置的元素 - 列表
- 修改最后一个元素 - 字符串、列表和元组
- 原地反转 - 列表

### [C] 回答以下问题：

(a) 这是一个有效的元组吗？

```python
tpl = ('Square')
```

**答案**

不是。创建元组的正确方式是

```python
tpl = ('Square',)
```

(b) 以下代码片段的输出是什么？

```python
num1 = num2 = (10, 20, 30, 40, 50)
print(id(num1), type(num2))
print(isinstance(num1, tuple))
print(num1 is num2)
print(num1 is not num2)
print(20 in num1)
print(30 not in num2)
```

**程序**

```python
40291816 <class 'tuple'>
True
True
False
True
False
```

(c) 假设一个日期表示为元组 (d, m, y)。编写一个程序来创建两个日期元组，并找出两个日期之间的天数。

**程序**

```python
#### 计算两个日期之间的天数
dt1 = (17, 3, 1998)
dt2 = (17, 4, 2011)

d1, m1, y1 = dt1[0], dt1[1], dt1[2]
d2, m2, y2 = dt2[0], dt2[1], dt2[2]
days1 = [31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
days2 = [31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

ndays1 = (y1 - 1) * 365
ldays1 = ((y1 - 1) // 4) - ((y1 - 1) // 100) + ((y1 - 1) // 400)
tdays1 = ndays1 + ldays1

if((y1 % 4 == 0 and y1 % 100 != 0) or (y1 % 400 == 0)) :
    days1[1] = 29
else :
    days1[1] = 28

s1 = sum(days1[0:m1 - 1])
tdays1 += s1

ndays2 = (y2 - 1) * 365
ldays2 = ((y2 -1) // 4) - ((y2 - 1) // 100) + ((y2 - 1) // 400)
tdays2 = ndays2 + ldays2

if((y2 % 4 == 0 and y2 % 100 != 0) or (y2 % 400 == 0)) :
    days2[1] = 29
else :
    days2[1] = 28

s2 = sum(days2[0:m2 - 1])
tdays2 += s2

diff = tdays2 - tdays1
print('Difference in days = ', diff)
```

**输出**

```
Difference in days = 4779
```

(d) 创建一个元组列表。每个元组应包含一个项目及其浮点数价格。编写一个程序，按价格降序对元组进行排序。提示：使用 **operator.itemgetter( )**。

**程序**

```python
import operator
lst = [('Key', 101.25), ('Lock', 320.85), ('Hammer', 100.55),
('Spanner', 67.77), ('Tong', 93.03)]
print(sorted(lst, reverse = True, key = operator.itemgetter(1)))
```

**输出**

```
[('Lock', 320.85), ('Key', 101.25), ('Hammer', 100.55), ('Tong', 93.03),
('Spanner', 67.77)]
```

(e) 将用户持有的股票数据存储为元组，包含以下关于股票的信息：

- 股票名称
- 购买日期
- 成本价
- 股票数量
- 卖出价

编写一个程序来确定：

- 投资组合的总成本。
- 总收益或损失金额。
- 盈利百分比或亏损百分比。

**程序**

```python
lst = [ ]
i = 0

num_companies = int(input('Enter no. of companies: '))
for i in range(num_companies):
    name = input('Enter name: ')
    no_of_shares = int(input('Enter no of shares: '))
    dt_of_pur = input('Enter date of purchase: ')
    cost_price = int(input('Enter Cost price: '))
    selling_price = int(input('Enter selling price: '))
    tpl = (name, no_of_shares, dt_of_pur, cost_price, selling_price)
    lst.append(tpl)

tot = 0
gaintot = 0
losstot = 0
for l in lst :
    no_of_shares = l[1]
    cost_price = l[3]
    selling_price = l[4]
    cop = int(no_of_shares * cost_price)
    tot = tot + cop

    if selling_price > cost_price :
        gaintot += (selling_price - cost_price) * no_of_shares
    else :
        losstot += (cost_price - selling_price) * no_of_shares

print(f'Total cost of portfolio:{tot:.2f}')

if gaintot > losstot :
    net = gaintot - losstot
    gain_per = net / tot * 100
    print(f'Net amount gained:{net:.2f}')
    print(f'Percentage profit:{gain_per:.2f}')
else :
    net = losstot - gaintot
    loss_per = net / tot * 100
    print(f'Net amount lost:{net:.2f}')
    print(f'Percentage loss:{loss_per:.2f}')
```

**输出**

```
Enter no. of companies: 3
Enter name: L and T
Enter no of shares: 100
Enter date of purchase: 12/12/2012
Enter Cost price: 145
Enter selling price: 186
Enter name: Tata Motors
Enter no of shares: 120
Enter date of purchase: 13/11/2016
Enter Cost price: 785
Enter selling price: 678
Enter name: Infosys
Enter no of shares: 90
Enter date of purchase: 14/05/2018
Enter Cost price: 775
Enter selling price: 800
Total cost of portfolio: 178450.00
Net amount lost: 6490.00
Percentage loss: 3.64
```

(f) 编写一个程序，从元组列表中移除空元组。

**程序**

```python
lst1= [( ), ('Paras', 5), ('Ankit', 11), ( ), ('Harsha', 115), ('Aditya', 115),
( ), ('Aditi', 3), ( )]
lst2 = [ ]
for item in lst1 :
    if len(item) != 0 :
        lst2.append(item)
print(lst2)
```

**输出**

```
[('Paras', 5), ('Ankit', 11), ('Harsha', 115), ('Aditya', 115), ('Aditi', 3)]
```

(g) 编写一个程序来创建以下3个列表：
- 一个姓名列表
- 一个学号列表
- 一个成绩列表

从这3个列表中生成并打印一个包含姓名、学号和成绩的元组列表。从这个列表中生成3个元组——一个包含所有姓名，另一个包含所有学号，第三个包含所有成绩。

**程序**

```python
name = ['Aditi' ,'Mrunal' , 'Aditya' , 'Girish' , 'Ankit' , 'Meenal']
rollno = ['12' , '43' , '45' , '50' , '66' , '21']
marks = ['90' , '45' , '82' , '75' , '95' , '65']
t1 = tuple(name)
t2 = tuple(rollno)
t3 = tuple(marks)
lst = [t1, t2, t3]
print(lst)
print(t1)
print(t2)
print(t3)
```

**输出**

```
[('Aditi', 'Mrunal', 'Aditya', 'Girish', 'Ankit', 'Meenal'), ('12', '43', '45', '50', '66', '21'), ('90', '45', '82', '75', '95', '65')]
('Aditi', 'Mrunal', 'Aditya', 'Girish', 'Ankit', 'Meenal')
('12', '43', '45', '50', '66', '21')
('90', '45', '82', '75', '95', '65')
```

### [D] 将以下各项与每个 **range( )** 函数在 **for** 循环中使用时将生成的值进行匹配。

- a. tpl1 = ('A',)
- b. tpl1 = ('A')
- c. t = tpl[::-1]
- d. ('A', 'B', 'C', 'D')
- e. [(1, 2), (2, 3), (4, 5)]
- f. tpl = tuple(range(2, 5))
- g. ([1, 2], [3, 4], [5, 6])
- h. t = tuple('Ajooba')
- i. [*a, *b, *c]
- j. (*a, *b, *c)

- 1. 长度为6的元组
- 2. 元组的列表
- 3. 元组
- 4. 元组的列表
- 5. 字符串
- 6. 对元组排序
- 7. (2, 3, 4)
- 8. 字符串的元组
- 9. 在列表中解包元组
- 10. 在元组中解包列表

**答案**

- tpl1 = ('A',) - 元组
- tpl1 = ('A') - 字符串
- t = tpl[::-1] - 对元组排序
- ('A', 'B', 'C', 'D') - 字符串的元组
- [(1, 2), (2, 3), (4, 5)] - 元组的列表
- tpl = tuple(range(2, 5)) - (2, 3, 4)
- ([1, 2], [3, 4], [5, 6]) - 元组的列表
- t = tuple('Ajooba') - 长度为6的元组
- [*a, *b, *c] - 在列表中解包元组
- (*a, *b, *c) - 在元组中解包列表

## 10 组

![](img/ac61a02854da7ea89272a34be93604dc_88_0.png)

### [A] 以下程序的输出是什么？

```
(a) s = {1, 2, 3, 7, 6, 4}
    s.discard(10)
    s.remove(10)
    print(s)
```

输出

元素 10 不在集合 s 中。**discard( )** 不会执行任何操作，而 **remove( )** 会报告一个错误。

```
(b) s1 = {10, 20, 30, 40, 50}
    s2 = {10, 20, 30, 40, 50}
    print(id(s1), id(s2))
```

输出

40530296 40530184

```
(c) s1 = {10, 20, 30, 40, 50}
    s2 = {10, 20, 30, 40, 50}
    s3 = {*s1, *s2}
    print(s3)
```

输出

{40, 10, 50, 20, 30}

```
(d) s = set('KanLabs')
    t = s[::-1]
    print(t)
```

输出

错误：集合不是可下标对象。换句话说，不能对集合使用 `[ ]`。

```
(e) num = {10, 20, {30, 40}, 50}
    print(num)
```

输出

错误：嵌套集合是非法的。

```
(f) s = {'Tiger', 'Lion', 'Jackal'}
del(s)
print(s)
```

输出

错误：名称 's' 未定义。这是因为 **del( )** 删除了集合对象。

```
(g) fruits = {'Kiwi', 'Jack Fruit', 'Lichi'}
fruits.clear( )
print(fruits)
```

输出

**set( )**。调用 **clear( )** 后，**fruits** 变成一个空集合。

```
(h) s = {10, 25, 4, 12, 3, 8}
s = sorted(s)
print(s)
```

输出

[3, 4, 8, 10, 12, 25]

```
(i) s = {}
t = {1, 4, 5, 2, 3}
print(type(s), type(t))
```

输出

<class 'dict'> <class 'set'>

### [B] 回答以下问题：

(a) 一个集合包含以 A 或 B 开头的名称。编写一个程序将这些名称分离到两个集合中，一个包含以 A 开头的名称，另一个包含以 B 开头的名称。

程序

```
#### 将给定集合拆分为两个集合
lst = {'Aditya', 'Aditi', 'Ankita', 'Aniket', 'Anuja', 'Bhushan', 'Bahu',
       'Bali', 'Bhoomi', 'Babhoti' }
t = set( )
s = set( )
for item in lst :
    if item.startswith('A') :
        t.add(item)
    elif item.startswith('B') :
        s.add(item)
print(s)
print(t)
```

输出

{'Bhoomi', 'Bahu', 'Babhoti', 'Bali', 'Bhushan'}
{'Anuja', 'Ankita', 'Aditya', 'Aniket', 'Aditi'}

(b) 创建一个空集合。编写一个程序，向该集合添加五个新名称，修改一个现有名称，并删除两个现有名称。

程序

```
#### 集合操作
s = set( )
s.add('Amol')
s.add('Priya')
s.add('Mira')
s.add('Dipti')
s.add('Anil')
print('添加 5 个名称后:', s)
s.remove('Anil')
s.add('ANIL')
print('修改 Anil 后:', s)
s.remove('Dipti')
s.remove('Mira')
print('删除 Dipti 和 Mira 后:', s)
```

输出

添加 5 个名称后: {'Priya', 'Dipti', 'Anil', 'Mira', 'Amol'}
修改 Anil 后: {'Priya', 'ANIL', 'Dipti', 'Mira', 'Amol'}
删除 Dipti 和 Mira 后: {'Priya', 'ANIL', 'Amol'}

(c) 两个集合函数 **discard( )** 和 **remove( )** 之间有什么区别？

答案

**remove( )** 在尝试移除的元素不在集合中时会引发异常，而 **discard( )** 不会。

(d) 编写一个程序，创建一个包含 10 个在 15 到 45 范围内随机生成的数字的集合。统计这些数字中有多少小于 30。删除所有大于 35 的数字。

程序

```
#### 集合操作
import random
s = set()
while True :
    s.add(random.randint(15, 45))
    if len(s) == 10 :
        break
print('原始集合:', s)
t = set()
count = 0
for item in s :
    if item < 30 :
        count += 1
    if item <= 35 :
        t.add(item)

s = t
print('小于 30 的数字数量:', count)
print('删除 > 35 的元素后的集合: ', s)
```

输出

原始集合: {32, 34, 40, 15, 16, 22, 23, 24, 25, 27}
小于 30 的数字数量: 7
删除 > 35 的元素后的集合: {32, 34, 15, 16, 22, 23, 24, 25, 27}

(e) 以下集合运算符的作用是什么？
|, &, ^, -

答案

- | 两个集合的并集
- & 两个集合的交集
- ^ 两个集合的对称差集
- - 两个集合的差集

(f) 以下集合运算符的作用是什么？
|=, &=, ^=, -=

答案

- |= 执行两个集合的并集，并将结果存储在左操作数中。
- &= 执行两个集合的交集，并将结果存储在左操作数中。
- ^= 查找两个集合的对称差集，并将结果存储在左操作数中。
- -= 查找两个集合的差集，并将结果存储在左操作数中。

(g) 你将如何删除字符串、列表和元组中存在的所有重复元素？

答案

```
#### 从字符串、列表和元组中删除重复项
s = 'Razmattaz'
s = "".join(sorted(set(s), key = s.index))
print(s)
lst = ['R', 'a', 'a', 'z', 'm', 'a', 't', 't', 'a', 'z']
lst = list(sorted(set(lst), key = lst.index))
print(lst)
tpl = ('R', 'a', 'a', 'z', 'm', 'a', 't', 't', 'a', 'z')
tpl = tuple(sorted(set(tpl), key = tpl.index))
print(tpl)
```

输出

Razmt
['R', 'a', 'z', 'm', 't']
('R', 'a', 'z', 'm', 't')

(h) 哪个运算符用于确定一个集合是否是另一个集合的子集？

答案

`<` 运算符用于确定一个集合是否是另一个集合的子集。相应的方法是 **issubset( )**，它给出相同的结果。

```
s = {12, 15, 13, 23, 22, 16, 17}
t = {13, 15, 22}
print(t.issubset(s))    # 打印 True
print(t < s)            # 打印 True
```

(i) 以下程序的输出是什么？

```
s = {'Mango', 'Banana', 'Guava', 'Kiwi'}
s.clear( )
print(s)
del(s)
print(s)
```

输出

set( )
NameError: name 's' is not defined

(j) 以下哪种是创建空集合的正确方法？

```
s1 = set( )
s2 = {}
```

**s1** 和 **s2** 的类型是什么？你将如何确认类型？

答案

**s1 = set( )** 是创建空集合的正确方法。
**s2 = {}** 是创建空字典的正确方法。
**s1** 和 **s2** 的类型可以如下所示确认：

```
s1 = set( )
s2 = { }
print(type(s1))
print(type(s2))
```

其输出将是：

```
<class 'set'>
<class 'dict'>
```

## 11 字典

![](img/ac61a02854da7ea89272a34be93604dc_96_0.png)

### [A] 判断以下陈述是真还是假：

(a) 字典元素可以使用基于位置的索引访问。
*答案*
假

(b) 字典是不可变的。
*答案*
假

(c) 字典保留插入顺序。
*答案*
假

(d) 字典 **d** 中的第一个键值对可以使用表达式 **d[0]** 访问。
*答案*
假

(e) **courses.clear( )** 将删除名为 **courses** 的字典对象。
*答案*
假

(f) 可以嵌套字典。
*答案*
真

(g) 可以在字典中为一个键存储多个值。
*答案*
真

### [B] 尝试回答以下问题：

(a) 编写一个程序，从键盘读取一个字符串，并创建一个包含字符串中每个字符出现频率的字典。同时以直方图的形式打印这些出现次数。

```
#### 统计字符串中字符的频率
s = input('输入任意字符串: ')
freq = { }
for ch in s :
    if ch in freq :
        freq[ch] += 1
    else :
        freq[ch] = 1
print ('所有字符的计数为: ', freq)

for k, v in freq.items( ) :
    print(k, ':', end = '')
    for i in range(0, v) :
        print('*', end ='')
    print( )
```

输出

输入任意字符串: Ashish Samant
所有字符的计数为: {'A': 1, 's': 2, 'h': 2, 'i': 1, ' ': 1, 'S': 1, 'a': 2, 'm': 1, 'n': 1, 't': 1}
A :*
s :**
h :**
i :*
 :*
S :*
a :**
m :*
n :*
t :*

(b) 创建一个包含学生姓名和他们在三个科目中获得的分数的字典。编写一个程序，将三个科目的分数替换为三个科目的总分和平均分。同时报告班级的优等生。

程序

```
#### 字典操作
import operator
students = {
    'Dipti' : { 'Maths' : 48, 'eng' : 60, 'hindi' : 95},
    'Smriti' : { 'Maths' : 75, 'eng' : 68, 'hindi' : 89},
    'Subodh' : { 'Maths' : 45, 'eng' : 66, 'hindi' : 87}
}
tot = {}
topper_name = ""
topper_marks = 0
for nam, info in students.items() :
    total = 0
    for sub, marks in info.items() :
        total = total + marks

    avg = int(total / 3)
    students[nam] = {'Total' : total, 'Average' : avg}
    if avg > topper_marks :
        topper_name = nam
        topper_marks = avg

print(students)
print ('班级优等生:' , topper_name)
print('优等生分数:', topper_marks)
```

输出

{'Dipti': {'Total': 203, 'Average': 67}, 'Smriti': {'Total': 232, 'Average': 77}, 'Subodh': {'Total': 198, 'Average': 66}}
班级优等生: Smriti
优等生分数: 77

(c) 给定以下字典：

```
portfolio = { 'accounts' :  [ 'SBI' , 'IOB'],
              'shares' : ['HDFC' , 'ICICI' , 'TM' , 'TCS'],
              'ornaments' : ['10 gm gold', '1 kg silver']}
```

编写一个程序执行以下操作：

- 向 portfolio 添加一个名为 'MF' 的键，其值为 'Reliance' 和 'ABSL'。
- 将 'accounts' 的值设置为包含 'Axis' 和 'BOB' 的列表。
- 对存储在 'shares' 键下的列表中的项目进行排序。
- 删除存储在 'ornaments' 键下的列表。

程序

```
#### 字典操作
portfolio = {
    'accounts' : [ 'SBI', 'IOB'],
```

## 第11章：字典

```python
'shares' : ['HDFC', 'ICICI', 'TM', 'TCS'],
'ornaments' : ['10 gm gold', '1 kg silver']
}
```

```python
portfolio['MF'] = ['Reliance','ABSL']
print(portfolio)
```

```python
portfolio['accounts'] = ['Axis', 'BOB']
print(portfolio)
```

```python
lst = portfolio['shares']
portfolio['shares'] = sorted(lst)
print(portfolio)
```

```python
del(portfolio['ornaments'])
print(portfolio)
```

输出

```
{'accounts': ['SBI', 'IOB'], 'shares': ['HDFC', 'ICICI', 'TM', 'TCS'],
'ornaments': ['10 gm gold', '1 kg silver'], 'MF': ['Reliance', 'ABSL']}
{'accounts': ['Axis', 'BOB'], 'shares': ['HDFC', 'ICICI', 'TM', 'TCS'],
'ornaments': ['10 gm gold', '1 kg silver'], 'MF': ['Reliance', 'ABSL']}
{'accounts': ['Axis', 'BOB'], 'shares': ['HDFC', 'ICICI', 'TCS', 'TM'],
'ornaments': ['10 gm gold', '1 kg silver'], 'MF': ['Reliance', 'ABSL']}
{'accounts': ['Axis', 'BOB'], 'shares': ['HDFC', 'ICICI', 'TCS', 'TM'], 'MF':
['Reliance', 'ABSL']}
```

(d) 创建两个字典——一个包含杂货物品及其价格，另一个包含杂货物品及购买数量。使用这两个字典中的值计算总账单。

程序

```python
#### 计算总账单金额
prices = { 'Bottles' : 30, 'Tiffin' : 100, 'Bag' : 400, 'Bicycle' : 2000 }
stock = { 'Bottles' : 10, 'Tiffin' : 8, 'Bag' : 1, 'Bicycle' : 5}
total = 0
for key in prices :
    value = prices[key] * stock[key]
    total += value
print('Total Bill Amount =' , total)
```

输出

```
Bottles 300
Tiffin 800
Bag 400
Bicycle 10000
Total Bill Amount = 11500
```

(e) 你会使用哪些函数从给定的字典中获取所有键、所有值和键值对？

答案

获取所有键 - **keys( )**
获取所有值 - **values( )**
获取键值对 - **items( )**

(f) 创建一个包含10个用户名和密码的字典。从键盘接收用户名和密码，并在字典中搜索它们。根据是否找到匹配项，在屏幕上打印相应的消息。

程序

```python
#### 检查有效用户
users = {
    'Sanjay' : 'ceftum1250', 'Rahul' : 'Crocin100',
    'Sanket' : 'Metrogyl50', 'Shyam' : 'Miopass10',
    'Satish' : 'mvpxx_9000', 'Srishti' : 'Relaxo!',
    'Smriti' : 'newyear200', 'Sakhi' : 'Bday1711',
    'Raakhi' : 'jallosh200', 'Rahika' : 'Ultu1900'
}
userid = input('Enter username: ')
password = input('Enter password: ')

for k, v in users.items( ) :
    if k == userid and v == password :
        print('Valid username and password')
        exit( )

print('Invalid username and password')
```

输出

```
Enter username: Smriti
Enter password: newyear200
Valid username and password
```

(g) 给定以下字典

```python
marks = { 'Subu' : { 'Maths' : 88, 'Eng' : 60, 'SSt' : 95 },
          'Amol' : { 'Maths' : 78, 'Eng' : 68, 'SSt' : 89 },
          'Raka' : { 'Maths' : 56, 'Eng' : 66, 'SSt' : 77 }}
```

编写一个程序来执行以下操作：

- 打印Amol在英语中获得的分数。
- 将Raka在数学中获得的分数设置为77。
- 按姓名对字典进行排序。

程序

```python
marks = {
    'Subu' : { 'Maths' : 88, 'Eng' : 60, 'SSt' : 95 },
    'Amol' : { 'Maths' : 78, 'Eng' : 68, 'SSt' : 89 },
    'Raka' : { 'Maths' : 56, 'Eng' : 66, 'SSt' : 77 }
}
print('Marks obtained by Amol in english:', marks['Amol']['Eng'])
marks['Raka']['Maths'] = '77'
print(marks)
marks = dict(sorted(marks.items( )))
print(marks)
```

输出

```
Marks obtained by Amol in english: 68
{'Subu': {'Maths': 88, 'Eng': 60, 'SSt': 95}, 'Amol': {'Maths': 78, 'Eng': 68, 'SSt': 89}, 'Raka': {'Maths': 56, 'Eng': 66, 'SSt': 77}}
{'Amol': {'Maths': 78, 'Eng': 68, 'SSt': 89}, 'Raka': {'Maths': 56, 'Eng': 66, 'SSt': 77}, 'Subu': {'Maths': 88, 'Eng': 60, 'SSt': 95}}
```

(h) 创建一个字典，存储以下数据：

| 接口 | IP地址 | 状态 |
|-----------|------------|--------|
| eth0      | 1.1.1.1    | up     |
| eth1      | 2.2.2.2    | up     |
| wlan0     | 3.3.3.3    | down   |
| wlan1     | 4.4.4.4    | up     |

编写一个程序来执行以下操作：

- 查找给定接口的状态。
- 查找所有状态为up的接口及其IP地址。
- 查找接口总数。
- 向字典中添加两个新条目。

程序

```python
#### 使用嵌套字典
ifs = {
    'eth0':{'IP' : '1.1.1.1', 'Status' : 'up'},
    'eth1':{'IP' : '2.2.2.2', 'Status' : 'up'},
    'wlan0':{'IP' : '3.3.3.3', 'Status' : 'down'},
    'wlan1':{'IP' : '4.4.4.4', 'Status' : 'up'}
}
test = input('Enter interface: ')
print(ifs[test]['Status'])

for k, v in ifs.items( ) :
    if v['Status'] == 'up' :
        print(k, v['IP'])

print('Total interfaces = ', len(ifs))

ifs['eth2'] = {'IP' : '5.5.5.5', 'Status' :'down'}
ifs['wlan2'] = {'IP' : '6.6.6.6', 'Status' : 'up'}
for k, v in ifs.items( ) :
    print(k, v)
```

输出

```
Enter interface: eth1
up
eth0 1.1.1.1
eth1 2.2.2.2
wlan1 4.4.4.4
Total interfaces = 4
eth0 {'IP': '1.1.1.1', 'Status': 'up'}
eth1 {'IP': '2.2.2.2', 'Status': 'up'}
wlan0 {'IP': '3.3.3.3', 'Status': 'down'}
wlan1 {'IP': '4.4.4.4', 'Status': 'up'}
eth2 {'IP': '5.5.5.5', 'Status': 'down'}
wlan2 {'IP': '6.6.6.6', 'Status': 'up'}
```

(i) 假设一个字典包含5个姓名和分数的键值对。编写一个程序，从最后一对打印到第一对。每打印一对就删除它，使得打印结束时字典为空。

程序

```python
marks = { 'Subu' : 88, 'Amol' : 78, 'Raka' : 56, 'Dinesh' : 68, 'Ranjit' : 88}
l = len(marks)
for i in range(l) :
    print(marks.popitem( ))
print(marks)
```

[C] 回答以下问题：

(a) 以下代码片段的输出是什么？

```python
d = { 'Milk' : 1, 'Soap' : 2, 'Towel' : 3, 'Shampoo' : 4, 'Milk' : 7}
print(d[0], d[1], d[2])
```

答案

错误：字典元素不能使用基于位置的索引访问。

(b) 以下哪些陈述是正确的？

- i. 字典总是包含唯一的键。
- ii. 字典中的每个键可以有多个值。
- iii. 如果为同一个键分配了不同的值，最新的值将生效。

答案

- i. 正确
- ii. 正确
- iii. 正确

(c) 如何创建一个空列表、空元组、空集合和空字典？

答案

```python
l = [ ]
t = ( )
s = set( )
d = { }
```

(d) 如何创建一个列表、元组、集合和字典，每个只包含一个元素？

程序

```python
l = [10]
t = (10,)
s = {10}
d = {10: 'A'}
```

(e) 给定以下字典：

```python
d = { 'd1': {'Fruitname' : 'Mango', 'Season' : 'Summer'},
      'd2': {'Fruitname' : 'Orange', 'Season' : 'Winter'}}
```

如何访问并打印Mango和Winter？

程序

```python
d = { 'd1': {'Fruitname' : 'Mango', 'Season' : 'Summer'},
      'd2': {'Fruitname' : 'Orange', 'Season' : 'Winter'}}
print(d['d1']['Fruitname'])
print(d['d2']['Season'])
```

(f) 在下表中，如果列中提到的数据类型具有该属性，请在框中打勾。

| 属性 | str | list | tuple | set | dict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 对象 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 集合 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 可变 | | ✓ | | ✓ | ✓ |
| 有序 | ✓ | ✓ | ✓ | | |
| 按位置索引 | ✓ | ✓ | ✓ | | |
| 按键索引 | | | | | ✓ |
| 可迭代 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 允许切片 | ✓ | ✓ | ✓ | | |
| 允许嵌套 | ✓ | ✓ | ✓ | | ✓ |
| 同质元素 | ✓ | | | ✓ | ✓ |
| 异质元素 | | ✓ | ✓ | | |

(g) 以下数据类型最常见的用途是什么？

- str - 字符集合
- list - 相似元素
- tuple - 成对或三元组
- set - 唯一元素
- dict - 键值对

## 12 推导式

[A] 判断以下陈述是真还是假：

(a) 元组推导式提供了一种快速紧凑的方式来生成元组。

答案
真

(b) 列表推导式和字典推导式可以嵌套。

答案
真

(c) 在列表推导式中使用的列表在迭代时不能被修改。

答案
真

(d) 集合是不可变的，不能用于推导式。

答案
假

(e) 推导式可用于创建列表、集合或字典。

答案
真

[B] 回答以下问题：

(a) 编写一个程序，使用列表推导式生成从(1, 1)到(5, 5)第一象限内所有点的整数坐标列表。

```python
coord = [(x, y) for x in range(1, 6)for y in range(1, 6)]
print(coord)
```

## 第12章：推导式

103

输出

```
[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
```

(b) 使用列表推导式，编写一个程序，通过将列表中的每个元素乘以10来创建一个新列表。

程序

```
lst = [-7, 10, 34, 2, 5, 45, 67]
lst = [(x * 10) for x in lst]
print(lst)
```

输出

```
[-70, 100, 340, 20, 50, 450, 670]
```

(c) 编写一个程序，使用列表推导式生成前20个斐波那契数。

程序

```
lst = [0, 1]
[lst.append(lst[k - 1] + lst[k - 2]) for k in range(2, 20)]
print('前20个斐波那契数:', lst)
```

输出

```
前20个斐波那契数:
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
```

(d) 编写一个程序，使用列表推导式生成两个列表。一个列表应包含前20个奇数，另一个应包含前20个偶数。

程序

```
lst1 = [x for x in range(40) if x % 2 != 0]
print('前20个奇数:')
print(lst1)
lst2 = [x for x in range(40) if x % 2 == 0]
print('前20个偶数:')
print(lst2)
```

输出

```
前20个奇数:
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
前20个偶数:
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
```

(e) 假设一个列表包含正数和负数。编写一个程序来创建两个列表——一个包含正数，另一个包含负数。

程序

```
lst = [1, 2, 5, -11, -9, 10, 13, 15, -17, -19, -21, -23, 25, 27, -29]
pos = [num for num in lst if num > 0]
print(pos)
neg = [num for num in lst if num < 0]
print(neg)
```

输出

```
[1, 2, 5, 10, 13, 15, 25, 27]
[-11, -9, -17, -19, -21, -23, -29]
```

(f) 假设一个列表包含5个字符串。编写一个程序将所有这些字符串转换为大写。

程序

```
lst = ['abc', 'def', 'ghi', 'jkl', 'lmn']
lst = [s.upper() for s in lst]
print(lst)
```

输出

```
['ABC', 'DEF', 'GHI', 'JKL', 'LMN']
```

(g) 编写一个程序，使用列表推导式将华氏温度列表转换为等效的摄氏温度。

程序

```
farh = [101, 120, 100, 67, 32]
celsius = [(e - 32) * 5 / 9 for e in farh]
print(celsius)
```

输出

```
[38, 48, 37, 19, 0]
```

(h) 编写一个程序，生成一个4 x 5的二维矩阵，其中包含40到160范围内4的随机倍数。

程序

```
import random
rows, cols = (5, 4)
arr = [[(4 * random.randint(10, 40)) for i in range(cols)] for j in range(rows)]
print(arr)
```

输出

```
[[72, 108, 92, 148], [88, 152, 76, 96], [148, 108, 104, 136], [132, 48, 160, 116], [140, 40, 104, 48]]
```

(i) 编写一个程序，将列表中的单词转换为大写并存储在集合中。

程序

```
lst = ['function', 'office', 'type', 'product', 'most']
s = set([word.upper() for word in lst])
print(s)
```

输出

```
{'MOST', 'TYPE', 'FUNCTION', 'OFFICE', 'PRODUCT'}
```

[C] 尝试回答以下问题：

(a) 考虑以下代码片段：

```
s = set([int(n) for n in input('Enter values: ').split()])
print(s)
```

如果输入为 1 2 3 4 5 6 7 2 4 5 0，上述代码片段的输出将是什么？

输出

```
{0, 1, 2, 3, 4, 5, 6, 7}
```

(b) 你将如何将以下代码转换为列表推导式？

```
a = []
for n in range(10, 30):
    if n % 2 == 0:
        a.append(n)
```

答案

```
a = [n for n in range(10, 30) if n % 2 == 0]
```

(c) 你将如何将以下代码转换为集合推导式？

```
a = set()
for n in range(21, 40):
    if n % 2 == 0:
        a.add(n)
print(a)
```

答案

```
a = {n for n in range(21, 40) if n % 2 == 0}
```

(d) 以下代码片段的输出将是什么？

```
s = [a + b for a in ['They ', 'We '] for b in ['are gone!', 'have come!']]
print(s)
```

答案

```
['They are gone!', 'They have come!', 'We are gone!', 'We have come!']
```

(e) 从句子

```
sent = 'Pack my box with five dozen liquor jugs'
```

你将如何生成下面给定的集合？

```
{'liquor', 'jugs', 'with', 'five', 'dozen', 'Pack'}
```

输出

```
sent = 'Pack my box with five dozen liquor jugs'
sent = {w for w in sent.split() if len(w) > 3}
```

(f) 以下哪项是字典推导式的正确形式？

- i. dict_var = {key : value for (key, value) in dictionary.items()}
- ii. dict_var = {key : value for (key, value) in dictionary}
- iii. dict_var = {key : value for (key, value) in dictionary.keys()}

输出

```
dict_var = {key : value for (key, value) in dictionary.items()}
```

(g) 使用推导式，你将如何将 {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5} 转换为 {'A' : 100, 'B' : 200, 'C' : 300, 'D' : 400, 'E' : 500}？

输出

```
d = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5}
d = {key.upper() : value * 100 for (key, value) in d.items()}
print(d)
```

(h) 以下代码片段的输出将是什么？

```
lst = [2, 7, 8, 6, 5, 5, 4, 4, 8]
s = {True if n % 2 == 0 else False for n in lst}
print(s)
```

输出

```
{False, True}
```

(i) 你将如何将 d = {'AMOL' : 20, 'ANIL' : 12, 'SUNIL' : 13, 'RAMESH' : 10} 转换为 {'Amol': 400, 'Anil': 144, 'Sunil': 169, 'Ramesh': 100}？

输出

```
d = {'AMOL' : 20, 'ANIL' : 12, 'SUNIL' : 13, 'RAMESH' : 10}
d = {key.capitalize() : value * value for (key, value) in d.items()}
print(d)
```

(j) 你将如何将下面列表中的单词转换为大写并存储在集合中？

```
lst = ['Amol', 'Vijay', 'Vinay', 'Rahul', 'Sandeep']
```

输出

```
lst = ['Amol', 'Vijay', 'Vinay', 'Rahul', 'Sandeep']
d = {word.upper() for word in lst}
print(d)
```

## 13 函数

[A] 回答以下问题：

(a) 编写一个程序，定义一个函数 **count_lower_upper()**，该函数接受一个字符串并计算其中大写和小写字母的数量。它应该将这些值作为字典返回。为一些示例字符串调用此函数。

程序

```
def count_lower_upper(s):
    dlu = {'Lower' : 0, 'Upper' : 0}
    for ch in s:
        if ch.islower():
            dlu['Lower'] += 1
        elif ch.isupper():
            dlu['Upper'] += 1
    return(dlu)

d = count_lower_upper('James BOnD ')
print(d)
d = count_lower_upper('Anant Amrut Mahalle')
print(d)
```

输出

```
{'Lower': 6, 'Upper': 3}
{'Lower': 14, 'Upper': 3}
```

(b) 编写一个程序，定义一个函数 **compute()**，该函数计算 n + nn + nnn + nnnn 的值，其中 n 是函数接收的数字。测试数字 4 和 7 的函数。

程序

```
import math
def compute(n):
    s = 0
    num = 0
    for outer in range(0, 4):
        num = num * 10 + n
        s = s + num
    return(s)

total = compute(7)
print('n + nn + nnn + nnnn 的值是', total)
total = compute(4)
print('n + nn + nnn + nnnn 的值是', total)
```

输出

```
n + nn + nnn + nnnn 的值是 8638
n + nn + nnn + nnnn 的值是 4936
```

(c) 编写一个程序，定义一个函数 **create_array()**，用于创建并返回一个3D数组，其维度传递给函数。同时，将此数组的每个元素初始化为传递给函数的值。

程序

```
def create_array(i, j, k, num):
    l = [[[num for col in range(k)] for row in range(j)] for twods in range(i)]
    return(l)

lst = create_array(4, 3, 2, 10)
print(lst)
```

输出

```
[[[10, 10], [10, 10], [10, 10]], [[10, 10], [10, 10], [10, 10]], [[10, 10], [10, 10], [10, 10]], [[10, 10], [10, 10], [10, 10]]]
```

(d) 编写一个程序，定义一个函数 **create_list()**，用于创建并返回一个列表，该列表是传递给它的两个列表的交集。

程序

```
def create_list(l1, l2):
    l3 = list(set(l1) & set(l2))
    return(l3)

lst1 = [10, 20, 30, 40, 50]
lst2 = [1, 2, 3, 40, 10]
lst3 = create_list(lst1, lst2)
print(lst3)
```

## 14 递归

### [A] 判断以下陈述的真假：

(a) 如果一个递归函数使用了三个变量 **a**、**b** 和 **c**，那么在每次递归调用中都会使用同一组变量。

*答案*

真

(b) 如果一个递归函数使用了三个变量 **a**、**b** 和 **c**，那么在每次递归调用中都会使用同一组变量。

*答案*

假

(c) 内存中会创建递归函数的多个副本。

*答案*

假

(d) 一个递归函数必须包含至少 1 个 **return** 语句。

*答案*

真

(e) 每个使用 **while** 或 **for** 循环完成的迭代都可以用递归替换。

*答案*

真

(f) 逻辑可以用自身形式表达的问题是编写递归函数的良好候选。

*答案*

真

(g) 尾递归类似于循环。

*答案*

真

(h) 如果基本情况没有正确定义，可能会发生无限递归。

*答案*

真

(i) 与使用循环的函数相比，递归函数更容易编写、理解和维护。

*答案*

假

### [B] 回答以下问题：

(a) 以下程序使用尾递归计算前 5 个自然数的和。重写该函数以使用头递归获取和。

```
def headsum(n) :
    if n != 0 :
        s = n + headsum(n - 1)
    else :
        return 0
    return s

print('Sum of First 5 Natural numbers = ', headsum(5))
```

程序

```
def headsum(n) :
    if n != 0 :
        s = n + headsum(n - 1)
    else :
        return 0
    return s

print('Sum of First 5 Natural numbers = ', headsum(5))
```

输出

Sum of First 5 Natural numbers = 15

(b) 有三个标记为 A、B 和 C 的柱子。四个圆盘放在柱子 A 上。最底下的圆盘最大，圆盘大小依次递减，最上面的圆盘最小。游戏的目标是将圆盘从柱子 A 移动到柱子 C，使用柱子 B 作为辅助柱子。游戏规则如下：

-   一次只能移动一个圆盘，并且它必须是某个柱子上的最上面的圆盘。
-   较大的圆盘永远不应放在较小的圆盘上面。

编写一个程序，打印出圆盘应移动的顺序，使得柱子 A 上的所有圆盘最终都转移到柱子 C。

程序

```
def move(n, sp, ap, ep) :
    if n == 1 :
        print('Move from', sp ,'to', ep)
    else :
        move(n-1, sp, ep, ap)
        move(1, sp, " ", ep)
        move(n-1, ap, sp, ep)

move(4, 'A', 'B', 'C')
```

输出

Move from A to B
Move from A to C
Move from B to C
Move from A to B
Move from C to A
Move from C to B
Move from A to B
Move from A to C
Move from B to C
Move from B to A
Move from C to A
Move from B to C
Move from A to B
Move from A to C
Move from B to C

(c) 通过键盘输入一个字符串。编写一个递归函数来计算该字符串中元音的数量。

程序

```
def fun(s, idx, count) :
    if idx == len(s):
        return count
    if s[idx] == 'a' or s[idx] == 'e' or s[idx] == 'i' or s[idx] == 'o' or s[idx] == 'u' :
        count += 1
    count = fun(s, idx + 1, count)
    return count

count = fun('Raindrops on roses', 0, 0)
print(count)
```

输出

6

(d) 通过键盘输入一个字符串。编写一个递归函数来移除该字符串中存在的任何制表符。

程序

```
def replace(source, i, n) :
    global target
    if i == n :
        return
    if source[ i ] == '\t' :
        pass
    else :
        target += source[ i ]

    i += 1
    replace(source, i, n)

s = 'Raindrops on Roses and whiskers on kittens'
print(s)
target = ''
replace(s, 0, len(s) - 1)
print(target)
```

输出

Raindrops on Roses and whiskers on kittens
RaindropsonRoses and whiskersonkitten

(e) 通过键盘输入一个字符串。编写一个递归函数来检查该字符串是否是回文。

程序

```
def ispalindrome(st, start, end) :
    if start > end :
        return True

    if st[start] != st[end] :
        return False

    status = ispalindrome(st, start + 1, end - 1)
    return status

st1 = 'malayalam'
st2 = 'malhindilam'
status = ispalindrome(st1, 0, len(st1) - 1)
print(status)
status = ispalindrome(st2, 0, len(st2) - 1)
print(status)
```

输出

True
False

(f) 通过键盘接收两个数字到变量 a 和 b 中。编写一个递归函数来计算 a^b 的值。

程序

```
def power(x, y) :
    if y == 0 :
        return 1
    if y == 1 :
        return x
    prod = x * power(x, y - 1)
    return prod

c = power(2, 5)
print(c)
d = power(3, 4)
print(d)
```

输出

32
81

(g) 编写一个递归函数来反转它接收的数字列表。

程序

```
def reverselist(lst, start, end) :
    if start > end :
        return

    lst[start], lst[end] = lst[end], lst[start]
    reverselist(lst, start + 1, end - 1)

numlst = [10, 20, 30, 40, 50]
reverselist(numlst, 0, 4)
print(numlst)
```

输出

[50, 40, 30, 20, 10]

(h) 一个列表包含一些负数和一些正数。编写一个递归函数，通过将所有负数替换为 0 来清理该列表。

程序

```
def replace(lst, i, n) :
    if i > n :
        return
    if lst[ i ] < 0 :
        lst[ i ] = 0

    i += 1
    replace( lst, i, n)

numlst = [10, 20, -3, -4, 50, -4, 60, 70, -4]
replace(numlst, 0, len(numlst) - 1)
print(numlst)
```

输出

[10, 20, 0, 0, 50, 0, 60, 70, 0]

(i) 编写一个递归函数来获取给定列表中所有数字的平均值。

程序

```
def get_average(lst, n) :
    if n == 1 :
        return lst[ 0 ]
    else :
        return (get_average(lst, n - 1) * (n - 1) + lst[n - 1]) / n
    print(sum, n)
    return sum

numlst = [10, 20, 30, 40, 50, 60]
avg = get_average(numlst, len(numlst))
print(avg)
```

输出

35.0

(j) 编写一个递归函数来获取给定字符串的长度。

程序

```
def get_length(st) :
    if st == "" :
        return 0
    else :
        return 1 + get_length(st[1:])
```

---

输出

[40, 10]

(e) 编写一个程序，定义一个函数 **sanitize_list( )** 来移除它接收的列表中的所有重复条目。

程序

```
def sanitize_list(l) :
    l = list(set(l))
    return(l)

lst = [10, 3, 30, 10, 4, 3, -5, 10, 0, -5]
lst = sanitize_list(lst)
print('List after removing duplicates: ', lst)
```

输出

List after removing duplicates: [0, 3, 4, 10, -5, 30]

(f) 以下程序中对 **print_it( )** 的哪些调用会报告错误？

```
def print_it(i, a, s, *args) :
    print( )
    print(i, a, s, end = ' ')
    for var in args :
        print(var, end = ' ')

print_it(10, 3.14)
print_it(20, s = 'Hi', a = 6.28)
print_it(a = 6.28, s = 'Hello', i = 30)
print_it(40, 2.35, 'Nag', 'Mum', 10)
```

*答案*

第一个调用

print_it(10, 3.14)

会报告错误 'missing 1 required positional argument: 's''

其他 3 个调用是正确的。

(g) 以下程序中对 **fun( )** 的哪些调用会报告错误？

```
def fun(a, *args, s = '!') :
    print(a, s)
    for i in args :
        print(i, s)

fun(10)
fun(10, 20)
fun(10, 20, 30)
fun(10, 20, 30, 40, s = '+')
```

*答案*

没有错误。所有调用都是正确的。

### [B] 尝试回答以下问题：

(a) 以下代码中传递给函数 **fun( )** 的是什么？

```
int a = 20
lst = [10, 20, 30, 40, 50]
fun(a, lst)
```

*答案*

整数的地址和列表的地址。

(b) 以下哪些是有效的 **return** 语句？

```
return (a, b, c)
return a + b + c
return a, b, c
```

*答案*

所有都是有效的 return 语句。

(c) 以下程序的输出是什么？

```
def fun( ) :
    print('First avatar')
fun( )

def fun( ) :
    print('New avatar')
fun( )
```

*答案*

First avatar
New avatar

(d) 你将如何定义一个包含三个 **return** 语句的函数，每个语句返回不同类型的值？

*答案*

```
def fun(a) :
    if a < 0 :
        return 10
    if a == 0 :
        return 10.0
    if a > 0 :
        return '10'

print(fun(-5))
print(fun(5))
print(fun(0))
```

(e) 函数定义可以嵌套吗？如果可以，为什么你想这样做？

*答案*

函数定义可以嵌套。有时我们需要一个只被一个函数需要的函数。然后可以将这个函数嵌套在需要调用它的函数内部。

(f) 你将如何调用 **print_it( )** 来打印 **tpl** 的元素？

```
def print_it(a, b, c, d, e) :
    print(a, b, c, d, e)
tpl = ('A', 'B', 'C', 'D', 'E')
```

*答案*

```
print_it(*tpl)
```

## 15 函数式编程

### 126 让我们用Python来解答

[A] 判断以下陈述的真假：

(a) lambda函数不能与**reduce( )**函数一起使用。

*答案*

假

(b) lambda、**map( )**、**filter( )**、**reduce( )**可以在一个单独的表达式中组合使用。

*答案*

真

(c) 虽然函数可以赋值给变量，但不能使用这些变量来调用函数。

*程序*

假

(d) 函数可以作为参数传递给另一个函数，也可以从函数中返回。

*程序*

真

(e) 函数可以在执行时构建，就像列表、元组等一样。

*程序*

真

(f) Lambda函数总是匿名的。

*程序*

真

[B] 使用lambda、**map( )**、**filter( )**和**reduce( )**或**它们的组合**来执行以下任务：

(a) 假设一个字典包含宠物类型（猫、狗等）、宠物名字和宠物年龄。编写一个程序，获取所有狗的年龄总和。

程序

```
def fun1(d) :
    if d['Type'] == 'Dog' :
        return d['Age']
    else :
        return 0

def fun2(n) :
    if n == 0 :
        return False
    else :
        return True

dct = {
    'A101' : {'Type' : 'Cat', 'Name' : 'Tauby', 'Age' : 6 },
    'A102' : {'Type' : 'Dog', 'Name' : 'Tommy', 'Age' : 8 },
    'A103' : {'Type' : 'Dog', 'Name' : 'Tiger', 'Age' : 10 }
}
lst2 = list(filter(fun2, list(map(fun1, list(dct.values( ))))))
print('The Sum of all Dogs ages:', sum(lst2)/len(lst2))
```

输出

The Sum of all Dogs ages: 9.0

(b) 考虑以下列表：
lst = [1.25, 3.22, 4.68, 10.95, 32.55, 12.54]
列表中的数字代表圆的半径。编写一个程序，获取这些圆的面积列表，并四舍五入到两位小数。

程序

```
lst = [1.25, 3.22, 4.68, 10.98, 32.55, 12.54]
area_lst = list(map(lambda n : round(n * n * 3.14, 2), lst))
print(area_lst)
```

输出

[4.91, 32.56, 68.77, 378.56, 3326.84, 493.77]

(c) 考虑以下列表：
nums = [10, 20, 30, 40, 50, 60, 70, 80]
strs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
编写一个程序，获取一个元组列表，其中每个元组包含一个来自一个列表的数字和一个来自另一个列表的字符串，顺序与它们在原始列表中出现的顺序相同。

程序

```
nums = [10, 20, 30, 40, 50, 60, 70, 80]
strs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ltpl = list(map(lambda x, y : (x, y), nums, strs))
print(ltpl)
```

输出

```
[(10, 'A'), (20, 'B'), (30, 'C'), (40, 'D'), (50, 'E'), (60, 'F'), (70, 'G'), (80, 'H')]
```

(d) 假设一个字典包含学生姓名和他们在考试中获得的分数。编写一个程序，获取考试中得分超过40分的学生列表。

程序

```
students = {
    'Dipti' : 55, 'Smriti' :12, 'Subodh' : 45,
    'Meenal' : 33, 'Harsha' : 40 , 'Bhushan' : 42
}
lst = filter(lambda x : x[1] >= 40, students.items( ))
print(list(lst))
```

输出

```
[('Dipti', 55), ('Subodh', 45), ('Harsha', 40), ('Bhushan', 42)]
```

(e) 考虑以下列表：
lst = ['Malayalam', 'Drawing', 'madamlamadam', '1234321']
编写一个程序，打印出那些是回文的字符串。

程序

```
lst = ['Malayalam', 'Drawing', 'madamlamadam', '1234321']
lst1 = list(filter(lambda x : (x == "".join(reversed(x))), lst))
print(lst1)
```

输出

```
['1234321']
```

(f) 一个列表包含员工姓名。编写一个程序，筛选出长度超过8个字符的姓名。

程序

```
lst = ['Parmeshwar', 'Kashmira', 'Seema', 'Roopa', 'Mahalaxmi']
lst1 = filter(lambda x : len(x) >= 8, lst)
print(list(lst1))
```

输出

```
['Parmeshwar', 'Kashmira', 'Mahalaxmi']
```

(g) 一个字典包含以下关于5名员工的信息：
名
姓
年龄
等级（熟练、半熟练、高度熟练）

编写一个程序，获取高度熟练的员工（名 + 姓）列表。

程序

```
d = {
    'Dinesh' : {'last_name' : 'Sahare', 'age' : 30, 'Grade' : 'Skilled'},
    'Ram' : {'last_name' : 'Jog', 'age' : 35, 'Grade' : 'Semi-Skilled'},
    'S.' : {'last_name' : 'Sam', 'age' : 25, 'Grade' : 'Highly-Skilled'},
    'Adi' : {'last_name' : 'Lim', 'age' : 25, 'Grade' : 'Highly-Skilled'},
    'Ann' : {'last_name' : 'Mir', 'age' : 25, 'Grade' : 'Highly-Skilled'}
}
lst = filter(lambda x : x[1]['Grade'] == 'Highly-Skilled', d.items( ))
print(list(lst))
```

输出

```
[('S.', {'last_name': 'Sam', 'age': 25, 'Grade': 'Highly-Skilled'}), ('Adi', {'last_name': 'Lim', 'age': 25, 'Grade': 'Highly-Skilled'}), ('Ann', {'last_name': 'Mir', 'age': 25, 'Grade': 'Highly-Skilled'})]
```

(h) 考虑以下列表：
lst = ['Benevolent', 'Dictator', 'For', 'Life']

编写一个程序，获取字符串 'Benevolent Dictator For Life'。

程序

```
lst = ['Benevolent', 'Dictator', 'For', 'Life']
s = ' '.join(map(str, lst))
print(s)
```

输出

Benevolent Dictator For Life

(i) 考虑以下班级学生列表。
lst = ['Rahul', 'Priya', 'Chaya', 'Narendra', 'Prashant']

编写一个程序，获取一个列表，其中所有姓名都转换为大写。

程序

```
lst = ['Rahul', 'Priya', 'Chaaya', 'Narendra', 'Prashant']
lst1 = map(lambda x : x.upper( ), lst)
print(list(lst1))
```

输出

```
['RAHUL', 'PRIYA', 'CHAYA', 'NARENDRA', 'PRASHANT']
```

## 16 模块与包

[A] 回答以下问题：

(a) 假设有三个模块 **m1.py**、**m2.py**、**m3.py**，分别包含函数 **f1( )**、**f2( )** 和 **f3( )**。你将如何在你的程序中使用这些函数？

*程序*

目录结构如下：

```
module
    __init__.py
    m1.py
    m2.py
    m3.py
client.py
```

这些函数可以如下使用：

```
#### client.py
import module.m1
import module.m2
import module.m3

module.m1.f1( )
module.m2.f2( )
module.m3.f3( )
```

(b) 编写一个包含函数 **fun1( )**、**fun2( )**、**fun3( )** 和一些语句的程序。向程序添加适当的代码，使其既可以作为模块使用，也可以作为普通程序运行。

*程序*

```
def fun1( ) :
    print('Inside function fun1')

def fun2( ) :
    print('Inside function fun2')

def fun3( ) :
    print('Inside function fun3')

def main( ) :
    fun1()
    fun2()
    fun3()

if (__name__ == '__main__'):
    main()
```

(c) 假设一个模块 **mod.py** 包含函数 **f1( )**、**f2( )** 和 **f3( )**。编写4种形式的导入语句，以便在你的程序中使用这些函数。

*程序*

目录结构如下：
module
    __init__.py
    mod.py
client.py

```
#### client.py - 方法 1
import module.mod
module.mod.f1()
module.mod.f2()
module.mod.f3()
```

```
#### client.py - 方法 2
from module.mod import f1
from module.mod import f2
from module.mod import f3
f1()
f2()
f3()
```

```
#### client.py - 方法 3
from module.mod import *
f1()
f2()
f3()
```

```
#### client.py - 方法 4
import module.mod as M
```

## 第16章：模块与包

[B] 尝试回答以下问题：

(a) 模块和包有什么区别？

答案

模块是一个包含函数定义和语句的`.py`文件。因此，所有`.py`文件都是模块。
如果一个目录中包含名为`__init__.py`的文件，则该目录被视为一个包。

(b) 创建多个包和模块的目的是什么？

答案

创建多个包和模块是为了管理代码的复杂性，并将代码组织成可重用的部分。

(c) 默认情况下，程序中的语句属于哪个模块？我们如何访问这个模块的名称？

答案

默认情况下，程序中的语句属于`__main__`模块。我们通过变量`__name__`来访问这个模块的名称。

(d) 在以下语句中，`a`、`b`、`c`、`x`分别代表什么？
`import a.b.c.x`

答案

`a`、`b`、`c`、`x`是嵌套模块。

(e) 如果模块`m`包含一个函数`fun()`，以下语句有什么问题？

答案

要调用`fun()`，我们必须使用语法

`m.fun()`

(f) **PYTHONPATH**变量的内容是什么？我们如何以编程方式访问其内容？

答案

`PYTHONPATH`环境变量包含一个目录列表。可以通过以下代码片段访问它们：

```
import sys
for p in sys.path :
    print(p)
```

(g) **sys.path**的内容表示什么？**sys.path**内容的顺序表示什么？

答案

**sys.path**变量包含一个目录列表。列表中的第一个目录是当前脚本执行所在的目录。其后是**PYTHONPATH**环境变量中指定的目录列表。

程序中使用的模块将按照列表中目录的相同顺序进行搜索。

(h) 第三方包的列表在哪里维护？

答案

PyPI维护着一个第三方包的列表。

(i) 哪个工具通常用于安装第三方包？

答案

`pip`是用于安装第三方包的常用工具。

(j) 以下导入语句是否达到相同的目的？

```
#### 版本1
import a, b, c, d

#### 版本2
import a
import b
import c
import d

#### 版本3
from a import *
from b import *
from c import *
from d import *
```

答案

是的

[C] 判断以下陈述的真假：

(a) 一个函数可以属于一个模块，而该模块可以属于一个包。

答案
真

(b) 一个包可以包含一个或多个模块。

答案
真

(c) 允许嵌套包。

答案
真

(d) **sys.path**变量的内容不能被修改。

答案
假

(e) 使用`*`来导入模块中定义的所有函数/类是一个好主意。

答案
假

## 第17章：命名空间

[A] 判断以下陈述的真假：

(a) 符号表包含我们程序中使用的每个标识符的信息。

答案

真

(b) 具有全局作用域的标识符可以在程序中的任何地方使用。

答案

真

(c) 可以在另一个函数内部定义一个函数。

答案

真

(d) 如果一个函数嵌套在另一个函数内部，那么外部函数中定义的变量对内部函数是可用的。

答案

真

(e) 如果嵌套函数创建了两个同名的变量，那么这两个变量被视为同一个变量。

答案

假

(f) 内部函数可以从外部函数外部调用。

答案

假

(g) 如果一个函数创建了一个与全局作用域中存在的变量同名的变量，那么该函数的变量将遮蔽全局变量。

答案

真

(h) 在全局作用域定义的变量对程序中定义的所有函数都是可用的。

答案

真

[B] 回答以下问题：

(a) 函数**locals()**和**globals()**有什么区别？

答案

**locals()** - 从函数/方法内部调用时，它返回一个包含该函数/方法可访问的标识符的字典。

**globals()** - 从函数/方法内部调用时，它返回一个包含该函数/方法可访问的全局标识符的字典。

(b) 以下打印语句的输出会相同还是不同？

```
a = 20
b = 40
print(globals( ))
print(locals( ))
```

答案

输出将是相同的。

(c) 一个标识符可以有哪些不同的作用域？

答案

局部(L)、嵌套(E)、全局(G)、内置(B)。

(d) 标识符可以拥有的最宽松的作用域是哪个？

答案

全局作用域是标识符可以拥有的最宽松的作用域。

## 第18章：类与对象

[A] 判断以下陈述的真假：

(a) 类属性和对象属性是相同的。

答案

假

(b) 当同一类的所有对象必须共享一个公共信息项时，类数据成员很有用。

答案

真

(c) 如果一个类有一个数据成员，并且从该类创建了三个对象，那么每个对象都将拥有自己的数据成员。

答案

真

(d) 一个类可以同时拥有类数据和类方法。

答案

真

(e) 通常，类中的数据是私有的，数据通过类的对象方法进行访问/操作。

答案

真

(f) 对象的成员函数必须显式调用，而`__init__()`方法会自动调用。

答案

真

(g) 每当一个对象被实例化时，构造函数就会被调用。

答案

真

(h) `__init__()`方法从不返回值。

答案

真

(i) 当一个对象超出其作用域时，其`__del__()`方法会自动调用。

答案

真

(j) `self`变量始终包含调用该方法/数据的对象的地址。

答案

真

(k) `self`变量甚至可以在类外部使用。

答案

假

(l) `__init__()`方法在对象的生命周期内只被调用一次。

答案

真

(m) 默认情况下，类中的实例数据和方法是公共的。

答案

真

(n) 在一个类中，两个构造函数可以共存——一个0参数构造函数和一个2参数构造函数。

答案

真

[B] 回答以下问题：

(a) 类中的哪些方法充当构造函数？

答案

类中的`__init__()`充当构造函数。

(b) 在以下代码片段中创建了多少个对象？

```
a = 10
b = a
c = b
```

答案

在上面的代码片段中创建了一个对象。

(c) 变量`age`和`_age`有什么区别？

答案

`age`是一个公共属性，而`_age`是对象的一个私有属性。

(d) 函数`vars()`和`dir()`有什么区别？

答案

`vars()` - 返回一个包含属性及其值的字典。
`dir()` - 返回一个属性列表。

(e) 在以下代码片段中，`display()`和`show()`有什么区别？

```
class Message :
    def display(self, msg) :
        pass
    def show(msg) :
        pass
```

答案

**display()**是一个对象方法，因为它接收调用它的对象的地址（在**self**中）。**show()**是一个类方法，它可以独立于对象进行调用。

(f) 在以下代码片段中，**display()**和**show()**有什么区别？

```
m = Message( )
m.display('Hi and Bye' )
Message.show('Hi and Bye' )
```

答案

**display()**是一个对象方法，因为它正在使用对象**m**进行调用。**show()**是一个类方法，它正在使用类**Message**进行调用。

(g) 在以下代码片段中，传递给**display()**的参数有多少个：

```
m = Sample( )
m.display(10, 20, 30)
```

答案

四个。除了10、20、30之外，**m**中包含的对象地址也会传递给**display()**。

[C] 尝试回答以下问题：

(a) 编写一个程序来创建一个表示复数的类，该类包含实部和虚部，然后使用它来执行复数的加法、减法、乘法和除法。

程序

```
import math

class Complex( ) :
    def __init__(self, x, y) :
        self.real = x
        self.imag = y
```

## 第18章：类与对象

149

```python
def display(self):
    if self.imag < 0:
        print(self.real, self.imag, 'i')
    else:
        print(self.real, '+', self.imag, 'i')
```

```python
def add(self, x):
    r = self.real + x.real
    i = self.imag + x.imag
    return Complex(r, i)
```

```python
def subtract(self, x):
    r = self.real - x.real
    i = self.imag - x.imag
    return Complex(r, i)
```

```python
def multiply(self, x):
    r = self.real * x.real - self.imag * x.imag
    i = self.real * x.imag + self.imag * x.real
    return Complex(r, i)
```

```python
def conj(self):
    r = self.real
    i = -self.imag
    return Complex(r, i)
```

```python
def mods(self):
    mod2 = self.real * self.real + self.imag * self.imag
    return math.sqrt(mod2)
```

```python
def divide(self, x):
    m = x.mods()
    c = x.conj()
    if m == 0:
        print('Unable to divide the complex numbers')
    else:
        quo = self.multiply(c)
        quo.real = quo.real / m
        quo.imag = quo.imag / m
        return quo
```

```python
a = Complex(2, 3)
b = Complex(6, -1)
print('a: ', end = "")
a.display( )
print('b: ', end = "")
b.display( )

c = a.add(b)
print('a + b = ', end = "")
c.display( )
d = a.subtract(b)
print('a - b = ', end = "")
d.display( )
e = a.multiply(b)
print('a * b = ', end = "")
e.display( )
f = a.divide ( b )
print('a / b = ', end = "")
f.display( )
```

输出

```
a: 2 + 3 i
b: 6 -1 i
a + b = 8 + 2 i
a - b = -4 + 4 i
a * b = 15 + 16 i
a / b = 1.4795908857482156 + 3.287979746107146 i
```

(b) 编写一个程序，实现一个**矩阵**类，并对3x3矩阵执行加法、乘法和转置操作。

程序

```python
class Matrix :
    size = 3
    def __init__(self, r, c) :
        self.rows = r
        self.cols = c
        self.arr = [ ]

    def initializeMatrix(self) :
        print('Enter the contents of the matrix row-wise: ')

        for i in range(self.rows) :
            print('Row ', i, ':')
            a = [ ]
            for j in range(self.cols) :
                a.append(int(input( )))
            print('Row ', i, 'completed.')
            self.arr.append(a)
        print('Matrix initialized successfully.')

    def displayMatrix(self) :
        for i in range(self.rows) :
            for j in range(self.cols) :
                print('{0:<5}'.format(self.arr[i][j]), end = "")
            print( )

    def add(self, m) :
        mat = Matrix(self.rows, self.cols)
        for i in range(self.rows) :
            lst = [ ]
            for j in range(self.cols) :
                lst.append(self.arr[i][j] + m.arr[i][j])
            mat.arr.append(lst)
        return mat

    def multiply(self, m) :
        mat = Matrix(self.rows, m.cols)
        for i in range(self.rows) :
            lst = [ ]
            for j in range(self.cols) :
                temp = 0
                for k in range(self.cols) :
                    temp = temp + self.arr[i][k] * m.arr[k][j]
                lst.append(temp)
            mat.arr.append(lst)
        return mat

    def transpose(self) :
        mat = Matrix(self.cols, self.rows)
        for i in range(self.cols) :
            lst = [ ]
            for j in range(self.rows) :
                lst.append(self.arr[j][i])
            mat.arr.append(lst)
        return mat
```

```python
print('Initialize Matrix 1:')
mat1 = Matrix(3, 3)
mat1.initializeMatrix( )

print('Initialize Matrix 2:')
mat2 = Matrix(3, 3)
mat2.initializeMatrix( )

print('First Matrix: ')
mat1.displayMatrix( )

print('Second Matrix: ')
mat2.displayMatrix( )

mat3 = mat1.add(mat2)
print('After addition: ')
mat3.displayMatrix( )

mat4 = mat1.multiply(mat2)
print('After multiplication: ')
mat4.displayMatrix( )

mat5 = mat1.transpose( )
print('Transpose of Matrix 1: ')
mat5.displayMatrix( )
```

#### 输出

初始化矩阵1：
逐行输入矩阵内容：
第0行：
1
2
3
第0行完成。
第1行：
1
2
3
第1行完成。
第2行：
1
2
3
第2行完成。
矩阵初始化成功。
初始化矩阵2：
逐行输入矩阵内容：
第0行：
1
1
1
第0行完成。
第1行：
1
1
1
第1行完成。
第2行：
1
1
1
第2行完成。
矩阵初始化成功。
第一个矩阵：
1 2 3
1 2 3
1 2 3
第二个矩阵：
1 1 1
1 1 1
1 1 1
加法后：
2 3 4
2 3 4
2 3 4
乘法后：
6 6 6
6 6 6
6 6 6
矩阵1的转置：
1 1 1
2 2 2
3 3 3

(c) 编写一个程序，创建一个可以计算立体图形表面积和体积的类。该类还应具备接受与立体图形相关数据的功能。

程序

```python
class Solid :
    def __init__(self, len_cbd = 0, br_cbd = 0, ht_cbd = 0, side_cube =
                0, ht_cyl = 0, rad_cyl = 0, rad_sphere = 0) :
        self.len_cbd = len_cbd
        self.br_cbd = br_cbd
        self.ht_cbd = ht_cbd
        self.side_cube = side_cube
        self.ht_cyl = ht_cyl
        self.rad_cyl = rad_cyl
        self.rad_sphere = rad_sphere

    def sarea_cuboid(self) :
        sa = 2 * (self.len_cbd * self.br_cbd + self.len_cbd * self.ht_cbd
                  + self.ht_cbd * self.br_cbd)
        print('Surface area of cuboid is:', sa)

    def vol_cuboid(self) :
        v = self.len_cbd * self.br_cbd * self.ht_cbd
        print('Volume of cuboid is:', v)

    def sarea_cube(self) :
        sa = 6 * (self.side_cube * self.side_cube)
        print('Surface area of cube is:', sa)

    def vol_cube(self) :
        v = self.side_cube * self.side_cube * self.side_cube
        print('Volume of cube is:', v)

    def sarea_cyl(self) :
        sa = 2 * (3.14 * self.rad_cyl * self.ht_cyl + 3.14 * self.rad_cyl
              * self.rad_cyl)
        print('Surface area of cylinder is:', sa)

    def vol_cyl(self) :
        v = 3.14 * self.rad_cyl * self.rad_cyl * self.ht_cyl
        print('Volume of cylinder is:', v)

    def sarea_sphere(self) :
        sa = 4 * (3.14 * self.rad_sphere * self.rad_sphere)
        print('Surface area of sphere is:', sa)

    def vol_sphere(self) :
        v = 4 / 3 * 3.14 * self.rad_sphere * self.rad_sphere
            * self.rad_sphere
        print('Volume of sphere is:', v)
```

```python
choice = 1
while choice != 0 :
    print('1. Cuboid')
    print('2. Cube')
    print('3. Cylinder')
    print('4. Sphere')
    print('0. Exit')
    choice = int(input('Enter choice: '))

    if choice == 1 :
        l = int(input('Length of cuboid: '))
        b = int(input('Breadth of cuboid: '))
        h = int(input('Height of cuboid: '))
        s = Solid(len_cbd = l, br_cbd = b, ht_cbd = h)
        s.sarea_cuboid( )
        s.vol_cuboid( )

    elif choice == 2 :
        sd = int(input('Side of cube: '))
        s = Solid(side_cube = sd)
        s.sarea_cube( )
        s.vol_cube( )

    elif choice == 3 :
        h = int(input('Height of cylinder: '))
        r = int(input('Radius of base: '))
        s = Solid(rad_cyl = r, ht_cyl = h)
        s.sarea_cyl( )
        s.vol_cyl( )

    elif choice == 4 :
        r = int(input('Radius of sphere: '))
        s = Solid(rad_sphere = r)
        s.sarea_sphere( )
        s.vol_sphere( )

    elif choice == 0 :
        print('Exiting!')

    else :
        print('Invalid choice!!')
```

#### 输出

1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：1
长方体的长度：5
长方体的宽度：4
长方体的高度：3
长方体的表面积是：94
长方体的体积是：60
1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：2
正方体的边长：5
正方体的表面积是：150
正方体的体积是：125
1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：3
圆柱体的高度：6
底面半径：3
圆柱体的表面积是：169.56
圆柱体的体积是：169.56
1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：4
球体的半径：6
球体的表面积是：452.15999999999997
球体的体积是：904.3199999999998
1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：8
无效选择！！
1. 长方体
2. 正方体
3. 圆柱体
4. 球体
0. 退出
请输入选择：0
退出！

(d) 编写一个程序，创建一个可以计算规则图形周长/周长和面积的类。该类还应具备接受与图形相关数据的功能。

程序

```python
class Shape :
```

## 第18章：类与对象

```python
def __init__(self, len_rect = 0, br_rect = 0, side_square = 0, rad_cir = 0) :
    self.len_rect = len_rect
    self.br_rect = br_rect
    self.side_square = side_square
    self.rad_cir = rad_cir

def area_rect(self) :
    a = self.len_rect * self.br_rect
    print('Area of rectangle is:', a)

def peri_rect(self) :
    p = 2 * (self.len_rect + self.br_rect)
    print('Perimeter of rectangle is:', p)

def area_square(self) :
    a = self.side_square * self.side_square
    print('Area of square is:', a)

def peri_square(self) :
    p = 4 * self.side_square
    print('Perimeter of square is:', p)

def area_cir(self) :
    a = 3.14 * self.rad_cir * self.rad_cir
    print('Area of circle is:', a)

def peri_cir(self) :
    p = 2 * 3.14 * self.rad_cir
    print('Perimeter of circle is:', p)

choice = 1
while choice != 0 :
    print('1. Rectangle')
    print('2. Square')
    print('3. Circle')
    print('0. Exit')
    choice = int(input('Enter choice: '))

    if choice == 1 :
        l = int(input('Length of rectangle: '))
        b = int(input('Breadth of rectangle: '))
        s = Shape(len_rect = l, br_rect = b)
        s.area_rect( )
        s.peri_rect( )

    elif choice == 2 :
        sd = int(input('Side of square: '))
        s = Shape(side_square = sd)
        s.area_square( )
        s.peri_square( )

    elif choice == 3 :
        r = int(input('Radius of circle: '))
        s = Shape(rad_cir = r)
        s.area_cir( )
        s.peri_cir( )

    elif choice == 0 :
        print('Exiting!')

    else :
        print('Invalid choice!!')
```

输出

```
1. Rectangle
2. Square
3. Circle
0. Exit
Enter choice: 1
Length of rectangle: 6
Breadth of rectangle: 5
Area of rectangle is: 30
Perimeter of rectangle is: 22

1. Rectangle
2. Square
3. Circle
0. Exit
Enter choice: 2
Side of square: 4
Area of square is: 16
Perimeter of square is: 16

1. Rectangle
2. Square
3. Circle
0. Exit
Enter choice: 3
Radius of circle: 5
Area of circle is: 78.5
Perimeter of circle is: 31.400000000000002
1. Rectangle
2. Square
3. Circle
0. Exit
Enter choice: 5
Invalid choice!!
1. Rectangle
2. Square
3. Circle
0. Exit
Enter choice: 0
Exiting!
```

(e) 编写一个程序，创建并使用一个 **Time** 类来执行各种时间算术运算。

程序

```python
class Time :
    def __init__(self, hr = 0, mnt = 0, sec = 0) :
        self.hours = hr
        self.minutes = mnt
        self.seconds = sec

    def add_seconds(self, sec) :
        if sec > 86400 :
            return 1

        h = int(sec / 3600)
        m = int((sec - h * 3600) / 60)
        s = int((sec - h * 3600 - m * 60))

        self.hours = self.hours + h
        self.minutes = self.minutes + m
        self.seconds = self.seconds + s
        if self.seconds >= 60 :
            self.minutes += 1
            self.seconds -= 60
        if self.minutes >= 60 :
            self.hours += 1
            self.minutes -= 60
        if self.hours >= 24 :
            self.hours = self.hours % 24

    def sub_seconds(self, sec) :
        if sec > 86400 :
            return 1

        h = int(sec / 3600)
        m = int((sec - h * 3600) / 60)
        s = int((sec - h * 3600 - m * 60))

        self.hours = self.hours - h
        self.minutes = self.minutes - m
        self.seconds = self.seconds - s
        if self.seconds < 0 :
            self.minutes -= 1
            self.seconds = 60 + self.seconds
        if self.minutes < 0 :
            self.hours -= 1
            self.minutes = 60 + self.minutes
        if self.hours < 0 :
            self.hours = 24 + self.hours

    def display(self) :
        print(self.hours, ':', self.minutes, ':', self.seconds)

t1 = Time(10, 15, 35)
print('Original time = ', end ='')
t1.display( )
val = t1.add_seconds(144)
if val == 1 :
    print('Cannot add more than 24 hours')
else :
    print('Time after adding 144 seconds = ', end ='')
    t1.display( )

print('Original time = ', end ="")
t1.display( )
val = t1.add_seconds(4000)
if val == 1 :
    print('Cannot add more than 24 hours')
else :
    print('Time after adding 4000 seconds = ', end ="")
    t1.display( )

print('Original time = ', end ="")
t1.display( )
val = t1.sub_seconds(4000)
if val == 1 :
    print('Cannot deduct more than 24 hours')
else :
    print('Time after deducting 4000 seconds = ', end ="")
    t1.display( )

print('Original time = ', end ="")
t1.display( )
val = t1.sub_seconds(144)
if val == 1 :
    print('Cannot deduct more than 24 hours')
else :
    print('Time after deducting 144 seconds = ', end ="")
    t1.display( )
```

输出

```
Original time = 10 : 15 : 35
Time after adding 144 seconds = 10 : 17 : 59
Original time = 10 : 17 : 59
Time after adding 4000 seconds = 11 : 24 : 39
Original time = 11 : 24 : 39
Time after deducting 4000 seconds = 10 : 17 : 59
Original time = 10 : 17 : 59
Time after deducting 144 seconds = 10 : 15 : 35
```

(f) 编写一个程序，通过创建一个链表类来实现链表数据结构。链表中的每个节点应包含汽车名称、价格以及指向下一个节点的链接。

程序

```python
class Node:
    def __init__(self, car, price):
        self.car = car
        self.price = price
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, c, pr):
        n = Node(c, pr)
        if self.head is None :
            self.head = n
        else :
            p = self.head
            while p.next is not None :
                p = p.next

            p.next = n

    def display(self):
        p = self.head
        while p is not None:
            print(p.car, p.price)
            p = p.next

llst = LinkedList( )
llst.add('BMW', '55 lac')
llst.add('Honda City', '12 lac')
llst.add('Mercedes', '75 lac')
llst.add('Esteem', '10 lac')
llst.add('i20', '6 lac')
llst.add('i10', '4 lac')
llst.display( )
```

输出

```
BMW 55 lac
Honda City 12 lac
Mercedes 75 lac
Esteem 10 lac
i20 6 lac
i10 4 lac
```

[D] 匹配以下内容：

- a. dir( )
- b. vars( )
- c. 函数中的变量
- d. import a.b.c
- e. 符号表
- f. 所有函数之外的变量

- 1. 嵌套包
- 2. 标识符、它们的类型和作用域
- 3. 返回字典
- 4. 局部命名空间
- 5. 返回列表
- 6. 全局命名空间

答案

- dir( ) - 返回列表
- vars( ) - 返回字典
- 函数中的变量 - 局部命名空间
- import a.b.c - 嵌套包
- 符号表 - 标识符、它们的类型和作用域
- 所有函数之外的变量 - 全局命名空间

## 第19章：类与对象的复杂性

[A] 判断以下陈述是真还是假：

(a) 全局函数可以调用类方法以及实例方法。

答案

真

(b) 在Python中，函数、类、方法和模块都被视为对象。

答案

真

(c) 给定一个对象，可以确定其类型和地址。

答案

真

(d) 在程序执行期间可以删除对象的属性。

答案

真

(e) 算术运算符、比较运算符和复合赋值运算符在Python中可以被重载。

答案

真

(f) `+` 运算符在 **str**、**list** 和 **int** 类中已被重载。

答案

假

[B] 回答以下问题：

(a) 应该定义哪些函数来重载 `+`、`-`、`/` 和 `//` 运算符？

答案

- + `__add__(self, other)`
- - `__sub__(self, other)`
- / `__truediv__(self, other)`
- // `__floordiv__(self, other)`

(b) `lst = [10, 10, 10, 30]` 创建了多少个对象？

答案

创建了两个对象，一个引用10，另一个引用30。可以验证如下：

```python
lst = [10, 10, 10, 30]
print(id(lst[ 0 ]), id(lst[ 1 ]), id(lst[ 2 ]), id(lst[ 3 ]))
```

其输出将是：

```
1481758656 1481758656 1481758656 1481758976
```

(c) 你将如何动态定义一个包含属性 Name、Age、Salary、Address、Hobbies 的 Employee 结构？

答案

```python
class Employee:
    pass
e = Employee( )
e.name = 'Rohan'
e.age = 29
e.salary = 340000
e.address = 'xyz'
e.hobbies = 'painting'
```

(d) 要重载 `+` 运算符，应在相应的类中定义哪个方法？

答案

`__add__(self, other)`

(e) 要重载 `%` 运算符，应在相应的类中定义哪个方法？

## 第19章：类与对象的复杂性

(f) 若要重载 `//=` 运算符，应在相应的类中定义哪个方法？

答案

`__ifloordiv__(self, other)`

(g) 如果一个类包含实例方法 `__ge__( )` 和 `__ne__( )`，它们表示什么？

答案

它们分别表示重载的 `>=` 和 `!=` 运算符方法。

(h) 如果以下语句能够执行，可以得出什么结论？

```python
a = (10, 20) + (30, 40)
b = 'Good' + 'Morning'
c = [10, 20, 30] + [40, 50, 60]
```

答案

`+` 运算符已在 **tuple**、**str** 和 **list** 类中被重载。

(i) 以下代码片段的输出是什么？

```python
a = (10, 20) - (30, 40)
b = 'Good' - 'Morning'
c = [10, 20, 30] - [40, 50, 60]
```

答案

错误：`-` 运算符未在 **tuple**、**str** 或 **list** 类中被重载。

(j) 以下语句能执行吗？如果能执行，你的结论是什么？

```python
print('Hello' * 7)
```

答案

是的，它能执行。它将输出 `Hello` 7次。我们可以得出结论，`*` 运算符已在 **str** 类中被重载。

(k) 在 **str** 类中，`+`、`-` 和 `*` 运算符中哪些被重载了？

答案

`+` 和 `*` 运算符已在 **str** 类中被重载。

(l) 下面所示的 `Sample` 类中定义的 `__truediv__( )` 方法何时会被调用？

```python
class Sample:
    def __truediv__(self, other):
        pass
```

答案

当对 **Sample** 对象使用 `/` 运算符时。

(m) 如果 `!=` 运算符在一个类中被重载，那么表达式 **c1 <= c2** 会被转换成哪个函数调用？

答案

调用函数 `__le__( )`。

(n) 你将如何为以下代码片段定义重载的 `*` 运算符？

```python
c1 = Complex(1.1, 0.2)
c2 = Complex(1.1, 0.2)
c3 = c1 * c2
```

答案

```python
class Complex:
    def __init__(self, x, y):
        self.real = x
        self.imag = y

    def display(self):
        if self.imag < 0:
            print(self.real, self.imag, 'i')
        else:
            print(self.real, '+', self.imag, 'i')

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

a = Complex(2, 3)
b = Complex(6, -1)
c = a * b
c.display()
```

(o) 实现一个 **String** 类，包含以下函数：

- 重载的 `+=` 运算符函数，用于执行字符串连接。
- 方法 **toLower( )**，用于将大写字母转换为小写。
- 方法 **toUpper( )**，用于将小写字母转换为大写。

*答案*

```python
class String:
    def __init__(self, x):
        self.s = x

    def display(self):
        print(self.s)

    def __iadd__(self, other):
        self.s = self.s + other.s
        return self

    def toUpper(self):
        self.s = self.s.upper()

    def toLower(self):
        self.s = self.s.lower()

a = String('www.ykanetkar')
b = String('.com')
a += b
a.display()
a.toUpper()
a.display()
a.toLower()
a.display()
```

[C] 匹配以下内容：

- a. 不能用作标识符名称 - 3. 关键字
- b. basic_salary - 2. 类变量
- c. CellPhone - 1. 类名
- d. count - 4. 函数中的局部变量
- e. self - 8. 仅在实例函数中有意义
- f. _fuel_used - 5. 私有变量
- g. __draw( ) - 6. 强私有标识符
- h. __iter__( ) - 7. Python调用的方法

## 第20章：容器与继承

[A] 判断以下陈述的真假：

(a) 继承是一个类通过扩展从父类继承属性和行为的能力。

答案

真

(b) 容器是一个类能够包含不同类的对象作为成员数据的能力。

答案

真

(c) 即使基类的源代码不可用，我们也可以从基类派生出一个类。

答案

真

(d) 多重继承与多级继承不同。

答案

真

(e) 如果成员名以 `__` 开头，派生类的对象无法访问基类的成员。

答案

真

(f) 从基类创建派生类需要对基类进行根本性的更改。

答案

假

(g) 如果基类包含一个成员函数 **func( )**，而派生类不包含同名函数，派生类的对象无法访问 **func( )**。

*答案*

假

(h) 如果没有为派生类指定构造函数，派生类的对象将使用基类中的构造函数。

*答案*

假

(i) 如果基类和派生类各自包含一个同名的成员函数，派生类的对象将调用派生类的成员函数。

*答案*

真

(j) 类 **D** 可以从类 **C** 派生，而 **C** 从类 **B** 派生，**B** 又从类 **A** 派生。

*答案*

真

(k) 将一个类的对象作为另一个类的成员是非法的。

*答案*

假

**[B]** 回答以下问题：

(a) 应导入哪个模块来创建抽象类？

*答案*

abc

(b) 为了使一个类成为抽象类，我们应该从哪个类继承它？

答案

ABC

(c) 假设有一个基类 **B** 和一个从 **B** 派生的类 **D**。**B** 有两个**公共**成员函数 **b1( )** 和 **b2( )**，而 **D** 有两个成员函数 **d1( )** 和 **d2( )**。为以下不同情况编写这些类：

- **b1( )** 应可从主模块访问，**b2( )** 不应可访问。
- **b1( )** 和 **b2( )** 都不应可从主模块访问。
- **b1( )** 和 **b2( )** 都应可从主模块访问。

程序

```python
#### 版本1：b1( ) 可访问，b2( ) 不可访问
class B:
    def b1(self):
        print('B - b1')

    def __b2(self):
        print('B - b2')

class D(B):
    def d1(self):
        print('D - d1')

    def d2(self):
        print('D - d2')

b = B()
b.b1()    # 可行
b.__b2()  # 错误

#### 版本2：b1( ) 不可访问，b2( ) 不可访问
class B:
    def __b1(self):
        print('B - b1')

    def __b2(self):
        print('B - b2')

class D(B):
    def d1(self):
        print('D - d1')

    def d2(self):
        print('D - d2')

b = B()
b.__b1()  # 错误
b.__b2()  # 错误

#### 版本3：b1( ) 可访问，b2( ) 可访问
class B:
    def b1(self):
        print('B - b1')

    def b2(self):
        print('B - b2')

class D(B):
    def d1(self):
        print('D - d1')

    def d2(self):
        print('D - d2')

b = B()
b.b1()    # 可行
b.b2()    # 可行
```

(d) 如果类 **D** 从两个基类 **B1** 和 **B2** 派生，那么编写这些类，每个类包含一个构造函数。确保在构建 **D** 类型的对象时，**B2** 的构造函数被调用。同时在每个类中提供一个析构函数。这些析构函数的调用顺序是什么？

程序

```python
class B1:
    def __init__(self):
        print('B1 Ctor')

    def __del__(self):
        print('B1 Dtor')

class B2:
    def __init__(self):
        print('B2 Ctor')

    def __del__(self):
        print('B2 Dtor')

class D(B1, B2):
    def __init__(self):
        B2.__init__(self)
        print('D Ctor')

    def __del__(self):
        B1.__del__(self)
        B2.__del__(self)
        print('D Dtor')

d = D()
d = None
```

输出

```
B2 Ctor
D Ctor
B1 Dtor
B2 Dtor
D Dtor
```

基类的析构函数在派生类的析构函数之前被调用。

(e) 创建一个名为 **Vehicle** 的抽象类，其中包含方法 **speed()**、**maintenance()** 和 **value()**。从 **Vehicle** 类派生出 **FourWheeler**、**TwoWheeler** 和 **Airborne** 类。检查你是否能够阻止 **Vehicle** 类对象的创建。使用其他类的对象调用这些方法。

## 第20章：容器与继承

```python
from abc import ABC, abstractmethod
class Vehicle(ABC) :
    @abstractmethod
    def speed(self) :
        pass

    def maintenance(self) :
        pass

    def value(self) :
        pass

class FourWheeler(Vehicle) :
    def speed(self) :
        print('In FourWheeler.speed')

    def maintenance(self) :
        print('In FourWheeler.maintenance')

    def value(self) :
        print('In FourWheeler.value')

class TwoWheeler(Vehicle) :
    def speed(self) :
        print('In TwoWheeler.speed')

    def maintenance(self) :
        print('In TwoWheeler.maintenance')

    def value(self) :
        print('In TwoWheeler.value')

#### v = Vehicle( )  # will result in error, as Vehicle is abstract class

fw = FourWheeler( )
fw.speed( )
fw.maintenance( )
fw.value( )

tw = TwoWheeler( )
tw.speed( )
tw.maintenance( )
tw.value( )
```

输出

```
In FourWheeler.speed
In FourWheeler.maintenance
In FourWheeler.value
In TwoWheeler.speed
In TwoWheeler.maintenance
In TwoWheeler.value
```

(f) 假设有一个类 **D** 派生自类 **B**。类 **D** 的一个对象可以访问以下哪些内容？

- **D** 的成员
- **B** 的成员

答案

两者都可以

[C] 将以下内容匹配：

- a. `__mro__( )`
- b. 继承
- c. `__var`
- d. 抽象类
- e. 父类
- f. object
- g. 子类
- h. 容器

- 1. ‘has a’ 关系
- 2. 不允许创建对象
- 3. 超类
- 4. 根类
- 5. ‘is a’ 关系
- 6. 名称修饰
- 7. 决定解析顺序
- 8. 子类

答案

- `__mro__( )` - 决定解析顺序
- 继承 - ‘is a’ 关系
- `__var` - 名称修饰
- 抽象类 - 不允许创建对象
- 父类 - 超类
- object - 根类
- 子类 - 子类
- 容器 - ‘has a’ 关系

## 第20章：容器与继承

[D] 尝试回答以下问题：

(a) 任何抽象类都派生自哪个类？

答案

ABC

(b) 一个类可以同时派生自多少个抽象类？

答案

任意数量

(c) 我们如何在Python中创建一个抽象类？

答案

通过从 **abc** 模块的 **ABC** 类派生，如下所示：

```python
from abc import ABC
class Sample(ABC) :
    pass
```

(d) 一个抽象类可以包含什么——实例方法、类方法、抽象方法？

答案

三者都可以。

(e) 一个抽象类可以创建多少个对象？

答案

零个

(f) 执行以下代码片段会发生什么？

```python
from abc import ABC, abstractmethod
class Sample(ABC) :
    @abstractmethod
    def display(self) :
        pass
s = Sample( )
```

答案

错误：无法从抽象类创建对象。

(g) 假设有一个名为 **Vehicle** 的类。应该做什么来确保不能从 **Vehicle** 类创建对象？

答案

通过从 **abc** 模块的 **ABC** 类派生，使其成为抽象类。

(h) 如何将抽象类中的实例方法标记为抽象？

答案

使用装饰器 **@abstractmethod** 进行标记。

(i) 以下代码片段中存在一些错误。你将如何修正它？

```python
class Shape(ABC) :
    @abstractmethod
    def draw(self) :
        pass

class Circle(Shape) :
    @abstractmethod
    def draw(self) :
        print('In draw')
```

答案

两个 **draw( )** 方法及其装饰器都必须缩进。

## 第21章：迭代器与生成器

[A] 回答以下问题：

(a) 编写一个程序，创建一个包含5个奇数的列表。将第三个元素替换为一个包含4个偶数的列表。展平、排序并打印该列表。

```python
lst = [1, 3, 9, 13, 17]
lst[2] = [2, 8, 12, 16]
lst1 = [ ]
for num in lst[2] :
    lst1.append(num)
lst = lst[0:2] + lst1 + lst[3:]
print(lst)
```

输出

```
[1, 3, 2, 8, 12, 16, 13, 17]
```

(b) 编写一个程序，展平以下列表：
`mat1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`

```python
lst = [ ]
mat1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
lst = [ ]
for m in mat1 :
    for ele in m :
        lst.append(ele)
print(lst)
```

输出

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

(c) 编写一个程序，生成一个在2到50范围内能被2和4整除的数字列表。

```python
lst = [n for n in range(2, 50) if n % 2 == 0 and n % 4 == 0]
print(lst)
```

输出

```
[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
```

(d) 假设有两个列表，每个列表包含5个字符串。编写一个程序，生成一个列表，该列表由从两个列表中选取对应元素并连接而成的字符串组成。

```python
lst1 = ['Cat', 'Dog', 'Lion', 'Tiger']
lst2 = ['Lily', 'Rose', 'Hibiscus', 'Lavender']
loft = zip(lst1, lst2)
lst3 = [ ]
for tpl in loft :
    lst3.append(tpl[0] + tpl[1])
print(lst3)
```

输出

```
['CatLily', 'DogRose', 'LionHibiscus', 'TigerLavender']
```

(e) 假设一个列表包含20个随机生成的整数。从键盘接收一个数字，并报告该数字在列表中所有出现的位置。

```python
import random
lst =[int(10 * random.random( )) for n in range(20)]
print(lst)
num = int(input('Enter number between 1 to 10: '))
indexlist = [i for i in range(len(lst)) if lst[i] == num]
print(num, 'is present at following positions')
print(indexlist)
```

输出

```
[2, 8, 1, 6, 0, 6, 4, 4, 4, 8, 6, 9, 1, 2, 5, 0, 1, 4, 8, 8]
Enter number between 1 to 10: 4
4 is present at following positions
[6, 7, 8, 17]
```

(f) 假设有两个列表——一个包含问题，另一个包含每个问题的4个可能答案的列表。编写一个程序，生成一个列表，该列表包含问题及其4个可能答案的列表。

```python
qlist = ['What is capital of India', 'Which is your favorite color?']
alist = [['Delhi', 'Mumbai', 'Hyderabad', 'Bangalore'], ['Red', 'Blue', 'White', 'Black']]
qalist = [ ]
for q, a in zip(qlist, alist) :
    lst = [q, *a]
    qalist.append(lst)
print(qalist)
```

输出

```
[['What is capital of India', 'Delhi', 'Mumbai', 'Hyderabad', 'Bangalore'], ['Which is your favorite color?', 'Red', 'Blue', 'White', 'Black']]
```

(g) 假设一个列表有20个数字。编写一个程序，从该列表中移除所有重复项。

```python
lst =[1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 4, 1, 3, 2, 1, 1, 2, 2, 5, 5]
lst = list(set(lst))
print(lst)
```

输出

```
[1, 2, 3, 4, 5]
```

(h) 编写一个程序，在不改变列表中数字顺序的情况下，获取数字列表的中位数。

```python
lst1 = [1, 2, 3, 4, 5, 6]
n = len(lst1)
s = sorted(lst1)
m = (sum(s[n // 2 - 1 : n // 2 + 1]) / 2.0, s[n // 2])[n % 2]
print(m)

lst2 = [7, 6, 5, 4, 3, 2, 1]
n = len(lst2)
s = sorted(lst2)
m = (sum(s[n // 2 - 1 : n // 2 + 1]) / 2.0, s[n // 2])[n % 2]
print(m)
```

输出

```
3.5
4
```

(i) 一个列表只包含正整数和负整数。编写一个程序，获取列表中负数的数量。

```python
lst1 = [-1, -2, -3, 1, 2, 3]
lst2 = [n for n in lst1 if n < 0]
c = len(lst2)
print(c)
```

输出

```
3
```

(j) 编写一个程序，将元组列表

`[(10, 20, 30), (150.55, 145.60, 157.65), ('A1', 'B1', 'C1')]`

转换为列表

`[(10, 150.55, 'A1'), (20, 145.60, 'B1'), (30, 157.65, 'C1')]`

```python
lst = [(10, 20, 30), (150.55, 145.60, 157.65), ('A1', 'B1', 'C1')]
lst1 = [ ]
for a, b, c in zip(*lst) :
    lst1.append((a, b, c))
print(lst1)
```

输出

```
[(10, 150.55, 'A1'), (20, 145.6, 'B1'), (30, 157.65, 'C1')]
```

(k) 以下程序的输出是什么？

```python
x = [[1, 2, 3, 4], [4, 5, 6, 7]]
y = [[1, 1], [2, 2], [3, 3], [4, 4]]
l1 = [xrow for xrow in x]
print(l1)
l2 = [(xrow, ycol) for ycol in zip(*y) for xrow in x]
print(l2)
```

输出

```
[[1, 2, 3, 4], [4, 5, 6, 7]]
[([1, 2, 3, 4], (1, 2, 3, 4)), ([4, 5, 6, 7], (1, 2, 3, 4)), ([1, 2, 3, 4], (1, 2, 3, 4)), ([4, 5, 6, 7], (1, 2, 3, 4))]
```

(l) 编写一个程序，使用生成器从通过键盘输入的一行文本中创建一组唯一的单词。

```python
line = input('Enter a sentence: ')
s = set(line.split( ))
print(s)
```

输出

```
Enter a sentence: I did not do this. He did it or she did it
{'it', 'or', 'do', 'He', 'she', 'did', 'this.', 'I', 'not'}
```

(m) 编写一个程序，使用生成器从多个学生的元组中找出学生的最高分及其姓名。

```python
def getname(stud, mm) :
    if stud[1] == mm :
        return stud[0]

lst = [('Ajay', 45), ('Sujay', 55), ('Nirmal', 40), ('Vijay', 75)]
```

## 第21章：迭代器与生成器

```python
maxmarks = max(student[1] for student in lst)
for student in lst :
    name = getname(student, maxmarks)
print(name, maxmarks )
```

输出

Vijay 75

(n) 编写一个程序，使用生成器按逆序生成字符串中的字符。

程序

```python
n = 'Sacchidanand'
revn = [ch for ch in n[::-1]]
print(revn)
```

输出

['d', 'n', 'a', 'n', 'a', 'd', 'i', 'h', 'c', 'c', 'a', 'S']

(o) 以下语句有什么区别？

```python
sum([x**2 for x in range(20)])
sum(x**2 for x in range(20))
```

答案

第一个表达式首先生成一个列表，然后计算列表中所有元素的和。

第二个表达式在生成每个数字的平方时，实时计算并累加总和。

两者将产生相同的结果，但第二个更高效，因为它占用的空间更少。

(p) 假设有两个列表，每个列表包含5个字符串。编写一个程序，通过从两个列表中选取对应元素进行拼接，生成一个新列表。

程序

```python
lst1 = ['Cat', 'Dog', 'Lion', 'Tiger']
lst2 = ['Lily', 'Rose', 'Hibiscus', 'Lavender']
lst3 = [(x + y) for x, y in zip(lst1, lst2)]
print(lst3)
```

输出

```python
['CatLily', 'DogRose', 'LionHibiscus', 'TigerLavender']
```

(q) 两个骰子可以产生36种独特的组合。创建一个字典，将这些组合以元组形式存储。

程序

```python
lst = [(d1, d2) for d1 in range(1,7) for d2 in range(1,7)]
print(lst)
```

输出

```python
[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
```

## 第22章：异常处理

[A] 判断以下陈述的真假：

(a) 异常处理机制旨在处理编译时错误。

*答案*

假

(b) 必须在将要抛出异常的类内部声明异常类。

*答案*

假

(c) 每个抛出的异常都必须被捕获。

*答案*

真

(d) 一个**try**块可以对应多个**except**块。

*答案*

真

(e) 当异常被抛出时，会调用异常类的构造函数。

*答案*

真

(f) **try**块不能嵌套。

*答案*

假

(g) 异常处理机制保证对象的正确销毁。

*答案*

假

(h) 所有异常都发生在运行时。

*答案*

真

(i) 异常提供了一种面向对象的方式来处理运行时错误。

*答案*

真

(j) 如果发生异常，程序会突然终止，没有任何从异常中恢复的机会。

*答案*

假

(k) 无论是否发生异常，**finally**子句（如果存在）中的语句都会被执行。

*答案*

真

(l) 一个程序可以包含多个**finally**子句。

*答案*

假

(m) **finally**子句用于执行清理操作，如关闭网络/数据库连接。

*答案*

真

(n) 在抛出用户定义的异常时，可以在异常对象中设置多个值。

*答案*

真

(o) 在一个函数/方法中，只能有一个**try**块。

*答案*

假

(p) 异常必须在抛出它的同一个函数/方法中被捕获。

*答案*

假

(q) 在异常对象中设置的所有值，在捕获该异常的**except**块中都可用。

*答案*

真

(r) 如果我们的程序没有捕获异常，那么Python运行时会捕获它。

*答案*

真

(s) 可以创建用户定义的异常。

*答案*

真

(t) 所有类型的异常都可以使用**Exception**类捕获。

*答案*

真

(u) 每个**try**块都必须有一个对应的**finally**块。

*答案*

假

**[B]** 回答以下问题：

(a) 如果我们不捕获运行时抛出的异常，谁会捕获它？

答案

如果我们不捕获运行时抛出的异常，那么Python运行时会捕获它。

(b) 简要说明使用异常处理相对于传统错误处理方法的最令人信服的理由。

答案

以下是倾向于使用异常处理而非传统错误处理的原因：

- 它允许将程序的逻辑与错误处理逻辑分离，使其更可靠且更易于维护。
- 异常信息从发生异常的地方传播到处理它的地方，这是由运行时环境完成的，而不是程序员的责任。
- 它允许在发生运行时错误时进行保证的清理操作。

(c) 是否所有可用于表示异常的类都必须派生自基类**Exception**？

答案

是

(d) **finally**块在Python异常处理机制中的作用是什么？

答案

清理活动，如释放外部资源、网络连接或数据库连接等，在finally块中完成，因为它无论是否发生异常都会被调用。

(e) Python中的嵌套异常处理是如何工作的？

答案

如果在嵌套的try块中引发异常，则使用嵌套的except块来处理它。如果未处理，则使用外部的except块来处理异常。

(f) 编写一个程序，接收10个整数，并将它们及其立方值存储在一个字典中。如果输入的数字小于3，则引发用户定义的异常**NumberTooSmall**；如果输入的数字大于30，则引发用户定义的异常**NumberTooBig**。无论是否发生异常，最后都打印字典的内容。

程序

```python
class NumberTooSmall(Exception) :
    def __init__(self, num) :
        self.num = num

    def get_details(self) :
        return {'Number too small' : self.num}

class NumberTooBig(Exception) :
    def __init__(self, num) :
        self.num = num

    def get_details(self) :
        return {'Number too big' : self.num}

class Numbers :
    def __init__(self) :
        self.dct = { }

    def append(self, num, cube ) :
        self.dct[num] = cube

    def display(self) :
        for k, v in self.dct.items( ) :
            print(k, v)
        print( )

n = Numbers( )
print('Enter 10 numbers between 3 to 30 :')
try :
    for x in range(10) :
        num = int(input( ))
        if num > 30 :
            raise NumberTooBig(num)
        elif num < 3 :
            raise NumberTooSmall(num)
        else :
            cube = num * num *num
            n.append(num, cube)
except NumberTooBig as ntb :
    print(ntb.get_details( ))
except NumberTooSmall as nts :
    print(nts.get_details( ))
finally :
    n.display( )
```

输出

```
Enter 10 numbers between 3 to 30 :
5
6
8
9
12
10
2
{'Number too small': 2}
5 125
6 216
8 512
9 729
12 1728
10 1000
```

(g) 以下代码片段有什么问题？

```python
try :
    # some statements
except :
    # report error 1
except ZeroDivisionError :
    # report error 2
```

答案

空的except块必须是最后一个except块。

(h) 以下哪个关键字不属于Python的异常处理——try、catch、throw、except、raise、finally、else？

答案

**catch**和**throw**不属于Python的异常处理。

(i) 以下代码的输出是什么？

```python
def fun( ) :
    try :
        return 10
    finally :
        return 20

k = fun( )
print(k)
```

输出

20

## 第23章：文件输入/输出

[A] 判断以下陈述的真假：

(a) 如果一个文件以读取模式打开，则该文件必须存在。

答案

真

(b) 如果一个以写入模式打开的文件已经存在，其内容将被覆盖。

答案

真

(c) 以追加模式打开文件时，该文件必须存在。

答案

假

[B] 回答以下问题：

(a) 以文本模式打开文件进行读取时，会进行哪些活动序列？

答案

以文本模式打开文件进行读取时，会执行以下活动：

1. 在磁盘上搜索文件是否存在。
2. 将文件加载到内存中。
3. 设置一个指针，指向文件中的第一个字符。
4. 以上所有。

(b) 是否必须始终以文本模式打开在文本模式下创建的文件以进行后续操作？

答案

是

(c) 使用以下语句时，

```python
fp = open('myfile', 'r')
```

## 第23章：文件输入/输出

如果发生以下情况：

- 磁盘上不存在 'myfile'
- 磁盘上存在 'myfile'

答案

系统会搜索磁盘以查找 'myfile' 的存在。如果不存在，则会引发 **FileNotFoundError** 异常。如果存在，则将其加载到内存中，并设置一个指向文件第一个字符的指针。

(d) 使用以下语句时，

```
f = open('myfile', 'wb')
```

如果发生以下情况：

- 磁盘上不存在 'myfile'
- 磁盘上存在 'myfile'

答案

系统会搜索磁盘以查找 'myfile' 的存在。如果不存在，则会引发 **FileNotFoundError** 异常。如果存在，则将其加载到内存中，并设置一个指向文件第一个字节的指针。

(e) 一个浮点数列表包含学生在考试中获得的百分制分数。要将这些分数存储到文件 'marks.dat' 中，你应该以哪种模式打开文件？为什么？

程序

'marks.dat' 应该以 'wb' 模式打开。这是因为在二进制模式下，当我们将一个数字存储到磁盘文件时，它占用的字节数与其在内存中占用的字节数相同。如果文件以 'w' 模式打开，那么数字将逐个字符存储，因此会占用与数字长度相同的字节数。

[C] 尝试回答以下问题：

(a) 编写一个程序来读取一个文件并显示其内容，在每行前显示行号。

程序

```
#### 显示文件内容
f = open('sample.txt', 'r')
while True :
    data = f.readline( )
    if data == '' :
        break
    print(data)
f.close( )
```

输出

CPython - 是参考实现，用C语言编写。
PyPy - 用Python语言的一个子集RPython编写。
Jython - 用Java编写。
IronPython - 用C#编写。

(b) 编写一个程序将一个文件的内容追加到另一个文件的末尾。

程序

```
#### 追加文件
f1 = open('sample.txt', 'r')
para1 = ''
while True :
    data = f1.readline( )
    if data == '' :
        break
    para1 += data

f2 = open('trial.txt', 'r+')
para2 = ''
while True :
    data = f2.readline( )
    if data == '' :
        break
    para2 += data

para2 += para1
print(para2)

f2.seek(0, 0)
f2.write(para2)
f1.close( )
f2.close( )
```

输出

'sample.txt' 包含几行小写字母。'trial.txt' 包含相同的大写字母行。连接后，'trial.txt' 的内容如下所示：

CPYTHON - IS THE REFERENCE IMPLEMENTATION, WRITTEN IN C.
PYPY - WRITTEN IN A SUBSET OF PYTHON LANGUAGE CALLED RPYTHON.
JYTHON - WRITTEN IN JAVA.
IRONPYTHON - WRITTEN IN C#.
CPython - is the reference implementation, written in C.
PyPy - Written in a subset of Python language called RPython.
Jython - Written in Java.
IronPython - Written in C#.

(c) 假设一个文件包含学生的记录，每条记录包含学生的姓名和年龄。编写一个程序来读取这些记录并按姓名排序显示。

程序

```
#### 对文件中的记录进行排序
import operator
f = open('students.txt', 'r')
dct = { }
while True :
    data = f.readline( )
    if data == '' :
        break
    stud = data.split( )
    dct[stud[0]] = stud[1]

f.close( )
lst = sorted(dct.items( ), key = operator.itemgetter(0))
for item in lst :
    print(item[0], item[1])
```

输出

Anil 23
Prabhu 22
Rakesh 25
Sameer 30
Sanjay 25
Suresh 33

(d) 编写一个程序将一个文件的内容复制到另一个文件。在复制过程中，将所有小写字符替换为对应的大写字符。

程序

```
#### 将文件内容转换为大写
f1 = open('stud1.txt', 'r')
f2 = open('stud2.txt', 'w')
while True :
    data = f1.readline( )
    if data == '' :
        break
    data = data.upper( )
    f2.write(data)

f1.close( )
f2.close( )
```

输出

SANJAY 25
SAMEER 30
ANIL 23
SURESH 33
PRABHU 22
RAKESH 25

(e) 编写一个程序，交替地从两个文件中合并行，并将结果写入新文件。如果一个文件的行数少于另一个文件，则只需将较大文件中剩余的行复制到目标文件中。

程序

```
#### 交替合并两个文件的行
f1 = open('sample.txt', 'r')
f2 = open('trial.txt', 'r')
f3 = open('combined.txt', 'w')
while True :
    data1 = f1.readline( )
    if data1 == "" :
        break
    f3.write(data1)
    data2 = f2.readline( )
    if data2 == "" :
        break
    f3.write(data2)

if data1 != "" :
    while True :
        data1 = f1.readline( )
        if data1 == "" :
            break
        f3.write(data1)

if data2 != "" :
    while True :
        data2 = f2.readline( )
        if data2 == "" :
            break
        f3.write(data2)

f1.close( )
f2.close( )
f3.close( )
```

输出

文件 'sample.txt' 包含以下行：

1. SANJAY 25
2. SAMEER 30
3. ANIL 23
4. SURESH 33
5. PRABHU 22
6. DINESH 40
7. Suresh 34

文件 'trial.txt' 包含以下行：

1. Sandhya 25
2. Seema 30
3. Swati 23
4. Supriya 33
5. Sunidhi 22

结果文件 'combined.txt' 包含以下行：

1. SANJAY 25
1. Sandhya 25
2. SAMEER 30
2. Seema 30
3. ANIL 23
3. Swati 23
4. SURESH 33
4. Supriya 33
5. PRABHU 22
5. Sunidhi 22
6. DINESH 40
7. Suresh 34

(f) 假设一个 Employee 对象包含以下详细信息：
员工代码
员工姓名
入职日期
工资
编写一个程序来序列化和反序列化这些数据。

程序

```
#### 员工记录的序列化、反序列化
import json
def encode_employee(x):
    if isinstance(x, Employee) :
        return(x.ecode, x.ename, x.doj, x.sal)
    else :
        raise TypeError('Complex object is not JSON serializable')

def decode_employee(dct):
    if '__Employee__' in dct :
        return Employee(dct['ecode'], dct['ename'], dct['doj'], dct['sal'])
    return dct

class Employee :
    def __init__(self, ecode, ename, doj, sal) :
        self.ecode = ecode
        self.ename = ename
        self.doj = doj
        self.sal = sal

    def print_data(self) :
        print(self.ecode, self.ename, self.doj, self.sal)

e = Employee('A101', 'Sameer', '17/11/2017', 25000)
f = open('data', 'w+')
json.dump(e, f, default = encode_employee)
f.seek(0)
ine = json.load(f, object_hook = decode_employee)
print(ine)
```

输出

['A101', 'Sameer', '17/11/2017', 25000]

(g) 一家医院保存着献血者文件，其中每条记录的格式为：
姓名：20列
地址：40列
年龄：2列
血型：1列（类型1、2、3或4）
编写一个程序来读取该文件，并打印出所有年龄低于25岁且血型为2型的献血者列表。

程序

```
#### 格式化读写
donors = {
    'Sanjay' : ['Gokulpeth', 25, 1],
    'Sunil' : ['Shankarnagar', 26, 2],
    'Akash' : ['Sitaburdi', 27, 3],
    'Rahul' : ['Ramnagar', 23, 2],
    'Riddhi' : ['Dharampeth', 22, 2],
    'Mangal' : ['Ramdaspeth', 21, 2]
}
f = open('donors.txt', 'w+')
for k, v in donors.items( ) :
    s = '{0:20s}{1:40s}{2:2s}{3:1s}\n'.format(k, v[0], str(v[1]), str(v[2]))
    f.write(s)
f.seek(0,0)
while True :
    data = f.readline( )
    if data == '' :
        break
    nam = data[:20]
    address = data[20:59]
    age = int(data[60:62:])
    bloodtype = int(data[62:])
    if age < 25 and bloodtype == 2 :
        print(nam, address, age, bloodtype)
f.close( )
```

输出

Rahul          Ramnagar          23 2
Riddhi         Dharampeth        22 2
Mangal         Ramdaspeth        21 2

(h) 给定一个班级学生姓名的列表，编写一个程序将这些姓名存储到磁盘上的一个文件中。提供显示列表中第n个姓名的功能，其中n从键盘读取。

程序

```
#### 修改文件中的记录
names = ['Sanjay', 'Sunil', 'Akash', 'Rahul', 'Riddhi', 'Mangal']
f = open('students.txt', 'w+')
for studname in names :
    f.write(studname + '\n')
num = int(input('Enter student number: '))
f.seek(0,0)
i = 1
while i < num :
    data = f.readline( )
    i += 1

data = f.readline( )
print('Num =', num, 'Name =', data)
f.close( )
```

输出

```
Enter student number: 4
Num = 4 Name = Rahul
```

(i) 假设一个主文件包含两个字段：学号和学生姓名。在学年结束时，有一批学生加入班级，另一批学生离开。一个事务文件包含学号以及用于添加或删除学生的适当代码。
编写一个程序来创建另一个文件，其中包含更新后的姓名和学号列表。假设主文件和事务文件都按学号升序排列。更新后的文件也应按学号升序排列。

程序

```
#### 处理主文件-事务文件
fm = open('master.txt', 'r')
mdata = fm.readlines( )

ft = open('tran.txt', 'r')
while True :
    trec = ft.readline( )
    if trec == '' :
        break
    tfields = trec.split( )

    if len(tfields) == 2 :
        count = 0
        for record in mdata :
            mfields = record.split( )
            if tfields[0] == mfields[0] :
                break
            count += 1
```

## 第24章：杂项

### [A] 判断以下陈述的正误：

(a) 我们可以向任何Python程序在命令行传递参数。

*答案*

正确

(b) `sys.argv`的第零个元素始终是正在执行的文件名。

*答案*

正确

(c) 在Python中，函数被视为对象。

*答案*

正确

(d) 函数可以作为参数传递给另一个函数，也可以从函数中返回。

*答案*

正确

(e) 装饰器为现有函数添加一些特性。

*答案*

正确

(f) 一旦创建了装饰器，它只能应用于程序中的一个函数。

*答案*

错误

(g) 被装饰的函数不能接收任何参数是强制性的。

*答案*

错误

(h) 被装饰的函数不能返回任何值是强制性的。

*答案*

错误

(i) 'Good!'的类型是bytes。

*答案*

错误

(j) **msg = 'Good!'**中msg的类型是**str**。

*答案*

正确

### [B] 回答以下问题：

(a) 是否必须在**def**语句正下方立即提及函数的文档字符串？

*答案*

是

(b) 编写一个使用命令行参数在文件中搜索一个单词并将其替换为指定单词的程序。程序的用法如下所示。

```
C:\> change -o oldword -n newword -f filename
```

**程序**

```python
#### change.py
import sys
import getopt
sys.argv = ['change.py', '-o', 'Unit', '-n', 'UNIT', '-f', 'Syllabus.txt']
if len(sys.argv) != 7 :
    print('Incorrect usage')
    print('change -o oldword -n newword -f filename')
    sys.exit(1)
try :
    options, arguments = getopt.getopt(sys.argv[1:],'ho:n:f:')
except getopt.GetoptError :
    print('change -o oldword -n newword -f filename')
else :
    for opt, arg in options :
        if opt == '-h' :
            print('change -o oldword -n newword -f filename')
            sys.exit(2)
        elif opt == '-o' :
            oldword = arg
        elif opt == '-n' :
            newword = arg
        elif opt == '-f' :
            filename = arg
    else :
        print('old word:', oldword)
        print('newword: ', newword)
        print('filename:', filename)
        if oldword and newword and filename:
            f = open(filename, 'r')
            data = f.read( )
            f.close( )
            data = data.replace(oldword, newword)
            f = open(filename, 'w')
            f.write(data)
            f.close( )
```

**提示**

程序存储在文件'change.py'中。文件'syllabus.txt'与'change.py'位于同一文件夹中。它包含以下文本：

- 单元1：面向对象编程
- 单元2：数据封装
- 单元3：继承
- 单元4：多态
- 单元5：后期绑定
- 单元6：构造函数
- 单元7：方法重载

(c) 编写一个可在命令提示符下用作计算实用程序的程序。程序的用法如下所示。

```
C:\> calc <switch> <n> <m>
```

其中，**n**和**m**是两个整数操作数。**switch**可以是任何算术运算符。输出应为运算结果。

**程序**

```python
import sys
if len(sys.argv) != 4 :
    print('Incorrect usage')
    print('calc operator number number')
    sys.exit(1)

operator = sys.argv[1]
m = int(sys.argv[2])
n = int(sys.argv[3])
if operator == '+' :
    result = m + n
    print('operator =', operator, 'm =', m, 'n =', n, 'result =', result)
elif operator == '-' :
    result = m - n
    print('operator =', operator, 'm =', m, 'n =', n, 'result =', result)
elif operator == '*' :
    result = m * n
    print('operator =', operator, 'm =', m, 'n =', n, 'result =', result)
elif operator == '/' :
    result = m / n
    print('operator =', operator, 'm =', m, 'n =', n, 'result =', result)
else :
    print('Illegal operator')
```

**输出**

该程序可以在命令行中执行，如下所示：

```
C:\> python calc.py + 23 45
```

执行后产生以下输出：

```
operator = + m = 23 n = 45 result = 68
```

(d) 使用位复合赋值运算符重写以下表达式：

```
a = a | 3      a = a & 0x48      b = b ^ 0x22
c = c << 2     d = d >> 4
```

*答案*

```
a |= 3        a &= 0x48        b ^= 0x22
c <<= 2       d >>= 4
```

(e) 考虑一个无符号整数，其最右边的位编号为0。编写一个函数**checkbits(x, p, n)**，如果从位置'p'开始的'n'位全部为1则返回True，否则返回False。例如，如果数字**x**的第4、3和2位为1，则**checkbits(x, 4, 3)**将返回true。

**程序**

```python
def display_bits(n) :
    for i in range(7, -1, -1) :
        andmask = 1 << i
        k = n & andmask
        print('0', end = "") if k == 0 else print('1', end = "")
    print( )

def checkbits(x, p, n) :
    no = 0
    for i in range(0, n) :
        if ((x >> (p - 1)) & 1) != 1 :
            return 0
        p -= 1
    return 1

num = int(input('Enter a number between 0 to 255: '))
display_bits(num)
p = int(input('Enter position: '))
n = int(input('Enter number of bits: '))
flag = checkbits(num, p, n)
if flag == 1 :
    print(n, 'bits starting from position', p, 'are on')
else :
    print(n, 'bits starting from position', p, 'are off')
```

**输出**

```
Enter a number between 0 to 255: 255
11111111
Enter position: 4
Enter number of bits: 3
3 bits starting from position 4 are on
Enter a number between 0 to 255: 96
01100000
Enter position: 6
Enter number of bits: 3
3 bits starting from position 6 are off
```

(f) 编写一个程序，接收一个数字作为输入，并检查其第3、第6和第7位是否为1。

**程序**

```python
#### 检查数字的第3、第6和第7位是否为1的程序
def display_bits(n) :
    for i in range(7, -1, -1) :
        andmask = 1 << i
        k = n & andmask
        print('0', end = '') if k == 0 else print('1', end = '')

num = int(input('Enter a number between 0 to 255: '))
display_bits(num)
j = num & 0x08
print( )

print('Its third bit is off') if j == 0 else print('Its third bit is on')
j = num & 0x40
print('Its sixth bit is off') if j == 0 else print('Its sixth bit is on')
j = num & 0x80
print('Its seventh bit is off') if j == 0 else print('Its seventh bit is on')
```

**输出**

```
Enter a number between 0 to 255: 65
01000001
Its third bit is off
Its sixth bit is on
Its seventh bit is off
```

(g) 编写一个程序，将一个8位数字接收到一个变量中，然后交换其高4位和低4位。

**程序**

```python
#### 交换数字高4位和低4位的程序
def display_bits(n) :
    for i in range(7, -1, -1) :
        andmask = 1 << i
        k = n & andmask
        print('0', end = '') if k == 0 else print('1', end = '')

num = int(input('Enter a number between 0 to 255: '))
display_bits(num)
n1 = num << 4
n2 = num >> 4
num = n1 | n2
print('\nAfter exchanging bits:')
display_bits(num)
```

**输出**

```
Enter a number between 0 to 255: 64
01000000
After exchanging bits:
00000100
```

(h) 编写一个程序，将一个8位数字接收到一个变量中，然后将其奇数位设置为1。

**程序**

```python
def display_bits(n) :
    for i in range(7, -1, -1) :
        andmask = 1 << i
        k = n & andmask
        print('0', end = '') if k == 0 else print('1', end = '')

def modify_oddbits(n) :
    for i in range(7, -1, -2) :
        ormask = 1 << i
```

## 25 并发与并行

![](img/ac61a02854da7ea89272a34be93604dc_230_0.png)

**[A]** 判断以下陈述的正误：

(a) 多线程能提高程序的执行速度。

*答案*

正确

(b) 一个运行中的任务可以包含多个正在运行的线程。

*答案*

正确

(c) 多处理与多线程是相同的。

*答案*

错误

(d) 如果我们创建一个继承自 **Thread** 类的类，我们仍然可以从其他类继承我们的类。

*答案*

正确

(e) 可以更改正在运行的线程的名称。

*答案*

正确

(f) 要启动一个线程，我们必须显式调用那个应该在单独线程中运行的函数。

*答案*

错误

(g) 要启动一个线程，我们必须显式调用在扩展 **Thread** 类的类中定义的 **run( )** 方法。

*答案*

错误

(h) 尽管我们没有显式调用那个应该在单独线程中运行的函数，但仍然可以向该函数传递参数。

*答案*

正确

(i) 我们无法控制在程序中启动的多个线程的优先级。

*答案*

错误

**[B]** 回答以下问题：

(a) 多处理和多线程有什么区别？

*答案*

多处理是同时执行多个进程的能力。

多线程是同时执行程序的多个部分（单元）的能力。

(b) 抢占式多线程和协作式多线程有什么区别？

*答案*

抢占式多线程 - 操作系统决定何时从一个任务切换到另一个任务。

协作式多线程 - 任务自身决定何时将控制权交给下一个任务。

(c) 在 Python 程序中，启动线程有哪两种可用的方法？

*答案*

- 通过将应该作为单独线程运行的函数的名称传递给 **Thread** 类的构造函数。
- 通过在 **Thread** 类的子类中重写 `__init__( )` 和 `run( )` 方法。

(d) 如果 **Ex** 类扩展了 **Thread** 类，那么我们可以为 **Ex** 类的对象启动多个线程吗？如果可以，如何操作？

*答案*

```python
import threading
class Ex(threading.Thread) :
    def __init__(self, s) :
        threading.Thread.__init__(self)
        self.msg = s

    def run(self) :
        while True :
            print(self.msg, end = '\n')

th1 = Ex('Hello')
th1.start( )
th2 = Ex('Hi')
th2.start( )
```

*输出*

```
HelloHi
HelloHi
HelloHi
HelloHi
HelloHi
HelloHi
HelloHi
... ... ...
```

(e) 以下语句中不同元素的含义是什么？

```python
th1 = threading.Thread(target = quads, args = (a, b))
```

*答案*

**threading** 是一个模块。它包含一个 **Thread** 类。
这里正在创建 **Thread** 类的一个对象。
该对象的地址将存储在 **th1** 中。
**quads** 是将在单独线程中运行的函数的名称。
**a**, **b** 是将传递给 **quad** 函数的参数。
参数必须以元组的形式给出。

(f) 编写一个多线程程序，将一个文件夹的内容复制到另一个文件夹。源文件夹和目标文件夹的路径应通过键盘输入。

程序

```python
import sys
import threading
import os
import shutil

def copy_file(input_file, output_file):
    shutil.copyfile(input_file, output_file)
    s = input_file + ' copied!\n'
    print(s)

source = sys.argv[1]
target = sys.argv[2]

if not os.path.exists(source) :
    print('source path does not exist')
    exit( )

if not os.path.exists(target) :
    os.mkdir(target)

os.chdir(source)
lst = os.listdir('.')
tharr = [ ]
for file in lst :
    sourcefilepath = source + '\\' + file
    targetfilepath = target + '\\' + file
    th = threading.Thread(target = copy_file, args = (sourcefilepath, targetfilepath))
    th.start( )
    tharr.append(th)

for th in tharr :
    th.join( )
```

输出

```
c:\Users\Kanetkar\Desktop\sourcedir\cubes.txt copied!
c:\Users\Kanetkar\Desktop\sourcedir\swam.txt copied!
c:\Users\Kanetkar\Desktop\sourcedir\Resolutions.docx copied!
```

(g) 编写一个程序，按顺序读取 3 个文件 a.txt、b.txt 和 c.txt 的内容，将其内容转换为大写，并分别写入文件 aa.txt、bb.txt 和 cc.txt。程序应报告执行此转换所需的时间。文件 a.txt、b.txt 和 c.txt 应添加到项目中并填充一些文本。程序应通过命令行参数接收文件名。从任何文件读取一行后，将程序挂起 0.5 秒。

程序

```python
import time
import sys
import threading

start_time = time.time( )
lst1= sys.argv[1:4]
lst2 = sys.argv[4:]

if len(lst1) != 3 or len(lst2) != 3 :
    print('Improper usage')
    print('Correct usage: convert a.txt b.txt c.txt aa.txt bb.txt cc.txt')
    exit( )

for i in range(0, 3) :
    f1 = open(lst1[i], 'r')
    f2 = open(lst2[i], 'w')
    while True :
        data = f1.readline( )
        if data == '' :
            break
        time.sleep(0.5)
        data = data.upper( )
        f2.write(data)
    f1.close( )
    f2.close( )
end_time = time.time( )
print('Time required = ', end_time - start_time, 'sec')
```

输出

```
Time required = 4.6332080364227295 sec
```

(h) 编写一个程序，通过启动 3 个不同的线程来执行转换操作，从而完成上述练习 [B](g) 中提到的相同任务。

程序

```python
import time
import sys
import threading

def readFile(input_file, output_file):
    f1 = open(input_file, 'r')
    f2 = open(output_file, 'w')
    while True :
        data = f1.readline( )
        if data == "" :
            break
        data = data.upper( )
        f2.write(data)
        time.sleep(0.5)
    f1.close( )
    f2.close( )

start_time = time.time( )
lst1= sys.argv[1:4]
lst2 = sys.argv[4:]

if len(lst1) != 3 or len(lst2) != 3 :
    print('Improper usage')
    print('Correct usage: convert a.txt b.txt c.txt aa.txt bb.txt cc.txt')
    exit( )

tharr = [ ]
for i in range(0, 3) :
    th = threading.Thread(target = readFile, args = (lst1[i], lst2[i]))
    th.start()
    tharr.append(th)

for th in tharr:
    th.join()

end_time = time.time()
print('Time required = ', end_time - start_time, 'sec')
```

输出

```
Time required = 1.5756025314331055 sec
```

**[C]** 将以下内容配对：

| | |
|---|---|
| a. 多处理 | 1. 使用 multiprocessing 模块 |
| b. 抢占式多线程 | 2. 使用多线程 |
| c. 协作式多线程 | 3. 使用 threading 模块 |
| d. CPU 密集型程序 | 4. 使用多处理 |
| e. I/O 密集型程序 | 5. 使用 asyncio 模块 |

答案

- 多处理 - 使用 multiprocessing 模块
- 抢占式多线程 - 使用 threading 模块
- 协作式多线程 - 使用 asyncio 模块
- CPU 密集型程序 - 使用多处理
- I/O 密集型程序 - 使用多线程

## 26 同步

![](img/ac61a02854da7ea89272a34be93604dc_238_0.png)

**[A]** 判断以下陈述的正误：

(a) 所有多线程应用程序都应该使用同步。

*答案*

错误

(b) 如果 3 个线程要从一个共享列表中读取数据，那么同步它们的活动是必要的。

*答案*

错误

(c) 一个线程获取的 **Lock** 可以由同一个线程或应用程序中运行的任何其他线程释放。

*答案*

正确

(d) 如果在可重入代码中使用 **Lock**，那么线程在第二次调用时很可能会被阻塞。

*答案*

正确

(e) **Lock** 和 **RLock** 的工作方式类似于互斥锁。

*答案*

正确

(f) 线程将在 **Event** 对象上等待，除非其内部标志被清除。

*答案*

正确

(g) **Condition** 对象内部使用一个锁。

## 第26章：同步

### [A] 判断以下陈述的正误：

(h) 使用**RLock**时，我们必须确保调用**release( )**的次数与调用**acquire( )**的次数相同。

答案：正确

(i) 使用**Lock**，我们可以控制可以访问资源的最大线程数。

答案：正确

(j) **Event**和**Condition**同步对象之间没有区别。

答案：错误

(k) 如果在一个Python程序中，一个线程读取文档，另一个线程写入同一文档，那么这两个线程应该同步。

答案：正确

(l) 如果在一个Python程序中，一个线程复制文档，另一个线程显示进度条，那么这两个线程应该同步。

答案：正确

(m) 如果在一个Python程序中，一个线程让你输入文档，另一个线程对同一文档进行拼写检查，那么这两个线程应该同步。

答案：正确

(n) 如果在一个Python程序中，一个线程扫描文档病毒，另一个线程让你暂停或停止扫描，那么这两个线程应该同步。

答案：正确

### [B] 回答以下问题：

(a) 哪些同步机制用于在多个线程之间共享资源？

答案：**Lock**、**RLock**和**Semaphore**同步机制用于在多个线程之间共享资源。

(b) 在多线程应用程序中，哪些同步对象用于线程间通信？

答案：**Event**和**Condition**同步对象用于多线程应用程序中的线程间通信。

(c) **Lock**和**RLock**有什么区别？

答案：**Lock**用于同步对共享资源的访问。**RLock**用于在可重入代码中同步对共享资源的访问。

(d) **Semaphore**同步原语的目的是什么？

答案：信号量用于限制对资源（如网络连接或数据库服务器）的访问，将其限制在有限数量的线程内。

(e) 编写一个包含三个线程的程序。第一个线程应生成1到20范围内的随机数，第二个线程应在屏幕上显示第一个线程生成的数字的平方，第三个线程应将第一个线程生成的数字的立方写入文件。

程序：

```python
import threading
import random
import queue
import time
import collections

def generate( ) :
    for i in range(10) :
        cond.acquire( )
        num = random.randrange(10, 20)
        print('Generated number =', num)
        qfors.append(num)
        qforc.append(num)
        cond.notifyAll( )
        cond.release( )

def square( ) :
    for i in range(10) :
        cond.acquire( )
        if len(qfors) :
            num = qfors.popleft( )
            print('num =', num, 'Square =', num * num)
        cond.notifyAll( )
        cond.release( )

def cube( ) :
    for i in range(10) :
        cond.acquire( )
        if len(qforc) :
            num = qforc.popleft( )
            f.write('num = ' + str(num) + ' cube = ' +
                    str(num * num * num) + '\n')
        cond.notifyAll( )
        cond.release( )

f = open('cubes.txt', 'w')
qfors = collections.deque()
qforc = collections.deque()
cond = threading.Condition()
th1 = threading.Thread(target = generate)
th2 = threading.Thread(target = square)
th3 = threading.Thread(target = cube)
th1.start()
th2.start()
th3.start()
th1.join()
th2.join()
th3.join()
f.close()
print('All Done!!')
```

输出：

```
Generated number = 19
Generated number = 12
Generated number = 16
Generated number = 12
Generated number = 11
Generated number = 10
Generated number = 17
Generated number = 10
Generated number = 19
Generated number = 15
num = 19 Square = 361
num = 12 Square = 144
num = 16 Square = 256
num = 12 Square = 144
num = 11 Square = 121
num = 10 Square = 100
num = 17 Square = 289
num = 10 Square = 100
num = 19 Square = 361
num = 15 Square = 225
All Done!!
```

(f) 假设一个线程正在生成从1到n的数字，另一个线程正在打印生成的数字。评论我们可能得到的输出。

答案：输出很可能会混乱，因为两个线程之间没有同步。

(g) 如果线程t1等待线程t2完成，而线程t2等待t1完成，会发生什么？

答案：会发生死锁情况。

### [C] 匹配以下配对：

- a. RLock
- b. Event
- c. Semaphore
- d. Condition
- e. Lock

- 1. 限制访问资源的线程数
- 2. 在可重入代码中共享资源时很有用
- 3. 用于线程间通信
- 4. 在状态变化时通知等待的线程
- 5. 在线程之间共享资源时很有用

答案：

- RLock - 在可重入代码中共享资源时很有用
- Event - 用于线程间通信
- Semaphore - 限制访问资源的线程数
- Condition - 在状态变化时通知等待的线程
- Lock - 在线程之间共享资源时很有用

## 第27章：Numpy库

### [A] 判断以下陈述的正误：

(a) 安装Python时会安装Numpy库。

答案：错误

(b) Numpy数组比列表运行更快。

答案：正确

(c) Numpy数组元素可以是不同类型。

答案：错误

(d) 创建后，Numpy数组的大小和形状可以动态更改。

答案：正确

(e) 如果**a**和**b**的形状和元素匹配，**np.array_equal(a, b))**将返回**True**。

答案：正确

### [B] 回答以下问题：

(a) 你将如何创建一个包含前10个自然数的Numpy数组？

答案：

```python
import numpy as np
intarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

(b) 我们可以使用Numpy创建复数数组吗？

答案：可以，如下所示：

```python
c = np.array( [[1, 2], [3, 4], [-1, 2]], complex)
```

(c) 你将如何创建5个大小为3 x 4 x 5的数组，并分别用值0、1、5、随机值和垃圾值填充它们？

答案：

```python
import numpy as np
a1 = np.zeros((3, 4, 5))    # 创建一个全零的3D数组
a2 = np.ones((3, 4, 5))     # 创建一个全一的3D数组
a3 = np.full((3, 4, 5), 5)  # 创建一个所有值都设为5的3D数组
a4 = np.empty((3, 4, 5))    # 创建一个包含垃圾值的3D数组
a5 = np.full((3, 4, 5), random.random())  # 数组 - 随机值
```

(d) 你将如何创建一个包含50个元素的数组，并用从1开始的奇数填充它？

答案：

```python
import numpy as np
a3 = np.linspace(1, 100, 2)
```

(e) 你将如何获取以下Numpy数组的元素类型、元素数量、基地址和占用的字节数？

```python
a1 = np.array([1, 2, 3, 4])
```

答案：

```python
import numpy as np
a1 = np.array([1, 2, 3, 4])
print(a1.dtype)        # 打印 int32
print(a1.itemsize)     # 打印 4
print(a1.data)         # 打印 <memory at 0x024BEE08>
print(a1.nbytes)       # 打印 16
```

(f) 你将如何获取以下Numpy数组的维度和形状？

```python
a1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
```

答案：

```python
import numpy as np
a1 = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
print(a1.ndim)
print(a1.shape)
print(a1.size)
```

(g) 给定两个3 x 4矩阵，你将如何对这些矩阵的对应元素进行加、减、乘、除运算？

答案：

```python
import numpy as np
a1 = np.array([[1, 2, 3, 4],[5, 6, 7, 8], [1, 4, 5, 2]])
a2 = np.array([[1, 1, 1, 1],[2, 2, 2, 2], [3, 3, 3, 3]])
a3 = a1 + a2
a4 = a1 - a2
a5 = a1 * a2
a6 = a1 / a2
```

(h) 以下哪些是Numpy数组的标量算术运算？

```python
import numpy as np
a1 = np.array([[10, 2, 3, 4],[5, 6, 7, 8]])
a2 = np.array([[1, 1, 1, 1],[2, 2, 2, 2]])
a3 = a1 + a2
a4 = a1 - a2
a5 = a1 * a2
a6 = a1 / a2
a7 = a1 % a2
a8 = a1 ** 2
a9 += a1
a10 += 5
a11 = a1 + 2
a12 = a1 ** 2
```

## 第27章：Numpy库

答案

最后三个操作是标量算术运算。

[C] 将以下配对匹配：

- a. s = np.trace(a)
- b. s = a.cumsum(axis = 1)
- c. a2 = np.copy(a1)
- d. print(a1 < 2)
- e. print(a1 > a2)
- f. print(a[1:3][3:6])
- g. a2 = invert(a1)

- 1. 统计操作
- 2. 线性代数操作
- 3. 深拷贝操作
- 4. 对应元素比较
- 5. 与单个值比较
- 6. 位运算操作
- 7. 切片操作

答案

- a - 1
- b - 2
- c - 3
- d - 5
- e - 4
- f - 7
- g - 6

## D 定期测试

![](img/ac61a02854da7ea89272a34be93604dc_252_0.png)

## 定期测试 I
（基于第1至6章）

时间：90分钟
满分：40分

[A] 填空：[5分，每题1分]

- (1) 在Python中，每个实体都被视为对象。
- (2) Python有33个关键字。
- (3) 用于获取对象地址的函数是`id()`。
- (4) 用于判断变量是否为整数类型的函数是`type()`。
- (5) Python中的三种类型是基本类型、容器类型和用户自定义类型。

[B] 判断正误：[5分，每题1分]

- (1) 在Python中，无需定义变量的类型。
正确
- (2) 在Python中，一个整数可以具有任意值。
正确
- (3) Python语言中的单行注释以`#`号开头。
正确
- (4) switch语句的效果可以通过`if - elif -else`实现。
正确
- (5) Python中没有`do-while`循环。
正确

[C] 以下程序的输出是什么：
[5分，每题1分]

(1) `print(6 // 2)`
输出
3

(2) `print(3 % -2)`
输出
-1

(3) `print(-2 % -4)`
输出
-2

(4) `print(17 / 4)`
输出
4.25

(5) `print(-5 // -3)`
输出
1

[D] 指出以下程序中的错误（如果有的话）：
[5分，每题1分]

```
(1) import math
    x = 2
    print(math.sqrt(x))

无错误
```

```
(2) msg = 'C:\newfolder\newfile'
    print('msg')

无错误。
```

```
(3) a = 4
    b = 2
    if a = b :
        print('Equal')
    else :
        print('Unequal')

错误。应使用`==`而不是`=`。
```

```
(4) lst = ['00', '01', '02', '03']
    i = 0
    while i < len(lst) :
        print(i, lst[i])
        i += 1

无错误
```

```
(5) lst = ['Lion', 'Tiger', 'Wolf', 'Cheetah']
    for i, ele in enum(lst) :
        print(i, ele)

错误。应使用`enumerate()`函数，没有`enum()`函数。
```

[E] 尝试回答以下问题：[20分，每题5分]

(1) 编写一个程序，持续输入数字直到用户停止，最后显示输入的正数、负数和零的个数。

程序

```
#### 统计正数、负数和零的个数
ans = 'y'
pos = neg = zero = 0
while ans == 'y' or ans == 'Y' :
    num = int(input('Enter a number: '))
    if num == 0 :
        zero += 1
    elif num > 0 :
        pos += 1
    elif num < 0 :
        neg += 1
    ans = input('Do you want to continue? ')
print('You entered', pos, 'positive numbers')
print('You entered', neg, 'negative numbers')
print('You entered', zero, 'zeros')
```

(2) 编写一个程序，计算通过键盘输入的一组数字的范围。范围是列表中最小和最大数字的差值。

程序

```
#### 计算一组数字范围的程序
import sys
tot = int(input("Enter total no. of numbers "))
i = 0
small = sys.maxsize
big = -sys.maxsize
while i < tot :
    n = int(input("Enter a number: "))
    if n < small :
        small = n
    if n > big :
        big = n
    i += 1

range = big - small
print('Range = ', range)
```

(3) 如果通过键盘输入三个整数，编写一个程序来判断它们是否构成勾股数。

程序

```
a = int(input('Enter a number: '))
b = int(input('Enter a number: '))
c = int(input('Enter a number: '))
if a * a == b * b + c * c or b * b == a * a + c * c or c * c == a * a + b * b :
    print('Numbers form a Pythagorean triplet')
else :
    print('Numbers do not form a Pythagorean triplet')
```

(4) 编写一个程序计算以下级数前10项的和：

1! 2! + 2! 3! + 3! 4! + 4! 5! + ...... + 9! 10!

程序

```
for i in range(1, 11) :
    prod1 = 1
    s = 0
    for j in range(1, i + 1) :
        prod1 = prod1 * j

    prod2 = prod1 * (j + 1)
    term = prod1 * prod2
    print(prod1, prod2)
    s = s + term
print ( 'sum of series = ', s )
```

## 定期测试 II
（基于第7至11章）

时间：90分钟

满分：40分

[A] 回答以下问题：[5分，每题1分]

(1) 如何创建一个空列表、空元组、空集合和空字典？

答案

```
lst = [ ]
tpl = ( )
s = set( )
dct = { }
```

(2) 集合（set）和冻结集合（frozenset）有什么区别？

答案

集合的元素可以更改，冻结集合的元素不能更改。

(3) 集合 `s = {[10, 20, 30], (10, 20, 30)}` 有什么问题？

答案

集合不能包含列表作为其元素之一。

(4) 如何在不使用`del()`或`remove()`方法的情况下，将列表 `[10, 20, 30, 40, 10, 30]` 转换为列表 `[40, 10, 20, 30]`？

答案

```
lst = list(set(lst))
```

(5) 如何将一个浮点数在10列中居中对齐，并保留小数点后3位进行打印？

答案

```
print(f'{round(a,3):^{10}}')
```

[B] 判断正误：

[5分，每题1分]

- (1) 相似的元素通常存储在集合中。
错误
- (2) 不相似的元素通常存储在列表中。
错误
- (3) 键值对通常存储在元组中。
错误
- (4) 唯一的元素通常存储在字典中。
错误
- (5) 原始字符串用于在`print()`函数中格式化输出。
错误

[C] 匹配以下内容：

[5分，每题1分]

- (a) 字典
- (b) 元组
- (c) 集合
- (d) 列表
- (e) 字符串

- (1) a = set( )
- (2) x = { }
- (3) b = 'msg'
- (4) d = (10, 20, 30)
- (5) f = [10, 20, 30]

答案

- (a) - (2)
- (b) - (4)
- (c) - (1)
- (d) - (5)
- (e) - (3)

[D] 指出以下程序中的错误（如果有的话）：

[5分，每题1分]

(1) l, b, h = input('Enter length, breadth & height: ')
print(l, b, h)

答案

错误：要解包的值过多

```
(2) lst = [11, 10, 5, 77, 24]
lst.add(45)
print(lst)
```

答案

错误：'list'对象没有属性'add'

```
(3) tpl = ((1, 5), (2, 3), (4, 5))
for x, y in tpl :
    print(x, y)
```

答案

无错误

```
(4) s = {77, 41, 22}
s.del(41)
print(s)
```

答案

错误：不能通过`s`调用`del()`

```
(5) dct = { 'Lion' : 4, 'Tiger' : 2, 'Wolf' : 9, 'Cheetah' : 1 }
for k, v in dct.keys( ) :
    print(k, v)
```

答案

错误：要解包的值过多

[E] 尝试回答以下问题：[20分，每题5分]

(1) 编写一个程序，接收圆的半径值，计算并打印其面积和周长。确保两个值都打印在15列中，每个值保留小数点后2位。

程序

```
r = int(input('Enter radius of circle: '))
a = round(3.14 * r * r, 2)
c = round(2 * 3.14 * r, 2)
print(f'Area = {a:15} {Circumference = }{c:15}')
```

输出

Enter radius of circle: 5
Area = 78.5 Circumference = 31.4

(2) 一个字典包含学号作为键，名字、中间名和姓氏作为值。编写一个程序，按名字的字母顺序打印字典项。

程序

```
import operator
d1 = {
    'A101' : ('Rahul', 'Ajay', 'Joshi'),
    'A102' : ('Ramesh', 'Atul', 'John'),
    'A121' : ('Ritesh', 'Abhin', 'Kate'),
    'A111' : ('Rajesh', 'Akash', 'Zade')
}
d2 = dict(sorted(d1.items( ), key = operator.itemgetter(1)))
print(d2)
```

输出

{'A101': ('Rahul', 'Ajay', 'Joshi'), 'A111': ('Rajesh', 'Akash', 'Zade'), 'A102': ('Ramesh', 'Atul', 'John'), 'A121': ('Ritesh', 'Abhin', 'Kate')}

(3) 一个元组包含书名和作者的元组。编写一个程序，打印书名和作者名，其中每个单词的首字母大写，其余字母小写。

程序

```
tpl = ( ('Principles of programming', 'rahul ajay joshi'),
        ('Art of computer science', 'donald e knuth'),
        ('Modern algebra', 'Ritesh abhin KATE') )
for t in tpl :
```

## 定期测试 III
（基于第12至17章）

时间：90分钟

满分：40分

### [A] 判断正误：[5分，每题1分]

(1) 我们不能使用推导式创建元组。
正确

(2) Python函数可以接收位置参数、关键字参数、可变长度位置参数和可变长度关键字参数。
正确

(3) 当我们执行一个程序时，其模块名称是`__module__`，并且它在变量`__name__`中可用。
正确

(4) 在模块`functions`中定义的函数`show()`和`display()`可以通过使用以下语句导入来使用：
`import show, display`
错误

(5) 要使一个目录被视为包，它必须包含一个名为`__init__.py`的文件。
正确

### [B] 回答以下问题：[10分，每题1分]

(1) 如何使用列表推导式从键盘接收4个数字作为输入？
答案
`n1, n2, n3, n4 = [int(n) for n in input('Enter four values: ').split( )]`

(2) 如何使用列表推导式生成20个在10到100范围内的随机数？
答案

```
import random
a = [random.randrange(10, 100) for n in range(20)]
```

(3) 如何使用推导式生成以下列表？
`[[25, 125], [36, 216], [49, 343], [64, 512], [81, 729], [100, 1000]]`
答案

```
a = [[n, n * n] for n in range(5, 11)]
```

(4) 如何使用推导式创建一个包含10到30范围内偶数的集合？
答案

```
a = [n for n in range(10, 30) if n % 2 == 0]
```

(5) 如何使用推导式从以下列表中删除所有值在20到50之间的数字？
`lst = [10, 3, 4, 5, 15, 20, 21, 23, 46, 50]`
答案

```
lst1 = [n for n in lst if n < 20 or n > 50]
```

(6) 使用字典推导式，如何将
`d = {'AMOL': 20, 'ANIL': 12, 'SUNIL': 13, 'RAMESH': 10}`
转换为
`{'Amol': 400, 'Anil': 144, 'Sunil': 169, 'Ramesh': 100}`
答案

```
d = {k.capitalize( ) : v ** 2 for (k, v) in d.items( )}
```

(7) 考虑以下代码片段：

```
def print_it(a, b, c, d, e) :
    print(a, b, c, d, e)

tpl = ('A', 'B', 'C', 'D', 'E')
```

如何将元组**t**的所有元素传递给函数**print_it()**？
答案

```
print_it(*tpl)
```

(8) 考虑以下代码片段：

```
def print_it(i, j, *args, x, y, **kwargs) :
    pass

print_it(10, 20, 100, 200, x = 30, y = 40)
```

什么被传递给了**args**和**kwargs**？
答案

100, 200 传递给了 args，没有内容传递给 kwargs

(9) 如果一个函数**cal_sum()**接收**3个整数**并返回它们的**和**，以下哪些对函数**cal_sum()**的调用是可接受的？
- i. `sum = cal_sum(a, b, c)`
- ii. `print(cal_sum(a, b, c))`
- iii. `sum = cal_sum(a, calSum(25, 10, 4), b)`
答案

i, ii, iii 和 iv

(10) 考虑以下代码片段：

```
def print_it(**kwargs) :
    pass

dct = {'Student' : 'Ajay', 'Age' : 23}
```

将`dct`传递给**print_it()**的正确方式是？
答案

```
print_it(**dct)
```

### [C] 尝试回答以下问题：[20分，每题5分]

(1) 编写一个程序，使用列表推导式将摄氏温度列表转换为等效的华氏温度。
程序

```
celsius = [32, 40, 25, 45, 18]
farh = [(e * 9 /5) + 32 for e in celsius]
print(farh)
```

输出

`[89.6, 104.0, 77.0, 113.0, 64.4]`

(2) 编写一个递归函数来计算前25个自然数的和。
程序

```
def resum(num) :
    if num == 1 :
        return num
    return num + resum(num - 1)

print('Sum of first 25 numbers: ', resum(25))
```

输出

`Sum of first 25 numbers: 325`

(3) 从键盘输入一个字符串。编写一个递归函数来计算该字符串中大写字母的数量。
程序

```
def count_caps(s, count) :
    if s == "" :
        return count

    if s[0] >= 'A' and s[0] <= 'Z' :
        count += 1

    count = count_caps(s[1:], count)
    return count

c = count_caps('Cidade de Goa', 0)
print('Count of caps = ', c)
```

输出

`Count of caps = 2`

(4) 使用函数式编程编写一个程序，为字符串列表中的每个元素添加前缀字符串'Hi '。
程序

```
lst1 = ['Shrinivas', 'Savitri', 'Shanmukh', 'Shweta']
lst2 = map(lambda s : 'Hi ' + s, lst1)
for item in lst2 :
    print(item)
```

输出

`Hi Shrinivas`
`Hi Savitri`
`Hi Shanmukh`
`Hi Shweta`

(5) 假设一个字典包含学生姓名和他们在考试中获得的分数。使用函数式编程编写一个程序，获取考试中分数低于40分的学生列表。
程序

```
dct = {'Shrinivas' : 35, 'Savitri' : 45, 'Shanmukh' : 38, 'Shweta' : 42}
lst2 = filter(lambda x : x[1] < 40, dct.items( ))
for item in lst2 :
    print(item)
```

输出

`('Shrinivas', 35)`
`('Shanmukh', 38)`

## 定期测试 IV
（基于第18至21章）

时间：90分钟

满分：40分

### [A] 填空：[4分，每题1分]

- (1) 每个类都派生自一个对象类。
- (2) Python中提供了继承和容器化重用机制。
- (3) 当我们使用容器化时，我们是在进行字节码级别的重用。
- (4) 生成器函数创建迭代器。

### [B] 判断正误：[12分，每题1分]

- (1) 对象的构造总是从派生类向基类进行。
错误
- (2) 不能从抽象类创建对象。
正确
- (3) 调用对象的实例方法时，对象的地址总是会传递给它。
正确
- (4) 可迭代对象不能传递给`zip()`函数。
错误
- (5) 在容器化中，一个对象嵌套在另一个对象内部。
正确
- (6) `+`、`-`和`*`运算符在`str`类中已被重载。
错误
- (7) `+`运算符在`str`、`list`和`int`类中已被重载。
正确
- (8) 是否可以从类方法中调用全局函数、另一个类方法和实例方法？
正确
- (9) 对象的大小受实例数据、类数据、全局数据和局部变量的影响。
错误
- (10) 类方法可以访问类数据和全局数据。
正确
- (11) 如果`NewSample`类派生自`Sample`类，并且创建了一个`NewSample`对象，那么将先调用`Sample`类的`__init__()`，然后调用`NewSample`类的`__init__()`。
正确
- (12) 生成器表达式按需生成下一个元素，而不是预先生成所有元素。
正确

### [C] 尝试回答以下问题：[14分，每题1分]

(1) 将以下内容分离为类和对象：
Bird, Player, Crow, Raj, Eagle, Flower, Rose, Lily, Flute, Instrument
答案

类 - Bird, Player, Flower, Instrument
对象 - Crow, Raj, Eagle, Rose, Lily, Flute

(2) 如何创建一个`Trial`类的对象？
答案

`t = Trial( )`

(3) 关于一个类，以下哪些操作可以完成？
- i. 一个类可以从另一个类继承
- ii. 一个类可以在另一个类内部定义
答案

i 和 ii 都可以

(4) 以下代码做了什么？
```
e = Sample( )
```
答案

为`Sample`类型的对象分配空间并调用构造函数

(5) 当控制从`fun()`返回时会发生什么？
```
def func( ) :
    s = new Trial( )
```
答案

`__del__()` 方法会被调用。

(6) 在以下语句中，将有多少个参数传递给`__init__()`？
```
t = Trial(10, 3.14, 'Hello')
```
答案

4

(7) 在执行以下代码片段时，`Student`类的`__init__()`被调用了多少次？
```
lst = [ ]
lst.append(Sample('Raju', 25))
lst.append(Sample('Anand', 34))
```

## 定期测试 V
（基于第22至26章）

时间：90分钟
满分：40分

**[A] 填空题：** [5分，每题1分]

1.  Python装饰器以@符号开头。
2.  传递给Python脚本的参数可在变量`sys.argv`中获取。
3.  传递给脚本的命令行参数可以使用`getopt()`函数进行解析。
4.  在Unicode中，每个字符都被分配一个称为码点的整数值，通常以十六进制表示。
5.  线程间的通信可以使用Event或Condition对象来完成。

**[B] 判断正误：** [10分，每题1分]

1.  在并行处理中，多个线程同时运行。
    正确
2.  在并发处理中，任何给定时间只有一个线程在运行。
    正确
3.  如果程序的不同单元在重叠的时间内执行，I/O密集型程序的性能可以得到提升。
    正确
4.  CPU密集型程序的性能可以通过并行处理得到提升。
    正确
5.  100米赛跑是并发处理的一个好例子。
    错误
6.  `except`块的顺序很重要。
    正确
7.  断言执行对假设的运行时检查。
    正确
8.  序列化意味着将对象写入文件。
    正确
9.  `'r+'`和`'w+'`模式是相同的，因为两者都允许从文件读取和写入文件。
    错误
10. 当使用`with`关键字打开文件时，文件会在使用完毕后立即关闭。
    正确

**[C] 回答以下问题：** [10分，每题1分]

(1) 语法错误、逻辑错误和异常中，哪一种发生在执行时？

答案
异常

(2) 你将如何把装饰器`@yk_decorator`应用到函数`fun()`上？

答案
@yk_decorator
def fun( ) :
    pass

(3) 考虑以下代码片段：

```
def yk_decorator(func) :
    def wrapper( ) :
        print('################')
        func( )
        print('################')
```

你需要在这个代码片段的末尾添加哪条语句才能使装饰器工作？

答案
return wrapper

(4) 如果我们执行一个脚本，命令为：
C:\>sample.py cat dog parrot
你将如何访问`'C:\>sample.py'`？

答案
sys.argv[0]

(5) 如果一个脚本执行命令为：
C:\>filecopy.py -s sourcefilename -t targetfilename
那么你将如何使用`getopt()`函数解析参数？

答案
options = getopt.getopt(sys.argv[1:],'s:t:')

(6) 以下代码片段的输出是什么？

```
import sys, getopt
sys.argv =['C:\a.py', '-h', 'word1', 'word2']
options, arguments = getopt.getopt(sys.argv[1:],'s:t:h')
print(options)
print(arguments)
```

答案
[('-h', '')]
['word1', 'word2']

(7) 你将如何在一个线程中启动函数`fun()`并传递参数`a, b, c`给它？

答案
```
th1 = threading.Thread(target = squares, args = (a, b))
```

(8) 你将如何检查`num`的第5位和第7位是开启还是关闭？

答案
```
if num & 32 == 1 and num & 128 == 1 :
    print('bits 5 and 7 are on')
```

(9) 你将如何将十六进制值`E0A485`存储在`bytes`数据类型中？

答案
```
by = b'\xe0\xa4\x85'
```

(10) 如果对3种异常要采取相同的操作，你需要编写多少个`except`块？

答案
应该只编写一个`except`子句，并在元组中列出这3种异常。

**[D] 尝试回答以下问题：** [20分，每题5分]

(1) 编写一个程序，定义一个函数，使用以下公式计算`c`的值：

```
c = ((a + b) / (a - b))
```

程序
```
def compute(a, b) :
    try :
        c = ((a + b) / (a - b))
    except ZeroDivisionError :
        print('Denominator is 0!!')
    else :
        print('c =', c)

a = int(input('Enter any integer: '))
b = int(input('Enter any integer: '))
c = compute(a, b)
```

输出
```
Enter any integer: 12
Enter any integer: 10
c = 11.0
```

(2) 一个文件`'sent.txt'`包含多行，每行包含多个单词。编写一个程序来找出文件中最长的单词。

程序
```
def longest_word(fname) :
    f = open(fname, 'r')
    words = f.read( ).split( )
    print(words)
    big = len(max(words, key = len))
    return [w for w in words if len(w) == big]

print(longest_word('sent.txt'))
```

输出
```
['Bad', 'officials', 'are', 'elected', 'by', 'good', 'citizens', 'who', 'do', 'not', 'vote', 'Good', 'citizens', 'is', 'a', 'rare', 'breed', 'Having', 'one', 'child', 'makes', 'you', 'a', 'parent', 'Having', 'two', 'you', 'are', 'a', 'referee', 'There', 'is', 'always', 'life', 'beyond', 'work', 'and', 'entertainment', 'Your', 'communication', 'skills', 'are', 'vital', 'Divisibility', 'of', 'integer', '9']
['entertainment', 'communication']
```

(3) 定义一个装饰器函数`@timer`，用于计算执行任何函数所需的时间。使用此装饰器来计时计算7、10和25的阶乘值的函数`factorial()`。

程序
```
import time
def timer(func) :
    def calculate(*args, **kwargs) :
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'Finished {func.__name__!r} in {runtime:.8f} secs')
        return value
    return calculate

@timer
def factorial(num) :
    p = i = 1
    while i <= num :
        p = p * i
        i += 1
    return(p)

f = factorial(7)
print('Factorial of 7 = ', f)
f = factorial(10)
print('Factorial of 10 = ', f)
f = factorial(25)
print('Factorial of 25 = ', f)
```

输出
```
Finished 'factorial' in 0.00000599 secs
Factorial of 7 = 5040
Finished 'factorial' in 0.00000813 secs
Factorial of 10 = 3628800
Finished 'factorial' in 0.00001583 secs
Factorial of 25 = 15511210043330985984000000
```