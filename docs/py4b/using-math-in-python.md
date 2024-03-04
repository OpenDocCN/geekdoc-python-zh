# 在 python 中使用数学

> 原文：<https://www.pythonforbeginners.com/basics/using-math-in-python>

## Python 中的数学

Python 发行版包括 Python 解释器，一个非常简单的开发
环境，称为 IDLE、库、工具和文档。

Python 预装在许多(如果不是全部)Linux 和 Mac 系统上，但它可能是旧版本。

## 计算器

要开始使用 python 解释器作为计算器，只需在
shell 中输入 Python。

```py
>>> 2 + 2
4

>>> 4 * 2
8

>>> 10 / 2
5

>>> 10 - 2
8 
```

## 用变量计数

在变量中输入一些值来计算矩形的面积

```py
>>> length = 2.20
>>> width = 1.10
>>> area = length * width
>>> area
2.4200000000000004 
```

## 计数器

计数器在编程中很有用，每次运行
时增加或减少一个值。

```py
>>> i = 0
>>> i = i + 1
>>> i
1
>>> i = 1 + 2
>>> i
3 
```

## 用 While 循环计数

这里有一个例子说明了使用计数器的用处

```py
>>> i = 0
>>> while i < 5:
...     print i
...     i =  i + 1
... 
0
1
2
3
4 
```

程序从 0 数到 4。在单词 while 和冒号之间，有一个
表达式，起初为真，但随后变为假。

只要表达式为真，下面的代码就会运行。

需要运行的代码必须缩进。

最后一条语句是一个计数器，每当循环
运行时，它的值就加 1。

## 乘法表

用 Python 制作乘法表很简单。

```py
table = 8
start = 1
max = 10
print "-" * 20
print "The table of 8"
print "-" * 20
i = start
while i <= max:
    result = i * table
    print i, " * ", table, " =" , result
    i = i + 1
print "-" * 20
print "Done counting..."
print "-" * 20 
```

> >输出:

———————
8
的表—————
1 * 8 = 8
2 * 8 = 16
3 * 8 = 24
4 * 8 = 32
5 * 8 = 40
6 * 8 = 48
7 * 8 = 56
8 * 8 = 64
9 * 8 = 72
10 * 8 = 80
———