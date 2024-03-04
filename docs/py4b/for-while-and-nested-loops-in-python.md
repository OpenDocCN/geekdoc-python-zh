# Python For 循环、While 循环和嵌套循环

> 原文：<https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python>

所有的编程语言都需要多次做类似的事情，这叫做迭代。

Python 中迭代的例子是循环。Python 使用 For 循环、While 循环和嵌套循环。

## 对于循环

Python 中的 For 循环允许我们迭代一个序列的元素，当你有一段想要重复 n 次的代码时，经常使用它。

for 循环语法如下:

```py
for x in list :

    do this..

    do this..
```

### for 循环的示例

假设您有一个如下所示的浏览器列表。对于我们赋予变量 browser 的每个元素，在浏览器列表中，打印出变量 browser

```py
browsers = ["Safari", "Firefox", "Chrome"]
for browser in browsers:
    print browser 
```

### for 循环的另一个例子

为了获得更多关于 for 循环的实践，请参见以下示例:

```py
numbers = [1,10,20,30,40,50]
sum = 0
for number in numbers:
    sum = sum + number
print sum 
```

### 循环单词

这里我们使用 for 循环来遍历单词 computer

```py
word = "computer"
for letter in word:
    print letter 
```

## 使用 python 范围函数

Python 编程语言有一个内置函数“range ”,可以生成一个包含我们在范围内指定的数字的列表。

给定的端点决不是生成列表的一部分；

range(10)生成 10 个值的列表，长度为 10 的序列的合法索引。

可以让范围函数从另一个数字开始，或者指定一个不同的增量(甚至是负数)。

这被称为“步骤”。

### 范围函数语法

范围*(开始、停止、步进*

开始是你开始的位置。

Stop 是停止的数字，不包括。

步长是增量，默认值为 1。

### 范围函数示例

#### **例 1**

打印从 0 开始到 5 结束的数字(不包括 5)。此示例使用了一个带有范围的 for 循环。

```py
>>> for number in range(0,5): print number
... 
0
1
2
3
4
>>>
```

#### **例 2**

打印从 1 开始到 10 结束的范围，不包括。这个范围示例没有使用 for 循环。

```py
>>> range(1,10)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
>>>
```

#### **例 3**

这个 range 函数示例被分配给一个变量。然后使用 for 循环迭代该变量。

```py
>>> a = range(1,10)
>>> for i in a: print i
... 
1
2
3
4
5
6
7
8
9
>>> 
```

#### **例 4**

这是一个使用范围内步长的示例。步长默认值为 1。在这个例子中，我们使用-2 作为步长，从 21 开始，直到-1。

```py
>>> for a in range(21,-1,-2): print a
... 
21
19
17
15
13
11
9
7
5
3
1
>>> 
```

## While 循环

Python 中的 while 循环告诉计算机只要满足条件就做某件事，它的构造由一段代码和一个条件组成。

在 while 和冒号之间，有一个值首先为真，但随后为假。

计算条件，如果条件为真，则执行块中的代码。

只要该语句为真，其余的代码就会运行。

将要运行的代码必须在缩进的块中。

它是这样工作的:“虽然这是真的，但这样做”

### While 循环的示例

下面的例子是这样的:只要变量 I 的值小于列表的长度(浏览器)，就打印出变量名。

循环语法:

```py
browsers = ["Safari", "Firefox", "Google Chrome"]
i = 0
while i < len(browsers):
    print browsers[i]
    i = i + 1
```

### While 循环的另一个例子

下面的脚本首先将变量 counter 设置为 0。

每当 while 循环运行一次，计数器的值就增加 2。只要变量计数器小于或等于 100，循环就会运行。

```py
counter = 0
while counter <= 100:
    print counter
    counter = counter + 2
```

### 使用 While 循环计数

这个小脚本会从 0 数到 9。i = i + 1 每运行一次，I 值就加 1。

```py
i = 0
while i < 10:
    print i
    i = i + 1
```

## 永恒的循环

注意不要在 Python 中形成一个永恒的循环，当你按下 Ctrl+C 时，循环会继续，确保你的 while 条件会在某个时候返回 false。

这个循环意味着 while 条件将永远为真，并将永远打印 Hello World。

```py
while True:
    print "Hello World" 
```

## 嵌套循环

在一些脚本中，你可能想要使用嵌套循环。

Python 中的嵌套循环是循环中的循环。

当你有一段你想运行 x 次的代码，然后在这段代码中你想运行 y 次的代码

在 Python 中，每当有人有一个列表的列表时——一个可迭代对象中的一个可迭代对象——就会大量使用这些。

```py
for x in range(1, 11):
    for y in range(1, 11):
        print '%d * %d = %d' % (x, y, x*y) 
```

## 打破循环

要从循环中脱离，可以使用关键字“break”。Break 停止循环的执行，与测试无关。break 语句可以在 while 和 for 循环中使用。

### 中断示例

这将要求用户输入。当用户键入“停止”时，循环结束。

```py
while True:
    reply = raw_input('Enter text, [type "stop" to quit]: ')
    print reply.lower()
    if reply == 'stop':
        break 
```

### 另一个破裂的例子

让我们再看一个如何使用 break 语句的例子。

```py
while True:
  num=raw_input("enter number:")
  print num
  if num=='20':
      break 
```

让我们举例说明如何在 for 循环中使用 break 语句。

```py
for i in range(1,10):
    if i == 3:
        break
    print i 
```

## 继续

continue 语句用于告诉 Python 跳过当前循环块中的其余语句，并继续循环的下一次迭代。

continue 语句拒绝当前循环迭代中的所有剩余语句，并将控制移回循环顶部。

continue 语句可以在 while 和 for 循环中使用。

```py
for i in range(1,10):
    if i == 3:
        continue
    print i 
```

### 继续示例

这个例子取自[tutorialspoint.com](https://www.tutorialspoint.com/python/python_loop_control.htm "tutorialspoint")

```py
#!/usr/bin/python

for letter in 'Python':     # First Example
   if letter == 'h':
       continue
   print 'Current Letter :', letter

var = 10                    # Second Example
while var > 0:              
   var = var -1
   if var == 5:
       continue
       print 'Current variable value :', var
   print "Good bye!" 
```

##### 输出

上述输出将产生以下结果:

```py
 Current Letter : P
Current Letter : y
Current Letter : t
Current Letter : o
Current Letter : n
Current variable value : 10
Current variable value : 9
Current variable value : 8
Current variable value : 7
Current variable value : 6
Current variable value : 4
Current variable value : 3
Current variable value : 2
Current variable value : 1
Good bye! 
```

## 及格

pass 语句什么也不做。当语法上需要一个语句，但程序不需要动作时，可以使用它。

```py
 >>> while True:
    ...       pass # Busy-wait for keyboard interrupt
    ... 
```