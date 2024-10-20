# Python 函数参数的乐趣

> 原文：<https://www.pythoncentral.io/fun-with-python-function-parameters/>

事实上，每种编程语言都有函数和过程，一种从程序的不同位置分离出可以多次调用的代码块的方法，以及一种向它们传递参数的方法。Python 也不例外，所以我们将快速浏览一下大多数语言都有的标准内容，然后看看 Python 提供的一些很酷的东西。

## **Python 中的位置函数参数**

这是一个非常简单的函数:

```py

def foo(val1, val2, val3):

return val1 + val2 + val3

```

使用时，我们会得到以下内容:

```py

>>> print(foo(1, 2, 3))

6

```

该函数有 3 个*位置*参数(每个参数在被调用时获取传递给该函数的下一个值——`val1`获取第一个值(1)，`val2`获取第二个值(2)，依此类推)。

### **命名 Python 函数参数(带默认值)**

Python 还支持*命名的*参数，这样当一个函数被调用时，参数可以通过名称显式赋值。这些通常用于实现默认值或可选值。例如:

```py

def foo(val1, val2, val3, calcSum=True):

# Calculate the sum

if calcSum:

return val1 + val2 + val3

# Calculate the average instead

else:

return (val1 + val2 + val3) / 3

```

使用该函数可以得到以下结果:

```py

>>> print(foo(1, 2, 3))

6

>>> print(foo(1, 2, 3, calcSum=False))

2

```

在这个例子中，`calcSum`是可选的——如果在调用函数时没有指定，它将获得默认值`True`。

### **潜在的 Python 函数参数问题**

要记住的一点是，默认值是在函数编译时计算的，如果值是可变的，这是一个重要的区别。下面的行为可能不是我们想要的:

```py

def foo(val, arr=[]):

arr.append(val)

return arr

```

使用该功能时:

```py

>>> print(foo(1))

[1]

>>> print(foo(2))

[1, 2]

```

发生这种情况是因为默认值(一个空列表)在函数编译时被求值一次，然后在每次调用函数时被重用。为了在每次调用中得到一个空列表，代码需要写成这样:

```py

def foo(val, arr=None):

if arr is None:

arr = []

arr.append(val)
返回 arr 

```

使用该函数时，我们得到:

```py

>>> print(foo(1))

[1]

>>> print(foo(2))

[2]

```

### **Python 函数参数顺序**

与位置参数不同，命名参数的指定顺序无关紧要:

```py

def foo(val1=0, val2=0):

return val1 – val2

```

使用时，我们会得到以下内容:

```py

>>> print(foo(val1=10, val2=3))

7

>>> # Note: parameters are in a different order

>>> print(foo(val2=3, val1=10))

7

```

不同寻常的是，Python 还允许通过名称来指定位置参数:

```py

def foo(val1, val2):

''' Note: parameters are positional, not named. '''

return val1 – val2

```

使用时，我们会得到以下内容:

```py

>>> # But we can still set them by name

>>> print(foo(val1=10, val2=3))

7

>>> # And in any order!

>>> print(foo(val2=3, val1=10))

7

```

这里有一个更复杂的例子:

```py

def foo(p1, p2, p3, n1=None, n2=None):

print('[%d %d %d]' % (p1, p2, p3))

if n1:

print('n1=%d' % n1)

if n2:

print('n2=%d' % n2)

```

使用时，我们会得到以下内容:

```py

>>> foo(1, 2, 3, n2=99)

[1 2 3]

n2=99

>>> foo(1, 2, n1=42, p3=3)

[1 2 3]

n1=42

```

这看起来确实令人困惑，但是理解它如何工作的关键是要认识到函数的参数列表是一个字典(一组键/值对)。Python 首先匹配位置参数，然后分配函数调用中指定的任何命名参数。

### **变量 Python 函数参数表**

当您开始查看变量参数列表时，Python 的酷就真正开始了。您甚至可以在不知道将传入什么参数的情况下编写函数！

在下面的函数中， *vals* 参数前面的星号表示*任何其他位置参数*。

```py

def lessThan(cutoffVal, *vals) :

''' Return a list of values less than the cutoff. '''

arr = []

for val in vals :

if val < cutoffVal:

arr.append(val)

return arr

```

使用该函数时，我们得到:

```py

>>> print(lessThan(10, 2, 17, -3, 42))

[2, -3]

```

我们在函数调用`(10)`中指定的第一个位置值被赋予函数中的第一个参数(`cutoffVal`)，然后所有剩余的位置值被放入一个元组中并被分配给 val。然后我们遍历这些值，寻找任何小于临界值的值。

我们也可以用命名参数做同样的事情。下面 dict 参数前面的双星号表示*任何其他命名的参数*。这一次，Python 将把它们作为字典中的键/值对提供给我们。

```py

def printVals(prefix='', **dict):

# Print out the extra named parameters and their values

for key, val in dict.items():

print('%s [%s] => [%s]' % (prefix, str(key), str(val)))

```

使用该函数时，我们得到:

```py

>>> printVals(prefix='..', foo=42, bar='!!!')

[foo] => [42]

[bar] => [!!!]

>>> printVals(prefix='..', one=1, two=2)

[two] => [2]

[one] => [1]

```

请注意，在最后一个示例中，这些值并没有按照在函数调用中指定的顺序打印出来。这是因为这些额外命名的参数是在字典中传递的，字典是一个*无序的*数据结构。

### **一个真实世界的例子**

那么，你会用这些做什么呢？例如，程序通常会从一个模板生成消息，该模板具有占位符，用于在运行时插入值。例如:

> 您好{name}。您的帐户余额为{1}，您还有{2}可用点数。

下面的函数使用这样一个模板和一组参数来替换占位符。

```py

def formatString(stringTemplate, *args, **kwargs):

# Replace any positional parameters

for i in range(0, len(args)):

tmp = '{%s}' % str(1+i)

while True:

pos = stringTemplate.find(tmp)

if pos < 0:

break

stringTemplate = stringTemplate[:pos] + \

str(args[i]) + \

stringTemplate[pos+len(tmp):]
#替换 kwargs.items()中 key，val 的任何命名参数
:
tmp = ' { % s } ' % key
while True:
pos = string template . find(tmp)
if pos<0:
break
string template = string template[:pos]+\
str(val)+\
string template[pos+len(tmp):]
返回字符串模板

```

这就是它的作用:

```py

>>> stringTemplate = 'pos1={1} pos2={2} pos3={3} foo={foo} bar={bar}'

>>> print(formatString(stringTemplate, 1, 2))

pos1=1 pos2=2 pos3={3} foo={foo} bar={bar}

>>> print(formatString(stringTemplate, 42, bar=123, foo='hello'))

pos1=42 pos2={2} pos3={3} foo=hello bar=123

```