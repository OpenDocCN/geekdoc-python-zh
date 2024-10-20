# Python 随机模块–生成随机数/序列

> 原文：<https://www.askpython.com/python-modules/python-random-module-generate-random-numbers-sequences>

本文讲的是 Python 中的`random`模块，用于为各种概率分布生成伪随机数。

* * *

## Python 随机模块方法

### 1.种子()

这将初始化一个随机数生成器。要生成新的随机序列，必须根据当前系统时间设置种子。`random.seed()`设置随机数生成的种子。

### 2\. getstate()

这将返回一个包含生成器当前状态的对象。要恢复状态，将对象传递给`setstate()`。

### 3\. setstate(state_obj)

通过传递状态对象，这将恢复调用`getstate()`时发生器的状态。

### 4 .烧录位元

这将返回一个带有`k`随机位的 Python 整数。这对于像`randrange()`这样的方法来处理随机数生成的任意大范围非常有用。

```py
>>> import random
>>> random.getrandbits(100) # Get a random integer having 100 bits
802952130840845478288641107953

```

这里有一个例子来说明`getstate()`和`setstate()`方法。

```py
import random

random.seed(1)

# Get the state of the generator
state = random.getstate()

print('Generating a random sequence of 3 integers...')
for i in range(3):
    print(random.randint(1, 1000))

# Restore the state to a point before the sequence was generated
random.setstate(state)
print('Generating the same identical sequence of 3 integers...')
for i in range(3):
    print(random.randint(1, 1000))

```

可能的输出:

```py
Generating a random sequence of 3 integers...
138
583
868
Generating the same identical sequence of 3 integers...
138
583
868

```

* * *

## 生成随机整数

random 模块提供了一些生成随机整数的特殊方法。

### 1.随机范围(开始、停止、步进)

从`range(start, stop, step)`返回一个随机选择的整数。这就引出了一个`ValueError`如果`start` > `stop`。

### 2\. randint(a, b)

返回一个介于 **a** 和 **b** 之间的随机整数(包括这两个值)。这也引出了一个`ValueError`如果`a` > `b`。

这里有一个例子说明了上述两个功能。

```py
import random

i = 100
j = 20e7

# Generates a random number between i and j
a = random.randrange(i, j)
try:
    b = random.randrange(j, i)
except ValueError:
    print('ValueError on randrange() since start > stop')

c = random.randint(100, 200)
try:
    d = random.randint(200, 100)
except ValueError:
    print('ValueError on randint() since 200 > 100')

print('i =', i, ' and j =', j)
print('randrange() generated number:', a)
print('randint() generated number:', c)

```

可能储量

```py
ValueError on randrange() since start > stop
ValueError on randint() since 200 > 100
i = 100  and j = 200000000.0
randrange() generated number: 143577043
randint() generated number: 170

```

* * *

## 生成随机浮点数

与生成整数类似，还有生成随机浮点序列的函数。

*   随机的。 **random** () - >返回[0.0 到 1.0]之间的下一个随机浮点数
*   随机的。**均匀** (a，b) - >返回一个随机浮点`N`使得 *a < = N < = b* 如果 a < = b 并且 *b < = N < = a* 如果 b < a
*   随机的。**指数变量**(λ)->返回对应于指数分布的数字。
*   随机的。**高斯** (mu，sigma) - >返回一个对应于高斯分布的数字。

其他分布也有类似的函数，如正态分布、伽玛分布等。

生成这些浮点数的示例如下:

```py
import random

print('Random number from 0 to 1 :', random.random())
print('Uniform Distribution between [1,5] :', random.uniform(1, 5))
print('Gaussian Distribution with mean = 0 and standard deviation = 1 :', random.gauss(0, 1))
print('Exponential Distribution with lambda = 0.1 :', random.expovariate(0.1))
print('Normal Distribution with mean = 1 and standard deviation = 2:', random.normalvariate(1, 5))

```

可能储量

```py
Random number from 0 to 1 : 0.44663645835100585
Uniform Distribution between [1,5] : 3.65657099941547
Gaussian Distribution with mean = 0 and standard deviation = 1 : -2.271813609629832
Exponential Distribution with lambda = 0.1 : 12.64275539117617
Normal Distribution with mean = 1 and standard deviation = 2 : 4.259037195111757

```

* * *

## 使用随机模块的随机序列

与整数和浮点序列类似，一般序列可以是项目的集合，如列表/元组。`random`模块提供了一些有用的功能，可以给序列引入一种随机性状态。

### 1.随机洗牌

这是用来随机播放序列的。序列可以是包含元素的任何列表/元组。

演示洗牌的示例代码:

```py
import random

sequence = [random.randint(0, i) for i in range(10)]

print('Before shuffling', sequence)

random.shuffle(sequence)

print('After shuffling', sequence)

```

可能的输出:

```py
Before shuffling [0, 0, 2, 0, 4, 5, 5, 0, 1, 9]
After shuffling [5, 0, 9, 1, 5, 0, 4, 2, 0, 0]

```

### 2.随机选择(序列)

在实践中，这是一个广泛使用的函数，您可能希望从一个列表/序列中随机选取一个项目。

```py
import random

a = ['one', 'eleven', 'twelve', 'five', 'six', 'ten']

print(a)

for i in range(5):
    print(random.choice(a))

```

可能储量

```py
['one', 'eleven', 'twelve', 'five', 'six', 'ten']
ten
eleven
six
twelve
twelve

```

### 3.随机样本(总体，k)

从长度为`k`的序列中返回一个随机样本。

```py
import random

a = ['one', 'eleven', 'twelve', 'five', 'six', 'ten']

print(a)

for i in range(3):
    b = random.sample(a, 2)
    print('random sample:', b)

```

可能储量

```py
['one', 'eleven', 'twelve', 'five', 'six', 'ten']
random sample: ['five', 'twelve']
random sample: ['ten', 'six']
random sample: ['eleven', 'one']

```

* * *

## 随机种子

由于伪随机生成是基于以前的数字，我们通常使用系统时间来确保程序在每次运行时都给出一个新的输出。因此我们利用了`**seeds**`。

Python 为我们提供了`random.seed()`，用它我们可以设置一个种子来获得一个初始值。这个种子值决定了随机数生成器的输出，因此如果它保持不变，输出也保持不变。

```py
import random

random.seed(1)

print('Generating a random sequence of 4 numbers...')
print([random.randint(1, 100) for i in range(5)])

# Reset the seed to 1 again
random.seed(1)

# We now get the same sequence
print([random.randint(1, 100) for i in range(5)])

```

可能储量

```py
Generating a random sequence of 4 numbers...
[18, 73, 98, 9, 33]
[18, 73, 98, 9, 33]

```

这确保了我们在处理伪随机序列时需要注意我们的种子，因为如果种子不变，序列可能会重复。

* * *

## 结论

我们学习了 Python 的 random 模块为我们提供的各种方法，用于处理整数、浮点数和列表等其他序列。我们还看到了**种子**如何影响伪随机数的序列。

## 参考

*   [Python 随机模块文档](https://docs.python.org/2/library/random.html)
*   JournalDev 关于随机数的文章