# Python 中的“不可变”是什么意思？

> 原文：<https://www.askpython.com/python/oops/immutable-objects-in-python>

在 Python 中每个实体都是一个[对象](https://www.askpython.com/python/oops/python-classes-objects)的情况下，不可变是什么意思？与其他一些编程语言不同，Python 不需要显式指定赋给变量的数据类型。相反，它会根据您提供的值自动分配[数据类型](https://www.askpython.com/python/python-data-types)。

简而言之，每个变量保存一个对象实例，并被赋予一个惟一的对象 ID，该 ID 是在程序运行时创建的。对象 ID 是一个整数，表示存储变量值的[内存位置](https://www.askpython.com/python-modules/garbage-collection-in-python)。

要获得每个对象的 id，需要打开 Python Shell，调用默认的 ID()函数，并传递变量名。这里有一个例子:

```py
#Initializing the variable
a = "this is not a random string"

#We call the id() function with the variable name as argument
print("The Object id of 'a' is: " + str(id(a)))

```

**输出:**

以下输出表示

```py
The Object id of 'a' is: 1695893310240

```

## 什么是不变性？

为了正确理解不变性的概念，我们需要知道可变对象和不可变对象之间的区别。

### 什么是可变对象？

如果一个对象的状态在它被创建后可以被改变，那么它被称为可变对象。

**举例:**

下面我们将下面的随机值列表分配给变量' **randomValues** '。一旦它被创建，我们检查并记下它的对象 ID。然后，我们需要修改列表(这可以通过追加值、删除值或简单地用其他值替换其中一个值来实现)。然后我们再次记下对象 ID。

如果对象 ID /链表的存储位置保持不变，那么我们可以说 [Python 链表](https://www.askpython.com/python/difference-between-python-list-vs-array)的状态已经改变。

```py
# Our list of random values
randomValues = ["Bojack Horseman", 42, "Robert Langdon", 1.61803]
id1 = id(randomValues)

# Modifying/Changing the state of our list
randomValues[1] = "The answer to everything"
randomValues.append("I love Python")
id2 = id(randomValues)

# Compare the object id before and after modifying
if id1 == id2:
    print("The Object ID Remains the same.")
else:
    print("The Object ID changes.")

```

输出:

```py
The Object ID Remains the same.

```

正如我们所看到的，当值改变时，列表的内存位置或 ID 保持不变。这意味着 Python 为该位置分配了更多的内存空间来考虑额外的值。

由此我们可以说列表是一个“可变的”对象或可变的对象。

### 什么是不可变对象？

如果一个对象的状态在创建后不能改变，那么它就被称为不可变对象。

**例 1:**

与我们之前的例子不同，我们在操作中使用了列表，下面，我们用随机值初始化一个元组。然后我们记下它的对象 ID。接下来，我们尝试修改元组，并比较之前和之后的对象 id。

```py
# Our tuple of random values
randomValues = ("Bojack Horseman", 42, "Robert Langdon", 1.61803)
id1 = id(randomValues)

# Modifying/Changing the state of our tuple
randomValues[1] = "The answer to everything"

# Compare the object id before and after modifying
if id1 == id2:
    print("The Object ID Remains the same.")
else:
    print("The Object ID changes.")

```

输出:

```py
TypeError: 'tuple' object does not support item assignment

```

这里我们看到 [tuple](https://www.askpython.com/python/tuple/python-tuple) (一种固有的不可变类型)不支持修改它的值或向它们追加项目。所以，让我们用一个整数继续同样的操作。

**例 2:**

现在我们需要给任何变量分配一个简单的整数值，并记下它的对象 ID。像前面的例子一样，我们给整型变量赋一个新值，并比较对象 ID。

```py
# first we assign an integer value to the variable 
randomNumber = 42
id1 = id(randomNumber)

# Change the value of our integer variable
randomNumber = 134
id2 = id(randomNumber)

if id1 == id2:
    print("The Object ID remains the same.") 
else:
    print("The Object ID changed.")

```

输出:

```py
The Object ID changed.

```

在这里，我们可以清楚地注意到，在新的值赋值之后，变量“randomNumber”的对象 id 也发生了变化。

也就是说，它是一个独立的物体。这不是原始对象状态的改变。

**注意:当你用一个不可变的对象给一个变量赋值时——这会创建一个新的对象，而不会覆盖当前的对象。**

### Python 中哪些对象是不可变的？

现在我们已经理解了单词**在 Python** 中不可变的含义，让我们看看 Python 中哪些类型的对象是不可变的:

*   用线串
*   整数
*   漂浮物
*   元组
*   范围是元组

## 结论

不可变对象的一个主要好处是，它们比可变对象的访问速度快得多。希望这篇文章能帮助你理解 Python 中不可变对象的概念。

### 参考

[https://docs.python.org/3/reference/datamodel.html](https://docs.python.org/3/reference/datamodel.html)