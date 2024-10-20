# 使用 Python hash()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-hash-function>

大家好！在今天的文章中，我们将关注 Python 的内置`hash()`函数。Python `hash()`函数计算 Python 对象的哈希值。但是语言在很大程度上使用了这一点。

让我们用一些例子来更多地了解这个函数！

* * *

## Python hash()的基本语法

这个函数接受一个*不可变的* [Python 对象](https://www.askpython.com/python/oops/python-classes-objects)，并返回这个对象的哈希值。

```py
value = hash(object)

```

请记住，哈希值依赖于一个哈希函数(来自`__hash__()`),它由`hash()`内部调用。这个散列函数需要足够好，以便给出几乎随机的分布。

那么，为什么我们想要一个散列函数在如此大的程度上随机化它的值呢？这是因为我们希望散列函数将几乎每个键映射到一个惟一的值。

如果你的值是随机分布的，那么两个不同的键被映射到同一个值的机会就非常小，这正是我们想要的！

现在，让我们看看正在使用的`hash()`函数，用于简单的对象，如整数、浮点数和字符串。

* * *

## 使用 hash()函数–一些例子

```py
int_hash = hash(1020)

float_hash = hash(100.523)

string_hash = hash("Hello from AskPython")

print(f"For {1020}, Hash : {int_hash}")
print(f"For {100.523}, Hash: {float_hash}")
print(f"For {'Hello from AskPython'}, Hash: {string_hash}")

```

**输出**

```py
For 1020, Hash : 1020
For 100.523, Hash: 1205955893818753124
For Hello from AskPython, Hash: 5997973717644023107

```

正如您所观察到的，整数的哈希值与它们的原始值相同。但是对于 float 和 string 对象，这些值显然是不同的。

现在，如果同一个对象(除了整数/浮点数)总是有相同的哈希值，那就不太安全了。因此，如果您再次运行上面的代码片段，您会注意到不同的值！

例如，这是我第二次运行同一个代码片段时的输出。

```py
For 1020, Hash : 1020
For 100.523, Hash: 1205955893818753124
For Hello from AskPython, Hash: -7934882731642689997

```

如您所见，字符串的值发生了变化！这是一件好事，因为它防止了同一对象被其他人访问！哈希值仅在程序运行期间保持不变。

之后，每次再次运行程序时，它都会不断变化。

## 为什么我们不能在可变对象上使用 hash()。

现在，记住我们之前提到的`hash()`仅用于*不可变的*对象。这是什么意思？

这意味着我们不能在可变对象上使用`hash()`,比如列表、集合、字典等等。

```py
print(hash([1, 2, 3]))

```

**输出**

```py
TypeError: unhashable type: 'list'

```

为什么会这样？每次可变对象的值改变时，程序都要改变哈希值，这是很麻烦的。

这将使得再次更新哈希值非常耗时。如果你这样做，那么 Python 需要花很多时间来引用同一个对象，因为引用会不断变化！

因此，我们不能使用`hash()`散列可变对象，因为它们只有一个值，对我们来说是隐藏的，所以程序可以在内部保存对它的引用。

然而，我们*可以*在不可变元组上使用`hash()`。这是一个[元组](https://www.askpython.com/python/tuple/python-tuple)，仅由不可变的对象组成，如 int、floats 等。

```py
>>> print(hash((1, 2, 3)))
2528502973977326415

>>> print(hash((1, 2, 3, "Hello")))
-4023403385585390982

>>> print(hash((1, 2, [1, 2])))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'

```

* * *

## 在自定义对象上使用 hash()

因为默认的 Python `hash()`实现通过覆盖`__hash__()`方法来工作，所以我们可以通过覆盖`__hash__()`来为我们的定制对象创建我们自己的`hash()`方法，只要相关的属性是不可变的。

现在让我们创建一个类`Student`。

我们将覆盖`__hash__()`方法来调用相关属性的`hash()`。我们还将实现`__eq__()`方法，用于检查两个自定义对象之间的相等性。

```py
class Student:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __eq__(self, other):
        # Equality Comparison between two objects
        return self.name == other.name and self.id == other.id

    def __hash__(self):
        # hash(custom_object)
        return hash((self.name, self.id))

student = Student('Amit', 12)
print("The hash is: %d" % hash(student))

# We'll check if two objects with the same attribute values have the same hash
student_copy = Student('Amit', 12)
print("The hash is: %d" % hash(student_copy))

```

**输出**

```py
The hash is: 154630157590
The hash is: 154630157597

```

我们确实可以观察到自定义对象的散列。不仅如此；两个不同的对象即使有相同的属性值，也有不同的哈希值！

这确实是我们希望从散列函数中得到的，而`hash()`已经成功地给了我们！

* * *

## 结论

我们学习了如何使用 Python `hash()`函数。这对于程序使用特殊的整数值来维护对每个对象的引用非常有用。

我们还看到了如何让`hash()`在定制对象上工作，只要它的属性是不可变的。

## 参考

*   关于 Python hash()函数的 JournalDev 文章

* * *