# 如何用 Python 计数器对 Python 3 中的对象计数

> 原文：<https://www.pythoncentral.io/how-to-count-objects-in-python-3-with-a-python-counter/>

Python 被认为是最容易使用的编程语言之一，这是有充分理由的。有一些库和内置功能可以让你做任何你想做的事情。

对多个重复的对象进行计数是一个编程问题，几十年来开发人员不得不寻找复杂的解决方案。但是在 Python 中，作为集合模块一部分的 Counter 子类为这个问题提供了一个简单有效的解决方案。

它是 dict 类的子类，学习使用它可以让你在程序中快速计算对象。在本指南中，我们将带您了解如何使用 counter 子类对 Python 3 中的对象进行计数。

## **在 Python 中使用计数器的基础知识**

有很多原因可以解释为什么你想要计算一个对象在给定数据源中出现的次数。也许您想确定某个特定项目在列表中出现的次数。

如果清单很短，你或许可以数一数手上的物品数量。但是如果列表很大，你能做什么呢？

这个问题的典型解决方案是使用计数器变量。该变量的起始值为 0，并且每当对象出现在数据源中时，该变量就递增 1。

当你想计算一个物体出现的次数时，使用计数器变量是最合适的。你要做的就是做一个单独的计数器。

然而，如果你需要对多个不同的对象进行计数，你将需要写和对象一样多的计数器。

使用 Python 字典是避免编写多个计数器的一个好方法。字典的键将存储您想要计数的对象，值将保存每个对象出现的频率。

要用 Python 字典对序列中的对象进行计数，只需循环遍历序列，检查每个对象是否在字典中，如果是，递增计数器。

让我们看一个简单的例子。下面，我们试图找出字母在单词“簿记员”中出现的次数

```py
word = "bookkeeper"
counter = {}

for letter in word:
	if letter not in counter:
		counter[letter] = 0
	counter[letter] += 1

print(counter)
```

运行代码给我们输出:

```py
{'b': 1, 'o': 2, 'k': 2, 'e': 3, 'p': 1, 'r': 1}
```

在示例中，for 循环遍历变量 word 中的字母。循环中的条件语句检查被检查的字母是否在字典中，在本例中，字典被称为 counter。

如果没有，它会创建一个保存该字母的新键，并将其值设置为零。最后，计数器变量加 1。

最后一条语句打印出计数器变量。因此，在输出中，您可以看到字母起到了键的作用，而值就是计数。

需要注意的是，当你用字典计算几个对象时，它们需要是可散列的，因为它们将作为字典的键。换句话说，对象在其整个生命周期中必须有一个常量哈希值。

还有第二种用字典计数对象的方法——使用 dict.get()并将零设置为默认值:

```py
word = "bookkeeper"
counter = {}

for letter in word:
	counter[letter] = counter.get(letter, 0) + 1

print(counter)
```

代码的输出将是:

```py
{'b': 1, 'o': 2, 'k': 2, 'e': 3, 'p': 1, 'r': 1}
```

使用时。get()与 dict，它要么给出零(缺省值)如果字母丢失或字母的当前计数。在代码中，该值随后递增 1，并作为字典中相应字母的值存储。

Python 还支持使用集合中的 defaultdict 来计算循环中的对象:

```py
from collections import defaultdict

word = "bookkeeper"
counter = defaultdict(int)

for letter in word:
	counter[letter] += 1

print(counter)
```

运行代码，我们得到输出:

```py
defaultdict(<class 'int'>, {'b': 1, 'o': 2, 'k': 2, 'e': 3, 'p': 1, 'r': 1})
```

用这种方式解决问题更具可读性，也更简洁。首先用 defaultdict 初始化计数器，使用 int()作为工厂函数。这样做允许您访问一个在 defaultdict 中不存在的键。

正如你使用 int()所预料的，缺省值将是零，这发生在你调用不带参数的函数时。

字典会自动创建一个键，并用工厂函数返回的值初始化它。

上面的解决方案是有效的，但是就像 Python 中的其他东西一样，还有更好的方法来解决这个问题。collections 模块创建了一个类来帮助同时计数不同的对象。这是计数器类。

## **Python 的计数器类入门**

Counter 类是 dict 的一个子类，用来计算可散列对象。如您所料，这是一个保存对象的字典，您将这些对象计数为键，计数为值。

要使用这个类，你必须向类的构造函数提供一个可迭代的或可散列的对象序列作为参数。

这个类在内部遍历序列，计算一个对象出现的频率。让我们看看构造计数器的不同方法。

### **构造计数器**

要同时计数多个对象，必须使用 iterable 或 sequence 来初始化计数器。

让我们看看如何用这种方法编写一个程序来计算“簿记员”中的字母数:

```py
from collections import Counter

print(Counter("bookkeeper")) # Passing a string argument 

print(Counter(list("bookkeeper"))) # Passing a list as an argument 
```

这段代码的输出将是:

```py
Counter({'e': 3, 'o': 2, 'k': 2, 'b': 1, 'p': 1, 'r': 1})

Counter({'e': 3, 'o': 2, 'k': 2, 'b': 1, 'p': 1, 'r': 1})
```

Counter 类遍历“bookkeeper”，生成一个字典，将字母存储为键，将计数存储为值。在这个例子中，我们首先导入计数器类，然后传递一个字符串作为参数。然后，我们传递一个列表，得到相同的输出。

除了列表，你还可以传递元组或者任何其他包含重复对象的可重复对象。

如前所述，有许多方法可以创建 Counter 类的实例。但是这些方法严格来说并不意味着计数。你可以做的一件事就是使用字典:

```py
from collections import Counter

print(Counter({"e": 3, "o": 2, "k": 2, "b": 1, "p": 1, "r": 1}))
```

运行该命令会得到输出:

```py
Counter({'e': 3, 'o': 2, 'k': 2, 'b': 1, 'p': 1, 'r': 1})
```

以这种方式使用字典给出键计数对的计数器初始值。还有另一种方法可以通过调用类的构造函数来做同样的事情，就像这样:

```py
from collections import Counter

print(Counter(e=3, o=2, k=2, b=1, p=1, r=1))
```

你也可以通过调用 set()函数来使用 Counter 类，就像这样:

```py
from collections import Counter

print(Counter(set("bookkeeper")))
```

其输出为:

```py
Counter({'e': 1, 'o': 1, 'p': 1, 'r': 1, 'b': 1, 'k': 1})
```

你可能知道，Python 中的集合存储唯一的对象。因此，以这种方式调用 set()会输出重复的字母。但是这样一来，原始 iterable 中的每个字母都有了一个实例。

由于 Counter 是 dict 的子类，所以它继承了常规字典的接口。但是没有实现。fromkeys()来防止歧义。

记住你可以在键中存储任何类型的可散列对象，而值可以存储任何类型的对象。但是作为一个计数器，值必须是整数。

让我们来看一个保存零和负计数的计数器实例的例子:

```py
from collections import Counter

collection = Counter(
	stamps=10,
	coins=-15,
	buttons=0,
	seeds=15
)
```

在这个例子中，你可能会问为什么有-15 个硬币。它可能用来表示你已经把它们借给朋友了。底线是 Counter 类允许以这种方式存储负数，并且您可能能够找到它的一些用例。

### **更新对象计数**

现在，您已经了解了如何获得一个计数器实例。用新的计数更新它并引入新的对象就像使用。更新()。

它是 Counter 类的一个实现，能够将现有的计数相加。它还使得创建新的键计数对成为可能。

。update()处理计数的映射和可重复项。当使用 iterable 作为参数时，该方法对项目进行计数，并根据需要更新计数器:

```py
from collections import Counter

letters = Counter({"e": 3, "o": 2, "k": 2, "b": 1, "p": 1, "r": 1})

letters.update("toad")
print(letters)
```

运行代码给出输出:

```py
Counter({'e': 3, 'o': 3, 'k': 2, 'b': 1, 'p': 1, 'r': 1, 't': 1, 'a': 1, 'd': 1})
```

我们将前面讨论的“簿记员”示例中的字母和相应的计数放在本示例的开头。然后，我们用了。更新以将单词“toad”中的字母添加到组合中。这引入了一些新的键计数对，如输出所示。

请记住，iterable 需要是一个元素序列，而不是键计数对，这样才能工作。但有趣的是:

使用整数以外的值作为计数会破坏计数器。我们来看看:

```py
from collections import Counter

letters = Counter({"e": "3", "o": "2", "k": "2", "b": "1", "p": "1", "r": "1"})

letters.update("toad")

print(letters)
```

运行代码给出输出:

```py
Traceback (most recent call last):
  File "<string>", line 5, in <module>
File "/usr/lib/python3.8/collections/__init__.py", line 637, in update
    _count_elements(self, iterable)
TypeError: can only concatenate str (not "int") to str

```

由于示例中定义的字母计数是字符串。update()不起作用，引发了 TypeError。

。还可以通过提供第二个计数器或计数映射作为参数来以另一种方式使用 update。下面是方法:

```py
from collections import Counter
sales = Counter(cheese=19, cake=20, foil=5)

# Using a counter 
monday_sales = Counter(cheese=5, cake=4, foil=3)
sales.update(monday_sales)
print(sales)

# Using a dictionary of counts
tuesday_sales = {"cheese ": 4, "cake": 3, "foil": 1}
sales.update(tuesday_sales)
print(sales)
```

在程序开始时，用另一个名为“monday_sales”的计数器更新现有的计数器在程序的后半部分，使用一个包含项目和计数的字典来更新计数器。如你所见。update()适用于两个计数器

### **访问计数器的内容**

由于 Counter 类具有与 dict 相同的接口，它可以执行与标准字典相同的操作。通过扩展，您可以像在字典中一样使用键访问来访问计数器中的值。

用典型的方法迭代键、项和值也很容易。我们来看看:

```py
from collections import Counter

letters = Counter("bookkeeper")
print(letters["e"])
# Output: 3
print(letters["k"])
# Output:2

for letter in letters:
	print(letter, letters[letter])
# Output:
b 1
o 2
k 2
e 3
p 1
r 1

for letter in letters.keys():
	print(letter, letters[letter])
# Output:
b 1
o 2
k 2
e 3
p 1
r 1

for letter, count in letters.items():
	print(letter, count)
# Output:
b 1
o 2
k 2
e 3
p 1
r 1

for count in letters.values():
	print(count)
# Output:
1
2
2
3
1
1
```

关于 Counter 值得注意的一件有趣的事情是，访问一个丢失的键会导致一个零错误而不是一个键错误。看一看:

```py
from collections import Counter

letters = Counter("bookkeeper")
print(letters["z"])
# Output: 0
```

### **寻找最常出现的物体**

使用 Counter 类的另一个方便之处是您可以使用。most_common()方法根据对象出现的频率列出对象。如果两个对象具有相同的计数，则它们以第一次出现的顺序显示。

如果一个数字“n”作为一个参数传递给该方法，它将输出“n”个最常见的对象。这里有一个探索该方法如何工作的例子:

```py
from collections import Counter
sales = Counter(cheese=19, cake=20, foil=5)

print(sales.most_common(1))
# Outputs the most common object.

print(sales.most_common(2))
# Outputs the two most common objects.

print(sales.most_common())
# Outputs all objects in order of frequency.

print(sales.most_common(None))
# Outputs all objects in order of frequency.

print(sales.most_common(20))
# Outputs all objects in order of frequency since the argument passed is greater than the total number of distinct objects. 

```

当没有参数或“无”作为参数传递时。most_common()返回所有对象。当参数超过计数器的当前长度时，该方法还返回所有对象。

有趣的是，你也可以有。most_common()按照频率从低到高的顺序返回对象。用切片很容易就能完成。你可以这样做:

```py
from collections import Counter
sales = Counter(cheese=19, cake=20, foil=5)

print(sales.most_common()[::-1])
# Returns objects in reverse order of commonality.

Print(sales.most_common()[:-3:-1])
# Returns the two least-common objects.
```

如你所见，第一次切片从最不常见到最常见返回变量中的对象。第二个切片从方法输出的结果中返回最后两个对象。

当然，您可以通过改变切片操作符中的第二个值来改变输出中最不常用对象的数量。为了得到三个最不常用的对象，运算符中的-3 将变成-4，依此类推。

要记住的重要一点是，计数器中的值必须是可排序的，这样方法才能正确工作。因为任何类型的数据都可以存储在计数器中，所以排序会变得复杂。

## **结论**

在本指南中，您已经了解了如何使用 collections 模块中的 Counter 类对对象进行计数。它消除了使用循环或嵌套数据结构的需要，简化了考验。将您在本指南中所学的知识整合到您的代码中，可以使代码更干净 *和* 更快。