# 如何在 Python 中对列表、元组或对象(已排序)进行排序

> 原文：<https://www.pythoncentral.io/how-to-sort-a-list-tuple-or-object-with-sorted-in-python/>

在 Python 中对列表或元组排序很容易！因为一个元组基本上就像一个不可修改的数组，所以我们将它视为一个列表。

## **对 Python 列表排序的简单方法**

好的，如果你只想对一系列数字进行排序，Python 有一个内置的函数可以帮你完成所有的困难工作。

假设我们有一个数字列表:

```py

>>> a = [3, 6, 8, 2, 78, 1, 23, 45, 9]

```

我们想把它们按升序排列。我们所做的就是在列表上调用`sort`进行就地排序，或者调用内置函数`sorted`不修改原始列表并返回一个新的排序列表。除了上面的原因之外，这两个函数接受相同的参数，可以被视为“相同的”。

我们开始吧:

```py

>>> sorted(a)

[1, 2, 3, 6, 8, 9, 23, 45, 78]

>>>

>>> a.sort()

>>> a

[1, 2, 3, 6, 8, 9, 23, 45, 78]

```

降序呢？

给你:

```py

>>> sorted(a, reverse=True)

[78, 45, 23, 9, 8, 6, 3, 2, 1]

>>> a.sort(reverse=True)

>>> l

[78, 45, 23, 9, 8, 6, 3, 2, 1]

```

Python 在幕后做什么？它在列表上调用一个版本的 mergesort。它在比较值时对每个对象调用函数`__cmp__`，并根据从`__cmp__`返回的值决定将哪个放在另一个的前面。返回值为`0`表示等于比较值，`1`表示大于比较值，`-1`表示小于比较值。我们稍后将使用这些信息来使我们自己的对象可排序。

你说元组呢？我正要说到这一点。

## **用简单的方法排序 Python 元组**

因为元组是不能修改的数组，所以它们没有可以直接调用的就地`sort`函数。他们必须总是使用`sorted`函数来返回一个排序后的*列表*。记住这一点，下面是如何做到这一点:

```py

>>> tup = (3, 6, 8, 2, 78, 1, 23, 45, 9)

>>> sorted(tup)

[1, 2, 3, 6, 8, 9, 23, 45, 78]

```

注意`sorted`是如何返回一个数组的。

好了，现在让我们来看看如何解决一些更复杂的问题。

## **对列表或元组列表进行排序**

这有点复杂，但仍然很简单，所以不要担心！`sorted`函数和`sort`函数都接受一个名为`key`的关键字参数。

`key`所做的是它提供了一种方法来指定一个函数，该函数返回你想要的排序依据。该函数获得一个传递给它的“不可见”参数，该参数表示列表中的一个项目，并返回一个值，您希望该值作为该项目的“键”进行排序。

让我来举例说明一下`key`关键字的论点，仅供你高超的眼光参考！

因此，我们来看一个新列表，通过对每个子列表中的第一项进行排序来测试它:

```py

>>> def getKey(item):

... return item[0]

>>> l = [[2, 3], [6, 7], [3, 34], [24, 64], [1, 43]]

>>> sorted(l, key=getKey)

[[1, 43], [2, 3], [3, 34], [6, 7], [24, 64]]

```

在这里我们可以看到，列表现在是按照子列表中的第一个条目以升序排序的。注意，您也可以使用`sort`函数，但是我个人更喜欢`sorted`函数，所以我将在以后的例子中使用它。

发生了什么事？还记得我说过的那个“看不见的”论点吗？这就是每次`sorted`需要一个值时传递给`getKey`函数的内容。刁钻，刁钻的 python).

根据每个子列表中的第二个项目进行排序就像将`getKey`函数改为这样一样简单:

```py

def getKey(item):

return item[1]

```

好吧。一切都好。那么，元组列表呢？很高兴你问了！

它实际上与我们上面的例子完全相同，但是列表是这样定义的:

```py

>>> a = [(2, 3), (6, 7), (3, 34), (24, 64), (1, 43)]

>>> sorted(l, key=getKey)

[(1, 43), (2, 3), (3, 34), (6, 7), (24, 64)]

```

唯一改变的是，我们现在得到的是一个由*个元组构成的列表*，而不是返回给我们的一个由*个列表构成的列表*。

完全相同的解决方案可以应用于一个元组，所以我不打算去那里，因为这只是多余的。这也将浪费更多制造这种漂亮的数字纸的数字树。

## **对定制 Python 对象的列表(或元组)进行排序**

这是我创建的一个自定义对象:

```py

class Custom(object):

def __init__(self, name, number):

self.name = name

self.number = number

```

为了便于分类，我们将它们列出来:

```py

customlist = [

Custom('object', 99),

Custom('michael', 1),

Custom('theodore the great', 59),

Custom('life', 42)

]

```

好了，我们已经在一个列表中获得了所有新的定制对象，我们希望能够对它们进行排序。我们如何做到这一点？

好吧，我们可以定义一个函数，就像我们上面做的那样，接收条目并返回一个列表。让我们开始吧。

```py

def getKey(custom):

return custom.number

```

这个有点不同，因为我们的对象不再是列表了。这允许我们在自定义对象中按照`number`属性进行排序。

因此，如果我们对我们的定制对象列表运行`sorted`函数，我们会得到这样的结果:

```py

>>> sorted(customlist, key=getKey)

[<__main__.Custom object at 0x7f64660cdfd0>,

<__main__.Custom object at 0x7f64660d5050>,

<__main__.Custom object at 0x7f64660d5090>,

<__main__.Custom object at 0x7f64660d50d0>]

```

一大堆我们无法理解的丑陋。完美。但是不要担心，亲爱的观众，有一个简单的修复，我们可以应用到我们的自定义对象，使它变得更好！

让我们像这样重新定义我们的对象类:

```py

class Custom(object):

def __init__(self, name, number):

self.name = name

self.number = number
def _ _ repr _ _(self):
return ' { }:{ } { } '。格式(自我。__class__。__ 姓名 _ _，
自我名，
自我号)

```

好吧，我们刚才到底做了什么？首先，`__repr__`函数告诉 Python 我们希望对象如何表示为。用更复杂的术语来说，它告诉解释器当对象被打印到屏幕上时如何显示它。

因此，我们告诉 Python 用类名、名称和编号来表示对象。

现在让我们再次尝试排序:

```py

>>> sorted(customlist, key=getKey)

[Custom: michael 1, Custom: life 42,

Custom: theodore the great 59, Custom: object 99]

```

那就好多了！现在，我们实际上可以说，它排序正确！

但是有一点小问题。这只是吹毛求疵，但我不想每次调用 sorted 时都必须键入那个`key`关键字。

那么，我该怎么做呢？嗯，还记得我跟你说过的那个`__cmp__`函数吗？让我们付诸行动吧！

让我们像这样再一次重新定义我们的对象:

```py

class Custom(object):

def __init__(self, name, number):

self.name = name

self.number = number
def _ _ repr _ _(self):
return ' { }:{ } { } '。格式(自我。__class__。__ 姓名 _ _，
自我名，
自我号)
def __cmp__(self，other): 
 if hasattr(other，' number'): 
返回 self . number . _ _ CMP _ _(other . number)

```

看起来不错。这样做的目的是告诉 Python 将当前对象的值与列表中的另一个对象进行比较，看看比较的结果如何。就像我上面说的，`sorted`函数将对它正在排序的对象调用`__cmp__`函数，以确定它们相对于其他对象的位置。

现在我们可以直接调用`sorted`而不用担心包含`key`关键字，就像这样:

```py

>>> sorted(customlist)

[Custom: michael 1, Custom: life 42, Custom: theodore the great 59, Custom: object 99]

```

瞧！它工作得很好。请注意，以上所有内容也适用于自定义对象的元组。但是你知道，我喜欢保存我的数码树。

## **对定制 Python 对象的异构列表进行排序**

好吧。因为 Python 是一种动态语言，所以它不太关心我们将什么对象放入列表中。它们可以都是同一类型，也可以都是不同类型。

所以让我们定义另一个不同的对象来使用我们的`Custom`对象。

```py

class AnotherObject(object):

def __init__(self, tag, age, rate):

self.tag = tag

self.age = age

self.rate = rate
def _ _ repr _ _(self):
return ' { }:{ } { } { } '。格式(自我。__class__。__name__，
 self.tag，
 self.age，self.rate)
def __cmp__(self，other): 
 if hasattr(other，' age '):
return self . age . _ _ CMP _ _(other . age)

```

这是一个类似的物体，但仍然不同于我们的`Custom`物体。

让我们列出这些物品和我们的`Custom`物品:

```py

customlist = [

Custom('object', 99),

Custom('michael', 1),

Custom('theodore the great', 59),

Custom('life', 42),

AnotherObject('bananas', 37, 2.2),

AnotherObject('pants', 73, 5.6),

AnotherObject('lemur', 44, 9.2)

]

```

现在，如果我们尝试在这个列表上运行`sorted`:

```py

>>> sorted(customlist)

Traceback (most recent call last):

File "<stdin>", line 1, in <module>

TypeError: an integer is required

```

我们得到一个可爱的错误。为什么？因为`Custom`没有名为`age`的属性，`AnotherObject`也没有名为`number`的属性。

我们该怎么办？慌！

开玩笑的。我们知道该怎么做。让我们再次重新定义那些对象！

```py

class Custom(object):

def __init__(self,name,number):

self.name = name

self.number = number
def _ _ repr _ _(self):
return ' { }:{ } { } '。格式(自我。__class__。__ 姓名 _ _，
自我名，
自我号)
def __cmp__(self，other): 
 if hasattr(other，' getKey'): 
返回 self.getKey()。__cmp__(other.getKey())
def getKey(self): 
返回自身编号
class another object(object):
def _ _ init _ _(self，tag，age，rate):
self . tag = tag
self . age = age
self . rate = rate
def _ _ repr _ _(self):
return ' { }:{ } { } { } '。格式(自我。__class__。__name__，
 self.tag，
 self.age，self.rate)
def __cmp__(self，other): 
 if hasattr(other，' getKey'): 
返回 self.getKey()。__cmp__(other.getKey())
def getKey(self):
return self . age

```

厉害！我们刚刚做了什么？我们定义了一个两个对象共有的函数`getKey`,这样我们就可以很容易地对它们进行比较。

所以现在如果我们再次运行`sorted`函数，我们会得到:

```py

>>> sorted(customlist)

[Custom: michael 1, AnotherObject: bananas 37 2.2,

Custom: life 42, AnotherObject: lemur 44 9.2,

Custom: theodore the great 59, AnotherObject: pants 73 5.6,

Custom: object 99]

```

不错！现在，我们的对象可以进行比较和排序，以满足他们的需求。

你说你还是喜欢使用`key`关键字？

你也可以这样做。如果您在每个对象中省略掉`__cmp__`函数，并像这样定义一个外部函数:

```py

def getKey(customobj):

return customobj.getKey()

```

然后像这样调用排序:

```py

>>> sorted(customlist, key=getKey)

[Custom: michael 1, AnotherObject: bananas 37 2.2,

Custom: life 42, AnotherObject: lemur 44 9.2,

Custom: theodore the great 59, AnotherObject: pants 73 5.6,

Custom: object 99]

```

现在你知道了！相当直截了当，但不像有些人猜测的那样直截了当。尽管如此，Python 通过其内置的`sorted`函数使其变得非常简单。

要了解更多的排序方法，请阅读[如何通过键或值对 Python 字典进行排序](https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/ "How to Sort Python Dictionaries by Key or Value")。你也可以查看 Python 中的 [Lambda 函数语法(内联函数),了解如何使用 Lambda 函数进行排序。](https://www.pythoncentral.io/lambda-function-syntax-inline-functions-in-python/ "Lambda Function Syntax (Inline Functions) in Python")

这就是你。离成为世界巨蟒大师又近了一步。

再见，暂时的。