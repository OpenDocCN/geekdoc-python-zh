# 用 Python 复制

> 原文：<https://www.pythonforbeginners.com/data-types/copy-in-python>

在 python 程序中，很多时候我们需要现有数据的相同副本。对于像 int、float、boolean 值或 string 这样的简单数据类型，赋值操作为我们完成了任务，因为它们是不可变的数据类型。复制之后，当我们对任何具有不可变数据类型的变量进行任何更改时，会创建一个新的数据实例，并且它们不会影响原始数据。在可变数据类型的情况下，如 list 或 dictionary，如果我们使用赋值操作符将数据复制到另一个变量，两个变量都引用同一个数据对象，如果我们对任何一个变量进行更改，对象也会发生变化，反映出对两个变量的影响。在本文中，我们将通过例子来理解 python 中复制的概念。

## python 中如何获取对象的对象 ID？

为了说明 python 中复制的概念，我们需要知道对象的对象 ID。对于这个任务，我们将使用`id()`函数。该函数将变量名作为输入，并返回对象的唯一 id。从下面的例子可以看出这一点。

```py
var1=1
print("ID of object at which var1 refers:",end=" ")
print(id(var1))
```

输出:

```py
ID of object at which var1 refers: 9784896
```

## 在 python 中使用=运算符复制对象

当我们试图通过使用=操作符将一个变量赋给另一个变量来复制一个对象时，它不会创建一个新的对象，但是两个变量都被赋给了同一个对象。

例如，如果我们有一个名为 var1 的变量引用一个对象，我们通过语句`var2=var1`将 var1 赋给 var2，这两个变量指向同一个对象，因此将具有相同的 ID。这可以看如下。

```py
 var1=11
var2=var1
print("ID of object at which var1 refers:",end=" ")
print(id(var1))
print("ID of object at which var2 refers:", end=" ")
print(id(var2))
```

输出:

```py
ID of object at which var1 refers: 9785216
ID of object at which var2 refers: 9785216
```

对于不可变对象，当我们改变赋给 var1 或 var2 中任何一个的值时，就会为新赋给该变量的值创建一个新对象。因此，如果我们更改分配给 var2 的值，将创建一个新对象，在重新分配后，var1 和 var2 将引用具有不同对象 ID 的对象。这可以这样理解。

```py
var1=11
var2=var1
print("ID of object at which var1 refers:",end=" ")
print(id(var1))
print("ID of object at which var2 refers:", end=" ")
print(id(var2))
print("After Assigning value to var2")
var2=10
print("ID of object at which var1 refers:",end=" ")
print(id(var1))
print("ID of object at which var2 refers:", end=" ")
print(id(var2))
```

输出:

```py
ID of object at which var1 refers: 9785216
ID of object at which var2 refers: 9785216
After Assigning value to var2
ID of object at which var1 refers: 9785216
ID of object at which var2 refers: 9785184
```

与上面不同的是，如果我们复制像列表这样的可变对象，在使用任何变量重新分配和修改导致同一个对象发生变化之后，它们指向同一个对象。这可以看如下。

```py
 list1=[1,2,3]
list2=list1
print("ID of object at which list1 refers:",end=" ")
print(id(list1))
print("ID of object at which list2 refers:", end=" ")
print(id(list2))
print("After appending another value to list2")
list2.append(4)
print("ID of object at which var1 refers:",end=" ")
print(id(list1))
print("ID of object at which var2 refers:", end=" ")
print(id(list2))
```

输出:

```py
ID of object at which list1 refers: 140076637998720
ID of object at which list2 refers: 140076637998720
After appending another value to list2
ID of object at which var1 refers: 140076637998720
ID of object at which var2 refers: 140076637998720
```

在上面的输出中，我们可以看到，当我们使用赋值操作符复制 list 时，变量 list1 和 list2 引用同一个对象并具有相同的 ID。当我们对任何列表进行更改时，这不会改变。

## 如何在 Python 中复制可变对象？

对于复制像列表或字典这样的可变对象，我们使用`copy()`方法。

当在任何对象上调用时，`copy()`方法创建一个新的对象，其数据与原始对象相同，并返回对它的引用。这意味着当我们使用`copy()`方法而不是=操作符复制对象时，原始对象和复制对象的地址是不同的。

例如，如果我们使用`copy()`方法将 list1 复制到 list2，那么 list1 和 list2 将引用不同的对象。这可以看如下。

```py
list1=[1,2,3]
list2=list1.copy()
print("ID of object at which list1 refers:",end=" ")
print(id(list1))
print("ID of object at which list2 refers:", end=" ")
print(id(list2))

print("After appending another value to list2")
list2.append(4)
print("ID of object at which var1 refers:",end=" ")
print(id(list1))
print("ID of object at which var2 refers:", end=" ")
print(id(list2))
```

输出:

```py
ID of object at which list1 refers: 140076492253376
ID of object at which list2 refers: 140076483798272
After appending another value to list2
ID of object at which var1 refers: 140076492253376
ID of object at which var2 refers: 140076483798272
```

在输出中，可以看到 list2 与 list1 具有不同的对象 ID。如果我们修改 list1 中的任何值，都不会导致 list2 发生任何变化，而当我们使用= operator 将一个对象复制到另一个对象时，情况就不同了。

## 结论

在本文中，我们研究了如何在 python 中复制可变和不可变对象。对于不可变对象，我们可以使用= operator，因为在每次重新分配时，都会为不可变对象创建一个新对象。在像 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 这样的可变对象的情况下，我们应该使用`copy()`方法来复制像列表或字典这样的对象，以避免程序中不必要的错误。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，以使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。