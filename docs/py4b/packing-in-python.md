# 用 Python 包装

> 原文：<https://www.pythonforbeginners.com/basics/packing-in-python>

打包是 python 中的一种技术，我们用它将几个值放入一个迭代器中。如果我们从字面意义上谈论打包，就像我们在现实世界中把某些项目打包到一个盒子里一样，在 python 中我们把某些变量打包到一个 iterable 中。在本文中，我们将研究如何执行打包并在我们的程序中使用它。

## 如何在 Python 中进行打包？

我们可以通过使用简单的语法声明列表或元组等可重复项来执行打包，也可以使用星号操作符*来打包。操作符*被用作将可变量解包成变量的操作符，但是它允许我们将多个变量打包成一个变量。

例如，假设我们有三个变量 num1，num2，num3。我们可以将它们打包成一个元组，如下所示。

```py
num1=1
num2=2
num3=3
myTuple=(num1,num2,num3)
```

或者我们可以将它们打包成一个列表，如下所示。

```py
 num1=1
num2=2
num3=3
myList=[num1,num2,num3]
```

或者，我们可以使用*运算符将这些数字打包在一起，如下所示。

```py
num1=1
num2=2
num3=3
*num,=num1,num2,num3
```

我们在*num 后面使用了逗号“，”因为赋值操作的左边必须是元组或列表，否则将会遇到错误。

在左侧，当我们使用*运算符时，我们还可以有其他变量，这些变量可以被赋值。例如，我们可以将两个数字打包到一个变量中，然后将第三个数字赋给另一个变量，如下所示。

```py
num1=1
num2=2
num3=3
*num,myNum=num1,num2,num3
```

在上面的例子中，我们必须记住 myNum 是一个强制变量，必须给它赋值，而*num 可以不赋值。这里，num3 将被分配给 myNum，num1 和 num2 将被打包在列表 num 中。

## 在传递参数中使用打包

当我们不知道有多少参数将被传递给一个函数时，我们使用打包来处理这种情况。我们可以声明最初的几个变量，然后使用星号操作符将剩余的参数打包到一个变量中，如下所示。

```py
def sumOfNumbers(num1,num2,*nums):
    temp=num1+num2
    for i in nums:
        temp=temp+i
    return temp
```

在上面的示例中，当只传递两个数字来计算它们的和时，nums 保持为空，并且返回两个数字的和。当我们向函数传递两个以上的数字时，第一个和第二个参数被分配给 num1 和 num2，参数中的其余数字将被打包到 nums 中，其行为类似于一个列表。之后，将计算这些数字的总和。

## 通过打包收集多个值

在编写程序时，可能会有这样的情况，我们希望将一个序列或字符串或一个 iterable 分成许多部分。例如，我们可能需要执行一个 [python 字符串分割](https://www.pythonforbeginners.com/dictionary/python-split)操作，从包含名字的字符串中收集一个人的名字。现在，如果我们不知道一个人的名字中可能有多少个单词，我们可以将名字收集到一个变量中，并将其余的单词收集到一个变量中，如下所示。

```py
name="Joseph Robinette Biden Jr"
first_name, *rest_name=name.split()
print("Full Name:")
print(name)
print("First Name:")
print(first_name)
print("Rest Parts of Name:")
print(rest_name)
```

输出:

```py
Full Name:
Joseph Robinette Biden Jr
First Name:
Joseph
Rest Parts of Name:
['Robinette', 'Biden', 'Jr']
```

## 在 python 中使用打包合并两个可重复项

我们可以在 python 中使用星号运算符(*)来合并列表、元组、集合和字典等不同的可迭代对象。

为了将两个元组合并成一个元组，我们可以使用打包。我们可以使用星号运算符合并两个元组，如下所示。

```py
 print("First Tuple is:")
tuple1=(1,2,3)
print(tuple1)
print("Second Tuple is:")
tuple2=(4,5,6)
print(tuple2)
print("Merged tuple is:")
myTuple=(*tuple1,*tuple2)
print(myTuple) 
```

输出:

```py
First Tuple is:
(1, 2, 3)
Second Tuple is:
(4, 5, 6)
Merged tuple is:
(1, 2, 3, 4, 5, 6)
```

就像元组一样，我们可以如下合并两个列表。

```py
print("First List is:")
list1=[1,2,3]
print(list1)
print("Second List is:")
list2=[4,5,6]
print(list2)
print("Merged List is:")
myList=[*list1,*list2]
print(myList)
```

输出:

```py
First List is:
[1, 2, 3]
Second List is:
[4, 5, 6]
Merged List is:
[1, 2, 3, 4, 5, 6]
```

我们还可以使用打包将两个或多个字典合并成一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python)。对于字典，使用字典解包操作符**来解包两个初始字典，然后将它们打包到第三个字典中。这可以从下面的例子中理解。

```py
print("First Dictionary is:")
dict1={1:1,2:4,3:9}
print(dict1)
print("Second Dictionary is:")
dict2={4:16,5:25,6:36}
print(dict2)
print("Merged Dictionary is:")
myDict={**dict1,**dict2}
print(myDict)
```

输出:

```py
First Dictionary is:
{1: 1, 2: 4, 3: 9}
Second Dictionary is:
{4: 16, 5: 25, 6: 36}
Merged Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
```

## 结论

在本文中，我们学习了 python 中的打包，并实现了不同的程序来理解使用星号(*)操作符的打包操作的用例。请继续关注更多内容丰富的文章。