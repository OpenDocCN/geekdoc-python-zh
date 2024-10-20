# 如何在 Python 中执行加法？

> 原文：<https://www.askpython.com/python/examples/addition-in-python>

在这篇文章中，我们将讨论一个最基本的话题。如果你是初学者，这对你有好处。但是如果你已经用 Python 写了代码，跳过这一步。

***也读作:[Python 中的 sum()方法](https://www.askpython.com/python/built-in-methods/python-sum-method)***

## **Python 中用户输入的两个数相加**

我们将使用 [input()方法](https://www.askpython.com/course/python-course-user-input)接受用户输入，然后使用这些数字在 Python 中执行加法。python 中两个数相加的基本代码是:

```py
def adding(x , y):
    return x + y

a = int(input("Enter first number :" ))
b = int(input("Enter second number :"))

sum = adding(a , b)

print("addition of {} and {} is {}".format(a,b,sum))

```

输出是:

```py
Enter first number : 6
Enter second number : 5
Addition of 6 and 5 is 11

```

在上面的代码中，我们使用 int()方法将字符串输入转换为整数，从而将输入类型转换为 int。

* * *

## **对列表中的元素进行加法运算**

我们可以使用[循环](https://www.askpython.com/python/python-loops-in-python)添加[列表](https://www.askpython.com/python/examples/linked-lists-in-python)中的所有项目，如下所示:

```py
def add_list(l1) :
  res = 0
  for val in l1 :
    res = res + val
  return res 

list = [1,3,5,7,9]
ans = add_list(list) 
print("The sum of all elements within the given list is {}".format(ans)) 

```

这段代码的输出是:

```py
The sum of all elements in the list is 25

```

在上面的代码中，我们定义了一个函数，在这个函数中，我们使用 for 循环来遍历列表中的所有元素，并在每次迭代后更新 **res** 的值。将我们的列表作为参数传递给 add_list 函数会给出我们的最终输出。

这是 python 中加法的基础。