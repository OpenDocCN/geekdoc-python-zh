# python 中如何进行乘法运算？

> 原文：<https://www.askpython.com/python/examples/multiplication-in-python>

在本文中，我们将看到如何用 python 编写代码来获得作为输入给出的数字或列表元素的乘积。

因此，在 python 中有不同的方法来执行乘法。最简单的一种是使用星号运算符( ***** )。也就是说，您传递两个数字，只需打印 num1 * num2 就会得到想要的输出。

## 用户输入两个数的 python 乘法运算

### 1.不使用函数

让我们编写一个简单的脚本来打印两个数字的乘积，而不使用函数。我们将简单地[打印](https://www.askpython.com/python/built-in-methods/python-print-function)结果。

```py
n1 = int(input("Enter a number:")
n2 = int(input("Enter another number:")
res = n1 * n2
print("The product is ", res)

```

输出将是:

```py
Enter a number: 3
Enter another number: 5
The product is 15 

```

### 2.有功能的

如果您必须在程序中多次使用乘法，那么您必须创建一个函数，该函数将返回调用时传递给它的数字的乘积。这将降低程序的复杂性并引入可重用性，也就是说，你可以用一组不同的参数反复调用同一个函数。

下面写了一个这样的例子:

```py
def mult(a , b):
  return a * b

n1 = int(input("Enter a number :"))
n2 = int(input("Enter another number :"))
multiplication1 = mult(n1 , n2)

num1 = 6.0
num2 = 5.0
multiplication2 = mult(num1 , num2)

print("The product of {} and {} is {}".format(n1 , n2 , multiplication1))
print("The product of {} and {} is {}".format(num1 , num2, multiplication2)

```

以上代码的输出:

```py
Enter a number : 4
Enter another number : 7
The product of 4 and 7 is 28
The product of 6.0 and 5.0 is 30.0

```

这里，我们定义了一个名为 **mult** 的函数，它返回乘积。我们在代码中调用了这个函数两次。首先，使用用户输入的整数值。第二，使用浮点值。因此，证明了可重用性。

* * *

## 对列表元素执行乘法运算

我们还可以使用不同的方式打印给定列表中所有元素的乘积:

### 1.通过遍历列表

在这个方法中，我们将使用一个 for 循环来遍历列表和一个初始化为 **1** 的变量' **res** '(不是 0，因为我们需要产品和 0 * any = 0)。“ **res** 的值随着每次迭代而更新。

```py
list1 = [3,4,5]
res = 1
for val in list1 :
  res = res * val
print("The product of elements of the given list is ", res)

```

输出:

```py
The product of elements of the given list is  60

```

### 2.使用 numpy.prod()

在 [NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 中，我们有 **prod()** ，它将一个列表作为参数，并返回列表中所有元素的乘积。这个函数用处很大，节省了很多代码。你只需要导入 **NumPy** 就可以使用 numpy.prod()。下面给出一个例子:

代码:

```py
import numpy
list1 = [2,3,4,5]
list2 = [10,10,10]
ans1 = numpy.prod(list1)
ans2 = numpy.prod(list2)
print("Multiplication of elements of list1 is ",ans1)
print("Multiplication of elements of list2 is ",ans2)

```

输出:

```py
Multiplication of elements of list1 is  120
Multiplication of elements of list2 is  1000

```

## 结论

这就是你在 Python 中多重数的方法！我希望这篇教程对你有用。