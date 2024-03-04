# 用 Python 计算平均值

> 原文：<https://www.pythonforbeginners.com/basics/calculate-average-in-python>

我们必须在 python 程序中执行许多数学计算来处理任何数据。在这篇文章中，我们将看看在 python 中计算给定数字平均值的不同方法。

## 如何计算给定数字的平均值

给定数字的平均值定义为所有数字的总和除以这些数字的总数。

例如，如果给定数字 1、2、4、5、6、7、8、10 和 12，我们可以通过首先计算它们的总和，然后用这些总和除以数字的总数来计算这些数字的平均值。这里，所有给定数字的和是 55，它们的总数是 9。因此，所有数字的平均值将是 55/9，即 6.111。

## 在 Python 中使用 for 循环计算平均值

如果给我们一个数字列表，我们可以使用 for 循环计算平均值。首先，我们将声明一个 **sumofNums** 和一个 **count** 变量，并将它们初始化为 0。然后，我们将遍历列表中的每个元素。在遍历时，我们将把每个元素添加到 **sumofNums** 变量中。同时，我们还会将**计数**变量加 1。遍历整个列表后，我们将在 **sumofNums** 变量中得到列表中所有元素的总和，并在 **count** 变量中得到元素的总数。现在，我们可以将 **sumofNums** 除以 **count** 来获得列表元素的平均值，如下所示。

```py
numbers = [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
sumOfNums = 0
count = 0
for number in numbers:
    sumOfNums += number
    count += 1
average = sumOfNums / count
print("The list of numbers is:", numbers)
print("The average of all the numbers is:", average) 
```

输出:

```py
The list of numbers is: [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
The average of all the numbers is: 19.53846153846154
```

## 使用内置函数计算平均值

不使用 for 循环，我们可以使用 python 中的内置[函数来计算给定列表中元素的平均值。](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)

我们可以使用 sum()方法计算列表中所有元素的总和，然后使用 len()方法计算列表中元素的总数。这样，我们将得到这些数字的总和以及这些数字的总数，我们可以用它们来计算平均值，如下所示。

```py
numbers = [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
sumOfNums = sum(numbers)
count = len(numbers)
average = sumOfNums / count
print("The list of numbers is:", numbers)
print("The average of all the numbers is:", average) 
```

输出:

```py
The list of numbers is: [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
The average of all the numbers is: 19.53846153846154
```

或者，我们可以使用统计模块的 mean()方法直接计算列表元素的平均值。我们将给定的数字列表作为输入传递给 mean()方法，它将返回数字的平均值，如下例所示。

```py
import statistics
numbers = [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
average = statistics.mean(numbers)
print("The list of numbers is:", numbers)
print("The average of all the numbers is:", average) 
```

输出:

```py
The list of numbers is: [1, 2, 34, 56, 7, 23, 23, 12, 1, 2, 3, 34, 56]
The average of all the numbers is: 19.53846153846154
```

## 结论

在本文中，我们讨论了在 Python 中计算给定数字平均值的不同方法。你可以在关于 [python 操作符](https://www.pythonforbeginners.com/basics/python-operators)的文章中读到其他操作。