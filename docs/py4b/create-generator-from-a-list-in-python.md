# 从 Python 中的列表创建生成器

> 原文：<https://www.pythonforbeginners.com/basics/create-generator-from-a-list-in-python>

Python 中的生成器是从容器对象中访问元素的非常有用的工具。在本文中，我们将讨论如何从一个列表中创建一个生成器，以及为什么我们需要这样做。这里我们将使用两种方法从列表中创建生成器。第一次使用生成器函数，第二次使用生成器理解。

## 使用生成器函数将列表转换为生成器

生成器函数是那些用**产生**语句而不是返回语句的函数。生成器函数有一个特点，一旦 yield 语句被执行，它们就暂停执行。为了恢复生成器函数的执行，我们只需要使用 next()函数，将生成器函数作为输入参数分配给生成器。

例如，假设我们想要一个给定列表的元素的平方。获得元素的平方的一种方式可以是使用列表理解或函数来创建具有现有列表的元素的平方的新列表，如下所示。

```py
def square(input_list):
    square_list = []
    for i in input_list:
        square_list.append(i ** 2)
    return square_list

myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:", myList)
squares = square(myList)
print("Elements obtained from the square function are:")
for ele in squares:
    print(ele) 
```

输出:

```py
The given list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements obtained from the square function are:
1
4
9
16
25
36
49
64
81
100
```

我们也可以创建一个生成器来代替一个新的列表，使用生成器函数来获得现有列表元素的平方。

为了使用 generator 函数从一个列表中创建一个生成器，我们将定义一个接受列表作为输入的生成器函数。在函数内部，我们将使用一个 for 循环，其中 yield 语句将用于给出现有列表元素的平方作为输出。我们可以如下执行此操作。

```py
def square(input_list):
    square_list = []
    for element in input_list:
        yield element ** 2

myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:", myList)
squares = square(myList)
print("Elements obtained from the square generator are:")
for ele in squares:
    print(ele) 
```

输出:

```py
The given list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements obtained from the square generator are:
1
4
9
16
25
36
49
64
81
100 
```

## 使用生成器理解将列表转换为生成器

我们可以使用 generator comprehension 从列表中创建一个生成器，而不是使用生成器函数。生成器理解的语法几乎与列表理解相同。

集合理解的语法是:*生成器* = **(** *表达式* **for** *元素***in***iterable***if***条件* **)**

您可以使用 generator comprehension 从列表中创建一个生成器，如下所示。这里，我们使用了与上一节中给出的相同的实现示例。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:", myList)
mygen = (element ** 2 for element in myList)
print("Elements obtained from the generator are:")
for ele in mygen:
    print(ele) 
```

输出:

```py
The given list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements obtained from the generator are:
1
4
9
16
25
36
49
64
81
100 
```

## 为什么要从列表中创建生成器？

生成器可以用来代替列表，原因有两个。让我们逐一了解他们两个。

1.  当我们从一个现有的列表创建一个新的列表时，程序使用内存来存储现有列表的元素。另一方面，生成器使用最少量的内存，几乎与函数所需的内存相似。因此，如果我们只需要从新创建的列表中访问元素，使用生成器代替列表会更有效。
2.  有了生成器，我们可以在不显式使用任何计数器的情况下随机访问列表的下一个元素。为此，我们可以使用 next()方法从生成器中提取下一个元素。

例如，考虑这样一种情况，我们有一个 100 个元素的列表。我们不断地要求用户输入一个数字，只要他们输入一个偶数，我们就打印列表的下一个元素。

这里，用户输入没有任何模式。因此，我们不能使用 for 循环来访问列表中的元素。相反，我们将使用一个生成器，通过 next()函数打印列表中的下一个元素，如下例所示。

```py
myList = range(0, 101)
myGen = (element ** 2 for element in myList)
while True:
    user_input = int(input("Input an even number to get an output, 0 to exit:"))
    if user_input == 0:
        print("Good Bye")
        break
    elif user_input % 2 == 0:
        print("Very good. Here is an output for you.")
        print(next(myGen))
    else:
        print("Input an even number.")
        continue
```

输出:

```py
Input an even number to get an output, 0 to exit:23
Input an even number.
Input an even number to get an output, 0 to exit:123
Input an even number.
Input an even number to get an output, 0 to exit:12
Very good. Here is an output for you.
0
Input an even number to get an output, 0 to exit:34
Very good. Here is an output for you.
1
Input an even number to get an output, 0 to exit:35
Input an even number.
Input an even number to get an output, 0 to exit:0
Good Bye 
```

## 结论

在本文中，我们讨论了从列表创建生成器的两种方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)