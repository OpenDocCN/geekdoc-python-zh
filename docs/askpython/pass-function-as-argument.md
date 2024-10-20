# Python:如何将函数作为参数传递？

> 原文：<https://www.askpython.com/python/examples/pass-function-as-argument>

大家好！在本教程中，我们将讨论在 Python 中将函数作为参数传递的不同方式。

* * *

## Python 中有哪些函数？

在 Python 编程中，函数起着非常关键的作用。我们在 Python 中有非常广泛和丰富的不同类型的函数集合。Python 中的**函数**提供了**模块化**特性。这意味着使用函数，我们可以将一个大的 Python 代码块分成更小的块，每个块必须执行特定的任务。我们可以使用预定义的函数，也可以定义自己的函数。在 Python 中，定义在特定类中的函数我们称之为**方法**。

## 充当一级对象

**一级**对象是那些在整个程序中被一致对待的对象。这意味着一级对象可以存储在变量中，作为参数传递给函数，或者在控制语句中使用。Python 支持一级对象的概念，将函数视为一级对象。正是因为这个原因，我们可以将函数作为参数传递给 Python 中的其他函数。

## Python 中如何将函数作为参数传递？

在 Python 中，我们可以通过以下方式将不同类型的函数作为参数传递给另一个函数。让我们逐一讨论。

### 1.用户定义的函数

在 Python 中，就像普通变量一样，我们可以将用户定义的函数作为参数传递给另一个函数。接受另一个函数作为其参数的函数称为高阶**函数**。让我们看看如何通过 Python 代码实现这一点。

```py
# Define higher order function
def fun(foo):
    result = foo('Welcome To AskPython!!')
    return result

# Define function-1
def fun1(str):
    return str.lower()

# Define function-2
def fun2(str):
    return str.upper()

# Pass funtion-1 as an argument
# to fun() function
str1 = fun(fun1)
print(str1)

# Pass funtion-2 as an argument
# to fun() function
str2 = fun(fun2)
print(str2)

```

**输出:**

```py
welcome to askpython!! 
WELCOME TO ASKPYTHON!!

```

### 2.分类方法

像用户定义的函数一样，我们也可以将类方法作为参数传递。让我们用两个方法在 Python 中定义一个类，并创建这个类的一个对象来调用这些方法。让我们看看实现这一点的 Python 代码。

```py
# Define a Python class
class demo_class:
    # Define method-1
    def method1(self):
        print("Method-1 Running")
        return "AskPython!!"
    # Define method-2
    def method2(self, foo):
        print("Method-2 Running")
        result = foo()
        return result

# Create a demo_class object
# using the class constructor
obj = demo_class()

# Pass method-1 as an argument to method-2
str = obj.method2(obj.method1)
print(str)

```

**输出:**

```py
Method-2 Running 
Method-1 Running 
AskPython!!

```

### 3.λ函数

在 Python 中，[**λ**函数](https://www.askpython.com/course/python-course-lambda-functions)是对λ表达式求值时返回的函数对象。像用户定义的函数和类方法一样，我们也可以将 lambda 函数作为参数传递给另一个函数。让我们看看实现这一点的 Python 代码。

```py
# Create a Python list
ls = [1, 2, 3, 4, 5]
print('This is the given list:')
print(ls)

# Pass lambda function 
# to map() function to claculate
# the square of each list element
iter_obj = map((lambda n: n**2), ls)

# Construct the list again with
# square of the elements of the given list
ls = list(iter_obj)
print('This is the final list:')
print(ls)

```

**输出:**

```py
This is the given list: 
[1, 2, 3, 4, 5] 
This is the final list: 
[1, 4, 9, 16, 25]

```

### 4.算子函数

在 Python 中，我们有包含预定义函数的**操作符**模块。这些函数允许我们对给定的参数列表执行数学、关系、逻辑或[位运算](https://www.askpython.com/course/python-course-bitwise-operators)。像用户定义的和 lambda 函数一样，我们也可以将一个操作符函数作为参数传递给另一个函数。这里我们将使用来自 operator 模块的`operator.mul()`函数，并将其传递给`reduce()`函数，该函数是在 **functools** 模块中定义的，并带有一个 Python 列表。这将计算并返回传递的列表元素的乘积。让我们通过 Python 代码来实现这一点。

```py
# Importing Python functools module which contains the reduce() function
import functools

# Importing Python operator module which contains the mul() function
import operator

# Defining a Python list
ls = [1, 3, 5, 7, 9, 11]
print("Given Python list:")
print(ls)

# Pass the mul() function as an argument 
# to the reduce() function along with the list
ls_product = functools.reduce(operator.mul, ls)

# Printing the results
print("Product of the given Python list: ", ls_product)

```

**输出:**

```py
Given Python list: 
[1, 3, 5, 7, 9, 11] 
Product of the given Python list:  10395

```

### 5.内置函数

在 Python 中，我们有很多标准的内置函数，比如 list()、tuple()、dict()、str()等等。像用户定义的函数一样，我们也可以将内置函数作为参数传递给 Python 中的另一个函数。这里我们将把`str()`函数传递给 [`map()`函数](https://www.askpython.com/python/built-in-methods/map-method-in-python)以及一个 Python 字符串和数字元组。这将返回一个迭代器对象，我们将使用 str.join()函数将给定的元组转换为 Python 字符串。让我们编写 Python 代码来实现这一点。

```py
# Create a Python tuple
tup = ('Linux', 'Ubuntu', 20.04)
print("Given Python tuple:")
print(tup)

# Pass the str() function
# to the map() function
iter_obj = map(str, tup)

# Construct the string from 
# the returned iterator object
# of the given tuple
str1 = "-".join(iter_obj)

# Print the result
print("Generated Python string from tuple:")
print(str1)

```

**输出:**

```py
Given Python tuple: 
('Linux', 'Ubuntu', 20.04) 
Generated Python string from tuple: 
Linux-Ubuntu-20.0

```

## 结论

在本教程中，我们学习了以下内容:

*   什么是一等品？
*   如何将自定义函数作为参数传递？
*   如何将类方法作为参数传递？
*   如何将 lambda 函数作为参数传递？
*   如何将运算符函数作为参数传递？
*   如何将内置函数作为参数传递？

希望您已经理解了上面讨论的所有概念，并准备好学习和探索更多关于 Python 中的函数。敬请关注我们！