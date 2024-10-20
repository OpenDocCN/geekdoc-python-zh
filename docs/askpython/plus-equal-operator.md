# Python 中的+=运算符——完全指南

> 原文：<https://www.askpython.com/python/examples/plus-equal-operator>

在这一课中，我们将看看 Python 中的 **+=操作符**，并通过几个简单的例子来看看它是如何工作的。

运算符“+=”是**加法赋值运算符**的简写。它将两个值相加，并将总和赋给一个变量(左操作数)。

让我们看三个例子来更好地理解这个操作符是如何工作的。

* * *

## 1.用+=运算符将两个数值相加

在下面提到的代码中，我们用初始值 5 初始化了变量 X，然后给它加上值 15，并将结果值存储在同一个变量 X 中。

```py
X = 5
print("Value Before Change: ", X)
X += 15
print("Value After Change: ", X)

```

代码的输出如下所示:

```py
Value Before Change:  5
Value After Change:  20

```

* * *

## 2.添加两个字符串

```py
S1 = "Welcome to "
S2 = "AskPython"

print("First String : ", S1)
print("Second String: ", S2)
S1+=S2
print("Final String: ", S1)

```

在上面提到的代码中，我们初始化了两个变量 S1 和 S2，初始值分别为“Welcome to”和“AskPython”。

然后，我们使用'+= '运算符将两个字符串相加，该运算符将连接字符串的值。

代码的输出如下所示:

```py
First String :  Welcome to 
Second String:  AskPython
Final String:  Welcome to AskPython

```

* * *

## 3.理解 Python 中“+=”运算符的结合性

“+=”运算符的结合性属性是从右向左的。让我们看看下面提到的示例代码。

```py
X = 5
Y = 10
X += Y>>1
print(X)

```

我们初始化了两个变量 X 和 Y，初始值分别为 5 和 10。在代码中，我们将 Y 的值右移 1 位，然后将结果添加到变量 X，并将最终结果存储到 X。

输出结果是 X = 10，Y = 10。

* * *

## **结论**

恭喜你！您刚刚学习了 python 中的'+= '操作符，还学习了它的各种实现。

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中的“in”和“not in”运算符](https://www.askpython.com/python/examples/in-and-not-in-operators-in-python)
2.  [Python //运营商——基于楼层的部门](https://www.askpython.com/python/python-floor-based-division)
3.  [Python 不等于运算符](https://www.askpython.com/python/python-not-equal-operator)
4.  [Python 中的运算符重载](https://www.askpython.com/python/operator-overloading-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *