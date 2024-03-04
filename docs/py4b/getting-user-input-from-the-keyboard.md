# 从键盘获取用户输入

> 原文：<https://www.pythonforbeginners.com/basics/getting-user-input-from-the-keyboard>

## 原始输入和输入

Python 中有两个函数可以用来从用户处读取数据:raw_input 和 input。您可以将结果存储到一个变量中。

## 原始输入

```py
raw_input is used to read text (strings) from the user:

```

```py
name = raw_input("What is your name? ")
type(name)

>>output
What is your name? spilcm
type 'str'>

```

## 投入

```py
input is used to read integers

```

```py
age = input("What is your age? ")
print "Your age is: ", age
type(age)

>>output
What is your age? 100
Your age is:  100
type 'int'>

```

##### 更多阅读

```py
http://www.python.org

```