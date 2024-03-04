# 布尔值，Python 中的 True 或 False

> 原文：<https://www.pythonforbeginners.com/basics/boolean>

## 什么是布尔？

布尔值是两个常量对象 False 和 True。

它们用于表示真值(其他值也可以被认为是
假或真)。

在数字上下文中(例如，当用作
算术运算符的参数时)，它们的行为分别类似于整数 0 和 1。

内置函数 bool()可用于将任何值转换为布尔值，
如果该值可被解释为真值

它们分别被写成假的和真的。

## 布尔字符串

Python 中的字符串可以测试真值。

返回类型将是布尔值(真或假)

让我们举个例子，首先创建一个新变量并给它赋值。

```py
 my_string = "Hello World"

my_string.isalnum()		#check if all char are numbers
my_string.isalpha()		#check if all char in the string are alphabetic
my_string.isdigit()		#test if string contains digits
my_string.istitle()		#test if string contains title words
my_string.isupper()		#test if string contains upper case
my_string.islower()		#test if string contains lower case
my_string.isspace()		#test if string contains spaces
my_string.endswith('d')		#test if string endswith a d
my_string.startswith('H')	#test if string startswith H

To see what the return value (True or False) will be, simply print it out.	

my_string="Hello World"

print my_string.isalnum()		#False
print my_string.isalpha()		#False
print my_string.isdigit()		#False
print my_string.istitle()		#True
print my_string.isupper()		#False
print my_string.islower()		#False
print my_string.isspace()		#False
print my_string.endswith('d')		#True
print my_string.startswith('H')		#True 
```

## 布尔和逻辑运算符

布尔值响应逻辑运算符和/或

>>>真假
假

>>>真实与真实
真实

>>>真假
假

>>>假或真
真

>>>假或假
假

记住，内置类型 Boolean 只能保存两个可能的
对象之一:True 或 False