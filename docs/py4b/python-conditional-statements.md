# Python 中的条件语句

> 原文：<https://www.pythonforbeginners.com/basics/python-conditional-statements>

## 条件语句

在编程中，我们经常想要检查条件并改变程序的
行为。

## 如何使用条件语句

我们可以根据一个变量的值编写有不止一个动作选择的程序。

也许最广为人知的语句类型是 if 语句。

如果一件事为真，使用 if 语句执行一个动作；如果另一件事为真，使用
语句执行任意数量的其他动作。

我们必须使用缩进来定义执行的代码，基于是否满足条件。

为了在 Python 中比较数据，我们可以使用比较操作符，在
中找到这个[布尔，真或假](https://www.pythonforbeginners.com/basics/boolean "Boolean")帖子。

#### 如果语句

if 语句的语法是:

if 表达式:
语句

#### Elif 语句

有时有两种以上的可能性，在这种情况下我们可以使用
elif 语句

它代表“else if”，这意味着如果原始 if 语句为
false，而 elif 语句为 true，则执行
elif 语句之后的代码块。

if…elif 语句的语法是:

```py
if expression1:
   statement(s)
elif expression2:
   statement(s)
elif expression3:
   statement(s)
else:
   statement(s) 
```

#### Else 语句

else 语句可以与 if 语句结合使用。

else 语句包含在 if 语句中的条件
表达式解析为 0 或 false 值时执行的代码块。

else 语句是可选语句，if 后面最多只能有一个
else 语句。

if 的语法..else 是:

```py
if expression:
   statement(s)
else:
   statement(s) 
```

#### 例子

这个脚本将根据用户的输入比较两个字符串

```py
# This program compares two strings.

# Get a password from the user.
password = raw_input('Enter the password: ')

# Determine whether the correct password
# was entered.

if password == 'hello':
    print'Password Accepted'

else:
    print'Sorry, that is the wrong password.' 
```

#### 另一个例子

让我们再展示一个例子，其中也将使用 elif 语句。

```py
#!/usr/bin/python

number = 20

guess = int(input('Enter an integer : '))

if guess == number:
    print('Congratulations, you guessed it.')

elif guess < number:
    print('No, it is a little higher than that')

else:
    print('No, it is a little lower than that') 
```