# Python If…Elif…Else 语句

> 原文：<https://www.pythonforbeginners.com/basics/python-if-elif-else-statement>

## 什么是条件？

Conditions tests if a something is True or False, and it uses Boolean values
(type bool) to check that. 

You see that conditions are either True or False **(with no quotes!)**. 

```py
2 < 5
3 > 7
x = 11
x > 10
2 * x < x
type(True)

```

The results of these tests decides what happens next. 

## 何时使用条件？

If you want to check if the user typed in the right word or to see if a number is higher / lower than 100\. 

## 语法解释

First, lets look at Pythons if statement code block. 

Rememeber, to indicate a block of code in Python, you must indent each line of the block by the same amount. 

```py
If ...else

    if condition:
        statements

    elif condition:
 statements

    else:
 statements

```

If, elif and else are **keywords** in Python.

A **condition** is a **test** for something ( is x less than y, is x == y etc. )

The **colon (:)** at the end of the if line is required. 

Statements are **instructions** to follow if the condition is true. 

These statements must be **indented** and is only being run when the if condition
is met. 

Typical conditions are: **x<y, x>y, x<=y, x>=y, x!=y and x==y**. 

If you want more choices, you will have to include at least two conditions . 

The "else" **MUST** be preceded by an if test and will **ONLY** run when condition of
the if statement is **NOT** met. 

**else** will run if all others fail. 

If you only have two choices in your construction, use **if ..else** 

If there are more than two options, use **if ..elif ..else..** that will make
it easier to read

**elif** is short for **"else if"** 

## 条件测试

An If statement sets an condition with one or more if statement that will be
used when a condition is met.

There can be zero or more elif parts, and the else part is optional. 

The keyword 'elif' is short for 'else if', and is useful to avoid excessive
indentation.

An if ... elif ... elif ... sequence is a substitute for the switch or case
statements found in other languages. 

```py
x = raw_input("What is the time?")

if x < 10:
 print "Good morning"

elif x<12: 
 print "Soon time for lunch"

elif x<18: 
  print "Good day"

elif x<22: 
 print "Good evening"

else: 
  print "Good night"

```

You can also use it to control that only specified users can login to a system. 

```py
# Allowed users to login
allowed_users = ['bill', 'steve']

# Get the username from a prompt
username = raw_input("What is your login? :  ")

# Control if the user belongs to allowed_users

if username in allowed_users:
    print "Access granted"

else:
    print "Access denied"

```

## Python If 语句代码块

To get an easy overview of Pythons if statement code block 

```py
if :
    [do something]
    ....
    ....
elif [another statement is true]:
    [do something else]
    ....
    ....
else:
    [do another thing]
    ....
    .... 
```

For more reading, please see Python's official [documentation](https://docs.python.org/2/tutorial/controlflow.html "docs_python"). 
