# 如何在 Python 中使用切片

> 原文：<https://www.pythonforbeginners.com/dictionary/python-slicing>

## Python 中的切片

```py
 When you want to extract part of a string, or some part of a list, you use a slice

The first character in string x would be x[0] and the nth character would be at
x[n-1]. 

Python also indexes the arrays backwards, using negative numbers. 

The last character has index -1, the second to last character has index -2. 
```

## 例子

```py
 x = "my string"

x[start:end] 	# items start through end-1
x[start:]    	# items start through the rest of the list
x[:end]      	# items from the beginning through end-1
x[:]         	# a copy of the whole list

One way to remember how slices work is to think of the indices as pointing betweencharacters, with the left edge of the first character numbered 0\. 

Then the right edge of the last character of a string of n characters has index n

 +---+---+---+---+---+
 | H | e | l | p | A |
 +---+---+---+---+---+
 0   1   2   3   4   5
-5  -4  -3  -2  -1

The first row of numbers gives the position of the indices 0...5 in the string, 
the second row gives the corresponding negative indices. 
```