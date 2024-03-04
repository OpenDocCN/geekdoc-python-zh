# Python:元组概述

> 原文：<https://www.pythonforbeginners.com/basics/python-tuples-overview>

```py
 Tuple  is another data type in Python. 

A tuple consists of values separated by commas.

Tuples are always enclosed in parentheses.

Tuples are for immutable.

Though tuples may seem similar to lists, they are often used in different
situations and for different purposes. 

Empty tuples are constructed by an empty pair of parentheses.

Tuple with one item is constructed by following a value with a comma. 
```

## 元组示例

```py
 x = ()				# empty tuple
x = (0,)			# one item tuple
x = (0, 1, 2, "abc")            # four item tuple: indexed x[0]..x[3]
x = 0, 1, 2, "abc"              # parenthesis are optional
x = (0, 1, 2, 3, (1, 2))        # nested subtuples
y = x[0]		      	# indexed item
y = x[4][0]                    	# indexed subtuple
x = (0, 1) * 2                  # repeat
x = (0, 1, 2) + (3, 4)         	# concatenation
for item in x: print item    	# iterate through tuple
b = 3 in x			# test tuple membership 
```