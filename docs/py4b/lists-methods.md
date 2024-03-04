# Python 中的内置列表方法

> 原文：<https://www.pythonforbeginners.com/basics/lists-methods>

## 列出内置方法

数据类型“List”有几个内置方法。

```py
 s = ['h','e','l','l','o']	#create a list
s.append('d')         		#append to end of list
len(s)                		#number of items in list
s.pop(2)              		#delete an item in the middle
s.sort()               		#sorting the list
s.reverse()           		#reversing the list
s.extend(['w','o'])    		#grow list
s.insert(1,2)         		#insert into list
s.remove('d')           	#remove first item in list with value e
s.pop()               		#remove last item in the list
s.pop(1)              		#remove indexed value from list
s.count('o')            	#search list and return number of instances found
s = range(0,10)          	#create a list over range 
s = range(0,10,2)        	#create a list over range with start index and increment 
```