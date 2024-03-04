# 如何在 Python 中使用排序

> 原文：<https://www.pythonforbeginners.com/dictionary/python-sorting>

## 整理

```py
 The easiest way to sort is with the sorted(list) function, which takes a list and returns a new list with those elements in sorted order. 

The original list is not changed. 

The sorted() function can be customized though optional arguments. 

The sorted() optional argument reverse=True, e.g. sorted(list, reverse=True), 
makes it sort backwards. 
```

## 例子

```py
 Sorting Examples 
```

```py
 **Create a list with some numbers in it** numbers = [5, 1, 4, 3, 2, 6, 7, 9]

**prints the numbers sorted** print sorted(numbers)

**the original list of numbers are not changed** print numbers

my_string = ['aa', 'BB', 'zz', 'CC', 'dd', "EE"]

**if no argument is used, it will use the default (case sensitive)** print sorted(my_string)

**using the reverse argument, will print the list reversed** print sorted(strs, reverse=True)   ## ['zz', 'aa', 'CC', 'BB'] 
```

## 更多阅读

```py
 Please see the [python wiki](https://wiki.python.org/moin/HowTo/Sorting/ "wki-sorting") for more things to do with sorting. 
```