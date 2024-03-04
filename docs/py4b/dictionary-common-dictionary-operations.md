# Python 中常见的字典操作

> 原文：<https://www.pythonforbeginners.com/dictionary/dictionary-common-dictionary-operations>

## 词典

```py
 A dictionary constant consists of a series of key-value pairs enclosed by curly
braces { }

With dictionaries you can store things so that you quickly can find them again 
```

## 字典操作

```py
 Below is a list of common dictionary operations: 
```

```py
 **create an empty dictionary** x = {}

**create a three items dictionary** x = {"one":1, "two":2, "three":3}

**access an element** x['two']

**get a list of all the keys** x.keys()

**get a list of all the values** x.values()

**add an entry** x["four"]=4

**change an entry** x["one"] = "uno"

**delete an entry** del x["four"]

**make a copy** y = x.copy()

**remove all items** x.clear()

**number of items** z = len(x)

**test if has key** z = x.has_key("one")

**looping over keys** for item in x.keys(): print item

**looping over values** for item in x.values(): print item

**using the if statement to get the values** if "one" in x:
    print x['one']

if "two" not in x:
    print "Two not found"

if "three" in x:
    del x['three'] 
```