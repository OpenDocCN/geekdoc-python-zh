# python 数据结构与算法 6 栈的应用之符号平衡（通用）

## Balanced Symbols (AGeneral Case)

## 栈的应用之平衡符号（通用）

The balanced parentheses problem shown above is a specific caseof a more general situation that arises in many programming languages. Thegeneral problem of balancing and nesting different kinds of opening andclosing symbols occurs frequently. For example, in Pythonsquare brackets, `[` and `]`, are used for lists; curly braces, `{` and `}`, are used fordictionaries; and parentheses, `(` and `)`, are used for tuples and arithmetic expressions. It is possibleto mix symbols as long as each maintains its own open and close relationship.Strings of symbols such as

相对编程语言的应用情形来说，上一节所讲的圆括号匹配只算是一个特例。不同种类的左符号和右符号的平衡实在是很常见的普遍问题。比如在 Python，左右方括号[ ]用于列表，左右大括号｛｝用于字典，左右圆括号（）用于元组。多种符号的混合应用中也要保持符号的平衡关系。如符号组成的字符串：

```py
{ { ( [ ] [ ] ) } ( ) }
```

```py
[ [ { { ( ( ) ) } } ] ]
```

```py
[ ] [ ] [ ] ( ) { }
```

are properly balanced in that not only does each opening symbolhave a corresponding closing symbol, but the types of symbols match as well.

不但符号的左右平衡，种类也是匹配的。

Compare those with the following strings that are not balanced:

对比以下字符串就是不平衡的：

```py
( [ ) ]
```

```py
( ( ( ) ] ) )
```

```py
[ { ( ) ]
```

The simple parentheses checker from the previous section caneasily be extended to handle these new types of symbols. Recall that eachopening symbol is simply pushed on the stack to wait for the matching closingsymbol to appear later in the sequence. When a closing symbol does appear, theonly difference is that we must check to be sure that it correctly matches thetype of the opening symbol on top of the stack. If the two symbols do notmatch, the string is not balanced. Once again, if the entire string isprocessed and nothing is left on the stack, the string is correctly balanced.

上节讲到的圆括号平衡算法很容易扩展到其他种类的符号中，只要每个左符号被压栈，然后等匹配的右符号出现，此时唯一的不同，就是左右匹配的同时，必须检查符号的种类也要匹配。如果发现不匹配，整个字符串就是不平衡的。最后，当整个字符串处理完毕并且同时栈被清空，字符串就是完全平衡的。

The Python program to implement this is shown in [*ActiveCode 5*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#lst-parcheck2). The only change appears inline 16 where we call a helper function, `matches`, to assist withsymbol-matching. Each symbol that is removed from the stack must be checked tosee that it matches the current closing symbol. If a mismatch occurs, theboolean variable `balanced` is set to `False`.

Python 语言的实现方法如下。与上一节的不同仅仅是调用一个辅助函数，matches，帮助检查符号各类的匹配。每个从栈顶弹出的元素必须检查是否与当前的右符号同一种类。如果不匹配，变量 balanced 被赋值为 False.

```py
from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index <len(symbolString) and balanced:
        symbol =symbolString[index]
        if symbol in"([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced =False
            else:
                top = s.pop()
                if notmatches(top,symbol):
                      balanced = False
        index = index + 1
    if balanced ands.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open)== closers.index(close)

print(parChecker('{{([][])}()}'))
print(parChecker('[{()]')
```