# python 数据结构与算法 3 栈的抽象数据类型

The Stack Abstract Data Type 栈的抽象数据类型

The stack abstract data type is defined by thefollowing structure and operations. A stack is structured, asdescribed above, as an ordered collection of items where items areadded to and removed from the end called the “top.” Stacks areordered LIFO. The stack operations are given below.

下面的结构和操作定义了栈的抽象数据类型。如前所述，栈是结构化的，有序的的数据集，它的增删操作都在叫在”栈顶“的一端进行，存储顺序是 LIFO。栈的操作方法如下：

*   Stack() <wbr>creates a new stack thatis empty. It needs no parameters and returns an emptystack.
*   Stack(),构造方法，创建一个空栈，无参数，返回值是空栈。
*   push(item) <wbr>adds a new item to thetop of the stack. It needs the item and returnsnothing.
*   Push(item)向栈顶压入一个新数据项，需要一个数据项参数，无返回值。
*   pop() <wbr>removes the top item from thestack. It needs no parameters and returns the item. The stack ismodified.
*   pop()抛出栈顶数据项，无参数，返回被抛出的数据项，栈本身发生变化。
*   peek() <wbr>returns the top item fromthe stack but does not remove it. It needs no parameters. The stackis not modified.
*   Peek()返回栈顶数据项，但不删除。不需要参数，栈不变。
*   isEmpty() <wbr>tests to see whether the stackis empty. It needs no parameters and returns a booleanvalue.
*   isEmpty()测试栈是否空栈。不需要参数，返回布尔值。
*   size() <wbr>returns the number of items onthe stack. It needs no parameters and returns aninteger.
*   size()返回栈内数据项的数目，不需要参数，返回值是整数。

For example,if <wbr>s <wbr>is a stack that has beencreated and starts out empty, then <wbr>[*Table1*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#tbl-stackops) <wbr>shows the results of a sequenceof stack operations. Under stack contents, the top item is listedat the far right.

例如，s 是一个空栈，表 1 是一系列的操作，栈内数据和返回值。注意栈顶在右侧。

| **Table 1: Sample Stack Operations** |
| --- |
| **Stack Operation** | **Stack Contents** | **Return Value** |
| --- | --- | --- |
| s.isEmpty() | [] | True |
| s.push(4) | [4] |  <wbr> |
| s.push('dog') | [4,'dog'] |  <wbr> |
| s.peek() | [4,'dog'] | 'dog' |
| s.push(True) | [4,'dog',True] |  <wbr> |
| s.size() | [4,'dog',True] | 3 |
| s.isEmpty() | [4,'dog',True] | False |
| s.push(8.4) | [4,'dog',True,8.4] |  <wbr> |
| s.pop() | [4,'dog',True] | 8.4 |
| s.pop() | [4,'dog'] | True |
| s.size() | [4,'dog'] | 2 |

Implementing a Stack in Python

栈的实现

Now that we have clearly defined the stack as anabstract data type we will turn our attention to using Python toimplement the stack. Recall that when we give an abstract data typea physical implementation we refer to the implementation as a datastructure.

现在已经定义了栈的抽象数据类型，我们转向栈的实现。注意当我们说抽象数据类型的物理实现时，指的是建立数据结构。

As we described in Chapter 1, in Python, as in anyobject-oriented programming language, the implementation of choicefor an abstract data type such as a stack is the creation of a newclass. The stack operations are implemented as methods. Further, toimplement a stack, which is a collection of elements, it makessense to utilize the power and simplicity of the primitivecollections provided by Python. We will use a list.

如第一章所述，python 是面向对象的程序设计语言，栈一类的抽象数据类型是通过类实现的。栈的操作作为类的方法。另外，栈作为数据项的集合，我们使用 python 中强大而简单的数据集 list 来实现。

Recall that the list class in Python provides anordered collection mechanism and a set of methods. For example, ifwe have the list[2,5,3,6,7,4], we need only todecide which end of the list will be considered the top of thestack and which will be the base. Once that decision is made, theoperations can be implemented using the list methods suchas <wbr>append <wbr>and <wbr>pop.

python 中的 list 类已经建立了一个数据集合机制和相应的方法，如果，有了一个列表[2,5,3,6,7,4],只需要约定哪一端是栈顶哪一端是栈底，list 中的方法如 append 和 pop 都可实现了。

The following stack implementation ([*ActiveCode1*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#lst-stackcode1)) assumes that the end of the list will hold thetop element of the stack. As the stack grows(as <wbr>push <wbr>operations occur), new items will beadded on the end of the list. <wbr>pop <wbr>operations will manipulate that sameend.

以下的栈实现假定 list 的右侧是栈顶。这样当栈增长（push）时，新数据项就加在尾部，而 pop 也在同一位置。

代码段：

```py
class Stack:

    def __init__(self):

        self.items = []

    def isEmpty(self):

        return self.items == []

    def push(self, item):

        self.items.append(item)

    def pop(self):

        return self.items.pop()

    def peek(self):

        return self.items[len(self.items)-1]

    def size(self):

        return len(self.items)
```