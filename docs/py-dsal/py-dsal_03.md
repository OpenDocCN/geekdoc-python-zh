# python 数据结构与算法 4 栈的实现

Remember thatnothing happens when we click the <wbr> <wbr>button other thanthe definition of the class. We must create a <wbr> <wbr>object and thenuse it. <wbr>[ActiveCode2](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#lst-stackcode1) <wbr>showsthe <wbr> <wbr>class in action aswe perform the sequence of operations from <wbr>[Table1](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#tbl-stackops).

上面代码仅仅是 stack 类的实现，如果运行的话，什么反应也没有。所以下面的代码，就是创建一个栈对象，并加入操作方法，就如表 1 中所作的操作一样。

s=Stack()

print(s.isEmpty())

s.push(4)

s.push('dog')

print(s.peek())

s.push(True)

print(s.size())

print(s.isEmpty())

s.push(8.4)

print(s.pop())

print(s.pop())

print(s.size())

 <wbr>

It isimportant to note that we could have chosen to implement the stackusing a list where the top is at the beginning instead of at theend. In this case, the previous pop and append methods would nolonger work and we would have to index position 0 (the first itemin the list) explicitly using pop and insert. The implementation isshown in CodeLens 1.

注意，我们也可以选择列表的头（左侧）作为栈顶，这样，前面的 pop 和 append 方法就不能用了，而必须指定索引 0(列表的第一个项)以便对栈内数据操作。如下面代码段：

```py
classStack:
def __init__(self):
self.items = []
def isEmpty(self):
return self.items ==[]
def push(self, item):
self.items.insert(0,item)
def pop(self):
returnself.items.pop(0)
def peek(self):
return self.items[0]
def size(self):
returnlen(self.items)

s = Stack()
s.push('hello')
s.push('true')
print(s.pop())
```

 <wbr>

This ability to change the physical implementationof an abstract data type while maintaining the logicalcharacteristics is an example of abstraction at work. However, eventhough the stack will work either way, if we consider theperformance of the two implementations, there is definitely adifference. Recall that the append and pop() operations were bothO(1). This means that the first implementation will perform pushand pop in constant time no matter how many items are on the stack.The performance of the second implementation suffers in that theinsert(0) and pop(0) operations will both require O(n) for a stackof size n. Clearly, even though the implementations are logicallyequivalent, they would have very different timings when performingbenchmark testing.

对抽象数据类型的实现方式的变更，仍能保持数据的逻辑特性不变，就是“抽象”的实例。两种栈的方式都能工作，但性能表现却有很大的不同。Append()和 pop()都是 O(1)，这意味着，不管栈内有多少数据项，第一种实现的性能是常数级的，第二种实现的 insert(0)和 pop(0)却需要 O(n)。很明显，逻辑上是等同的，但在性能基准测试时，时间测试的结果是非常之不同的。

**Self Check**

**自测题**

stk-1: Given the following sequence of stack operations,what is the top item on the stack when the sequence iscomplete?

经过以下操作后，栈顶数据项是哪个？

 <wbr>

m = Stack()

m.push('x')

m.push('y')

m.pop()

m.push('z')

m.peek()

stk-2: Given the following sequence of stack operations,what is the top item on the stack when the sequence iscomplete?

第二题：经以下测试，栈顶元素是哪个？

m = Stack()

m.push('x')

m.push('y')

m.push('z')

**whilenot**m.isEmpty():

 <wbr> <wbr>m.pop()

 <wbr> <wbr>m.pop()

Write afunction <wbr>revstring(mystr) <wbr>thatuses a stack to reverse the characters in a string.

第三题，为以下代码补充一个函数 <wbr>revstring(mystr)，实现将一个字符串翻转顺序。

from testimport testEqual

frompythonds.basic.stack import Stack

defrevstring(mystr):

 <wbr> <wbr> <wbr>#在此处补充代码

testEqual(revstring('apple'),'elppa')

testEqual(revstring('x'),'x')

testEqual(revstring('1234567890'),'0987654321')