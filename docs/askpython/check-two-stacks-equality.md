# Python 中如何检查两个栈是否相等？

> 原文：<https://www.askpython.com/python/examples/check-two-stacks-equality>

在本教程中，我们将讨论使用 Python 检查两个堆栈是否相等的不同方法。

* * *

## 什么是堆栈？

一个 **[Python 栈](https://www.askpython.com/python/python-stack)** 是一个线性数据结构，基于 **LIFO** 原理工作。根据 LIFO 原则，最后插入堆栈的元素将首先被移除/访问。这就是为什么我们称之为**后进先出**。这意味着我们只能在一个方向上对栈执行不同的操作。栈可以用任何编程语言实现，包括 Python。在 Python 中，可以使用 list、deque 和 LifoQueue 来实现堆栈。为了简单起见，这里我们将使用 Python 列表来实现堆栈。

## Python 堆栈的属性

*   堆栈是单向的，即堆栈的元素只能从一端插入或删除。
*   堆栈维护一个指向堆栈最后一个元素的 **top** 指针。
*   要访问堆栈的第( **i** )个元素，我们必须移除最后( **N-i** )个元素。
*   堆栈可以是无溢出条件的动态或有溢出条件的静态的**。**

## 检查 Python 中两个堆栈的相等性

我们有两个堆栈，必须检查这两个堆栈是否相同。只有当两个堆栈中的元素数量相同、值相同、顺序相同时，它们才被称为**等于**。例如:

```py
stack1 = [1, 5, 7, 9]
stack2 = [1, 5, 7, 9]
(stack1 & stack2) are the same.
stack3 = [9, 5, 7, 1]
(stack1 & stack3) and (stack2 & stack3) are not the same.

```

### 方法 1:比较并弹出两个堆栈的顶部元素

让我们看看 Python 中方法一检查两个给定堆栈相等性的算法:

1.  首先创建一个 checker 变量，并将其设置为 **True** (最初假设两个堆栈相等)。
2.  然后比较两个堆栈的大小，如果它们不相等，则将 checker 变量设置为 **False** 并返回控制。
3.  否则比较两个堆栈的顶部元素。如果相等，将它们从两个堆栈中取出。
4.  如果栈顶元素不相等，则将 checker 变量设置为 **False** 并返回控制。
5.  重复**步骤 3** 和 **4** 直到两个堆栈都变空，即堆栈的所有元素都弹出。
6.  最后检查我们在**步骤 1** 中定义的 checker 变量的值，如果为**真**则表示两个堆栈相等，否则不相等(或不相等)。

让我们通过 Python 代码来实现上面的算法。

```py
# Define a function in Python
# To check if the two stacks
# Equal or not
def equal_stacks(s1, s2):
    # Create a checker variable
    # And initialize it with True
    val = True

    # Check the size of both stacks
    # Passed as arguments
    if len(s1) != len(s2):
        val = False
        return val

    # Compare the top of each stack
    while(len(s1)):
        if s1[-1] == s2[-1]:
            s1.pop()
            s2.pop()
        else:
            val = False
            break
    # Return the final value
    # Of checker variable val
    return val

# Driver Code
# Define two stacks
stack1 = [8, 15, 7, 11]
stack2 = [8, 15, 9, 11]

# Pass the above two Stacks to equal_stacks() function
# And check their equality
if equal_stacks(stack1, stack2):
    print("Two stacks are equal!")
else:
    print("Two stacks are not equal!!")

# Print the contents of both the stacks
# After their comparison
print(f'\nStack-1 after comparison: {stack1}')
print(f'\nStack-2 after comparison: {stack2}')

```

**输出:**

```py
Two stacks are not equal!

Stack-1 after comparison: [8, 15, 7]      

Stack-2 after comparison: [8, 15, 9]

```

在上面的输出中，我们可以清楚地看到，在比较之后，两个堆栈的内容都发生了改变。

### 方法 2:比较没有改变的两个堆栈的顶部元素

让我们看看 Python 中方法二检查两个给定堆栈相等性的算法:

1.  首先创建一个 checker 变量，并将其设置为 **True** (最初假设两个堆栈相等)。
2.  然后将两个堆栈的大小保存在两个独立的变量中，比如(P 和 Q ),并对它们进行比较。如果它们不相等，则将 checker 变量设置为 **False** 并返回控制。
3.  否则，在范围[1，P + 1]上运行循环的**，并执行以下操作:**
    1.  首先将栈 1 的顶部(P-1)个元素转移到栈 2。
    2.  将堆栈 1 的当前顶部元素存储到一个单独的变量中，比如说 temp。
    3.  现在将堆栈 2 的顶部 2*(P-1)个元素转移到堆栈 1。
    4.  将堆栈 2 的顶部元素与 temp 变量(即堆栈 1 的顶部元素)内的值进行比较。
    5.  如果两个堆栈的两个对应的顶部元素相等，则通过将堆栈 1 的顶部(P-1)个元素转移到堆栈 2 来重建两个堆栈。
    6.  否则将检验器变量设置为**假**并返回控制。
4.  最后检查我们在**步骤 1** 中定义的 checker 变量的值，如果为**真**则表示两个堆栈相等，否则不相等(或不相等)。

让我们通过 Python 代码来实现上面的算法。

```py
# Define a function to push the elements of
# One stack into another stack
def push_stack(s1, s2, len):
	i = 1
	while (i <= len):
        # Append the top of s1 to s2
		s2.append(s1[-1])
        # Delete the top of s1
		s1.pop()
        # Increment the loop counter
		i = i + 1

# Define a function to check 
# If the two stacks equal or not
def equal_stacks(s1, s2):
    # Create a checker variable
    # And initialize it with True
    val = True
	# Find the size of S1 stack
    P = len(s1)
	# Find the size of S2 stack
    Q = len(s2)
	# Compare the size of s1 & s2 stacks
    if (P != Q):
        val = False
        return val
    # Compare the top elements of each stack
    for i in range(1, P + 1):
        # Push P-i elements of stack s1 to stack s2
        push_stack(s1, s2, P - i)
		# Save the value of S1 top
        val = s1[-1]
		# Push 2 * (P-i) elements of stack S2 to stack S1
        push_stack(s2, s1, 2 * (P - i))
		# Compare the top elements of both stacks s1 & s2
        if (val != s2[-1]):
            val = False
            return val
		# Reconstruct both the stacks s1 & s2
        push_stack(s1, s2, P - i)
	# Return the final value of val
    return val

# Driver Code
# Define two stacks
stack1 = [5, 7, 11, 8]
stack2 = [5, 7, 11, 8]

# Pass the above two Stacks to equal_stacks() function
# And check their equality
if equal_stacks(stack1, stack2):
    print("Two stacks are equal!")
else:
    print("Two stacks are not equal!!")

# Print the contents of both the stacks
# After their comparison
print(f'\nStack-1 after comparison: {stack1}')
print(f'\nStack-2 after comparison: {stack2}')

```

**输出:**

```py
Two stacks are equal!

Stack-1 after comparison: [5, 7, 11, 8]   

Stack-2 after comparison: [5, 7, 11, 8]

```

在上面的输出中，我们可以清楚地看到两个堆栈的内容在比较后都没有被修改或改变。

## 结论

在本教程中，我们学习了 Python 中检查两个给定堆栈相等性的不同方法。

*   在第一种方法中，我们在改变它们之后检查了两个堆栈的相等性，即，在最后我们没有原始堆栈。
*   在第二种方法中，我们已经检查了两个堆栈的相等性，而没有改变它们，也就是说，最后我们得到了原始堆栈。