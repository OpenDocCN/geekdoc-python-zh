# Python 中的双向链表

> 原文：<https://www.pythonforbeginners.com/basics/doubly-linked-list-in-python>

在 python 编程中，链表用于各种应用程序中。在本文中，我们将用 python 实现一个双向链表。为了理解双向链表，你需要有简单链表的知识。因此，如果你不知道单链表，你可以在这篇关于 python 中的[链表的文章中读到它们。](https://www.pythonforbeginners.com/lists/linked-list-in-python)

## 什么是双向链表？

双向链表是一种链表，其中的节点使用两个引用相互链接。

双向链表中的每个节点由三个属性组成，即`data`、`next`和`previous`。您可以将双向链表中的节点可视化，如下所示。

这里，

*   `previous`属性用于引用链表中的前一个节点。
*   `data`属性用于存储节点中的数据。
*   `next`属性用于引用链表中的下一个节点。

在双向链表中，可以有任意数量的节点。链表中的所有节点使用 previous 和 next 属性相互连接。您可以想象一个有三个节点的双向链表，如下所示。

![Doubly Linked List in Python](img/847a88a7b3ffb6e9e38bc69f508d83e7.png "Doubly Linked List in Python")



Doubly Linked List in Python

在上图中，我们创建了一个包含三个节点的双向链表。

*   第一个节点的`data`属性中包含 5，p1 和 n1 分别作为其`previous`和`next`属性。
*   第二个节点在其`data`属性中包含 10，p2 和 n2 分别作为其`previous`和`next`属性。
*   第三个节点在其`data`属性中包含 15，p3 和 n3 分别作为其`previous`和`next`属性。
*   `head`节点不包含任何数据，用于引用链表中的第一个节点。
*   第一个节点的`previous`属性不引用任何其他节点。它指向`None`。类似地，最后一个节点的`next`属性不引用任何其他节点。也指向了`None`。

因为我们对双向链表有一个大致的概念，所以让我们试着用 Python 实现一个双向链表。

## 如何用 Python 创建双向链表？

为了在 python 中创建双向链表，我们将首先为双向链表创建一个节点，如下所示。

```py
class Node:
    def __init__(self, value):
        self.previous = None
        self.data = value
        self.next = None
```

这里，`Node`类包含属性`data`来存储链表中的数据。在 python 中，`previous`和`next`属性用于连接双向链表中的节点。

创建节点后，我们将创建一个`DoublyLinkedList`类来实现 python 中的双向链表。该类将包含一个初始化为`None`的`head`属性。

```py
class DoublyLinkedList:
    def __init__(self):
        self.head = None
```

创建空链表后，我们可以创建一个节点，并将其分配给`head`属性。要向链表添加更多的节点，可以手动分配节点的`next`和`previous`属性。

我们可以编写一个方法，在 python 中的双向链表中插入一个元素，而不是手动将节点分配给链表。我们还可以执行不同的操作，比如在 python 中更新和删除双向链表中的元素。让我们逐一讨论每个操作。

## 检查 Python 中的双向链表是否为空

要检查 python 中的双向链表是否为空，我们只需检查链表的`head`属性是否指向`None`。如果是，我们就说链表是空的。否则不会。

对于这个操作，我们将实现一个 `isEmpty()` 方法。当在双向链表上调用`isEmpty()`方法时，它将检查链表的`head`属性是否指向`None`。如果是，则返回`True`。否则返回`False`。

下面是检查双向链表是否为空的 `isEmpty()`方法的 python 实现。

```py
 def isEmpty(self):
        if self.head is None:
            return True
        return False
```

## 在 Python 中求双向链表的长度

为了找到双向链表的长度，我们将遵循以下步骤。

*   首先，我们将创建一个临时变量`temp`和一个计数器变量`count`。
*   我们将用双向链表的`head`初始化变量`temp`。同样，我们将把变量`count`初始化为 0。
*   现在我们将使用 while 循环和`temp`变量遍历链表。
*   遍历时，我们会先检查当前节点(`temp`)是否为`None`。如果是，我们将退出循环。否则，我们将首先把`count`加 1。之后，我们将把`temp`变量赋给当前节点的`next`节点。

在 while 循环执行之后，我们将在变量`count`中获得双向链表的长度。

下面是 python 中双向链表的`length()` 方法的实现。当在双向链表上调用时，它计算链表的长度并返回值。

```py
 def length(self):
        temp = self.head
        count = 0
        while temp is not None:
            temp = temp.next
            count += 1
        return count
```

## 在 Python 中搜索双向链表中的元素

为了在 python 中搜索双向链表中的元素，我们将使用下面的算法。

*   首先，我们将定义一个变量`temp`，并将其初始化为链表的`head`属性。
*   我们还将定义一个变量`isFound`，并将其初始化为`False`。这个变量将用于检查给定的元素是否在双向链表中找到。
*   之后，我们将使用 while 循环和`temp`变量遍历链表的节点。
*   在迭代双向链表的节点时，我们将执行以下操作。
    *   我们将检查当前节点是否是`None`。如果是，则意味着我们已经到达了链表的末尾。因此，我们将移出 while 循环。
    *   如果当前节点不是`None`，我们将检查当前节点中的数据是否等于我们正在搜索的值。
    *   如果当前节点中的元素等于我们正在搜索的元素，我们将把值`True`赋给`isFound`变量。之后，我们将使用 break 语句跳出 while 循环。否则，我们将移动到链表中的下一个节点。

在 while 循环执行之后，如果`isFound`变量的值为`True`，则该元素被认为是在链表中找到的。否则不会。

下面是`search()`方法的实现。当在双向链表上调用时，`search()`方法将一个元素作为它的输入参数。执行后，如果在链表中找到该元素，则返回 True。否则，它返回 False。

```py
 def search(self, value):
        temp = self.head
        isFound = False
        while temp is not None:
            if temp.data == value:
                isFound = True
                break
            temp = temp.next
        return isFound
```

## 在 Python 中的双向链表中插入元素

在 python 中向双向链表中插入元素时，可能有四种情况。

1.  我们需要在链表的开头插入一个元素。
2.  我们需要在链表的给定位置插入一个元素。
3.  我们需要在链表的柠檬后面插入一个元素。
4.  我们需要在链表的末尾插入一个元素。

让我们逐一讨论每一种情况。

### 在双向链表的开头插入

为了在 python 中的双向链表的开头插入一个元素，我们将首先检查链表是否为空。您可以使用上一节讨论的`isEmpty()`方法检查链表是否为空。

如果双向链表是空的，我们将简单地用给定的数据创建一个新的节点，并将它赋给链表的`head`属性。

如果链表不为空，我们将遵循以下步骤。

*   首先，我们将创建一个具有给定数据的新节点，这些数据必须插入到链表中。
*   之后，我们会将链表的`head`属性所引用的节点赋给新节点的`next`属性。
*   然后，我们将新节点赋给链表的`head`属性所引用的节点的`previous`属性。
*   最后，我们将新节点分配给链表的`head`属性。

执行上述步骤后，新元素将被添加到双向链表的开头。

下面是 python 中`insertAtBeginning()`方法的实现。在双向链表上调用`insertAtBeginning()`方法时，该方法将一个元素作为其输入参数，并将其插入到链表的开头。

```py
 def insertAtBeginning(self, value):
        new_node = Node(value)
        if self.isEmpty():
            self.head = new_node
        else:
            new_node.next = self.head
            self.head.previous = new_node
            self.head = new_node
```

### 在双向链表的末尾插入

为了在 python 中的双向链表的末尾插入一个元素，我们将使用下面的算法。

*   首先，我们将使用给定的元素创建一个新节点。
*   之后，我们将检查双向链表是否为空。如果是，我们将使用`insertAtBeginning()`方法将新元素添加到列表中。
*   否则，我们将定义一个变量`temp`并将`head`属性赋给它。之后，我们将使用 while 循环移动到双向链表的最后一个节点。
*   在 while 循环中，我们将检查当前节点的`next`属性是否指向`None`。如果是，我们已经到达列表的最后一个节点。因此，我们将移出循环。否则，我们将转移到下一个节点。
*   到达最后一个节点(`temp`)后，我们会将新节点赋给最后一个节点的`next`属性。然后，我们将把`temp`分配给新节点的`previous`属性。

执行上述步骤后，新元素将被添加到双向链表的末尾。

下面是`insertAtEnd()`方法的实现。在链表上调用`insertAtEnd()`方法时，该方法将一个元素作为其输入参数，并将其添加到链表的末尾。

```py
 def insertAtEnd(self, value):
        new_node = Node(value)
        if self.isEmpty():
            self.insertAtBeginning(value)
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node
            new_node.previous = temp
```

### 在双向链表的元素后插入

要在 python 中的双向链表中的另一个元素之后插入一个新元素，我们将使用以下步骤。

首先，我们将定义一个变量`temp`，并将其初始化为链表的`head`属性。之后，我们将使用 while 循环和`temp`变量遍历链表的节点。在迭代双向链表的节点时，我们将执行以下操作。

*   我们将检查当前节点是否是`None`。如果是，则意味着我们已经到达了链表的末尾。因此，我们将移出 while 循环。
*   如果当前节点不是`None`，我们将检查当前节点中的`data`是否等于我们必须在其后插入新元素的元素。
*   如果当前节点中的元素等于我们必须在其后插入新元素的元素，我们将使用 break 语句退出 while 循环。否则，我们将移动到链表中的下一个节点。

执行 while 循环后，可能会出现两种情况。

*   如果`temp`变量包含值`None`，这意味着链表中不存在我们必须在其后插入新值的元素。在这种情况下，我们将打印该元素不能插入双向链表。

*   如果`temp`变量不是`None`，我们将在链表中插入元素。为此，我们将遵循以下步骤。
    *   首先，我们将用需要插入的元素创建一个新节点。
    *   我们将把当前节点的下一个节点赋给新节点的`next`属性。之后，我们将当前节点赋给新节点的`previous`属性。
    *   然后，我们将新节点赋给当前节点的下一个节点的`previous`属性。
    *   最后，我们将新节点分配给当前节点的`next`属性。

执行上述步骤后，新元素将被插入到双向链表中给定值之后。

下面是`insertAfterElement()`方法的实现。`insertAfterElement()`方法将两个值作为它的输入参数。第一个参数是要插入到链表中的新值。第二个参数是必须在其后插入新值的元素。

在执行之后，如果必须在其后插入新值的元素出现在双向链表中，则`insertAfterElement()`方法将新元素插入双向链表。否则，它会显示新元素不能插入到链表中。

```py
 def insertAfterElement(self, value, element):
        temp = self.head
        while temp is not None:
            if temp.data == element:
                break
            temp = temp.next
        if temp is None:
            print("{} is not present in the linked list. {} cannot be inserted into the list.".format(element, value))
        else:
            new_node = Node(value)
            new_node.next = temp.next
            new_node.previous = temp
            temp.next.previous = new_node
            temp.next = new_node
```

### 在双向链表中的给定位置插入

要在 python 中的双向链表中的给定位置 N 插入一个元素，我们将遵循以下步骤。

如果 N==1，则意味着必须在第一个位置插入元素。我们将使用 `insertAtBeginning()` 方法在双向链表中插入元素。否则，我们将遵循以下步骤。

*   首先，我们将定义一个变量`temp`，并将其初始化为链表的`head`属性。然后，我们将初始化一个变量`count`为 1。
*   之后，我们将使用 while 循环和`temp`变量遍历链表的节点。
*   在迭代双向链表的节点时，我们将执行以下操作。
    *   我们将检查当前节点是否是`None`。如果是，则意味着我们已经到达了链表的末尾。因此，我们将移出 while 循环。
    *   如果当前节点不是`None`，我们将检查变量 count 的值是否等于 N-1。如果是，我们将使用 break 语句跳出 while 循环。否则，我们将移动到链表中的下一个节点。

执行 while 循环后，可能会出现两种情况。

*   如果`temp`变量包含值`None`，则意味着链表中的元素少于 N-1 个。在这种情况下，我们不能在第 N 个位置插入新节点。因此，我们将打印出该元素不能插入到双向链表中。
*   如果`temp`变量不是`None`，我们将在链表中插入元素。为此，我们将有两种选择。
*   首先我们会检查当前节点(`temp`)的下一个节点是否是`None`，如果是，我们需要在链表的末尾插入新元素。因此，我们将使用 `insertAtEnd()`方法进行同样的操作。
*   如果当前节点的下一个节点不是`None`，我们将使用以下步骤在给定位置插入新元素。
    *   首先，我们将用需要插入的元素创建一个新节点。
    *   我们将把当前节点的下一个节点赋给新节点的`next`属性。
    *   之后，我们将当前节点赋给新节点的`previous`属性。
    *   然后，我们将新节点赋给当前节点的下一个节点的`previous`属性。
    *   最后，我们将新节点分配给当前节点的`next`属性。

执行上述步骤后，新元素将被插入到双向链表的给定位置。

下面是`insertAtPosition()`方法的实现。`insertAtPosition()`方法将两个值作为它的输入参数。第一个参数是要插入到链表中的新值。第二个参数是新值必须插入的位置。

执行后，`insertAtPosition()`方法在双向链表中的期望位置插入新元素。否则，它会显示新元素不能插入到链表中。

```py
 def insertAtPosition(self, value, position):
        temp = self.head
        count = 0
        while temp is not None:
            if count == position - 1:
                break
            count += 1
            temp = temp.next
        if position == 1:
            self.insertAtBeginning(value)
        elif temp is None:
            print("There are less than {}-1 elements in the linked list. Cannot insert at {} position.".format(position,
                                                                                                               position))
        elif temp.next is None:
            self.insertAtEnd(value)
        else:
            new_node = Node(value)
            new_node.next = temp.next
            new_node.previous = temp
            temp.next.previous = new_node
            temp.next = new_node
```

## 用 Python 打印双向链表的元素

为了在 python 中打印双向链表的元素，我们将首先定义一个变量`temp`并将链表的`head`赋给它。之后，我们将使用 while 循环来遍历链表的节点。

在迭代时，我们将首先检查双向链表中的当前节点是否为`None`。如果是，我们将移出 while 循环。否则，我们将打印当前节点的`data`属性。最后，我们将使用节点的 next 属性移动到链表中的下一个节点。

下面是`printLinkedList()`方法的实现。在双向链表上调用`printLinkedList()`方法时，会打印链表的所有元素。

```py
 def printLinkedList(self):
        temp = self.head
        while temp is not None:
            print(temp.data)
            temp = temp.next
```

## 在 Python 中更新双向链表中的元素

为了在 python 中更新双向链表中的元素，我们将首先定义一个变量`temp`并将链表的`head`赋给它。我们还将定义一个变量`isUpdated`，并将其初始化为`False`。之后，我们将使用 while 循环来遍历链表的节点。

在迭代时，我们将首先检查双向链表中的当前节点是否为`None`。如果是，我们将移出 while 循环。否则，我们将检查当前节点中的`data`属性是否等于需要用新值替换的值。如果是，我们将更新当前节点中的`data`属性，并将`isUpdated`中的值更新为`True`。最后，我们将使用 break 语句跳出 while 循环。

在执行 while 循环之后，我们将检查`isUpdated`是否为假。如果是，我们将打印该值没有更新。否则，我们将打印该值已更新。

下面是`updateElement()` 方法的实现。`updateElement()`方法将两个值作为它的输入参数。第一个参数是要更新的旧值。第二个参数是新值。

执行后，`updateElement()` 方法将给定元素更新为新值。

```py
 def updateElement(self, old_value, new_value):
        temp = self.head
        isUpdated = False
        while temp is not None:
            if temp.data == old_value:
                temp.data = new_value
                isUpdated = True
            temp = temp.next
        if isUpdated:
            print("Value Updated in the linked list")
        else:
            print("Value not Updated in the linked list")
```

### 更新给定位置的元素

为了在 python 中更新双向链表中给定位置 N 处的元素，我们将使用下面的算法。

首先，我们将定义一个变量`temp`，并将其初始化为链表的`head`属性。然后，我们将变量 count 初始化为 1。之后，我们将使用 while 循环和`temp`变量遍历链表的节点。

在迭代双向链表的节点时，我们将执行以下操作。

*   我们将检查当前节点是否是`None`。如果是，则意味着我们已经到达了链表的末尾。因此，我们将移出 while 循环。
*   如果当前节点不是`None`，我们将检查变量 count 的值是否等于 n，如果是，我们将使用 break 语句退出 while 循环。否则，我们将移动到链表中的下一个节点。

执行 while 循环后，可能会出现两种情况。

*   如果`temp`变量包含值`None`，则意味着链表中的元素少于 N 个。在这种情况下，我们不能更新链表中第 n 个位置的元素。因此，我们将打印该元素无法更新。
*   如果`temp`变量不是`None`，我们将通过更新当前节点的`data`属性来更新链表中的第 n 个元素。

下面是`updateAtPosition()`方法的实现。`updateAtPosition()`方法将两个值作为它的输入参数。第一个参数是链表中要更新的新值。第二个参数是新值必须被赋值的位置。

执行后，`updateAtPosition()`方法更新双向链表中所需位置的元素。

```py
 def updateAtPosition(self, value, position):
        temp = self.head
        count = 0
        while temp is not None:
            if count == position:
                break
            count += 1
            temp = temp.next
        if temp is None:
            print("Less than {} elements in the linked list. Cannot update.".format(position))
        else:
            temp.data = value
            print("Value updated at position {}".format(position))
```

## 在 Python 中从双向链表中删除元素

在 python 中从双向链表中删除元素时，可能有四种情况。

1.  我们需要从链表的开头删除一个元素。
2.  我们需要从链表的末尾删除一个元素。
3.  我们需要删除一个特定的元素。
4.  我们需要从链表的给定位置删除一个元素。

让我们逐一讨论每一种情况。

### 从双向链表的开头删除

要从双向链表的开头删除一个元素，我们将首先检查双向链表是否为空。如果是，我们会说我们不能从链表中删除任何元素。

否则，我们将检查链表中是否只有一个元素，即头节点的`next`属性是否指向`None`。如果`head`节点的下一个属性是 None，我们将把`None`分配给`head`。

如果链表中有多个元素，我们将把链表的`head`移动到当前`head`的`next`节点。之后，我们将把`None`赋给新头节点的`previous`属性。

执行上述步骤后，链表的第一个节点将被删除。

下面是`deleteFromBeginning()`方法的实现。在双向链表上调用`deleteFromBeginning()`方法时，会删除链表的第一个节点。

```py
 def deleteFromBeginning(self):
        if self.head is None:
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            self.head = None
        else:
            self.head = self.head.next
            self.head.previous = None
```

### 删除双向链表的最后一个元素

要删除 python 中双向链表的最后一个元素，我们将使用以下步骤。

*   首先，我们将检查双向链表是否为空。如果是，我们会说我们不能从链表中删除任何元素。
*   否则，我们将检查链表是否只有一个元素，即`head`节点的`next`属性是否指向`None`。如果头节点的`next`属性是`None`，我们将`None`赋给`head`。
*   如果链表中有多个元素，我们将创建一个变量`temp`并将`head`赋给该变量。之后，我们将遍历双向链表，直到到达链表的最后一个节点，即当前节点的`next`属性变为`None`。
*   到达最后一个节点后，我们会将`None`赋给当前节点的上一个节点的`next`属性。随后，我们将把`None`赋给当前节点的`previous`属性。

通过执行上述步骤，双向链表的最后一个元素将从链表中删除。

下面是`deleteFromLast()`方法的实现。在双向链表上调用`deleteFromLast()` 方法时，会删除链表的最后一个节点。

```py
 def deleteFromLast(self):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            self.head = None
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.previous.next = None
            temp.previous = None
```

### 删除双向链表中的给定元素

为了从 python 中的双向链表中删除给定的元素，我们将使用以下步骤。

*   首先，我们将检查双向链表是否为空。如果是，我们会说我们不能从链表中删除任何元素。
*   否则，我们将检查链表是否只有一个元素，即头节点的`next`属性是否指向`None`。
*   如果`head`节点的`next`属性为 None，我们将检查第一个节点中的元素是否是要删除的元素。如果是，我们将把`None`分配给`head`。
*   如果链表中有多个元素，我们将创建一个变量`temp`并将`head`赋给该变量。
*   之后，我们将使用`temp`变量和 while 循环遍历双向链表直到最后。
*   在遍历链表时，我们将首先检查当前节点是否是`None`，即我们已经到达了链表的末尾。如果是，我们将移出 while 循环。
*   如果当前节点不是`None`，我们将检查当前节点是否包含需要删除的元素。如果是，我们将使用 break 语句跳出 while 循环。

执行 while 循环后，可能会出现两种情况。

*   如果`temp`变量包含值`None`，则意味着我们已经到达了双向链表的末尾。在这种情况下，我们不能从链表中删除元素。因此，我们将打印出该元素不能从双向链表中删除。
*   如果`temp`变量不是 none，我们将从链表中删除该元素。为此，我们将有两种选择。
    *   首先，我们将检查当前节点是否是链表的最后一个节点，即当前节点(`temp`)的下一个节点是`None`，如果是，我们基本上需要删除链表的最后一个节点。因此，我们将使用`deleteFromLast()`方法进行同样的操作。
    *   如果当前节点的下一个节点不是`None`，我们将使用以下步骤删除给定的元素。
    *   我们将把`temp`节点的下一个节点赋给`temp`的前一个节点的`next`属性。
    *   然后，我们将把`temp`节点的前一个节点赋给 temp 的下一个节点的`previous`属性。
    *   最后，我们将把`None`分配给`temp`节点的`previous`和`next`属性。

通过执行上述步骤，任何给定的元素，如果存在于链表中，将从链表中删除。

下面是`delete()`方法的实现。在双向链表上调用`delete()` 方法时，该方法将一个元素作为输入参数，并从链表中删除它的第一个匹配项。

```py
 def delete(self, value):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            if self.head.data == value:
                self.head = None
        else:
            temp = self.head
            while temp is not None:
                if temp.data == value:
                    break
                temp = temp.next
            if temp is None:
                print("Element not present in linked list. Cannot delete element.")
            elif temp.next is None:
                self.deleteFromLast()
            else:
                temp.next = temp.previous.next
                temp.next.previous = temp.previous
                temp.next = None
                temp.previous = None
```

### 从双向链表中的给定位置删除

为了在 python 中从双向链表的给定位置 N 删除一个元素，我们将使用下面的算法。

首先，我们将检查链表是否为空。如果是，我们会说我们不能删除任何元素。

之后，我们将检查 N==1，这意味着我们需要从第一个位置删除该元素。如果是，我们将使用`deleteFromBeginning()`方法删除第一个元素。

否则，我们将遵循以下步骤。

*   首先，我们将定义一个变量`temp`，并将其初始化为链表的`head`属性。然后，我们将初始化一个变量`count`为 1。
*   之后，我们将使用 while 循环和`temp`变量遍历链表的节点。
*   在迭代双向链表的节点时，我们将执行以下操作。
    *   我们将检查当前节点是否是`None`。如果是，则意味着我们已经到达了链表的末尾。因此，我们将移出 while 循环。
    *   如果当前节点不是`None`，我们将检查变量`count`的值是否等于 n，如果是，我们将使用 break 语句退出 while 循环。否则，我们将移动到链表中的下一个节点。

执行 while 循环后，可能会出现两种情况。

*   如果`temp`变量包含值`None`，则意味着链表中的元素少于 N 个。在这种情况下，我们不能删除链表中第 n 个位置的元素。因此，我们将打印出该元素不能从双向链表中删除。
*   如果`temp`变量不为 none，我们将从链表中删除第 n 个元素。为此，我们将有两种选择。
    *   首先，我们将检查当前节点(`temp`)的下一个节点是否为 None。如果是，则第 n 个元素是链表的最后一个元素。因此，我们将使用 `deleteFromEnd()` 方法来删除元素。
    *   如果第 n 个元素不是双向链表的最后一个元素，我们将执行以下操作来删除第 n 个元素。
    *   我们将把`temp`(当前节点)的下一个节点赋给 temp 的前一个节点的`next`属性。
    *   然后，我们将把`temp`节点的前一个节点赋给`temp`的下一个节点的`previous`属性。
    *   最后，我们将把`None`分配给`temp`节点的`previous`和`next`属性。

通过执行上述步骤，任何给定的元素，如果存在于链表中，将从链表中删除。

下面是`deleteFromPosition()`方法的实现。在双向链表上调用`deleteFromPosition()`方法时，该方法将需要删除元素的位置作为其输入参数。执行后，它从双向链表中的指定位置删除元素。

```py
 def deleteFromPosition(self, position):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif position == 1:
            self.deleteFromBeginning()
        else:
            temp = self.head
            count = 1
            while temp is not None:
                if count == position:
                    break
                temp = temp.next
            if temp is None:
                print("There are less than {} elements in linked list. Cannot delete element.".format(position))
            elif temp.next is None:
                self.deleteFromLast()
                temp.previous.next = temp.next
                temp.next.previous = temp.previous
                temp.next = None
                temp.previous = None
```

## 双向链表在 Python 中的完整实现

现在我们已经讨论了用 python 实现双向链表的所有方法，让我们执行程序来观察实现。

```py
class Node:
    def __init__(self, value):
        self.previous = None
        self.data = value
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def isEmpty(self):
        if self.head is None:
            return True
        return False

    def length(self):
        temp = self.head
        count = 0
        while temp is not None:
            temp = temp.next
            count += 1
        return count

    def search(self, value):
        temp = self.head
        isFound = False
        while temp is not None:
            if temp.data == value:
                isFound = True
                break
            temp = temp.next
        return isFound

    def insertAtBeginning(self, value):
        new_node = Node(value)
        if self.isEmpty():
            self.head = new_node
        else:
            new_node.next = self.head
            self.head.previous = new_node
            self.head = new_node

    def insertAtEnd(self, value):
        new_node = Node(value)
        if self.isEmpty():
            self.insertAtBeginning(value)
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node
            new_node.previous = temp

    def insertAfterElement(self, value, element):
        temp = self.head
        while temp is not None:
            if temp.data == element:
                break
            temp = temp.next
        if temp is None:
            print("{} is not present in the linked list. {} cannot be inserted into the list.".format(element, value))
        else:
            new_node = Node(value)
            new_node.next = temp.next
            new_node.previous = temp
            temp.next.previous = new_node
            temp.next = new_node

    def insertAtPosition(self, value, position):
        temp = self.head
        count = 0
        while temp is not None:
            if count == position - 1:
                break
            count += 1
            temp = temp.next
        if position == 1:
            self.insertAtBeginning(value)
        elif temp is None:
            print("There are less than {}-1 elements in the linked list. Cannot insert at {} position.".format(position,
                                                                                                               position))
        elif temp.next is None:
            self.insertAtEnd(value)
        else:
            new_node = Node(value)
            new_node.next = temp.next
            new_node.previous = temp
            temp.next.previous = new_node
            temp.next = new_node

    def printLinkedList(self):
        temp = self.head
        while temp is not None:
            print(temp.data, sep=",")
            temp = temp.next

    def updateElement(self, old_value, new_value):
        temp = self.head
        isUpdated = False
        while temp is not None:
            if temp.data == old_value:
                temp.data = new_value
                isUpdated = True
            temp = temp.next
        if isUpdated:
            print("Value Updated in the linked list")
        else:
            print("Value not Updated in the linked list")

    def updateAtPosition(self, value, position):
        temp = self.head
        count = 0
        while temp is not None:
            if count == position:
                break
            count += 1
            temp = temp.next
        if temp is None:
            print("Less than {} elements in the linked list. Cannot update.".format(position))
        else:
            temp.data = value
            print("Value updated at position {}".format(position))

    def deleteFromBeginning(self):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            self.head = None
        else:
            self.head = self.head.next
            self.head.previous = None

    def deleteFromLast(self):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            self.head = None
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.previous.next = None
            temp.previous = None

    def delete(self, value):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif self.head.next is None:
            if self.head.data == value:
                self.head = None
        else:
            temp = self.head
            while temp is not None:
                if temp.data == value:
                    break
                temp = temp.next
            if temp is None:
                print("Element not present in linked list. Cannot delete element.")
            elif temp.next is None:
                self.deleteFromLast()
            else:
                temp.next = temp.previous.next
                temp.next.previous = temp.previous
                temp.next = None
                temp.previous = None

    def deleteFromPosition(self, position):
        if self.isEmpty():
            print("Linked List is empty. Cannot delete elements.")
        elif position == 1:
            self.deleteFromBeginning()
        else:
            temp = self.head
            count = 1
            while temp is not None:
                if count == position:
                    break
                temp = temp.next
            if temp is None:
                print("There are less than {} elements in linked list. Cannot delete element.".format(position))
            elif temp.next is None:
                self.deleteFromLast()
                temp.previous.next = temp.next
                temp.next.previous = temp.previous
                temp.next = None
                temp.previous = None

x = DoublyLinkedList()
print(x.isEmpty())
x.insertAtBeginning(5)
x.printLinkedList()
x.insertAtEnd(10)
x.printLinkedList()
x.deleteFromLast()
x.printLinkedList()
x.insertAtEnd(25)
x.printLinkedList()
x.deleteFromLast()
x.deleteFromBeginning()
x.insertAtEnd(100)
x.printLinkedList()
```

输出:

```py
True
5
5
10
5
5
25
100
```

## 结论

在本文中，我们讨论了双向链表在 python 中的实现。我希望这篇文章能帮助你学习 python 中双向链表的所有概念。如果您在实现中发现任何错误或改进，请在评论中告诉我们。

要了解更多关于 python 编程的知识，可以阅读这篇关于如何在 python 中找到列表中最大值的[索引的文章。您可能还会喜欢这篇关于如何用 python](https://www.pythonforbeginners.com/basics/find-the-index-of-max-value-in-a-list-in-python) 对对象列表进行[排序的文章。](https://www.pythonforbeginners.com/basics/sort-list-of-objects-in-python)

请继续关注更多内容丰富的文章。

快乐学习！