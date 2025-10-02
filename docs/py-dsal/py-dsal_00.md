# python 数据结构与算法 1 基本数据结构

## Basic Data Structures 第一章 基本数据结构

## Objectives

学习目标

*   To understand the abstract data types stack, queue, deque, andlist.
*   To be able to implement the ADTs stack, queue, and deque usingPython lists.
*   To understand the performance of the implementations of basiclinear data structures.
*   To understand prefix, infix, and postfix expression formats.
*   To use stacks to evaluate postfix expressions.
*   To use stacks to convert expressions from infix to postfix.
*   To use queues for basic timing simulations.
*   To be able to recognize problem properties where stacks, queues,and deques are appropriate data structures.
*   To be able to implement the abstract data type list as a linkedlist using the node and reference pattern.
*   To be able to compare the performance of our linked listimplementation with Python’s list implementation.
*   理解栈、队列、双向队列和列表的抽象数据类型
*   以 python 之 list 为工具，建立栈队列双向队列和列表的 ADT
*   理解基本数据结构的性能
*   理解前置、中置和后置表达式
*   使用栈计算后置表达式
*   用栈把前置转后置
*   用队列实现基本的时间仿真
*   学会根据问题性质，选择使用栈队双向队等合适的数据结构
*   用列表的 ADT 实现链表的节点和指针
*   学会比较链表和 list 的性能

## What Are Linear Structures?

什么是线性数据结构?

We willbegin our study of data structures by considering four simple butvery powerful concepts. Stacks, queues, deques, and lists areexamples of data collections whose items are ordered depending onhow they are added or removed. Once an item is added, it stays inthat position relative to the other elements that came before andcame after it. Collections such as these are often referred toas <wbr>lineardata structures.

我们开始数据结构的学习，从四种简单而功能强大的结构开始。栈、队、双向队和列表是一种数据的集合，它的元素根据自己被加入或删除的的顺序排列。当一个元素加入集合之后，它就与之前和之后加入的元素保持一个固定的相对位置。这种数据集合叫做线性数据结构。

Linearstructures can be thought of as having two ends. Sometimes theseends are referred to as the “left” and the “right” or in some casesthe “front” and the “rear.” You could also call them the “top” andthe “bottom.” The names given to the ends are not significant. Whatdistinguishes one linear structure from another is the way in whichitems are added and removed, in particular the location where theseadditions and removals occur. For example, a structure might allownew items to be added at only one end. Some structures might allowitems to be removed from either end.

既然是线性，就有两头。有时叫做“左侧”或“右侧”，有时叫做“前端”“后端”，叫做"顶部“和”底部“也无不可。因为名字不重要。重要的是数据结构增加删除数据的方式，特别是增删的位置。例如，一种结构可能只允许从一头增加成员，另一种结构则两头都行。

These variationsgive rise to some of the most useful data structures in computerscience. They appear in many algorithms and can be used to solve avariety of important problems.

这种方式的变化在计算机科学中构成了多种非常有用的数据结构，他们出现在各种算法和重要问题的解决方案中。