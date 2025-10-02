# python 数据结构与算法 8 栈的应用之中缀前缀后缀

Infix, Prefix and PostfixExpressions

中缀、前缀和后缀表达式

When you write anarithmetic expression such as B * C, the form of the expression provides youwith information so that you can interpret it correctly. In this case we knowthat the variable B is being multiplied by the variable C since themultiplication operator * appears between them in the expression. This type ofnotation is referred to as **infix** since the operator is *inbetween* the two operands that it is working on.

当写下 B*C 这个算术表达式的时候，你很清楚这表示什么。我们都知道这是要计算变量 B 被变量 C 乘，因为乘法符号*出现在两个表达式中间。这种表达式我们称之为中缀，因为操作符在变量的“中间”。

Consider anotherinfix example, A + B * C. The operators + and * still appear between theoperands, but there is a problem. Which operands do they work on? Does the +work on A and B or does the * take B and C? The expression seems ambiguous.

来看另一个中缀表达式 A+B*C，操作符+和*仍在操作数中间，但这时有就疑问了，操作符是操作哪个数？是+操作 A 和 B 呢还是*操作 B 和 C？这个表达式似乎有点含混。

In fact, you havebeen reading and writing these types of expressions for a long time and they donot cause you any problem. The reason for this is that you know something aboutthe operators + and *. Each operator has a **precedence** level.Operators of higher precedence are used before operators of lower precedence.The only thing that can change that order is the presence of parentheses. Theprecedence order for arithmetic operators places multiplication and divisionabove addition and subtraction. If two operators of equal precedence appear,then a left-to-right ordering or associativity is used.

事实上这种表达式我们经常见，也经常写，从来没有含混过。原因是我们知道操作符的优先级。优先级高的操作符优先计算，除非用括号改变顺序。优先级顺序是乘除加减，如果两个操作符在同一级别，那就从左到右依次进行。

Let’s interpretthe troublesome expression A + B * C using operator precedence. B and C aremultiplied first, and A is then added to that result. (A + B) * C would forcethe addition of A and B to be done first before the multiplication. Inexpression A + B + C, by precedence (via associativity), the leftmost + wouldbe done first.

现在用优先级顺序来解释 A+B*C。B 和 C 先相乘，然后 A 与乘积相加。（A+B）*C 将强制 A 和 B 先相见，再相乘。但 A+B+C 就是从左到右的顺序。

Although all thismay be obvious to you, remember that computers need to know exactly whatoperators to perform and in what order. One way to write an expression thatguarantees there will be no confusion with respect to the order of operationsis to create what is called a **fully parenthesized** expression.This type of expression uses one pair of parentheses for each operator. Theparentheses dictate the order of operations; there is no ambiguity. There isalso no need to remember any precedence rules.

是的，这些对你来说太显而易见了。但是请记住，计算机需要精确地知道操作符的行为和顺序。有一种书写表达式的方法叫做“完全括号”,这种表达式把每一个操作符都加了括号，表达完全精确，也不必记忆优先级规则。

The expression A +B * C + D can be rewritten as ((A + (B * C)) + D) to show that themultiplication happens first, followed by the leftmost addition. A + B + C + Dcan be written as (((A + B) + C) + D) since the addition operations associatefrom left to right.

表达式 A+B*C+D 写成（（A+（B*C））+D）表明先算乘法，再算左边的加法。A+B+C+D 写成（（（A+B）+C）+D）以表明从左到右。

There are twoother very important expression formats that may not seem obvious to you atfirst. Consider the infix expression A + B. What would happen if we moved theoperator before the two operands? The resulting expression would be + A B.Likewise, we could move the operator to the end. We would get A B +. These looka bit strange.

A+B 是操作符放在中间，如果把操作符放在操作数前面呢？变成+A B。放在后面呢？A B +。是不是看起来很奇怪。

These changes tothe position of the operator with respect to the operands create two newexpression formats, **prefix** and **postfix**. Prefixexpression notation requires that all operators precede the two operands thatthey work on. Postfix, on the other hand, requires that its operators comeafter the corresponding operands. A few more examples should help to make thisa bit clearer (see [*Table 2*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#tbl-example1)).

这两种变型形成新的格式，叫做前缀与后缀。前缀就是操作符放在他们的操作数前面，后缀就是放在后面。看一下表 2 就更清楚.

| **Table 2: Examples of Infix, Prefix, and Postfix** |
| --- |
| **Infix Expression****中缀** | **Prefix Expression****前缀** | **Postfix Expression****后缀** |
| --- | --- | --- |
| A + B | + A B | A B + |
| A + B * C | + A * B C | A B C * + |

A + B * C would bewritten as + A * B C in prefix. The multiplication operator comes immediatelybefore the operands B and C, denoting that * has precedence over +. Theaddition operator then appears before the A and the result of themultiplication.

用前缀方式，A+B*C 就要写成 +A*BC，*放在 B 和 C 前面，+放在 A 和*BC 的前面，乘法相对加法的优先级就体现出来了。

In postfix, theexpression would be A B C * +. Again, the order of operations is preservedsince the * appears immediately after the B and the C, denoting that * hasprecedence, with + coming after. Although the operators moved and now appeareither before or after their respective operands, the order of the operandsstayed exactly the same relative to one another.

后缀表达式的情况下，A+B*C 就要写成 A B C * +。因为*紧跟 B 和 C，当然是先算*，后算+。操作符可以前缀或后缀，但操作顺序完全相同。

Now consider theinfix expression (A + B) * C. Recall that in this case, infix requires theparentheses to force the performance of the addition before the multiplication.However, when A + B was written in prefix, the addition operator was simplymoved before the operands, + A B. The result of this operation becomes thefirst operand for the multiplication. The multiplication operator is moved infront of the entire expression, giving us * + A B C. Likewise, in postfix A B +forces the addition to happen first. The multiplication can be done to thatresult and the remaining operand C. The proper postfix expression is then A B +C *.

再看一下中缀表达式（A+B）*C，中缀要求用括号来强制先算加法再算乘法。但前缀时，A+B 写在+ A B，相加之和是后面乘法的第一个操作数，所以*就要移动到+A B 和 C 的前面，所以写成 * +A B C。相类似的，后缀时， A B +强制先算加法，结果与 C 相乘，于是就写在 A B + C *

Consider thesethree expressions again (see [*Table 3*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#tbl-parexample)). Something very important has happened. Where did theparentheses go? Why don’t we need them in prefix and postfix? The answer isthat the operators are no longer ambiguous with respect to the operands thatthey work on. Only infix notation requires the additional symbols. The order ofoperations within prefix and postfix expressions is completely determined bythe position of the operator and nothing else. In many ways, this makes infixthe least desirable notation to use.

把这三种表达方式放在表 3 对比一下，见证奇迹的时刻到了，括号去哪儿了？为什么前缀和后缀不需要括号？答案就是，在前缀和后缀中，操作符和他们的操作数之间关系清晰，他们的位置就说明了计算顺序，不需要象中缀那样，额外用括号来帮助分辨。因此，在很多情况下，中缀是最不想用的表达式。

| **Table 3: An Expression with Parentheses** |
| --- |
| **Infix Expression** | **Prefix Expression** | **Postfix Expression** |
| --- | --- | --- |
| (A + B) * C | * + A B C | A B + C * |

[*Table 4*](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#tbl-example3) shows some additional examples of infix expressionsand the equivalent prefix and postfix expressions. Be sure that you understandhow they are equivalent in terms of the order of the operations beingperformed.

表 4 提供更多的对比例子，要仔细对比他们怎样安排位置来保证计算正确的。

| **Table 4: Additional Examples of Infix, Prefix, and Postfix** |
| --- |
| **Infix Expression** | **Prefix Expression** | **Postfix Expression** |
| --- | --- | --- |
| A + B * C + D | + + A * B C D | A B C * + D + |
| (A + B) * (C + D) | * + A B + C D | A B + C D + * |
| A * B + C * D | + * A B * C D | A B * C D * + |
| A + B + C + D | + + + A B C D | A B + C + D + |