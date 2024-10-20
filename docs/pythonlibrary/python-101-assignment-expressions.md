# Python 3 -赋值表达式

> 原文：<https://www.blog.pythonlibrary.org/2018/06/12/python-101-assignment-expressions/>

我最近偶然发现了 [PEP 572](https://www.python.org/dev/peps/pep-0572/) ，这是克里斯·安吉利科、蒂姆·皮特斯和吉多·范·罗苏姆本人提出的在 **Python 3.8** 中加入赋值表达式的建议！我决定检查一下，看看什么是赋值表达式。这个想法其实很简单。Python 核心开发人员想要一种使用以下符号在表达式中分配变量的方法:

```py

NAME := expr

```

这个话题已经有了很多争论，如果你愿意，你可以阅读关于 [Python-Dev Google group](https://groups.google.com/forum/#!msg/dev-python/WhTyLfI6Ctk/BI_gdR8vBAAJ) 的详细内容。我个人发现通读 Python 核心开发社区提出的各种利弊是非常有见地的。

不管怎样，让我们看看 PEP 572 中的一些例子，看看我们是否能弄清楚如何使用赋值表达式。

```py

# Handle a matched regex
if (match := pattern.search(data)) is not None:
    ...

# A more explicit alternative to the 2-arg form of iter() invocation
while (value := read_next_item()) is not None:
    ...

# Share a subexpression between a comprehension filter clause and its output
filtered_data = [y for x in data if (y := f(x)) is not None]

```

在这 3 个例子中，我们在表达式语句本身中创建了一个变量。第一个示例通过将 regex 模式搜索的结果赋给变量 **match** 来创建变量。第二个示例将变量**的值**赋给在 while 循环表达式中调用函数的结果。最后，我们将调用 f(x)的结果赋给列表理解中的变量 **y** 。

赋值表达式最有趣的特性之一(至少对我来说)是它们可以用在赋值语句不能用的上下文中，比如 lambda 或前面提到的 comprehension。然而，它们也不支持赋值语句可以做的一些事情。例如，你不能给多个目标赋值:

```py

x = y = z = 0  # Equivalent: (x := (y := (z := 0)))

```

您可以在 [PEP](https://www.python.org/dev/peps/pep-0572/#differences-between-assignment-expressions-and-assignment-statements) 中看到完整的差异列表

PEP 中有更多的信息，包括其他几个例子，讨论被拒绝的备选方案和范围。

* * *

### 相关阅读

*   PEP 572 - [赋值表达式](https://www.python.org/dev/peps/pep-0572/)
*   Reddit: [意外覆盖局部变量](https://www.reddit.com/r/Python/comments/8fokpw/you_can_accidentally_override_local_variables/)
*   Reddit on [PEP 572](https://www.reddit.com/r/Python/comments/8ex72p/pep_572_assignment_expressions/)