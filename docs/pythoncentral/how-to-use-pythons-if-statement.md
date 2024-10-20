# 如何使用 Python 的 If 语句

> 原文：<https://www.pythoncentral.io/how-to-use-pythons-if-statement/>

在 Python 中，If 语句包含基于特定条件是否为真而执行的代码。例如，如果您想打印一些东西，但是您希望打印文本的执行取决于其他条件，您将需要使用 if 语句。

If 语句的基本语法如下:

```py
if(condition) : [code to execute]
```

简单 if 语句的语法相当简单。确保条件可以是真或假，比如变量等于、不等于、小于或大于某个值(请记住，这种情况下的“等于”符号是这样的:==)要了解它在上下文中的用法，请看看下面的示例。

```py
var=40
if(var == 40) : print "Happy Birthday!"
print "end of if statement"
```

您可能会猜到，上面代码的输出如下:

```py
Happy Birthday!
end of if statement
```

生日快乐！因为这是在 var 等于 40 的条件下执行的代码，事实上 var 等于 40。本例中使用了 if 语句的 End 来表示 if 语句已经结束。如果 var 不等于 40，那么只打印“If 语句结束”,属于 if 语句的代码不会被执行，因为只有当 if 条件为真时才允许执行。