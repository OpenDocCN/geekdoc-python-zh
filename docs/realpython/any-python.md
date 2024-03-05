# 如何在 Python 中使用 any()

> 原文：<https://realpython.com/any-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**Python any(): Powered Up 布尔函数**](/courses/python-any-boolean-function/)

作为一名 Python 程序员，你会经常处理[布尔值](https://realpython.com/python-boolean/)和[条件语句](https://realpython.com/python-conditional-statements/)——有时非常复杂。在这些情况下，您可能需要依赖能够简化逻辑和整合信息的工具。好在 Python 中的 **`any()`** 就是这样一个工具。它遍历 iterable 中的元素，并返回一个值，指示在布尔上下文中是否有任何元素为 true，或 **truthy。**

在本教程中，您将学习:

*   如何使用`any()`
*   如何在`any()`和`or`之间做出决定

让我们开始吧！

**Python 中途站:**本教程是一个**快速**和**实用**的方法来找到你需要的信息，所以你会很快回到你的项目！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 如何在 Python 中使用`any()`

想象一下，你正在为雇主的招聘部门编写一个程序。您可能希望安排符合以下任何标准的候选人参加面试:

1.  已经了解 Python 了
2.  有五年或五年以上的开发经验
3.  有学位

可以用来编写这个条件表达式的一个工具是 [`or`](https://realpython.com/python-or-operator/) :

```py
# recruit_developer.py
def schedule_interview(applicant):
    print(f"Scheduled interview with {applicant['name']}")

applicants = [
    {
        "name": "Devon Smith",
        "programming_languages": ["c++", "ada"],
        "years_of_experience": 1,
        "has_degree": False,
        "email_address": "devon@email.com",
    },
    {
        "name": "Susan Jones",
        "programming_languages": ["python", "javascript"],
        "years_of_experience": 2,
        "has_degree": False,
        "email_address": "susan@email.com",
    },
    {
        "name": "Sam Hughes",
        "programming_languages": ["java"],
        "years_of_experience": 4,
        "has_degree": True,
        "email_address": "sam@email.com",
    },
]
for applicant in applicants:
    knows_python = "python" in applicant["programming_languages"]
    experienced_dev = applicant["years_of_experience"] >= 5

    meets_criteria = (
        knows_python
        or experienced_dev
        or applicant["has_degree"]
    )
    if meets_criteria:
        schedule_interview(applicant)
```

在上面的例子中，您检查每个申请人的证书，如果申请人符合您的三个标准中的任何一个，就安排面试。

**技术细节:** Python 的`any()`和`or`并不局限于计算布尔表达式。相反，Python 对每个参数执行[真值测试](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)，评估表达式是 [**真值**还是**假值**](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context) 。例如，非零整数值被认为是真的，而零被认为是假的:

>>>

```py
>>> 1 or 0
1
```

在本例中，`or`将非零值`1`评估为真值，即使它不属于布尔类型。`or`回归`1`，无需评价`0`的真实性。在本教程的后面，您将了解更多关于`or`的返回值和参数求值。

如果您执行这段代码，那么您将看到 Susan 和 Sam 将获得面试机会:

```py
$ python recruit_developer.py
Scheduled interview with Susan Jones
Scheduled interview with Sam Hughes
```

该计划选择与苏珊和萨姆安排面试的原因是苏珊已经知道 Python 和萨姆有学位。注意每个候选人只需要满足一个标准。

评估申请人资格的另一种方法是使用`any()`。当您在 Python 中使用`any()`时，您必须将申请人的证书作为[可迭代](https://realpython.com/lessons/looping-over-iterables/)参数传递:

```py
for applicant in applicants:
    knows_python = "python" in applicant["programming_languages"]
    experienced_dev = applicant["years_of_experience"] >= 5

    credentials = (
        knows_python,
        experienced_dev,
        applicant["has_degree"],
    )
    if any(credentials):
        schedule_interview(applicant)
```

当您在 Python 中使用`any()`时，请记住您可以将任何 iterable 作为参数传递:

>>>

```py
>>> any([0, 0, 1, 0])
True

>>> any(set((True, False, True)))
True

>>> any(map(str.isdigit, "hello world"))
False
```

在每个例子中，`any()`循环遍历不同的 Python iterable，测试每个元素的真实性，直到找到一个真值或检查每个元素。

**注意:**最后一个例子使用 Python 内置的 [`map()`](https://realpython.com/python-map-function/) ，返回一个迭代器，其中每个元素都是将字符串中的下一个字符传递给`str.isdigit()`的结果。这是使用`any()`进行更复杂检查的有效方法。

你可能想知道`any()`是否仅仅是`or`的装扮版。在下一节中，您将了解这些工具之间的区别。

[*Remove ads*](/account/join/)

## 如何区分`or`和`any()`

Python 中的`or`和`any()`有两个主要区别:

1.  句法
2.  返回值

首先，您将了解语法如何影响每个工具的可用性和可读性。其次，您将了解每个工具返回的值的类型。了解这些差异将有助于您决定哪种工具最适合给定的情况。

### 语法

`or`是一个[操作符](https://realpython.com/lessons/operators-and-built-functions/)，所以它有两个参数，一边一个:

>>>

```py
>>> True or False
True
```

另一方面，`any()`是一个接受一个参数的函数，一个对象的 iterable，它通过循环来评估真实性:

>>>

```py
>>> any((False, True))
True
```

这种语法上的差异非常显著，因为它会影响每个工具的可用性和可读性。例如，如果您有一个 iterable，那么您可以将 iterable 直接传递给`any()`。要从`or`获得类似的行为，你需要使用一个循环或者一个类似 [`reduce()`](https://realpython.com/any-python/) 的函数:

>>>

```py
>>> import functools
>>> functools.reduce(lambda x, y: x or y, (True, False, False))
True
```

在上面的例子中，您使用了 [`reduce()`](https://realpython.com/lessons/python-reduce-function/) 将一个 iterable 作为参数传递给`or`。用`any`可以更有效地做到这一点，它直接接受 iterables 作为参数。

为了说明每个工具的语法影响其可用性的另一种方式，假设您想要避免测试一个条件，如果任何前面的条件是`True`:

```py
def knows_python(applicant):
    print(f"Determining if {applicant['name']} knows Python...")
    return "python" in applicant["programming_languages"]

def is_local(applicant):
    print(f"Determine if {applicant['name']} lives near the office...")

should_interview = knows_python(applicant) or is_local(applicant)
```

如果`is_local()`执行的时间相对较长，那么当`knows_python()`已经返回`True`时，你就不要调用它了。这叫做**懒**评估，或者 [**短路**评估](https://realpython.com/python-operators-expressions/#compound-logical-expressions-and-short-circuit-evaluation)。默认情况下，`or`延迟评估条件，而`any`不会。

在上面的例子中，程序甚至不需要确定 Susan 是否是本地人，因为它已经确认她知道 Python。这足够安排一次面试了。在这种情况下，用`or`延迟调用函数将是最有效的方法。

为什么不用`any()`来代替？您在上面了解到`any()`将 iterable 作为参数，Python 根据 iterable 类型评估条件。因此，如果您使用一个列表，Python 将在创建该列表期间执行`knows_python()`和`is_local()`，然后调用`any()`:

```py
should_interview = any([knows_python(applicant), is_local(applicant)])
```

在这里，Python 会为每一个申请人调用`is_local()`，即使是懂 Python 的人。因为`is_local()`将花费很长时间来执行，并且有时是不必要的，这是一个低效的逻辑实现。

在使用 iterables 时，有一些方法可以让 Python 延迟调用函数，比如用`map()`构建一个迭代器，或者使用[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions):

```py
any((meets_criteria(applicant) for applicant in applicants))
```

这个例子使用了一个[生成器表达式](https://realpython.com/courses/python-generators/)来生成布尔值，表明申请人是否符合面试标准。一旦申请人符合标准，`any()`将返回`True`,而不检查其余的申请人。但是请记住，这些类型的解决方法也有其自身的问题，并不适合每种情况。

需要记住的最重要的事情是，`any()`和`or`之间的语法差异会影响它们的可用性。

语法并不是影响这些工具可用性的唯一差异。接下来，让我们看看`any()`和`or`的不同返回值，以及它们如何影响您决定使用哪个工具。

[*Remove ads*](/account/join/)

### 返回值

Python 的`any()`和`or`返回不同类型的值。`any()`返回一个 Boolean 值，该值指示是否在 iterable 中找到了真值:

>>>

```py
>>> any((1, 0))
True
```

在这个例子中，`any()`找到了一个真值(整数`1`，所以它返回了布尔值`True`。

另一方面，`or`返回它找到的第一个真值，不一定是布尔值。如果没有真值，那么`or`返回最后一个值:

>>>

```py
>>> 1 or 0
1

>>> None or 0
0
```

在第一个例子中，`or`评估了`1`，它是真的，并且在不评估`0`的情况下返回它。第二个例子中，`None`是 falsy，所以`or`接下来对`0`求值，也是 falsy。但是因为没有更多的表达式要检查，`or`返回最后一个值，`0`。

当您决定使用哪个工具时，考虑您是否想知道对象的*实际值*或者只是真值是否存在于对象集合中的某个地方是很有帮助的。

## 结论

恭喜你！您已经了解了在 Python 中使用`any()`的来龙去脉，以及`any()`和`or`之间的区别。随着对这两种工具理解的加深，您已经准备好在自己的代码中做出选择。

你现在知道了:

*   如何在 Python 中使用 **`any()`**
*   为什么你会用 **`any()`** 而不是 **`or`**

如果您想继续学习条件表达式以及如何使用 Python 中的`or`和`any()`等工具，那么您可以查看以下资源:

*   [T2`operator.or_()`](https://docs.python.org/3.4/library/operator.html#operator.or_)
*   [T2`all()`](https://realpython.com/python-all/)
*   [`while`循环](https://realpython.com/python-while-loop/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**Python any(): Powered Up 布尔函数**](/courses/python-any-boolean-function/)****