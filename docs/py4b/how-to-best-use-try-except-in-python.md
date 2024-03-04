# 如何在 Python 中最好地使用 Try-Except

> 原文：<https://www.pythonforbeginners.com/error-handling/how-to-best-use-try-except-in-python>

异常处理允许我们对变量施加约束，以在计算机程序中实现我们的业务逻辑，它还使我们能够编写一个健壮的程序，该程序可以处理执行过程中出现的不同类型的错误。在 python 中，我们使用 try-except 块来实现异常处理。在本文中，我们将通过查看插图来了解一些编写高效 python 程序的方法，以便最好地使用 try——除了在 python 中。所以，让我们深入研究一下。

## 使用 raise 关键字手动引发异常

python 程序在出错时会自动抛出内置异常，但我们也可以使用 raise 关键字手动引发内置异常。通过使用 raise 关键字抛出内置异常，我们可以使用内置异常对变量施加约束，以确保程序在逻辑上是正确的。这里要记住的一点是，一个程序可能不会抛出任何错误，但如果不对变量应用适当的约束，它可能会输出一个不切实际的值。因此，我们可以使用 raise 关键字在 [python try-except](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python) 块中抛出异常来强制约束。

例如，假设我们想根据一个人的年龄来计算他的出生年份，我们可以这样做:

```py
age= 10
print("Age is:")
print(age)
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
10
Year of Birth is:
2011
```

我们可以看到，对于一个 10 岁的人来说，程序给出了正确的输出。现在我们将尝试给程序输入一个负的年龄，看看会发生什么。

```py
age= -10
print("Age is:")
print(age)
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
-10
Year of Birth is:
2031
```

在上面的输出中，我们可以看到程序已经成功执行，并且给出了一个逻辑上不可能的输出，因为没有人能够知道他的出生年份。因此，当 age 为负时，我们将使用 try-except 错误处理代码手动引发一个错误，然后我们将如下所示显示正确的输出。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    if age<0:
        raise ValueError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except ValueError:
    print("Input Correct age.")
```

输出:

```py
Age is:
-10
Input Correct age.
```

这里可以看到，当输入年龄为负值时，程序手动引发`ValueError`，错误由 except 块中的代码处理。我们也可以自己定义定制的异常来实现约束，如下一节所述。

## 使用自定义异常

为了加强约束，我们可以通过定义类来声明自己的异常，这些类应该是任何内置异常的子类。通过这种方式，我们可以创建一个名称对我们的约束更有意义的异常，然后我们可以通过检查和引发自定义异常来对变量实施约束。

在下面的例子中，我们通过继承 python 异常类来定义`NegativeAgeError`异常类。每当输入中的 age 值为负时，就会引发名为`NegativeAgeError`的自定义错误，然后由 except 块中的代码处理该错误。只要输入年龄的正值，程序就会正常工作。

```py
class NegativeAgeError(Exception):
    pass
try:
    age= -10
    print("Age is:")
    print(age)
    if age<0:
        raise NegativeAgeError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except NegativeAgeError:
    print("Input Correct age.")
```

输出:

```py
Age is:
-10
Input Correct age
```

## 在 python 中使用 try-except 区分不同类型的错误

在许多情况下，我们希望以不同的方式处理不同的运行时和自定义异常。我们可能想要找出在程序执行期间发生了哪个异常，或者我们可能想要在不同的异常发生时执行不同的任务。在这些情况下，我们可以通过将异常名称作为 except 块的参数来单独捕获每个异常，如下例所示，然后我们可以针对该特定异常执行 except 块中给出的语句。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    assert age>0
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except AssertionError:
    print("Input Correct age.")
except ValueError:
    print("ValueError occured")
except NameError:
    print("NameError occurred")
except:
    print("Some other exception occurred")
```

在上面的程序中，我们已经明确地将`AssertionError`、`ValueError`和`NameError`与其他类型的异常区分开来，并且我们已经明确地处理了它们。在区分异常时必须记住，最一般的异常必须在 except 块中最后写入，最具体的异常必须在 except 块中首先写入。否则将会生成不可达的代码并导致错误。

例如，在 python 中使用 try-except 来区分异常的方法是错误的。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    raise IOError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except:
    print("Some other exception occurred")
except AssertionError:
    print("Input Correct age.")
except ValueError:
    print("ValueError Occured")
except NameError:
    print("NameError occurred") 
```

输出:

```py
 File "/usr/lib/python3/dist-packages/IPython/core/interactiveshell.py", line 3331, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)

  File "<ipython-input-4-f6b48848354a>", line 1, in <module>
    runfile('/home/aditya1117/untitled0.py', wdir='/home/aditya1117')

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 827, in runfile
    execfile(filename, namespace)

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 110, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "/home/aditya1117/untitled0.py", line 8
    print(yearOfBirth)
    ^
SyntaxError: default 'except:' must be last
```

我们可以看到，当我们在特定异常之前使用更一般的 except 块时，程序中会出现语法错误。

## 在 python 中使用 try-except 对异常进行分组

在编写程序时，可能会有很多情况，我们希望以相同的方式处理两个或更多的异常。我们可以在 python 中将某些类型的异常分组，然后通过将组中的每个异常作为参数传递给 except 块，由相同的错误处理代码来处理它们。

在下面的例子中，我们将`AssertionError`、`ValueError`和`NameError`分组在一个 except 块中，当任何错误发生时，将执行相同的代码。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    raise ValueError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except (AssertionError,ValueError,NameError):
    print("AssertionError or ValueError or NameError has occurred.")
except:
    print("Some other exception occurred")
```

## 最终块

在一个程序中，无论在什么情况下，每次程序终止前都可能有一些语句要执行。在这种情况下，我们可以将语句放在 Finally 块中。无论 try 块中是否出现任何错误/异常，Finally 块都会执行。当 try 块引发异常时，首先执行 except 块中的代码，然后执行 finally 块中的代码。否则，当没有发生任何错误时，在 try 块之后执行 Finally 块。

例如，当我们执行 [python 写文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)操作时，文件总是需要在程序终止前关闭，这样数据才不会丢失。在这种情况下，我们可以编写代码来关闭 Finally 块中的文件，如下所示。

```py
try:
    file_ptr=open("filename.txt","w")
    file_ptr.write("This is PythonForBeginners.com")

except (IOError):
    print("IOError has occurred.")
except:
    print("Some other exception occurred")
finally:
    file_ptr.close()
```

## 结论

在本文中，我们已经看到了如何在 python 中最好地使用 try-except，方法是应用像自定义异常这样的方法，对异常进行分组并区分不同的异常。请继续关注更多内容丰富的文章。