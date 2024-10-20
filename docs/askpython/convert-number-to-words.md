# 在 Python 中将数字转换为单词[一位数接一位数]

> 原文：<https://www.askpython.com/python/examples/convert-number-to-words>

在本教程中，我们将学习如何转换一个数字到它的措辞(数字方面)。例如，如果数字是 12，单词将是“1-2”。对于其余的输入，将进行类似的操作。

* * *

## 代码实现

我们将遵循下面提到的一些步骤:

* * *

### 步骤 1:创建数字到单词映射的全局列表

创建一个全局列表，包含从 0 到 9 的每个数字的单词。列表将包含映射到索引的元素，如下表所示。

| 索引 | Zero | one | Two | three | four | five | six | seven | eight | nine |
| 措辞/价值 | 零 | 一个 | 二 | 三 | 四 | 五 | 六 | 七 | 八 | 九 |

Global list for digit to word mapping

```py
# Global Array storing word for each digit
arr = ['zero','one','two','three','four','five','six','seven','eight','nine']

```

* * *

### 步骤 2:输入数字并创建主函数

为了输入数字，我们将使用`input`函数，然后将其转换为整数，我们还将创建一个空函数，将数字转换为单词。

```py
# Global Array storing word for each digit
arr = ['zero','one','two','three','four','five','six','seven','eight','nine']

def number_2_word(n):
    pass

n = int(input())
print("Number Entered was : ", n)
print("Converted to word it becomes: ",end="")
print(number_2_word(n))

```

* * *

### 步骤 3:编写函数内部的主要逻辑

对于这段代码，我们将利用**递归**。如果你对递归知之甚少或者一无所知，我建议你看看下面提到的教程:

***阅读更多关于递归的内容:[Python 中的递归](https://www.askpython.com/python/python-recursion-function)***

对于每个递归调用，我们将检查我的数字是否变成 0，如果是，我们将返回一个空字符串，否则我们将在**模数函数**的帮助下继续添加每个数字的单词，并将数字**除以 10** 以缩小数字并移动到下一个数字。

代码实现如下所示，为了便于理解，添加了一些注释。

```py
# Global Array storing word for each digit
arr = ['zero','one','two','three','four','five','six','seven','eight','nine']

def number_2_word(n):

    # If all the digits are encountered return blank string
    if(n==0):
        return ""

    else:
        # compute spelling for the last digit
        small_ans = arr[n%10]

        # keep computing for the previous digits and add the spelling for the last digit
        ans = number_2_word(int(n/10)) + small_ans + " "

    # Return the final answer
    return ans

n = int(input())
print("Number Entered was : ", n)
print("Converted to word it becomes: ",end="")
print(number_2_word(n))

```

* * *

**输出**:

```py
Number Entered was :  123
Converted to word it becomes: one two three

```

```py
Number Entered was :  46830
Converted to word it becomes: four six eight three zero 

```

* * *

## 结论

因此，在本教程结束时，我们看到，通过使用递归，可以很容易地将数字转换为单词(数字方式)。

感谢您的阅读！快乐学习！😇

* * *