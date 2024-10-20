# 如何解决 Python 中的替换和删除问题？

> 原文：<https://www.askpython.com/python/examples/replace-and-remove-problem>

你好编码器！所以在本教程中，我们将理解一个简单的问题。问题的名称是`Replace and Remove Problem`,我们将用不同的字符串替换一个特定的字符，并从用户输入中删除一个特定的字符。

因此，我们知道需要用不同的字符串或字符组替换一个字符，并从输入中删除一个字符。我们要遵循的两条规则如下:

1.  用双 d ( `dd`)替换`a`
2.  删除任何出现的`b`

## 解决方案实施

我们将遵循下面提到的一些步骤:

*   步骤 1:输入“N”(初始字符串的输入)
*   步骤 2:将字符串转换成字符列表(字符数组)
*   步骤 3:遍历字符数组
    *   步骤 3.1:如果“a”出现在图片中，则将其更改为“dd”
    *   步骤 3.2:如果“b”出现在图片中，则将其从字符数组中移除
*   步骤 4:将更新后的字符数组加入到原始字符串中，并打印输出

现在我们已经理解了解决问题的方法，让我们一步一步地进入实现部分。

### 步骤 1 和 2:取 N 的输入，并将其转换为字符数组

使用`input`函数在 Python 中获取输入，然后使用 `list`函数创建字符数组，该函数将输入字符串作为参数。

```py
# 1\. Taking input
n = input()
# 2\. Convert into a list of characters
l= list(n)

```

### 步骤 3:遍历数组，按照规则替换和删除字符

现在我们有了我们的字符数组，我们将遍历列表，每当获得字符`a`时，我们用 dd 替换它，每当遇到`b`时，我们将从字符数组中删除该字符。

为了替换字符，我们将直接改变数组中的字符位置，为了从数组中删除一个字符，我们使用了`remove`函数。

```py
# Rule 1 : Replace 'a' with 'dd'
# Rule 2 : Remove each 'b'

# Iterate over each character
i = len(l)-1
while(i!=-1):

    # Rule 1
    if(l[i] == 'a'):
        l[i] = 'dd'

    # Rule 2
    elif(l[i] == 'b'):
        l.remove(l[i])
    i = i-1

```

### 步骤 4:加入新的更新的字符数组

最后一步是将更新后的字符数组的所有元素连接成一个字符串。更好的选择是改变作为输入的原始字符串。我们用来实现的函数是`join`函数。

```py
# Join the updated list
n = ''.join(l)
print("New string is: ",n)

```

我们走吧！您的解决方案已经完成！现在让我们看看一些随机样本输出。

## 最终代码

```py
# 1\. Taking input
n = input()
# 2\. Convert into a list of characters
l= list(n)

print("String entered by user is: ",n)

# Rule 1 : Replace 'a' with 'dd'
# Rule 2 : Remove each 'b'

# Iterate over each character
i = len(l)-1
while(i!=-1):

    # Rule 1
    if(l[i] == 'a'):
        l[i] = 'dd'

    # Rule 2
    elif(l[i] == 'b'):
        l.remove(l[i])
    i = i-1

# Join the updated list
n = ''.join(l)
print("New string is: ",n)

```

## 输出

```py
String entered by user is:  abccba
New string is:  ddccdd

```

```py
String entered by user is:  abccbabdgsea
New string is:  ddccdddgsedd

```

## 结论

我希望您理解了问题和解决方案以及代码实现。你可以自己完成所有的事情！感谢您的阅读！编码快乐！