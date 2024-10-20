# 根据 Python 中的种子关键字自动建议用户名

> 原文：<https://www.askpython.com/python/examples/auto-suggest-usernames>

在本教程中，我们将学习如何通过在 Python 中添加约束来建议用户名。在生成用户名之前，我们已经设置了以下约束:

1.  至少两个大写字母
2.  仅允许使用特殊字符`.`、`-`和`_`
3.  至少有 3 位数字

## Python 中的自动建议用户名

为了让它对用户更有意义，我们将首先从用户那里获取输入数据，在输入数据的基础上，我们将向他们建议一个用户名。让我们一步步来看完整的代码。

### 步骤 1:启动用户名

现在用户名大多以“#”或“@”开头。我们将保留用户名开头的标签(#)。你可以保留任何你想要的符号。

### 步骤 2:获取用户信息

显然，我们希望用户名对用户有某种意义，用户必须以某种方式与它联系起来，使他们更容易记住。

人们可以获得的关于一个人的最简单的信息是他的名字，这里我们将考虑用户的全名。

### 步骤 3:添加约束

下一步，我们将在用户名生成代码中添加以下约束。

#### 1.至少两个大写字母

我们将使用名和姓的首字母来满足这个约束。

#### 2.特殊字符添加

只允许 3 个特殊字符，即“.”，'-'和' _ '。

所以在大写字母后面，我们会插入一个特殊的字符。您可以将字符放在您希望的任何位置，只是改变语句的顺序。

#### 3.至少三个数字和一些随机的小写字母

最后一个约束是小写字母和至少三位数字的组合。

小写字母的数量取决于用户名的长度，在我们的例子中，我们将用户名的长度保持为 10。

到目前为止，在这 10 个字符中，有四个字符已经被' # '填充，两个大写字符和一个特殊字符。

为了简化小写字母，我们将从用户名的剩余字母中选择随机字符。我们将从 0 到 9 中随机选择三个数字。

我们将保留用户名的最终顺序，如下所示。

```py
# + 2 Uppercase characters + . or - or _ + 3 Lowercase characters + 3 Digits

```

## 使用 Python 自动建议用户名[实现]

完整的代码实现如下所示，为了便于理解，添加了一些注释。

```py
# Taking input of name of the user
name  = input("Enter your full name: ")

# Initializing the username
username = "#"

# 1\. First two uppercase letter
l = name.split()
# Name must have both first and last name
while(len(l)!=2):
    name = input("Enter full name please: ")
    l = name.split()
username += l[0][0].upper()
username+=l[1][0].upper()

# 2\. Adding special character ( . , _ or -)
import random
choice = random.choices(".-_", k=1)
username += choice[0]

# 3\. Atleast three digits : The 3 digits chosen ( will be added after lowecase letters)
digits_chosen = random.choices("0123456789",k=3) 

# 4\. Lowercase letters ( 3 )
remaining = l[0][1:] + l[1][1:]
letters_chosen = random.choices(remaining,k=3)

# 5\. Include the three lower and then three digits
username = username+  letters_chosen[0] + letters_chosen[1] + letters_chosen[2]
username = username + digits_chosen[0] + digits_chosen[1] + digits_chosen[2]

print("The Final Username Generated is: ", username)

```

## 输出

针对一些随机输入对代码进行了测试。你可以自己看一看。

```py
Enter your full name: Isha Bansal
The Final Username Generated is:  #IB-sha914

```

如果用户没有输入它的全名，程序会要求再次输入。

```py
Enter your full name: Kartik
Enter full name please: Kartik Gupta
The Final Username Generated is:  #KG_iat397

```

我希望你理解了问题的逻辑和实现。您可以根据自己的偏好设置和更改约束。

感谢您的阅读！