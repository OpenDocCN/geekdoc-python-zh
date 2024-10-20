# 使用 Python 创建更安全的密码

> 原文：<https://www.askpython.com/python/examples/create-safer-passwords>

你好，编码员们！在本教程中，我们将创建一个 python 程序，使您的密码更加安全可靠。

我们都知道，创建一个强大的密码在一个人的生活中起着重要的作用，以保持您的帐户和个人信息的安全，并防止它落入坏人之手。

简单的密码很容易被破解，所以我们需要让我们的密码难以破解。在这个应用程序中，我们将使用不同的特殊符号替换一串字符，例如$、& 、@、0、1、|，等等，以使您的密码难以破解。

该应用程序将您的密码作为用户的输入，然后用提到的特殊符号替换其字符，然后为用户打印新的更强密码的输出。

## 使用 Python 创建安全密码

为了使密码更加安全，我们将首先创建一个映射，存储需要替换的字符和特殊符号。

在下一步中，将创建一个函数，该函数将对用户输入的密码进行所有替换，然后返回更安全的密码。

```py
# 1\. Mapping characters with the secret codes
SECRET_CODE = [('g', '$'), ('t', '&'), ('a', '@'), ('s', '0'), ('h', '1'),('l', '|')]

# 2\. Creating a function which will return the stronger password
def secure_password(user_pass):
    for a,b in SECRET_CODE:
        user_pass = user_pass.replace(a, b)
    return user_pass

# 3\. Taking the input of the user password
cur_pass = input("Enter your current password: \n")

# 4\. Making the password more secure
new_pass = secure_password(cur_pass)
print(f"Your secure and safe password is {new_pass}")

```

## 一些示例输出

上面给出的代码将返回一个更安全的密码，同样的情况可以在下面给出的两个输出中看到。

```py
Enter your current password: 
This is my generic password
Your secure and safe password is T1i0 i0 my $eneric [email protected]

```

```py
Enter your current password: 
Please give me a strong password
Your secure and safe password is P|[email protected] $ive me @ 0&ron$ [email protected]

```

## 结论

您可以根据自己的喜好用更多的符号或数字来替换字符，并使密码更难破解。我希望您喜欢这个简单的应用程序。

感谢您的阅读！