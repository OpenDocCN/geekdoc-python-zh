# 代码 2020 的来临:第二天

> 原文：<https://www.blog.pythonlibrary.org/2020/12/04/advent-of-code-2020-day-2/>

《代码 2020》问世的第二天是关于验证密码的。它为您提供一系列密码策略和密码，然后要求您确定这些密码中有多少是有效的。

对于这个挑战，我选择不使用 Python 库，只进行文本解析。

#### ！！！前方剧透！！！

如果你还没有解决这个难题，你应该停止阅读！

#### 第一部分

对于我的解决方案，我最终创建了两个函数。一个用于验证密码，一个用于计算有效密码的数量。这些本来是可以合并的，但是我的最佳编码实践接管了它，我阻止自己在一个函数中放太多东西。

代码如下:

```py
def verify_password(line: str) -> bool:
    policy, password = [item.strip() for item in line.split(':')]
    bounds = policy[:-2]
    letter = policy[-1]
    low, high = [int(item) for item in bounds.split('-')]

    letter_count = password.count(letter)
    if low <= letter_count <= high:
        return True
    return False

def count_good_passwords(data: list) -> int:
    good_passwords = 0
    for line in data:
        if verify_password(line):
            good_passwords += 1
    print(f'Number of good password: {good_passwords}')
    return good_passwords

if __name__ == "__main__":
    data = []
    with open('passwords.txt') as f:
        for line in f:
            data.append(line.strip())
    count_good_passwords(data)
```

这段代码将解决问题的第一部分。在这里打开文件，提取每一行，去掉两端的空格。一旦你有了字符串列表，你就把它们传递给你的函数，这个函数计算好的密码的数量。

操作的大脑在`verify_password()`中，它使用列表理解将行分解成密码策略和密码本身。然后你根据策略做一些检查，如果好就返回**真**，如果不好就返回**假**。

#### 第二部分

在第 2 部分中，必须对密码策略进行不同的解释。与其在这里重复问题陈述，你应该去[检查一下](https://adventofcode.com/2020/day/2)，看看两者有何不同。

为了实现这一点，我更新了两个函数，以接受一个**版本的**参数:

```py
def verify_password(line: str, version: int = 1) -> bool:
    policy, password = [item.strip() for item in line.split(':')]
    bounds = policy[:-2]
    letter = policy[-1]
    low, high = [int(item) for item in bounds.split('-')]

    if version == 1:
        letter_count = password.count(letter)
        if low <= letter_count <= high:
            return True
    elif version == 2:
        letters = [password[low-1], password[high-1]]
        if letters.count(letter) == 1:
            return True
    return False

def count_good_passwords(data: list, version: int = 1) -> int:
    good_passwords = 0
    for line in data:
        if verify_password(line, version):
            good_passwords += 1
    print(f'Number of good password: {good_passwords}')
    return good_passwords

if __name__ == "__main__":
    data = []
    with open('passwords.txt') as f:
        for line in f:
            data.append(line.strip())

    count_good_passwords(data, version=2)
```

代码基本相同，只是现在它使用一个条件语句以不同的方式检查密码策略。第一个版本对字母进行计数，而第二个版本检查字母的位置。

总的来说，这是一个非常直接的变化。

我所有的代码也在 Github 上，这个包含了几个单元测试，你可以去看看。