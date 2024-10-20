# 如何用 Python 将十进制数转换成单词

> 原文：<https://www.blog.pythonlibrary.org/2012/06/02/how-to-convert-decimal-numbers-to-words-with-python/>

将这篇文章命名为“如何将浮点数转换成单词”可能是一个更好的主意，但是因为我谈论的是货币，所以我认为使用十进制更准确。总之，几年前，我[写过](https://www.blog.pythonlibrary.org/2010/10/21/python-converting-numbers-to-words/)关于如何将数字转换成 Python。我重提这个话题的主要原因是因为我最终需要再做一次，我发现我自己的例子相当缺乏。它没有展示如何实际使用它将类似“10.12”的东西转换为“十美元十二美分”。因此，我将在本文中向您展示如何做到这一点，然后我们还将看看我的读者给我的一些替代方案。

### 从头再来

首先，我们将获取原始代码，并在最后添加一些测试，以确保它按照我们想要的方式工作。然后我会告诉你一个稍微不同的方法。最后，我们将看看另外两个试图做这类事情的项目。

```py

'''Convert number to English words
$./num2eng.py 1411893848129211
one quadrillion, four hundred and eleven trillion, eight hundred and ninety
three billion, eight hundred and forty eight million, one hundred and twenty
nine thousand, two hundred and eleven
$

Algorithm from http://mini.net/tcl/591
'''

# modified to exclude the "and" between hundreds and tens - mld

__author__ = 'Miki Tebeka '
__version__ = '$Revision: 7281 $'

# $Source$

import math

# Tokens from 1000 and up
_PRONOUNCE = [
    'vigintillion',
    'novemdecillion',
    'octodecillion',
    'septendecillion',
    'sexdecillion',
    'quindecillion',
    'quattuordecillion',
    'tredecillion',
    'duodecillion',
    'undecillion',
    'decillion',
    'nonillion',
    'octillion',
    'septillion',
    'sextillion',
    'quintillion',
    'quadrillion',
    'trillion',
    'billion',
    'million',
    'thousand',
    ''
]

# Tokens up to 90
_SMALL = {
    '0' : '',
    '1' : 'one',
    '2' : 'two',
    '3' : 'three',
    '4' : 'four',
    '5' : 'five',
    '6' : 'six',
    '7' : 'seven',
    '8' : 'eight',
    '9' : 'nine',
    '10' : 'ten',
    '11' : 'eleven',
    '12' : 'twelve',
    '13' : 'thirteen',
    '14' : 'fourteen',
    '15' : 'fifteen',
    '16' : 'sixteen',
    '17' : 'seventeen',
    '18' : 'eighteen',
    '19' : 'nineteen',
    '20' : 'twenty',
    '30' : 'thirty',
    '40' : 'forty',
    '50' : 'fifty',
    '60' : 'sixty',
    '70' : 'seventy',
    '80' : 'eighty',
    '90' : 'ninety'
}

def get_num(num):
    '''Get token <= 90, return '' if not matched'''
    return _SMALL.get(num, '')

def triplets(l):
    '''Split list to triplets. Pad last one with '' if needed'''
    res = []
    for i in range(int(math.ceil(len(l) / 3.0))):
        sect = l[i * 3 : (i + 1) * 3]
        if len(sect) < 3: # Pad last section
            sect += [''] * (3 - len(sect))
        res.append(sect)
    return res

def norm_num(num):
    """Normelize number (remove 0's prefix). Return number and string"""
    n = int(num)
    return n, str(n)

def small2eng(num):
    '''English representation of a number <= 999'''
    n, num = norm_num(num)
    hundred = ''
    ten = ''
    if len(num) == 3: # Got hundreds
        hundred = get_num(num[0]) + ' hundred'
        num = num[1:]
        n, num = norm_num(num)
    if (n > 20) and (n != (n / 10 * 10)): # Got ones
        tens = get_num(num[0] + '0')
        ones = get_num(num[1])
        ten = tens + ' ' + ones
    else:
        ten = get_num(num)
    if hundred and ten:
        return hundred + ' ' + ten
        #return hundred + ' and ' + ten
    else: # One of the below is empty
        return hundred + ten

def num2eng(num):
    '''English representation of a number'''
    num = str(long(num)) # Convert to string, throw if bad number
    if (len(num) / 3 >= len(_PRONOUNCE)): # Sanity check
        raise ValueError('Number too big')

    if num == '0': # Zero is a special case
        return 'zero '

    # Create reversed list
    x = list(num)
    x.reverse()
    pron = [] # Result accumolator
    ct = len(_PRONOUNCE) - 1 # Current index
    for a, b, c in triplets(x): # Work on triplets
        p = small2eng(c + b + a)
        if p:
            pron.append(p + ' ' + _PRONOUNCE[ct])
        ct -= 1
    # Create result
    pron.reverse()
    return ', '.join(pron)

if __name__ == '__main__':

    numbers = [1.37, 0.07, 123456.00, 987654.33]
    for number in numbers:
        dollars, cents = [int(num) for num in str(number).split(".")]

        dollars = num2eng(dollars)
        if dollars.strip() == "one":
            dollars = dollars + "dollar and "
        else:
            dollars = dollars + "dollars and "

        cents = num2eng(cents) + "cents"
        print dollars + cents 
```

我们只关注测试程序的最后一部分。这里我们有一个列表，列出了我们在程序中运行的各种值，并确保它输出了我们想要的。请注意，我们有不到一美元的金额。这是我见过的一个边缘案例，因为我的雇主想用真实金额测试我们的代码，但不希望转移巨额资金。以下是一种略有不同的数据输出方式:

```py

temp_amount = 10.34
if '.' in temp_amount:
    amount = temp_amount.split('.')
    dollars = amount[0]
    cents = amount[1]
else:
    dollars = temp_amount
    cents = '00'

amt = num2eng.num2eng(dollars)
total = amt + 'and %s/100 Dollars' % cents
print total

```

在这种情况下，我们不把美分部分写成单词，而只是把数字写在一百以上。是的，我知道这很微妙，但这篇文章对我来说也是一个大脑垃圾场，所以下次我必须这样做时，我会在我的指尖上有所有的信息。

### 试用 PyNum2Word

在我发布了我的原创文章后，有人过来告诉我关于 PyNum2Word 项目的事情，以及我应该如何使用它。PyNum2Word 项目当时还不存在，但我决定这次尝试一下。遗憾的是，这个项目没有我能找到的文档。连一个自述文件都没有！另一方面，它声称可以为美国、德国、英国、欧盟和法国做货币。我以为德国、英国和法国都在欧盟，所以我不知道在他们现在都使用欧元的情况下，使用法郎等货币有什么意义。

无论如何，在我们的例子中，我们将使用下面的文件， **num2word_EN.py** ，来自他们的测试包。文件底部实际上有一个测试，与我构建的测试相似。事实上我的测试是基于他们的。让我们试着编辑这个文件，在他们的第二个列表中添加一个小于 1 的数字，比如 **0.45** ，看看这是否可行。下面是第二个列表的输出结果(为了简洁，我跳过了第一个列表的输出):

 `0.45 is zero point four five cents
0.45 is zero point four five
1 is one cent
1 is one
120 is one dollar and twenty cents
120 is one hundred and twenty
1000 is ten dollars
1000 is one thousand
1120 is eleven dollars and twenty cents
1120 is eleven hundred and twenty
1800 is eighteen dollars
1800 is eighteen hundred
1976 is nineteen dollars and seventy-six cents
1976 is nineteen hundred and seventy-six
2000 is twenty dollars
2000 is two thousand
2010 is twenty dollars and ten cents
2010 is two thousand and ten
2099 is twenty dollars and ninety-nine cents
2099 is two thousand and ninety-nine
2171 is twenty-one dollars and seventy-one cents
2171 is twenty-one hundred and seventy-one` 

它起作用了，但不是以我期望的方式。在美国，当我们谈论货币时，我们会称**0.45**“45 美分”而不是“0.45 美分”。当我研究这个的时候，我确实了解到其他一些国家的人们确实使用后一种术语。我觉得有趣的是，这个模块接受任何高于 100 的东西，并将其分为美元和美分。例如，注意 120 被翻译成“一美元二十美分”而不是“一百二十美元”。还要注意，上面写的是“二十美分”，不是“零点二零美分”。我不知道如何解释这种矛盾。如果你给它传递一个小于 100 的整数，它就能工作。因此，如果您让用户放入一个 float，您会希望像我前面所做的那样分解它:

```py

if '.' in temp_amount:
    amount = temp_amount.split('.')
    dollars = amount[0]
    cents = amount[1]
else:
    dollars = temp_amount
    cents = '00'

```

然后通过脚本传递每个部分以获得片段，然后将它们放在一起。

### 使用数字. py

我的另一个名叫埃里克·沃尔德的读者联系了我关于他的[数字. py 脚本](http://www.brainshell.org/numbers.py)。让我们看看这是怎么回事！

看一下代码，你会很快发现它不能处理 float，所以我们必须分解我们的 float 并分别传递美元和美分。我用几个不同的数字试了一下，似乎可以正确地转换它们。该脚本甚至在千位标记处添加了逗号。它没有在任何地方添加“和”字，但我现在不关心这个。

### 结论

所有这三种方法都需要某种包装器来添加“美元”和“美分”(或数字/100)单词，并将浮点数分成两部分。我认为 Eric 的代码非常简单，也是最好的文档。PyNum2Word 项目的代码也非常简洁，并且运行良好，但是没有文档。我很久以前找到的解决方案也能工作，但是我发现代码非常难看，不太容易阅读。我真的没有推荐，但我想我最喜欢埃里克的。如果你需要做多币种的灵活性，那么 PyNum2Word 项目值得一看。