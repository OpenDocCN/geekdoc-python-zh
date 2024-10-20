# Python:将数字转换成单词

> 原文：<https://www.blog.pythonlibrary.org/2010/10/21/python-converting-numbers-to-words/>

我在工作中的第一个自我强加的项目是重新创建一个令人讨厌的应用程序，它是一个弗兰肯斯坦怪物:一个带有 VBA 图形用户界面的微软 Access 文件。在很大程度上，该应用程序甚至没有数据库。无论如何，应用程序的一部分允许用户键入支票的金额，VBA 代码会神奇地将这些数字翻译成你通常在支票上写的文本。例如，假设我开了一张 1234.56 美元的支票。它会将其转换为“一千二百三十四美元五十六美分”。我需要用 Python 做同样的事情！

我花了相当多的时间试图想出正确的单词公式输入到谷歌，这将返回我所需要的。不幸的是，经过大量的劳动和疯狂的打字，我一无所获。我发现了一些在皮斯顿没有人做过的事情！！！或者那天我的 Google-fu 很糟糕，但我想是前者。

VBA 密码有一条线索，我已经记不起来了。我想是 VBA 图书馆的名字起了作用。总之，不管是什么原因，我找到了下面的代码(注意:下面的链接不再有效):

```py

#!/usr/bin/env python
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
    'million ',
    'thousand ',
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

#FIXME: Currently num2eng(1012) -> 'one thousand, twelve'
# do we want to add last 'and'?
def num2eng(num):
    '''English representation of a number'''
    num = str(long(num)) # Convert to string, throw if bad number
    if (len(num) / 3 >= len(_PRONOUNCE)): # Sanity check
        raise ValueError('Number too big')

    if num == '0': # Zero is a special case
        return 'zero'

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
    from sys import argv, exit
    from os.path import basename
    if len(argv) < 2:
        print 'usage: %s NUMBER[s]' % basename(argv[0])
        exit(1)
    for n in argv[1:]:
        try:
            print num2eng(n)
        except ValueError, e:
            print 'Error: %s' % e 
```

正如代码中的注释所指出的，我稍微修改了代码，以匹配 VBA 代码所做的。除此之外，和我发现的一模一样。我不会解释这一块，因为它是一种有趣的发现自己。希望你喜欢！