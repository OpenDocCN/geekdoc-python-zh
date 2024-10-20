# 代码的出现:第 4 天-护照处理

> 原文：<https://www.blog.pythonlibrary.org/2020/12/09/advent-of-code-day-4-passport-processing/>

顾名思义， [Day 4](https://adventofcode.com/2020/day/4) 都是办理护照。您将看到以下字段，您必须使用这些字段执行一些验证步骤:

```py
byr (Birth Year)
iyr (Issue Year)
eyr (Expiration Year)
hgt (Height)
hcl (Hair Color)
ecl (Eye Color)
pid (Passport ID)
cid (Country ID)

```

#### ！！！前方剧透！！！

如果你还没有解决这个难题，你应该停止阅读！

#### 第一部分

挑战的第一部分是验证所有字段都存在。您将检查输入文件并统计所有有效护照。

我对这个问题相当简单的解决方案如下:

```py
def process(lines: str) -> bool:
    """
    Part 1
    """
    d = lines.split(' ')
    dd = dict([item.split(':') for item in d if item])
    keys = ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid']
    for key in keys:
        if key not in dd:
            return False
    return True

def count_passports(data: str) -> int:
    pport = {}
    s = ''
    count = 0
    passports = []
    for line in data.split('\n'):
        line = line.strip()
        if line == '':
            # process
            count += bool(process(s))
            passports.append(s)
            s = ''
        s += ' ' + line
    print(f"Valid passports: {count}")
    return count

if __name__ == '__main__':
    with open('input.txt') as f:
        data = f.read()
    count_passports(data)
```

在这里，我解析了这一行，并在我的 **process()** 函数中将它转换成一个 Python 字典。然后我检查了所有的钥匙，确认所有需要的都在。如果任何一个键不存在，那么我返回 **False。**

#### 第二部分

挑战的这一部分要复杂得多。现在，我必须对以下内容进行代码验证:

```py
byr (Birth Year) - four digits; at least 1920 and at most 2002.
iyr (Issue Year) - four digits; at least 2010 and at most 2020.
eyr (Expiration Year) - four digits; at least 2020 and at most 2030.
hgt (Height) - a number followed by either cm or in:
    If cm, the number must be at least 150 and at most 193.
    If in, the number must be at least 59 and at most 76.
hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
pid (Passport ID) - a nine-digit number, including leading zeroes.
cid (Country ID) - ignored, missing or not.

```

对于这个问题，我采用了最直接的方法，并使用了一系列条件语句:

```py
def process2(lines: str) -> bool:
    """
    Part 2
    """
    d = lines.split(' ')
    dd = dict([item.split(':') for item in d if item])

    keys = ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid']
    for key in keys:
        if key not in dd:
            return False

    if not any([dd[key] != 4 for key in ['byr', 'iyr', 'eyr']]):
        return False

    if '1920' > dd['byr'] or dd['byr'] > '2002':
        return False
    if dd['iyr'] < "2010" or dd['iyr'] > "2020":
        return False
    if dd['eyr'] < "2020" or dd['eyr'] > "2030":
        return False
    if not any(['in' in dd['hgt'], 'cm' in dd['hgt']]):
        return False
    if 'cm' in dd['hgt']:
        height = int(dd['hgt'][:-2])
        if height < 150 or height > 193:
            return False
    if 'in' in dd['hgt']:
        height = int(dd['hgt'][:-2])
        if height < 59 or height > 76:
            return False

    if '#' not in dd['hcl']:
        return False
    hcl = dd['hcl'].split('#')[-1]
    if len(hcl) != 6:
        return False

    try:
        int(hcl, 16)
    except ValueError:
        return False
    eye_colors = ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
    if dd['ecl'] not in eye_colors:
        return False
    if len(dd['pid']) != 9:
        return False

    if dd['pid'] == '182693366':
        print()
    return True

def count_passports(data: str) -> int:
    pport = {}
    s = ''
    count = 0
    passports = []
    for line in data.split('\n'):
        line = line.strip()
        if line == '':
            # process
            count += bool(process2(s))
            passports.append(s)
            s = ''
        s += ' ' + line
    print(f"Valid passports: {count}")
    return count

if __name__ == '__main__':
    with open('input.txt') as f:
        data = f.read()
    count_passports(data)
```

这是可行的，但这绝对是一个无聊的解决方案。我看到其他人使用[华丽的](https://pypi.org/project/voluptuous/)包来创建一个模式，然后用它来验证一切。您还可以创建一个也执行验证步骤的类。

如果你想看的话，我所有的代码也在 Github 上。