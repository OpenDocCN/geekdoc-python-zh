# 如何在 Python 中获取电话号码信息？

> 原文：<https://www.askpython.com/python/examples/get-phone-number-information>

在本教程中，我们将看看如何在 Python 中获取电话号码信息。“phonenumbers”这个名字指的是一个非常有趣和方便的库。这是一个库，它将帮助我们在 Python 中体验电话号码的乐趣。

## 用 Python 获取电话号码信息的步骤

我们将利用 phonenumbers 库来获取有关电话号码的信息。让我们更深入地了解这一课。

首先，在命令提示符下运行下面一行来安装 phonenumbers 库。

```py
pip install phonenumbers

```

* * *

### 将字符串转换为电话号码格式

为了研究 phonenumbers 模块的功能，我们必须首先获得 phone number 格式的用户电话号码。在这一节中，我们将了解如何将用户的电话号码转换为 phone number 格式。

输入必须是字符串类型，国家代码必须在电话号码之前。

```py
import phonenumbers
pN = phonenumbers.parse("+919876643290")
print(pN)

```

```py
Country Code: 91 National Number: 9876643290

```

* * *

### 获取时区

下面是一个简单的 Python 程序，它使用 phonenumbers 模块来确定电话号码的时区。

首先，我们将字符串输入转换为电话号码格式，然后我们利用一个内置的方法来确定用户的时区。它只返回有效数字的结果。

```py
import phonenumbers
from phonenumbers import timezone
pN = phonenumbers.parse("+919876643290")
tZ = timezone.time_zones_for_number(pN)
print(tZ)

```

```py
('Asia/Calcutta',)

```

* * *

### 从文本中提取电话号码

使用这个模块，我们可以从文本/段落中提取电话号码。您可以遍历它来获得电话号码列表。PhoneNumberMatcher 对象为此提供了必要的方法。

```py
import phonenumbers
T = "Contact us at +919876643290 or +14691674587"
N = phonenumbers.PhoneNumberMatcher(T, "IN")
for n in N:
	print(n)

```

```py
PhoneNumberMatch [14,27) +919876643290

```

* * *

### 电话号码的运营商和地区

我们将学习如何使用本模块的地理编码器和运营商功能来确定电话号码的运营商和区域。

```py
import phonenumbers
from phonenumbers import geocoder, carrier
pN = phonenumbers.parse("+919876643290")
C = carrier.name_for_number(pN, 'en')
R = geocoder.description_for_number(pN, 'en')
print(C)
print(R)

```

```py
Airtel
India

```

* * *

## 结论

恭喜你！您刚刚学习了如何在 Python 中获取电话号码信息。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python:将数字转换成文字](https://www.askpython.com/python/python-convert-number-to-words)
2.  [在 Python 中把一个数字转换成单词【一个数字接一个数字】](https://www.askpython.com/python/examples/convert-number-to-words)
3.  [在 Python 中寻找最小数字的 3 种简单方法](https://www.askpython.com/python/examples/smallest-number-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *