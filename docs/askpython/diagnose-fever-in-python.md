# 用 Python 诊断发热[简易 CLI 方法]

> 原文：<https://www.askpython.com/python/diagnose-fever-in-python>

嘿编码器！在本教程中，我们将了解一个常见的 Python 编程问题，你能使用 Python 编程语言诊断发烧吗？

发烧是指体温高于正常水平。正常体温因人而异，但通常在 98.6 华氏度(37 摄氏度)左右。发烧不是一种疾病。这通常是一个迹象，表明你的身体正在努力对抗疾病或感染。

## 用 Python 实现发热检测

我们将首先询问用户，他们输入的温度是摄氏度还是华氏度。这可以对决策产生重大影响。现在我们将检查输入是 C 还是 F，或者是否有错误的输入。

```py
temp = input("Would you like to enter your temperature in Celcius or Fahrenheit: ")
if temp.upper() == "C":
    pass
elif temp.upper() == "F":
    pass
else:
    pass

```

让我们一个接一个地寻找最终的代码。第一块是输入的温标为“C”时。在这种情况下，用户可以输入温度，如果温度大于或等于 37.8，则该人发烧。否则，这个人没有发烧。为了更好的诊断，温度被转换为浮动。看看下面的代码。

```py
temp = input("Would you like to enter your temperature in Celcius or Fahrenheit: ")
if temp.upper() == "C":
    result = input("Enter your body temprature in Celcuis: ")
    r = float(result)
    if r >= 37.8:
        print("You've a fever")
    else:
        print("You don't have a fever")
elif temp.upper() == "F":
    pass
else:
    pass

```

下一个模块是当输入为‘F’时。在这种情况下，阈值温度为 98.6。其余同上。接受输入并将输入转换为浮点以便更好地分析。请看下面的代码片段。

```py
temp = input("Would you like to enter your temperature in Celcius or Fahrenheit: ")
if temp.upper() == "C":
    result = input("Enter your body temprature in Celcuis: ")
    r = float(result)
    if r >= 37.8:
        print("You've a fever")
    else:
        print("You don't have a fever")
elif temp.upper() == "F":
    result1 = input("Enter your body temprature in Fahrenheit:")
    r1 = float(result1)
    if r1 >= 98.6:
        print("You've a fever")
    else:
        print("You don't have a fever")
else:
    pass

```

我们遇到的最后一个障碍是用户输入错误。在这种情况下，一个简单的语句作为输出被打印出来。看看下面的代码。

```py
temp = input("Would you like to enter your temperature in Celcius or Fahrenheit: ")
if temp.upper() == "C":
    result = input("Enter your body temprature in Celcuis: ")
    r = float(result)
    if r >= 37.8:
        print("You've a fever")
    else:
        print("You don't have a fever")
elif temp.upper() == "F":
    result1 = input("Enter your body temprature in Fahrenheit:")
    r1 = float(result1)
    if r1 >= 98.6:
        print("You've a fever")
    else:
        print("You don't have a fever")
else:
    print("Please enter the correct input")

```

## Python 中发热检测的完整代码

```py
temp = input("Would you like to enter your temperature in Celcius or Fahrenheit: ")
if temp.upper() == "C":
    result = input("Enter your body temprature in Celcuis: ")
    r = float(result)
    if r >= 37.8:
        print("You've a fever")
    else:
        print("You don't have a fever")
elif temp.upper() == "F":
    result1 = input("Enter your body temprature in Fahrenheit:")
    r1 = float(result1)
    if r1 >= 98.6:
        print("You've a fever")
    else:
        print("You don't have a fever")
else:
    print("Please enter the correct input")

```

## 一些样本输出

```py
Would you like to enter your temperature in Celcius or Fahrenheit: C
Enter your body temprature in Celcuis: 100
You've a fever

Would you like to enter your temperature in Celcius or Fahrenheit: F
Enter your body temprature in Fahrenheit:56
You don't have a fever

Would you like to enter your temperature in Celcius or Fahrenheit: j
Please enter the correct input

```

## 结论

在本教程中，我们学习了如何使用 Python 编程语言来诊断发烧。如果你喜欢这个教程，我相信你也会喜欢下面的！

1.  [Python 中的天气应用| Tkinter–GUI](https://www.askpython.com/python/examples/gui-weather-app-in-python)
2.  [Python Tkinter:摄氏到华氏转换器](https://www.askpython.com/python-modules/tkinter/celsius-to-fahrenheit-converter)
3.  [Python:将数字转换成文字](https://www.askpython.com/python/python-convert-number-to-words)
4.  [Python 中的误差线介绍](https://www.askpython.com/python/examples/error-bars-in-python)

感谢您的阅读！编码快乐！😁