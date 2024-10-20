# 如何使用 Python 显示日期和时间

> 原文：<https://www.pythoncentral.io/how-to-display-the-date-and-time-using-python/>

如果您需要在任何 Python 项目中打印和显示日期和时间，那么执行这些操作的代码非常简单。要开始，您需要导入时间模块。从那以后，只有几行简短的代码妨碍了显示准确的当前日期和时间。看看下面的代码片段:

导入时间

## 24 小时格式# #
print(time . strftime(" % H:% M:% S "))

## 12 小时格式# #
print(time . strftime(" % I:% M:% S "))

## dd/mm/yyyy 格式
打印(time.strftime("%d/%m/%Y "))

## mm/dd/yyyy 格式
打印(time.strftime("%m/%d/%Y "))

## dd/mm/yyyy hh:mm:ss 格式
print(time . strftime(' % d/% M/% Y % H:% M:% S '))

## mm/dd/yyyy hh:mm:ss 格式
print(time . strftime(' % M/% d/% Y % H:% M:% S '))

在上面的例子中，你会看到 4 种不同的显示日期和时间的格式选项。对于时间，有 24 小时格式，这是指包括所有 24 小时的时钟，而不是一天重复两次的 12 小时时钟，也有 12 小时格式的选项。有一个示例演示了如何以 dd/mm/yyyy 格式显示日期，还有一个示例演示了如何以 mm/dd/yyyy 格式显示日期，这是美国最流行的日期格式。

最后，有一个例子以日/月/年，小时:分钟:秒的格式组合了日期和时间(还有一个例子是一天之前的月份，适用于美国的程序员)。您可以随意将这些添加到您想要轻松快速显示日期和时间的任何 Python 项目中。