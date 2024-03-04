# Python 中的基本日期和时间类型

> 原文：<https://www.pythonforbeginners.com/basics/python-strftime-and-strptime>

## 日期和时间

date、datetime 和 time 对象都支持 strftime(format)方法，以便在显式格式字符串的控制下创建表示时间的字符串。

以下是格式代码及其指令和含义的列表。

```py
 %a 	Locale’s abbreviated weekday name.
%A 	Locale’s full weekday name. 	 
%b 	Locale’s abbreviated month name. 	 
%B 	Locale’s full month name.
%c 	Locale’s appropriate date and time representation. 	 
%d 	Day of the month as a decimal number [01,31]. 	 
%f 	Microsecond as a decimal number [0,999999], zero-padded on the left
%H 	Hour (24-hour clock) as a decimal number [00,23]. 	 
%I 	Hour (12-hour clock) as a decimal number [01,12]. 	 
%j 	Day of the year as a decimal number [001,366]. 	 
%m 	Month as a decimal number [01,12]. 	 
%M 	Minute as a decimal number [00,59]. 	 
%p 	Locale’s equivalent of either AM or PM.
%S 	Second as a decimal number [00,61].
%U 	Week number of the year (Sunday as the first day of the week)
%w 	Weekday as a decimal number [0(Sunday),6]. 	 
%W 	Week number of the year (Monday as the first day of the week)
%x 	Locale’s appropriate date representation. 	 
%X 	Locale’s appropriate time representation. 	 
%y 	Year without century as a decimal number [00,99]. 	 
%Y 	Year with century as a decimal number. 	 
%z 	UTC offset in the form +HHMM or -HHMM.
%Z 	Time zone name (empty string if the object is naive). 	 
%% 	A literal '%' character.
```

## strftime() vs strptime()

strptime()–字符串“解析”时间–用于将字符串转换为日期/时间对象。使用它将日期字符串解析为日期/时间对象。

strftime()–字符串“格式”时间–用于格式化日期对象。当您想要格式化日期时，请使用此选项。