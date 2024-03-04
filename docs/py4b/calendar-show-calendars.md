# Python 日历:显示日历

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/calendar-show-calendars>

这个脚本将要求输入一年。然后，它将接受该输入并返回整年的日历。

```py
import calendar

print "Show a given years monthly calendar"

print ''

year = int(raw_input("Enter the year"))

print ''

calendar.prcal(year)

print '' 
```

在脚本中添加一个 while 循环，再尝试一年。

```py
import calendar

while True:
    print "Show a given years monthly calendar"

    print ''

    year = int(raw_input("Enter the year"))

    print ''

    calendar.prcal(year)

    print ''

    raw_input("Press enter to go on ...") 
```

有关日历模块的更多信息，请参见[官方](https://docs.python.org/2/library/calendar.html "python")文档。