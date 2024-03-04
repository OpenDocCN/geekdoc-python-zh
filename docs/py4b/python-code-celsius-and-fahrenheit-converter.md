# Python 代码:摄氏和华氏转换器

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-code-celsius-and-fahrenheit-converter>

## 概观

这个脚本将华氏温度转换为摄氏温度。要创建摄氏和华氏的 python 转换器，首先必须找出要使用哪个公式。

华氏温度到摄氏温度的公式:

(F–32)x 5/9 = C 或者说白了，先减去 32，再乘以 5，再除以 9。

摄氏至华氏公式:

(C × 9/5) + 32 = F 或者说白了，乘以 9，再除以 5，再加 32。

将华氏温度转换为摄氏温度

```py
#!/usr/bin/env python
Fahrenheit = int(raw_input("Enter a temperature in Fahrenheit: "))

Celsius = (Fahrenheit - 32) * 5.0/9.0

print "Temperature:", Fahrenheit, "Fahrenheit = ", Celsius, " C" 
```

将摄氏温度转换为华氏温度

```py
 #!/usr/bin/env python
Celsius = int(raw_input("Enter a temperature in Celsius: "))

Fahrenheit = 9.0/5.0 * Celsius + 32

print "Temperature:", Celsius, "Celsius = ", Fahrenheit, " F" 
```

阅读并理解剧本。快乐脚本！