# Python 代码:将 KM/H 转换为 MPH

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-code-convert-kmh-to-mph>

## 将公里/小时转换为英里/小时

```py
 This script converts speed from KM/H to MPH, which may be handy for calculating
speed limits when driving abroad, especially for UK and US drivers. 

The conversion formula for kph to mph is : 1 kilometre = 0.621371192 miles 
```

```py
#!/usr/bin/env python
kmh = int(raw_input("Enter km/h: "))
mph =  0.6214 * kmh
print "Speed:", kmh, "KM/H = ", mph, "MPH"

```