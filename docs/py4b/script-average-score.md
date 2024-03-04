# Python 代码:计算平均分

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/script-average-score>

## 概观

```py
 This script will calculate the average of three values. 

Make sure to put in "int" before the raw_input function, since we are
using integers. 
```

```py
# Get three test score
round1 = int(raw_input("Enter score for round 1: "))

round2 = int(raw_input("Enter score for round 2: "))

round3 = int(raw_input("Enter score for round 3: "))

# Calculate the average
average = (round1 + round2 + round3) / 3

# Print out the test score
print "the average score is: ", average 

```

```py
 Happy Scripting 
```