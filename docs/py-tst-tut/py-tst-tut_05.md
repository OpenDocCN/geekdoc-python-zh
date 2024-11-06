# 参数化测试

# 参数化测试

### 练习 1：示例数据集

你有一个适用于文本文件`mobydick_summary.txt`的成对列表（单词，计数）：

```
PAIRS = [
         ('months', 1),
         ('whale', 5),
         ('captain', 4),
         ('white', 2),
         ('harpoon', 1),
         ('goldfish', 0)
] 
```

我们将从这些样本创建六个测试。

不要手动创建六个测试，我们将使用**pytest 中的测试参数化**。编辑文件`test_parameterized.py`，并将以下装饰器添加到测试函数中：

```
@pytest.mark.parametrize('word, number', PAIRS) 
```

在函数头部添加两个参数`word`和`number`，并删除下面的赋值。

运行测试，确保所有六个测试都通过。

### 练习 2：编写另一个参数化测试

函数**get_top_words()**计算文本语料库中最常见的单词。它应该为书籍**mobydick_full.txt**产生以下前五个结果：

| position | word |
| --- | --- |
| 1. | of |
| 2. | the |
| 3. | is |
| 4. | sea |
| 5. | ship |

编写一个参数化测试，检查这五个位置。
