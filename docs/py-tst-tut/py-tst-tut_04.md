# Fixtures

# Fixtures

### 练习 1：用于测试数据的模块

创建一个新模块`conftest.py`，其中包含一个包含大量特殊字符的句子的字符串变量：

```
sample = """That #§&%$* program still doesn't work!
I already de-bugged it 3 times, and still numpy.array keeps raising AttributeErrors. What should I do?""" 
```

创建一个函数，返回一个包含上述示例文本的`mobydick.TextCorpus`对象。使用以下内容作为标题：

```
@pytest.fixture
def sample_corpus():
    ... 
```

### 练习 2：使用 fixture

现在创建一个名为`test_sample.py`的模块，其中包含使用 fixture 的函数：

```
def test_sample_text(sample_corpus):
    assert sample_corpus.n_words == 77 
```

使用`pytest`执行模块。请注意，**不需要**导入`conftest`。Pytest 会自动处理。

### 练习 3：创建更多 fixtures

为`mobydick_full.txt`和`mobydick_summary.txt`文件中的两个文本语料库创建 fixtures。

### 练习 4：从 fixtures 修复 fixtures

在`conftest.py`中创建一个使用另一个 fixture 的 fixture：

```
from mobydick import WordCounter

@pytest.fixture
def counter(mobydick_summary):
    return WordCounter(mobydick_summary) 
```

编写一个简单的测试，确保 fixture 不是`None`
