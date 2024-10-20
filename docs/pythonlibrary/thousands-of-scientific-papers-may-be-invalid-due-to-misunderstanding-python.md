# 数千篇科学论文可能因误解 Python 而无效

> 原文：<https://www.blog.pythonlibrary.org/2019/10/13/thousands-of-scientific-papers-may-be-invalid-due-to-misunderstanding-python/>

最近发现，数千篇科学文章的结论可能是无效的，因为科学家不理解 Python 的 **glob.glob()** 不返回排序结果。

这是由 [Vice](https://www.vice.com/en_us/article/zmjwda/a-code-glitch-may-have-caused-errors-in-more-than-100-published-studies) 、 [Slashdot](https://science.slashdot.org/story/19/10/12/1926252/python-code-glitch-may-have-caused-errors-in-over-100-published-studies) 报道的，在 [Reddit](https://www.reddit.com/r/Python/comments/dh2qwd/more_than_100_scientific_articles_may_be/) 上也有一场有趣的讨论。

有些人认为这是 Python 中的一个小故障，但是 **glob** 从未保证返回的结果是经过排序的。和往常一样，我建议仔细阅读文档，以充分理解代码的作用。如果你能围绕你的代码写测试也是一个好主意。Python 包含了一个 [unittest](https://docs.python.org/3/library/unittest.html) 模块，使得这变得更加容易。