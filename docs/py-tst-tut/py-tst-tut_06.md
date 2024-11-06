# 测试命令行程序

# 测试命令行程序

### 练习 1：测试命令行应用程序

程序**word_counter.py**可以从命令行使用来计算最频繁出现的单词：

```
python word_counter.py mobydick_summary.txt 
```

命令行应用程序也需要进行测试。你可以在**test_commandline.py**中找到测试。

你的任务是确保命令行测试通过。

### 练习 2：测试命令行选项

程序**word_counter.py**计算测试文件中最频繁出现的单词。可以从命令行使用它来计算前五个单词：

```
python word_counter.py moby_dick_summary.txt 5 
```

你的任务是为该程序开发一个新的测试。

### 练习 3：用户验收

任何软件的最终测试是你的用户能否完成他们需要完成的任务。

你的任务是*手动*使用程序**word_counter.py**来找出梅尔维尔在书籍《白鲸》的全文中更频繁地使用*'whale'*还是*'captain'*。

**用户验收测试不能被机器替代。**
