# 更新日志

> 原文：[`www.thedigitalcatbooks.com/pycabook-changelog/`](https://www.thedigitalcatbooks.com/pycabook-changelog/)
> 
> 你最后记得的是什么？嗯？
> 
> 外星人，1979

我将在这里跟踪书籍发布之间的变化，遵循 [语义版本控制](https://semver.org/)。主版本号的变化意味着不兼容的变化，即书籍的大幅重写，也称为第 2 版、第 3 版等。次版本号的变化意味着内容中添加了某些重要内容，如新章节或部分。补丁号的变化表示文本或代码中的小修复，如错别字、句子改写等。

**当前版本**：2.1.2

### 版本 2.1.2 (2022-01-06)

+   在 Mau 代码中的一项修复防止了在合并多个文件时 Markua 访问者中的脚注冲突

### 版本 2.1.1 (2021-10-19)

+   对脚注和链接进行全局审查，以适应 Mau v2.0

### 版本 2.1.0 (2021-08-20)

+   本版本是用 Mau 编写的，但转换为 Markua 以使用 Leanpub 的处理链发布 PDF。

+   第八章通过迁移得到了改进，这些迁移可以正确地在生产数据库中创建表格。

+   [Maxim Ivanov](https://github.com/ivanovmg) 在书籍和代码仓库中修复了许多错误，并解决了两者之间的一些不一致性。了不起的工作，非常感谢您的帮助！

+   GitHub 用户 [robveijk](https://github.com/robveijk) 发现了一个提及的文件并未包含在第二版中。谢谢！

+   GitHub 用户 [mathisheeren](https://github.com/mathisheeren) 修正了一个错别字。谢谢！

+   GitHub 用户 [4myhw](https://github.com/4myhw) 发现了一个损坏的链接，并修复了代码中使用 `self` 而不是 `cls` 的错误。谢谢！

+   几个人，特别是 [Jakob Waibel](https://github.com/JakWai01) 发现了罗伯特·马丁的名字中的错别字。感谢所有人，并对马丁先生表示歉意。

+   GitHub 用户 [1110sillabo](https://github.com/1110sillabo) 指出，基于 AsciiDoctor 的工具链创建 PDF 时并不完美，已通过回到 Lanpub 的 Markua 进行修复。

+   [Giovanni Natale](https://github.com/gnatale) 在代码和文本中发现了几个问题，并友好地提交了建议和修复。谢谢！

### 版本 2.0.1 (2021-02-14)

+   GitHub 用户 [1110sillabo](https://github.com/1110sillabo) 和不知疲倦的 [Faust Gertz](https://github.com/soulfulfaust) 友好地提交了一些 PR 来修复错别字。谢谢！

+   首个版本从 Mau 源转换为 Asciidoctor

### 版本 2.0.0 (2020-12-30)

+   书籍结构的重大重构

+   HTML 版本

+   介绍性示例，概述了系统组件

+   一些不错的图表

+   管理脚本以编排 Docker

+   在随机位置添加了许多错别字

### 版本 1.0.12 (2020-04-13)

+   GitHub 用户 [Vlad Blazhko](https://github.com/pisarik) 在项目 `fileinfo` 中发现了一个错误，并添加了一个修复和一个测试条件。因此，我扩展了关于模拟的章节，增加了一个小节来描述他所做的工作。非常感谢 Vlad！

### 版本 1.0.11 (2020-02-15)

+   GitHub 用户 [lrfuentesw](https://github.com/lrfuentesw) 在内存仓库中发现了错误。字符串值的定价过滤器没有工作，因为它们没有被转换为整数。谢谢！

### 版本 1.0.10 (2019-09-23)

+   GitHub 用户 [Ramces Chirino](https://github.com/chirinosky) 提交了一个大型的 PR，修复了许多语法错误。谢谢！

### 版本 1.0.9 (2019-04-12)

+   GitHub 用户 [plankington](https://github.com/plankington) 修复了一些错误。谢谢！

### 版本 1.0.8 (2019-03-19)

+   GitHub 用户 [Faust Gertz](https://github.com/faustgertz) 和 [Michael "Irish" O'Neill](https://github.com/IrishPrime) 在第一部分的第一章的示例 `calc` 代码中发现了错误。谢谢！

+   GitHub 用户 [Ahmed Ragab](https://github.com/Ragabov) 修复了一些错误。非常感谢！

### 版本 1.0.7 (2019-03-02)

+   GitHub 用户 [penguindustin](https://github.com/penguindustin) 建议在工具部分添加 `pipenv`，因为它在 Python 打包用户指南中被官方推荐。谢谢！

+   GitHub 用户 [godiedelrio](https://github.com/godiedelrio) 在文件 `rentomatic/rentomatic/repository/postgresrepo.py` 中发现了一个错误。代码在查询结果返回时没有将单个对象转换为领域实体。由于我还没有引入检查返回对象性质的测试，因此这个错误没有被测试发现。

### 版本 1.0.6 (2019-02-06)

+   无私的 [Eric Smith](https://github.com/genericmoniker) 在第二部分的第四章中修复了错误和语法。非常感谢。

### 版本 1.0.5 (2019-01-28)

+   [Eric Smith](https://github.com/genericmoniker) 和 [Faust Gertz](https://github.com/faustgertz) 在第二部分中修复了许多错误。谢谢你们两位的帮助。

### 版本 1.0.4 (2019-01-22)

+   [Grant Moore](https://github.com/grantmoore3d) 和 [Hans Chen](https://github.com/hanschen) 修正了两个错误。谢谢你们！

### 版本 1.0.3 (2019-01-11)

+   [Eric Smith](https://github.com/genericmoniker) 修复了更多错误，并在第一部分的第三章中纠正了一些措辞。谢谢 Eric！

### 版本 1.0.2 (2019-01-09)

+   [Max H. Gerlach](https://github.com/maxhgerlach) 发现并修复了更多错误。再次感谢 Max！

### 版本 1.0.1 (2019-01-01)

+   [Max H. Gerlach](https://github.com/maxhgerlach)、[Paul Schwendenman](https://github.com/paul-schwendenman) 和 [Eric Smith](https://github.com/genericmoniker) 乐于修复了许多错误和语法错误。非常感谢！

### 版本 1.0.0 (2018-12-25)

+   初次发布
