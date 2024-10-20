# Prisma 对象关系映射的优势

> 原文：<https://www.pythoncentral.io/benefits-of-object-relational-mapping-with-prisma/>

[![Object Relational Mapping](img/dcd23ff8ac9342d621ebf2cbf0b844ab.png)](https://www.pythoncentral.io/wp-content/uploads/2022/12/data.jpg)

在使用数据库时，经常会出现是使用结构化查询语言(SQL)还是对象关系映射(ORM)的问题。因为开发人员需要维护各种系统和服务之间的信任链，所以 ORM 通常是最佳选择，较少依赖开发人员的 SQL 知识。

在构建 web 应用程序时，开发人员可以从利用对他们需要的平台、库和工具提供本机支持的开发环境中获得巨大的好处。这些环境中的应用程序开发通常通过自动代码完成来简化，从而加速开发并提高代码的语法正确性。在行业中，开发人员经常被本地支持 [Nestjs 和 Prisma](https://amplication.com/blog/build-a-nodejs-graphql-api-with-nestjs-and-prisma) 的开发环境所吸引，因为它提供了更高级别的兼容性，并减少了由开发人员的简单错误导致的错误。

# *引入对象关系映射的概念。*

ORM 是开发人员使用的技术，允许他们使用面向对象的编程范例从数据存储中检索和修改数据。ORM 与处理对象有着密切的关系，可用于开发人员可能使用的任何编程语言，例如 [Python](https://medium.com/pragmatech/orm-for-python-b63cfbc39e7f) 。

映射器生成可引用的编程对象，这些对象虚拟地映射数据存储中的所有数据表。然后，开发人员将利用这些编程对象与数据进行交互。主要方法是尝试减轻开发人员设计和开发复杂 SQL 查询和存储过程的任务，以访问数据。

虽然使用较小对象集合的项目可能不会从已安装的 ORM 库中受益，但是更大更复杂的项目将会从这些库的可用性中受益匪浅。

在开发人员构建一个小项目的场景中，安装 ORM 库不会比创建专门的 SQL 查询提供更多的好处。在这种情况下，使用 SQL 语句驱动您的[应用程序](https://www.pythoncentral.io/writing-tests-for-your-django-applications-views/)就足够了。然而，对于需要访问来自数百个数据存储的源数据的大中型项目来说，ORM 变得非常有用。这里，需要一个允许开发人员一致地利用和维护应用程序数据层的框架。

## 那么，利用 ORM 有什么好处呢？

ORM 能够创建语法正确的 SQL 查询，并针对高效的数据检索和修改进行了优化。消除了开发人员构建、部署和调试其应用程序所使用的任何 SQL 的需要。

这反过来又为开发人员改进了 [CI/CD 过程](https://en.wikipedia.org/wiki/CI/CD),使得代码更容易维护和重用。给予开发人员改进的创造性自由来操作数据，而没有技术和时间开销。

样板代码是指需要重复使用的代码片段，很少或没有变化。这意味着开发人员必须编写许多行代码来执行甚至是最简单的任务。利用 ORM 为开发人员提供了标准化的接口，有效地减少了样板文件和代码，极大地缩短了应用程序上市的时间。

ORMs 提供了数据库抽象，方便了数据库切换。最后，它可以通过自主过滤传入请求来防范 SQL 注入攻击。

## 是什么让 Prisma 有别于其他 ORM？

Prisma 是下一代 ORM，用于访问 Nest.js 应用程序上的数据。无论开发团队是构建 REST 还是 GraphQL APIs，Prisma 都与 NestJS 的模块化架构无缝集成。然而，Prisma 的独特之处在于它不需要复杂的对象模型，而是通过模式文件映射应用程序的数据结构，如表和列。

Prisma 提供的工具之一叫做 [Prisma Migrate](https://www.prisma.io/docs/concepts/components/prisma-migrate) 。Migrate 利用模式文件为 SQL 生成迁移文件，从而生成类型定义。保持 Prisma 模式文件与源数据库模式一致。

Prisma 可以在普通 JavaScript 中使用，但包括 TypeScript，提供了超过 TypeScript 生态系统中其他 ORM 的类型安全级别。

## *总之*

不同的开发人员可能在不同的时间使用 SQL 和 ORM。两种选择都是正确的。然而，这并不是每种情况下的最佳选择。记住这一点，开发人员在决定方法之前应该考虑项目范围、组织需求和能力。

Prisma 开源 ORM 提供了业内其他 ORM 无法比拟的优势，因此是业内领先的选择之一。