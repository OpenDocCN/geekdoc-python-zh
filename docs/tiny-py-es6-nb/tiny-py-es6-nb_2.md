# 微型 ES6 笔记本

来自：[mattharrison/Tiny-es6-Notebook](https://github.com/mattharrison/Tiny-es6-Notebook)

## 介绍

这不是一本教学手册，而是 ECMAScript2015 或 ES6 语法（也称为现代 JavaScript）的注释、表格和示例。它是作者在培训期间的额外资源，旨在作为一本实体笔记��分发。喜欢纸质材料的参与者可以添加自己的注释、想法，并获得经过精心筛选的示例的宝贵参考。

## 严格模式

ES6 模块根据规范始终是严格的。不需要将此代码放在代码顶部以启用严格模式：

```
'use strict';

```

严格模式会执行以下操作：

+   需要显式创建全局变量（通过 `let`）

+   向不可写变量赋值时会抛出异常

+   删除不可删除属性时会抛出错误

+   如果重复函数参数名称会抛出 `SyntaxError`

+   在数字文字前面放一个 `0` 会抛出 `SyntaxError`（使用 `0o` 表示八进制）

+   在原始值上设置属性时会抛出 `TypeError`。

## 变量

在 ES6 中有各种声明变量的方式。函数局部变量使用 `var` 声明，块变量使用 `let` 声明，常量变量使用 `const` 声明：

```
const PI = 3.14

//PI = 3.1459 // TypeError

function add(x, y) {
  result = x + y  // Implicit global
  return result
}

add(3, 4)
console.log(result)   // prints 7!

function sub(x, y) {
  let val = x + y
  return val
}

sub(42, 2)
//console.log(val)  // ReferenceError

```

注意

常量变量不一定是不可变的（它们的值可以改变），但它们不能重新绑定到另一个对象或原始值。

提示

一个很好的经验法则是将 `const` 作为默认声明。只在需要时使用 `let`，并尽量避免使用 `var`。

### 作用域

以下函数说明了使用 `let` 声明的作用域：

```
function scopetest() {
  //console.log(1, x) //ReferenceError
  let x
  console.log(2, x)
  x = 42
  console.log(3, x)
  if (true) {
    //console.log(4, x) // ReferenceError
    let x
    console.log(5, x)
    x = 17
    console.log(6, x)
  }
  console.log(7, x)
}

```

输出为：

```
2 undefined
3 42
5 undefined
6 17
7 42

```

如果使用 `var` 声明，我们会看到不同的行为：

```
function varscopetest() {
  console.log(1, x)
  var x
  console.log(2, x)
  x = 42
  console.log(3, x)
  if (true) {
    console.log(4, x)
    var x
    console.log(5, x)
    x = 17
    console.log(6, x)
  }
  console.log(7, x)
}

```

输出为：

```
1 undefined
2 undefined
3 42
4 42
5 42
6 17
7 17

```

### 解构

我们可以通过 *解构* 从列表中提取变量：

```
let paul = ['Paul', 1942, 'Bass']
let [name, ,instrument] = paul

```

可以在解构过程中提供默认值：

```
let [name, ,instrument='guitar'] = ['Paul', 1942]

```

使用 *spread* 运算符复制列表：

```
let p2 = [...paul]

```

还有对象解构的概念：

```
let paul = {name: 'Paul', year: 1942};

let {name, inst, year} = paul;

// inst is undefined

```

对象解构也可以提供默认值：

```
let paul = {name: 'Paul', year: 1942};

let {name = 'Joe', inst = 'Guitar', year} = paul;

```

我们还可以在对象解构过程中重命名属性：

```
let paul = {name: 'Paul', year: 1942};

let {name: firstname, inst: instrument, year} = paul;

```

...我们可以将重命名与默认值结合：

```
let paul = {name: 'Paul', year: 1942};

let {name: firstname = 'Joe', inst: instrument = 'Guitar', year} = paul;

```

## 类型

在 ES6 中有两种数据类型，原始值和对象。ES6 有六种原始、不可变的数据类型：字符串、数字、布尔值、null、undefined 和 symbol（ECMAScript 2015 中新增）。当我们使用文字时，我们得到一个原始值。当我们在它们上调用方法时，ES6 会进行 *自动装箱*。还有用于字符串、数字、布尔值和 symbol 的对象包装器。它们分别是 `String`、`Number`、`Boolean` 和 `Symbol`。如果使用 `new` 调用构造函数，将得到一个对象，要获取原始值，请调用 `.valueOf()`。

### `null`

通常表示对象将被期望的地方的值。

```
let count = null
typeof count === "object"

```

注意

尽管 `null` 是一个原始值，但 `typeof` 的结果是一个对象。这是根据规范[[1]](#id2)（尽管被认为是一个缺陷）。

| [[1]](#id1) | [`www.ecma-international.org/ecma-262/6.0/#sec-typeof-operator-runtime-semantics-evaluation`](http://www.ecma-international.org/ecma-262/6.0/#sec-typeof-operator-runtime-semantics-evaluation) |
| --- | --- |

### `undefined`

一个全局对象的属性，其值是原始值 `undefined`。已声明但未赋值的变量的值为 `undefined`：

```
let x  // x is undefined

```

`undefined` 的 `typeof` 是字符串 `"undefined"`：

```
typeof(x) === 'undefined'  // true

```

谨慎使用与 `null` 的松散等式和严格等式比较：

```
undefined == null   // true - loose
undefined === null  // false - strict

```

### 布尔

布尔变量可以有值 `true` 或 `false`。

我们可以使用 `Boolean` 包装器将其他值强制转换为布尔值。大多数值都是真值：

真值和假值

| 真值 | 假值 |
| --- | --- |
| `true` | `false` |
| 大多数对象 | `null` |
| `1` | `0` 或 `-0` |
| `"string"` | `""`（空字符串） |
| `[]`（空列表） | `undefined` |
| `{}`（空对象） |   |
| `Boolean(new Boolean(false))` |   |

注意调用 `new Boolean(obj)`（即作为构造函数）返回一个 `Boolean` 对象，而调用 `Boolean(obj)`（仿佛它是一个函数）返回一个原始的 `true` 或 `false`。另外，请注意，将任何对象强制转换为布尔值都会强制转换为 `true`，即使内部值是 `false`。

获取原始布尔值的常用技巧是使用*双重否定*而不是使用 `Boolean`。这在 `if` 语句中是不必要的。但是，如果你想要创建一个保存布尔值的变量（或者返回一个布尔值），这个技巧会很方便。`!`（非运算符）强制将值转换为否定的布尔值，因此如果我们再次应用它，我们应该得到正确的布尔值：

```
"" == false    // true
"" === false   // false
!!"" == false  // true
!!"" === false // true

```

布尔属性

| 属性 | 描述 |
| --- | --- |
| `Boolean.length` | `1` |
| `Boolean.prototype` | `Boolean` 的原型 |

布尔原型方法

| 方法 | 描述 |
| --- | --- |
| `b.p.constructor()` | `Boolean` 对象 |
| `b.p.toString()` | 值为 `"true"` 或 `"false"` 的字符串 |
| `b.p.valueOf()` | 原始布尔值 |

### 对象

ES6 添加了从变量名创建对象键的能力：

```
const name = 'Paul'
const person = { name }  // like name: 'Paul'

```

注意

数组展开是 ES6 的一个特性。对象展开不是，尽管许多 JS 引擎支持它。

如果我们想要在另一个对象中包含属性，我们可以进行浅层*展开*：

```
const p2 = { ...person, copy: true }

```

另外还支持*计算属性键*：

```
const inst = 'Guitar'
const person = {
  // like playsGuitar: true
  ['plays' + inst]: true
}

```

*方法定义* 还有一个简写形式：

```
const account = {
   amount: 100,
   // old style
   add: function(amt) {
      this.amount += amt
   },
   // shorthand, no function
   remove(amt) {
    this.amount -= amt
   }
}

```

通常我们会将这些封装在一个函数中以创建对象：

```
function getAccount(amount) {
  return {
     amount,
     add(amt) {
        this.amount += amt
     },
     // shorthand, no function
     remove(amt) {
      this.amount -= amt
     }
  }
}

```

我们还可以在对象中定义属性：

```
function getAccount(amount) {
  return {
     _amount: amount,
     get amount() {
       return this._amount
     },
     set amount(val) {
       this._amount = val
     }
  }
}

```

`Object` 可以被调用为构造函数（使用 `new`）和作为函数。两者的行为相同，并将其调用的内容包装在一个对象中。

使用方法 `Object.defineProperties` 和 `Object.defineProperty`，我们可以使用*数据描述符*或*访问器描述符*设置属性。

访问器描述符允许我们创建函数（`get` 和 `set`）来定义成员访问。它看起来像这样：

```
{
  configurable: false, // default
  enumerable: false,   // default
  get: getFunc, // function to get prop, default undefined
  set: setFunc, // function to set prop, default undefined
}

```

数据描述符允许我们为属性创建一个值，并设置它是否可写。它看起来像这样：

```
{
  configurable: false, // default
  enumerable: false,   // default
  value: val           // default undefined
  writeable: false,    // default
}

```

如果 `configurable` 的值为 `false`，则除了 `writeable` 之外，不能使用 `Object.defineProperty` 更改任何值。此外，属性不能被删除。

`enumerable` 属性确定属性是否会出现在 `Object.keys()` 或 `for ... in` 循环中。

一个访问器描述符的示例：

```
function Person(fname) {
  let name = fname;
  Object.defineProperty(this, 'name', {
    get: function() {
      // if you say this.name here
      // you will blow the stack
      if (name === 'Richard') {
        return 'Ringo';
      }
      return name;
  },
    set: function(n) {
      name = n
    }
  });
}

let p = new Person('Richard');

console.log(p.name);  // writes 'Ringo'
p.name = 'Fred';
console.log(p.name);  // writes 'Fred'

```

这些也可以在类中指定：

```
class Person {
  constructor(name) {
    this.name = name;
    Object.defineProperty(this, 'name', {
      get: function() {
        // if you say this.name here
        // you will blow the stack
        if (name === 'Richard') {
          return 'Ringo';
        }
        return name;
      },
      set: function(n) {
        name = n
      }
    });
  }
}

```

以下表格列出了对象的属性。

对象属性

| 属性 | 描述 |
| --- | --- |
| `Object.prototype` | 对象的原型 |
| `Object.prototype.__proto__` | 值：`null` |

对象方法

| 方法 | 描述 |
| --- | --- |
| `Object.assign(target, ...sources)` | 将 `sources` 的属性复制到 `target` 中 |
| `Object.create(obj, [prop])` | 创建一个具有 `obj` 的 `prototype` 和 `prop` 属性的新对象 |
| `Object.defineProperties( obj, prop)` | 从 `prop` 更新 `obj` 的属性 |
| `Object.defineProperty( obj, name, desc)` | 在 `obj` 上创建一个名为 `name` 的属性，具有描述符 `desc` |
| `Object.freeze(obj)` | 阻止对 `obj` 的属性（添加或删除）的未来更改。在严格模式���会抛出错误，否则在尝试稍后调整属性时会静默失败 |
| `Object.getOwnProperty Descriptor(obj, name)` | 获取 `obj` 上 `name` 的描述符。不能在原型链中。 |
| `Object.getOwnProperty Descriptors( obj)` | 枚举在 `obj` 上不在原型链中的描述符 |
| `Object.getOwnProperty Names(obj)` | 返回在 `obj` 上找到的不在原型链中的字符串属性的字符串数组 |
| `Object.getOwnProperty Symbols(obj)` | 返回在 `obj` 上找到的不在原型链中的符号属性的符号数组 |
| `Object.getPrototypeOf( obj)` | 返回 `obj` 的原型 |
| `Object.is(a, b)` | 布尔值，判断两个值是否相同。不会像 `==` 那样强制转换。而且，不像 `===`，不会将 `-0` 视为等于 `+0`，或者 `NaN` 不等于 `NaN`。 |
| `Object.isExtensible(obj)` | 布尔值，判断对象是否可以添加属性 |
| `Object.isFrozen(obj)` | 布尔值，判断对象是否被冻结（冻结也是封闭的和不可扩展的） |
| `Object.isSealed(obj)` | 布尔值，判断对象是否被封闭（不可扩展，不可移除，但可能可写） |
| `Object.keys(obj)` | 在 `for ... in` 循环中给出的可枚举属性，不在原型链中 |
| `Object.preventExtensions( obj)` | 不能直接添加新属性（可以添加到原型中），但可以删除它们 |
| `Object.seal(obj)` | 阻止对象属性的更改。请注意，值可以更改。 |
| `Object.setPrototypeOf(obj, proto)` | 设置对象的 `prototype` 属性 |

对象原型方法

| 方法 | 描述 |
| --- | --- |
| `o.p.constructor()` | `Object` 构造函数 |
| `o.p.hasOwnProperty(prop)` | 布尔值，判断 `prop` 是否是 `o` 的直接属性（不在原型链中） |
| `o.p.isPrototypeOf(obj)` | 布尔值，判断 `o` 是否存在于 `obj` 的原型链中 |
| `o.p.propertyIs Enumerable(property)` | 属性是否可枚举的布尔值 |
| `o.p.toLocaleString()` | 代表对象的区域敏感字符串 |
| `o.p.toString()` | 代表对象的字符串 |
| `o.p.valueOf()` | 返回原始值 |

## 数字

### `NaN`

`NaN`是一个全局属性，表示*不是一个数字*。这是某些数学运算失败的结果，比如负数的平方根。函数`isNaN`将测试一个值是否为`NaN`。

### `Infinity`

`Infinity`是一个全局属性，表示一个非常大的数字。还有`-Infinity`表示非常大的负值。

您可以将整数文字指定为整数、十六进制、八进制或二进制数字。还支持创建浮点值。请参阅数字类型表。

数字类型

| 类型 | 示例 |
| --- | --- |
| 整数 | `14` |
| 整数（十六进制） | `0xe` |
| 整数（八进制） | `0o16` |
| 整数（二进制） | `0b1110` |
| 浮点数 | `14.0` |
| 浮点数 | `1.4e1` |

作为构造函数调用（`new Number(obj)`），将返回一个`Number`对象。当作为函数调用（不带`new`）时，将执行到原始类型的类型转换。

数字属性

| 属性 | 描述 |
| --- | --- |
| `Number.EPSILON` | 数字之间的最小值`2.220446049250313e-16` |
| `Number.MAX_SAFE_INTEGER` | 最大整数`9007199254740991` (`2⁵³ - 1`) |
| `Number.MAX_VALUE` | 最大数字`1.7976931348623157e+308` |
| `Number.MIN_SAFE_INTEGER` | 最小负整数`-9007199254740991` (`-(2⁵³ - 1)`) |
| `Number.MIN_VALUE` | 最小数字`5e-324` |
| `Number.NEGATIVE_INFINITY` | 负溢出`-Infinity` |
| `Number.NaN` | 非数字值`NaN` |
| `Number.POSITIVE_INFINITY` | 正溢出 |
| `Number.name` | 值：`Number` |
| `Number.prototype` | `Number`构造函数的原型 |

数字方法

| 方法 | 描述 |
| --- | --- |
| `n.isFinite(val)` | 测试`val`是否有限 |
| `n.isInteger(val)` | 测试`val`是否为整数 |
| `n.isNaN(val)` | 测试`val`是否为`NaN` |
| `n.isSafeInteger(val)` | 测试`val`是否在安全值之间的整数 |
| `n.parseFloat(s)` | 将字符串`s`转换为数字（或`NaN`） |
| `n.parseInt(s, [radix])` | 将字符串`s`转换为给定基数（`radix`）的整数（或`NaN`） |

数字原型方法

| 方法 | 描述 |
| --- | --- |
| `n.p.constructor()` |   |
| `n.p.toExponential( [numDigits])` | 以`numDigits`精度返回指数表示的字符串 |
| `n.p.toFixed([digits])` | 以`digits`精度返回固定点表示的字符串 |
| `n.p.toLocaleString([locales, [options]])` | 以区域敏感的表示返回字符串 |
| `n.p.toPrecision([numDigits])` | 以`numDigits`精度返回固定点或指数表示的字符串 |
| `n.p.toString([radix])` | 返回字符串表示。`radix`可以在`2`和`36`之间表示基数 |
| `n.p.valueOf()` | 返回数字的原始值 |

### `Math`库

ES6 具有内置的数学库来执行常见操作。

数学属性

| 属性 | 描述 |
| --- | --- |
| `Math.E` | 值为 `2.718281828459045` |
| `Math.LN10` | 值为 `2.302585092994046` |
| `Math.LN2` | 值为 `0.6931471805599453` |
| `Math.LOG10E` | 值为 `0.4342944819032518` |
| `Math.LOG2E` | 值为 `1.4426950408889634` |
| `Math.PI` | 值为 `3.141592653589793` |
| `Math.SQRT1_2` | 值为 `0.7071067811865476` |
| `Math.SQRT2` | 值为 `1.4142135623730951` |

数学方法

| 方法 | 描述 |
| --- | --- |
| `Math.abs(n)` | 计算绝对值 |
| `Math.acos(n)` | 计算反余弦 |
| `Math.acosh(n)` | 计算反双曲余弦 |
| `Math.asin(n)` | 计算反正弦 |
| `Math.asinh(n)` | 计算反双曲正弦 |
| `Math.atan(n)` | 计算反正切 |
| `Math.atan2(y, x)` | 计算商的反正切 |
| `Math.atanh(n)` | 计算反双曲正切 |
| `Math.cbrt(n)` | 计算立方根 |
| `Math.ceil(n)` | 计算大于 `n` 的最小整数 |
| `Math.clz32(n)` | 计算前导零的计数 |
| `Math.cos(n)` | 计算余弦 |
| `Math.cosh(n)` | 计算双曲余弦 |
| `Math.exp(x)` | 计算 e 的 x 次方 |
| `Math.expm1(x)` | 计算 e 的 x 次方减 1 |
| `Math.floor(n)` | 计算小于 `n` 的最大整数 |
| `Math.fround(n)` | 计算最接近的浮点数 |
| `Math.hypot(x, [y], [...)` | 计算斜边（和的平方根） |
| `Math.imul(x, y)` | 计算整数乘积 |
| `Math.log(n)` | 计算自然对数 |
| `Math.log10(n)` | 计算以 10 为底的对数 |
| `Math.log1p(n)` | 计算 1 + `n` 的自然对数 |
| `Math.log2(n)` | 计算以 2 为底的对数 |
| `Math.max(...)` | 计算最大值 |
| `Math.min(...)` | 计算最小值 |
| `Math.pow(x, y)` | 计算 x 的 y 次方 |
| `Math.random()` | 0 到 1 之间的随机数 |
| `Math.round(n)` | 计算最接近的整数 |
| `Math.sign(n)` | 对于 `n` 的负值，零值或正值，返回 `-1`，`0` 或 `1` |
| `Math.sin(n)` | 计算正弦 |
| `Math.sinh(n)` | 计算双曲正弦 |
| `Math.sqrt(n)` | 计算平方根 |
| `Math.tan(n)` | 计算正切 |
| `Math.tanh(n)` | 计算双曲正切 |
| `Math.trunc(b)` | 计算没有小数的整数值 |

## 内置类型

### 字符串

ES6 字符串是一系列 UTF-16 代码单元。可以用单引号或双引号创建字符串字面量：

```
let n1 = 'Paul'
let n2 = "John"

```

要制作长字符串，您可以使用反斜杠表示字符串在以下行继续：

```
let longLine = "Lorum ipsum \
fooish bar \
the end"

```

或者 `+` 运算符允许字符串连接：

```
let longLine = "Lorum ipsum " +
"fooish bar " +
"the end"

```

### 模板字面量

使用反引号，您可以创建 *模板字面量*。这允许插值：

```
let name = 'Paul';
let instrument = 'bass';

var `Name: ${name} plays: ${instrument}`

```

请注意，模板字面量可以是多行的：

```
`Starts here
and ends here`

```

### 原始字符串

如果您需要带有反斜杠的字符串，您可以用反斜杠 (`/`) 转义反斜杠，或者您可以使用 *原始* 字符串：

```
String.raw `This is a backslash: \
and this is the newline character: \n`

```

### 方法

字符串属性

| 属性 | 描述 |
| --- | --- |
| `String.length` | 值为 `1` |
| `String.name` | 值为 `String` |
| `String.prototype` | `String` 构造函数的原型 |

静态字符串方法

| 方法 | 描述 |
| --- | --- |
| `String.fromCharCode(n1, ...)` | 返回包含 Unicode 值`n1`的字符的字符串 |
| `String.fromCodePoint(n1, ...)` | 返回包含 Unicode 点`n1`的字符的字符串 |
| `String.raw` | 创建原始模板文字（在反引号包围的字符串后跟随此） |

字符串原型方法

| 方法 | 描述 |
| --- | --- |
| `s.p.anchor(aName)` | 返回`<a name="aName">s</a>` |
| `s.p.big()` | 返回`<big>s</big>` |
| `s.p.blink()` | 返回`<blink>s</blink>` |
| `s.p.bold()` | 返回`<b>s</b>` |
| `s.p.charAt(idx)` | 返回在`idx`处的字符的字符串。如果索引无效则返回空字符串 |
| `s.p.charCodeAt(idx)` | 返回 UTF-16 代码的 0 到 65535 之间的整数。如果索引无效则返回`NaN` |
| `s.p.codePointAt(idx)` | 返回 Unicode 代码点的整数值。如果索引无效则返回`undefined` |
| `s.p.concat(s1, ...)` | 返回字符串的连接 |
| `s.p.constructor()` | 字符串构造函数 |
| `s.p.endsWith(sub, [length])` | 如果`s`（限制为`length`大小）以`sub`结尾则返回布尔值 |
| `s.p.fixed()` | 返回`<tt>s</tt>` |
| `s.p.fontcolor(c)` | 返回`<font color="c">s</font>` |
| `s.p.fontsize(num)` | 返回`<font size="num">s</font>` |
| `s.p.includes(sub, [start])` | 如果从`start`找到`sub`在`s`中则返回布尔值 |
| `s.p.indexOf(sub, [start])` | `sub`在从`start`开始的`s`中的索引。如果未找到则返回`-1` |
| `s.p.italics()` | 返回`<i>s</i>` |
| `s.p.lastIndexOf(sub, start)` | 返回从最右边的`start`字符开始的`sub`的索引（默认`+Infinity`）。如果未找到则返回`-1` |
| `s.p.link(url)` | 返回`<a href="url">s</a>` |
| `s.p.localeCompare( other, [locale, [option])` | 返回`-1`，`0`或`1`，`s`在`other`之前，相等或之后 |
| `s.p.match(reg)` | 返回一个数组，整个匹配在索引 0 中，其余条目对应括号 |
| `s.p.normalize([unf])` | 返回字符串的 Unicode 标准化形式。`unf`可以是`"NFC"`，`"NFD"`，`"NFKC"`或`"NFKD"` |
| `s.p.repeat(num)` | 返回`num`次连接的`s` |
| `s.p.replace(this, that)` | 返回一个新字符串，用`that`替换`this`。`this`可以是正则表达式或字符串。`that`可以是一个字符串或一个函数，该函数接受`match`，每个括号匹配的参数，匹配的索引偏移和原始字符串。它返回一个新值。 |
| `s.p.search(reg)` | 返回正则表达式首次匹配`s`的索引或`-1` |
| `s.p.slice(start, [end])` | 返回在半开区间切片的字符串，包括`start`并且不包括`end`。负值意味着`s.length - val` |
| `s.p.small()` | 返回`<small>s</small>` |
| `s.p.split([sep, [limit]])` | 返回一个由字符串围绕`sep`分割的子字符串数组。`sep`可以是正则表达式。`limit`确定结果中的分割数。如果正则表达式包含括号，则匹配的部分也包含在结果中。使用`s.join`来撤销 |
| `s.p.startsWith(sub, [pos])` | 如果`s`（从`pos`索引开始）以`sub`开头则返回布尔值 |
| `s.p.strike()` | 返回`<string>s</strike>` |
| `s.p.sub()` | 返回`<sub>s</sub>` |
| `s.p.substr(pos, [length])` | 返回从`pos`索引开始的子字符串（可以为负）。`length`是要包括的字符数。 |
| `s.p.substring(start, [end])` | 返回包括`start`在内但不包括`end`的字符串切片（半开放）。不允许负值。 |
| `s.p.sup()` | 返回`<sup>s</sup>` |
| `s.p.toLocaleLowerCase()` | 根据本地返回小写值 |
| `s.p.toLocaleUpperCase()` | 根据本地返回大写值 |
| `s.p.toLowerCase()` | 返回小写值 |
| `s.p.toString()` | 返回字符串表示 |
| `s.p.toUpperCase()` | 返回大写值 |
| `s.p.trim()` | 返回删除前导和尾随空格的字符串 |
| `s.p.trimLeft()` | 返回删除前导空格的字符串 |
| `s.p.trimRight()` | 返回删除尾随空格的字符串 |
| `s.p.valueOf()` | 返回字符串表示的原始值 |
| `s.p[@@iterator]()` | 返回字符串的迭代器。在迭代器上调用`.next()`会返回代码点 |

注意

许多生成 HTML 的方法创建的标记与 HTML5 不兼容。

### 数组

ES6 数组可以使用文字语法创建，也可以通过调用`Array`构造函数创建：

```
let people = ['Paul', 'John', 'George']
people.push('Ringo')

```

数组不需要是密集的：

```
people[6] = 'Billy'

console.log(people)
//["Paul", "John", "George", "Ringo", 6: "Billy"]

```

`includes` 方法对于检查数组成员资格很有用：

```
people.includes('Yoko')   // true

```

如果在迭代过程中需要索引号，则`entries`方法会给我们一个索引、项目对的列表：

```
for (let [i, name] of people.entries()) {
  console.log(`${i} - ${name}`);
}

// Output
// 0 - Paul
// 1 - John
// 2 - George
// 3 - Ringo
// 4 - undefined
// 5 - undefined
// 6 - Billy

```

我们可以在数组上进行索引操作：

```
let paul = people[0];

```

注意索引操作不支持负索引值：

```
people[-1]; // undefined, not Billy

```

在 ES6 中我们可以对`Array`进行子类化：

```
class PostiveArray extends Array {
  push(val) {
    if (val >  0) {
      super(val);
    }
  }
}

```

您可以使用`slice`方法*切片*数组。请注意，即使只有一个项目，切片也会返回一个数组。请注意，`slice`方法可以接受负索引：

```
people.slice(1, 2)  // ['John']
people.slice(-1)    // ['Billy']
people.slice(3)     // ["Ringo", undefined × 2, "Billy"]

```

数组属性

| 属性 | 描述 |
| --- | --- |
| `Array.length` | 值：`1` |
| `Array.name` | 值：`Array` |
| `Array.prototype` | `Array`构造函数的原型 |

数组方法

| 方法 | 描述 |
| --- | --- |
| `Array.from(iter, [func, [this]])` | 从可迭代对象返回一个新的`Array`。对每个项目调用`func`。在执行`func`时使用`this`的值。 |
| `Array.isArray(item)` | 如果`item`是`Array`则返回布尔值 |
| `Array.of(val, [..., valN])` | 返回带有值的`Array`。整数值被插入，而`Array(3)`创建一个具有三个插槽的`Array`。 |

数组原型方法

| 方法 | 描述 |
| --- | --- |
| `a.p.concat(val, [..., valN])` | 返回一个新的`Array`，插入值。如果值是数组，则追加项目 |
| `a.p.copyWithin(target, [start, [end]])` | 返回经过变异的`a`，从`start`到`end`的项目浅复制到索引`target`中 |
| `a.p.entries()` | 返回数组迭代器 |
| `a.p.every(func, [this])` | 返回一个布尔值，如果对数组中的每个项目调用`func(item, [idx, [a]])`都为真（`idx`是索引，`a`是数组）。`this`是函数的`this`值 |
| `a.p.fill(val, [start, [end]])` | 返回经过变异的`a`，从`start`索引开始插入`val`，直到但不包括`end`索引。索引值可以为负数 |
| `a.p.filter(func, [this])` | 返回一个新数组，其中包含谓词`func(item, [idx, [a]])`为真的项目。`this`是函数的`this`值 |
| `a.p.find(func, [this])` | 返回数组中谓词`func(item, [idx, [a]])`为真的第一个项目（或`undefined`）。`this`是函数的`this`值 |
| `a.p.findIndex(func, [this])` | 返回数组中谓词`func(item, [idx, [a]])`为真的第一个索引（或`-1`）。`this`是函数的`this`值 |
| `a.p.forEach(func, [this])` | 返回`undefined`。对`a`的每个项目应用`func(item, [idx, [a]])`。`this`是函数的`this`值 |
| `a.p.includes(val, [start])` | 返回一个布尔值，如果从`start`索引开始，`a`包含`val` |
| `a.p.indexOf(val, [start])` | 返回`a`中`val`的第一个索引（或`-1`），从`start`索引开始 |
| `a.p.join([sep])` | 返回带有`sep`（默认为`,`）插入的字符串 |
| `a.p.keys()` | 返回索引值的迭代器（不跳过稀疏值） |
| `a.p.lastIndexOf(val, [start])` | 返回`a`中`val`的最后一个索引（或`-1`），从`start`索引开始向后搜索 |
| `a.p.map(func, [this])` | 返回一个新数组，其中对`a`的每个项目调用`func(item, [idx, [a]])`。`this`是函数的`this`值 |
| `a.p.pop()` | 返回`a`的最后一个项目（或`undefined`）（改变`a`） |
| `a.p.push(val, [..., valN])` | 返回`a`的新长度。将值添加到`a`的末尾 |
| `a.p.reduce(func, [init])` | 返回归约的结果。对每个项目调用`func(accumulator, val, idx, a)`。如果提供了`init`，则初始时将`accumulator`设置为它，否则`accumulator`是`a`的第一个项目。 |
| `a.p.reduceRight(func, [init])` | 返回应用反向的归约的结果 |
| `a.p.reverse()` | 返回并改变`a`的顺序为倒序 |
| `a.p.shift()` | 返回并移除`a`的第一个项目（改变`a`） |
| `a.p.slice([start, [end]])` | 返回从`start`到但不包括`end`的`a`的浅拷贝。允许负索引值 |
| `a.p.some(func, [this])` | 返回一个布尔值，如果对数组中的任何项目调用`func(item, [idx, [a]])`都为真（`idx`是索引，`a`是数组）。`this`是函数的`this`值 |
| `a.p.sort([func])` | 返回并改变排序后的 `a`。可以使用 `func(a, b)`，返回 `-1`，`0` 或 `1` |
| `a.p.splice(start, [deleteCount, [item1, ..., itemN]])` | 返回删除对象的数组。在索引 `start` 处改变 `a`，删除 `deleteCount` 个项目，并插入 `items`。 |
| `a.p.toLocaleString( [locales, [options]])` | 返回表示区域设置中数组的字符串 |
| `a.p.toString()` | 返回表示区域设置中数组的字符串 |
| `a.p.unshift([item1, ... itemN])` | 返回 `a` 的长度。通过在 `a` 前面插入元素来改变 `a`。（将按照调用时出现的顺序） |
| `a.p.values()` | 返回具有 `a` 中项目的迭代器 |

### 数组缓冲区

`ArrayBuffer` 包含通用的基于字节的数据。为了操作内容，我们将视图（类型化数组或数据视图）指向特定位置：

```
let ab = new ArrayBuffer(3)
let year = new Uint16Array(ab, 0, 1)
let age = new Uint8Array(ab, 2, 1)
year[0] = 1942
age[0] = 42

```

ArrayBuffer 属性

| 属性 | 描述 |
| --- | --- |
| `ArrayBuffer.length` | `1` |
| `ArrayBuffer.prototype` | `ArrayBuffer` 原型 |

ArrayBuffer 原型方法

| 方法 | 描述 |
| --- | --- |
| `a.p.constructor()` | `ArrayBuffer` 对象 |
| `a.p.slice(begin, [end])` | 返回从 `begin` 到 `n` 之前的 `a` 的副本的新 `ArrayBuffer` |

### 类型化数组

ES6 支持各种类型的 *类型化数组*。这些数组保存的是二进制数据，而不是 ES6 对象：

```
let size = 3;
let primes = new Int8Array(size);
primes[0] = 2;
primes[1] = 3;
primes[2] = 5;  // size now 3
primes[3] = 7;  // full! ignored
console.log(primes) // Int8Array [ 2, 3, 5 ]

```

与普通数组有一些不同：

+   项目具有相同的类型

+   数组是连续的

+   它被初始化为零

要将类型化数组放入普通数组中，我们可以使用 *spread* 运算符：

```
let normal = [...primes]

```

类型化数组

| 类型 | 大小（字节） | 描述 | C 类型 |
| --- | --- | --- | --- |
| `Int8Array` | 1 | 有符号整数 | `int8_t` |
| `Uint8Array` | 1 | 无符号整数 | `uint8_t` |
| `Uint8ClampedArray` | 1 | 无符号整数 | `uint8_t` |
| `Int16Array` | 2 | 有符号整数 | `int16_t` |
| `Uint16Array` | 2 | 无符号整数 | `unint16_t` |
| `Int32Array` | 4 | 有符号整数 | `int32_t` |
| `Uint32Array` | 4 | 无符号整数 | `unint32_t` |
| `Float32Array` | 4 | 32 位浮点数 | `float` |
| `Float64Array` | 8 | 64 位浮点数 | `float` |

以下 `Array` 方法缺失：`push`，`pop`，`shift`，`splice` 和 `unshift`。

还有两个额外的方法可以在 `TypedArrays` 上找到

类型化数组原型方法

| 方法 | 描述 |
| --- | --- |
| `t.p.set(array, [offset])` | 返回 `undefined`。将 `array` 复制到 `t` 的 `offset` 位置 |
| `t.p.subarray(start, [end])` | 返回从位置 `start` 开始到 `end` 之前的 `t` 数据视图的 `TypeArray`。请注意，`t` 和新对象共享数据。 |

### 数据视图

`DataView` 是与 `ArrayBuffer` 交互的另一个接口。如果需要控制字节序，应该使用它而不是类型化数组。

```
let buf = new ArrayBuffer(2)
let dv = new DataView(buf)
let littleEndian = true;
let offset = 0
dv.setInt16(offset, 512, littleEndian)

let le = dv.getInt16(offset, littleEndian) //512

let be = dv.getInt16(offset) // Big endian 2

```

构造函数支持缓冲区和可选的偏移量和长度：

```
new DataView(buffer, [offset, [length]])

```

DataView 属性

| 属性 | 描述 |
| --- | --- |
| `DataView.name` | `DataView` |
| `DataView.prototype` | `DataView` 构造函数 |
| `DataView.prototype.buffer` | 底层的`ArrayBuffer` |
| `DataView.prototype.byteLength` | 视图的长度 |
| `DataView.prototype.byteOffset` | 视图的偏移量 |

DataView 原型方法

| 方法 | 描述 |
| --- | --- |
| `d.p.getFloat32(offset, [littleEndian])` | 从`offset`处检索带符号 32 位浮点数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getFloat64(offset, [littleEndian])` | 从`offset`处检索带符号 64 位浮点数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getInt16(offset, [littleEndian])` | 从`offset`处检索带符号 16 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getInt32(offset, [littleEndian])` | 从`offset`处检索带符号 32 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getInt8(offset, [littleEndian])` | 从`offset`处检索带符号 8 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getUint16(offset, [littleEndian])` | 从`offset`处检索无符号 16 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getUint32(offset, [littleEndian])` | 从`offset`处检索无符号 32 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.getUint8(offset, [littleEndian])` | 从`offset`处检索无符号 8 位整数。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setFloat32(offset, value, [littleEndian])` | 在`offset`处设置带符号 32 位浮点数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setFloat64(offset, value, [littleEndian])` | 在`offset`处设置带符号的 64 位浮点数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setInt16(offset, value, [littleEndian])` | 在`offset`处设置带符号 16 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setInt32(offset, value, [littleEndian])` | 在`offset`处设置带符号 32 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setInt8(offset, value, [littleEndian])` | 在`offset`处设置带符号 8 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setUint16(offset, value, [littleEndian])` | 在`offset`处设置无符号 16 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setUint32(offset, value, [littleEndian])` | 在`offset`处设置无符号 32 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |
| `d.p.setUint8(offset, value, [littleEndian])` | 在`offset`处设置无符号 8 位整数`value`。如果`littleEndian`为`true`，则使用小端格式 |

### Date

`Date`的选项。要获取当前时间，只需使用：

```
let now = new Date();

```

如果提供整数，则将获得自 Unix 纪元以来的秒数：

```
let msPast1970 = 1
let groovyTime = new Date(msPast1970)

// Wed Dec 31 1969 17:00:00 GMT-0700 (MST)

```

ES6 规范仅支持 ISO8601 的一种变体，但实际上也支持 RFC 2822/1123 中的字符串：

```
let modern = new Date("Tue, 14 Mar 2017 14:59:59 GMT")

```

最后，`Date`构造函数允许我们指定年、月、日、小时、分钟、秒和毫秒：

```
let piDay = new Date(2017, 3, 14)

```

使用构造函数创建的日期是在本地时间。要在 UTC 中创建一个 `Date`，请使用 `Date.UTC` 方法。

注意

一个 RFC 2822/RFC 1123 字符串是一个类似于以下的可读字符串：

```
"Tue, 14 Mar 2017 14:59:59 GMT"

```

`toUTCString` 方法将给出这个字符串。

注意

ES6 中的 ISO 8601 规定如下：

```
YYYY-MM-DDTHH:mm:ss.sssZ

```

`Date` 上有一个 `toISOString` 方法，将返回这种格式。

日期属性

| 属性 | 描述 |
| --- | --- |
| `Date.name` | `Date` |
| `Date.prototype` | `Date` 构造函数 |

日期方法

| 方法 | 描述 |
| --- | --- |
| `d.UTC(year, month, [day, [hour, [minute, [second, [millisecond]]]]])` | 返回自指定 UTC 时间以来的自 Unix 纪元以来的毫秒数 |
| `d.now()` | 返回自 Unix 纪元以来的毫秒数 |
| `d.parse(str)` | 返回 ISO 8601 字符串的自 Unix 纪元以来的毫秒数 |

日期原型方法

| 方法 | 描述 |
| --- | --- |
| `d.p.getDate()` | 返回月份中的日期，介于 `1` 和 `31` 之间的数字 |
| `d.p.getDay()` | 返回一周中的日期（`0` 代表星期日）。 |
| `d.p.getFullYear()` | 返回年份，介于 `0` 和 `9999` 之间的数字 |
| `d.p.getHours()` | 返回小时数，介于 `0` 和 `23` 之间的数字 |
| `d.p.getMilliseconds()` | 返回毫秒数，介于 `0` 和 `999` 之间���数字 |
| `d.p.getMinutes()` | 返回分钟数，介于 `0` 和 `59` 之间的数字 |
| `d.p.getMonth()` | 返回月份，介于 `0`（一月）和 `11`（十二月）之间的数字 |
| `d.p.getSeconds()` | 返回秒数，介于 `0` 和 `59` 之间的数字 |
| `d.p.getTime()` | 返回自 Unix 纪元以来的毫秒数 |
| `d.p.getTimezoneOffset()` | 返回时区偏移量（以分钟为单位） |
| `d.p.getUTCDate()` | 返回 UTC 月份中的日期，介于 `1` 和 `31` 之间的数字 |
| `d.p.getUTCDay()` | 返回 UTC 一周中的日期（`0` 代表星期日）。 |
| `d.p.getUTCFullYear()` | 返回 UTC 年份，介于 `0` 和 `9999` 之间的数字 |
| `d.p.getUTCHours()` | 返回小时数，介于 `0` 和 `23` 之间的数字 |
| `d.p.getUTCMilliseconds()` | 返回 UTC 毫秒数，介于 `0` 和 `999` 之间的数字 |
| `d.p.getUTCMinutes()` | 返回 UTC 分钟数，介于 `0` 和 `59` 之间的数字 |
| `d.p.getUTCMonth()` | 返回 UTC 月份，介于 `0`（一月）和 `11`（十二月）之间的数字 |
| `d.p.getUTCSeconds()` | 返回 UTC 秒数，介于 `0` 和 `59` 之间的数字 |
| `d.p.getYear()` | 已弃用的年份实现，请使用 `getFullYear` |
| `d.p.setDate(num)` | 在改变月份中的日期后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setFullYear(year, [month, [day]])` | 在改变日期值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setHours(hours, [min, [sec, [ms]]])` | 在改变时间值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setMilliseconds(ms)` | 在改变毫秒值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setMinutes(min, [sec, [ms]])` | 在改变时间值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setMonth(month, [day])` | 在改变日期值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setSeconds(sec, [ms])` | 在改变时间值后返回自 Unix 纪元以来的毫秒数 |
| `d.p.setTime(epoch)` | 在变异时间值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCDate(num)` | 在变异日期后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCFullYear(year, [month, [day]])` | 在变异日期值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCHours(hours, [min, [sec, [ms]]])` | 在变异时间值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCMilliseconds( ms)` | 在变异 ms 值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCMinutes(min, [sec, [ms]])` | 在变异时间值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCMonth(month, [day])` | 在变异日期值后返回 Unix 纪元后的毫秒数 |
| `d.p.setUTCSeconds(sec, [ms])` | 在变异时间值���返回 Unix 纪元后的毫秒数 |
| `d.p.setYear(year)` | 已损坏的年份实现，请使用`setFullYear` |
| `d.p.toDateString()` | 返回美式英语日期的可读形式 |
| `d.p.toGMTString()` | 已损坏，请使用`toUTCString` |
| `d.p.toISOString()` | 返回 ISO 8601 形式的日期字符串 |
| `d.p.toJSON()` | 返回日期 JSON 字符串形式（ISO 8601） |
| `d.p.toLocaleDateString( [locales, [options]])` | 返回区域设置中日期部分的字符串 |
| `d.p.toLocaleString( [locales, [options]])` | 返回区域设置中的日期和时间字符串 |
| `d.p.toLocaleTimeString( [locales, [options]])` | 返回区域设置中的时间字符串 |
| `d.p.toString()` | 返回美式英语的字符串表示 |
| `d.p.toTimeString()` | 返回美式英语时间部分的字符串表示 |
| `d.p.toUTCString()` | 返回 UTC 时区中的（通常是 RFC-1123 格式化的）字符串版本 |
| `d.p.valueOf()` | 返回 Unix 纪元后的毫秒数 |

### 地图

地图没有原型（就像对象那样）可能与您的键冲突。对象只支持字符串或符号作为键，而地图支持任何类型的键（函数、对象或基元）。地图的另一个好处是您可以轻松地通过`size`属性获取长度。要获取对象的长度，您需要对其进行迭代。

您应该使用对象还是地图？如果需要记录类型数据，请使用对象。对于需要变异和迭代的哈希集合，请选择地图。

只需调用构造函数或传递键值对的可迭代对象即可创建`Map`。

```
let instruments = new Map([['Paul', 'Bass'],
   ['John', 'Guitar']]);
instruments.set('George', 'Guitar');

instruments.has("Ringo");  // false
for (let [name, inst] of instruments) {
  console.log(`${name} - ${inst}`);
}

```

地图属性

| 属性 | 描述 |
| --- | --- |
| `Map.name` | `Map` |
| `Map.prototype` | `Map`的构造函数原型 |
| `Map.prototype.size` | 返回地图中的项目数 |

地图原型方法

| 方法 | 描述 |
| --- | --- |
| `m.p.clear()` | 返回`undefined`。从`m`中删除所有项目（变异`m`） |
| `m.p.delete(key)` | 如果`key`在`m`中，则返回布尔值。变异`m`并将其删除 |
| `m.p.entries()` | 返回键值对数组的迭代器 |
| `m.p.forEach(func, [this])` | 返回`undefined`。将`func（value，key，m）`应用于`m`中的每个键和值 |
| `m.p.get(key)` | 返回值或 `undefined` |
| `m.p.has(key)` | 如果 `key` 在 `m` 中找到，则返回布尔值 |
| `m.p.keys()` | 返回按插入顺序排列的 `m` 中的键的迭代器 |
| `m.p.set(key, value)` | 使用 `value` 来改变 `m` 中的 `key`，返回 `m` |
| `m.p.values()` | 按插入顺序返回值的迭代器 |

### `WeakMaps`

`WeakMaps` 允许您跟踪对象，直到它们被垃圾收集。键不创建引用，因此如果键恰好被垃圾收集（或其包含的对象被收集），数据可能会从 weakmap 中消失。

构造函数具有与`Map`相同的接口，您可以通过不提供参数来创建一个空的`WeakMap`，或者您可以传递一个键值对的可迭代对象。

弱映射属性

| 属性 | 描述 |
| --- | --- |
| `WeakMap.name` | `WeakMap` |
| `WeakMap.prototype` | `WeakMap` 的构造函数原型 |

弱映射原型方法

| 方法 | 描述 |
| --- | --- |
| `w.p.delete(key)` | 如果 `key` 存在且被移除，则返回 `true`，否则返回 `false` |
| `w.p.get(key)` | 返回 `key` 的值，如果缺少则返回 `undefined` |
| `w.p.has(key)` | 如果 `key` 在 `w` 中找到，则返回布尔值 |
| `w.p.set(key, value)` | 使用新的 `key` 和 `value` 改变 `w`，并返回 `w` |

### 集合

集合是一个可变的无序集合，不能包含重复项。集合用于删除重复项并测试成员资格。您可以通过调用没有参数的构造函数来创建一个空的`Set`。如果要从可迭代对象创建 `Set`，请将其传递到构造函数中：

```
let digits = new Set([0, 1, 1, 2, 3, 4, 5, 6,
  7, 8 , 9]);

digits.has(9);  // true

let odd = new Set([1, 3, 5, 7, 9]);
let prime = new Set([2, 3, 5, 7]);

```

不提供集合操作。以下是添加差异的示例：

```
Set.prototype.difference = function(other) {
  let result = new Set(this);
  for (let val of other) {
    result.delete(val);
  }
  return result;
}

let even = digits.difference(odd);
console.log(even);  // Set { 0, 2, 4, 6, 8 }

```

集合属性

| 属性 | 描述 |
| --- | --- |
| `Set.name` | `Set` |
| `Set.prototype` | `Set` 的构造函数原型 |
| `Set.prototype.size` | 返回集合的大小 |

集合原型方法

| 方法 | 描述 |
| --- | --- |
| `s.p.add(item)` | 返回 `s`。将 `item` 添加到 `s`（变异） |
| `s.p.clear()` | 返回 `undefined`。从 `s` 中删除（变异）所有项 |
| `s.p.delete(item)` | 如果删除，则返回值，否则返回 `false`（变异） |
| `s.p.entries()` | 返回 `s` 中每个项（相同接口的 `Map`）的迭代器，包括每个项，每个项 |
| `s.p.forEach(func, [this])` | 返回 `undefined`。对 `s` 中的每个项应用 `func(item, item, s)`。与 `Array` 和 `Map` 具有相同的接口 |
| `s.p.has(item)` | 如果 `item` 在 `s` 中找到，则返回布尔值 |
| `s.p.values()` | 返回按插入顺序排列的项的迭代器 |

### 弱集合

弱集合是对象的集合，而不是任何类型。我们无法对它们进行枚举。对象可能在垃圾收集时从中自动消失。

`WeakSet` 构造函数具有与 `Set` 相同的接口（无参数或可迭代对象）。

弱集合属性

| 属性 | 描述 |
| --- | --- |
| `WeakSet.name` | `WeakSet` |
| `WeakSet.prototype` | `WeakSet` 的原型 |

弱集合原型方法

| 方法 | 描述 |
| --- | --- |
| `w.p.add(item)` | 返回 `w`，将 `item` 插入 `w` |
| `w.p.delete(item)` | 如果`item`在`w`中则返回`true`（同时移除它），否则返回`false` |
| `w.p.has(item)` | 如果`item`在`w`中则返回布尔值 |

### 代理

代理允许您为基本操作（获取/设置属性，调用函数，循环值，装饰等）创建自定义行为。我们使用*handler*为*target*配置*traps*。

构造函数接受两个参数，一个是目标，一个是处理程序：

```
const handler = {
  set: function(obj, prop, value) {
    if (prop === 'month') {
      if (value < 1 || value > 12) {
        throw RangeError("Month must be between 1 & 12")
      }
    }
    obj[prop] = value
    // need to return true if successful
    return true
  }
}

const cal = new Proxy({}, handler)

cal.month = 12
console.log(cal.month)  // 12
cal.month = 0  // RangeError

```

代理方法

| 属性 | 描述 |
| --- | --- |
| `Proxy.revocable(target, handler)` | 创建一个可撤销的代理。当调用`revoke`时，代理会抛出`TypeError` |

### 反射

`Reflect`对象允许您检查对象。`Reflect`不是一个构造函数，所有方法都是静态的。

数学方法

| 方法 | 描述 |
| --- | --- |
| `Reflect.apply(obj, this, args)` | 返回使用`this`值调用`obj(...args)`的结果 |
| `Reflect.construct(obj, args)` | 返回`new obj(...args)`的结果 |
| `Reflect.defineProperty(obj, key, descriptor)` | 类似于`Object.defineProperty(obj, key, descriptor)`，但返回`Boolean` |
| `Reflect.deleteProperty(obj, key)` | 从`obj`中删除`key`，返回`Boolean` |
| `Reflect.get(obj, key)` | 返回`obj[key]` |
| `Reflect.getOwnPropertyDescriptor(obj, key)` | 返回`obj[key]`的属性描述符 |
| `Reflect.getPrototypeOf(obj)` | 返回`obj`的原型或`null`（如果没有继承属性） |
| `Reflect.has(obj, key)` | 返回 `key in obj` |
| `Reflect.isExtensible(obj)` | 如果您可以向`obj`添加新属性，则返回`Boolean` |
| `Reflect.ownKeys(obj)` | 返回`obj`中的键的`Array` |
| `Reflect.preventExtensions(obj)` | 禁止在`obj`上添加扩展，如果成功则返回`Boolean` |
| `Reflect.set(obj, key, value, [this])` | 在`obj`上成功设置属性`key`时返回`Boolean`。 |

### 符号

ES6 引入了一种新的原始类型，Symbol。它们具有类似字符串的属性（不可变，不能在其上设置属性，可以作为属性名）。它们还具有类似对象的行为（即使描述相同，也是唯一的）。符号是唯一的值，可以用作属性键而不会发生冲突。它们必须使用索引操作（方括号）访问，而不是点符号。它们也不会在`for ... in`循环中迭代。要检索它们，我们需要使用`Object.getOwnPropertySymbols`或`Reflect.ownKeys`。

构造函数接受一个可选的描述参数：

```
Symbol('name') == Symbol('name')   // false
Symbol('name') === Symbol('name')  // false

```

注意

`Symbol`不是一个构造函数，如果您尝试将其用作构造函数，则会引发`TypeError`。

符号属性

| 属性 | 描述 |
| --- | --- |
| `Symbol.hasInstance` | 用于为`instanceof`定义类行为。将方法定义为`static Symbol.hasInstance ...` |
| `Symbol.isConcatSpreadable` | 用于为`Array.concat`定义类行为。如果项目被展开（或扁平化），则设置为`true`。 |
| `Symbol.iterator` | 用于为`for...of`定义类行为。应遵循迭代协议。可以是一个生成器 |
| `Symbol.match` | 用于定义类行为，以便在 `String` 方法中作为正则表达式响应：`startsWith`、`endsWith`、`includes` |
| `Symbol.name` | 值：`Symbol` |
| `Symbol.prototype` | `Symbol` 的原型 |
| `Symbol.replace` | 用于定义类行为，以响应 `String.p.replace` 方法 |
| `Symbol.search` | 用于定义类行为，以响应 `String.p.search` 方法 |
| `Symbol.species` | 用于定义类行为，在创建派生对象时使用哪个构造函数 |
| `Symbol.split` | 用于定义类行为，以响应 `String.p.split` 方法 |
| `Symbol.toPrimitive` | 用于定义类行为，以响应强制转换。将方法定义为 `static [Symbol.toPrimitive] (hint) ...`，其中 `hint` 可以是 `'number'`、`'string'` 或 `'default'` |
| `Symbol.toStringTag` | 用于定义类行为，以响应 `Object.p.toString` 方法 |
| `Symbol.unscopables` | 用于定义 `with` 语句中的类行为。应该设置为将属性映射到布尔值的对象，如果它们在 `with` 中不可见，则为 `true`（即 `true` 表示抛出 `ReferenceError`） |

符号方法

| 方法 | 描述 |
| --- | --- |
| `Symbol.for(key)` | 返回全局注册表中的 `key` 的符号，其他创建符号并返回它。 |
| `Symbol.keyFor(symbol)` | 返回全局注册表中符号的字符串值，如果符号不在注册表中则返回 `undefined` |

符号原型方法

| 方法 | 描述 |
| --- | --- |
| `s.p.toString()` | 返回符号的字符串表示 |
| `s.p.valueOf()` | 返回符号的原始值（符号） |

## 内置函数

内置函数

| 方法 | 描述 |
| --- | --- |
| `eval(str)` | 评估在 str 中找到的代码 |
| `isFinite(val)` | 如果 `val` 是 `Infinity`、`-Infinity` 或 `NaN`，则返回 `false`，否则返回 `true` |
| `isNaN(val)` | 如果 `val` 是 `NaN`，则返回 `true`，否则返回 `False` |
| `parseFloat(str)` | 如果 `str` 可以转换为数字，则返回浮点数，否则返回 `NaN` |
| `parseInt(val, radix)` | 如果 `str` 可以转换为整数，则返回整数，否则返回 `NaN`。它忽略 `radix` 中不是数字的字符 |
| `decodeURI(uri)` | 返回字符串的未编码版本。应该用于完整的 URI |
| `decodeURIComponent(str)` | 返回字符串的未编码版本。应该用于 URI 的部分 |
| `encodeURI(uri)` | 返回 URI 的编码版本。应该用于完整的 URI |
| `encodeURIComponent(uri)` | 返回 URI 的编码版本。应该用于 URI 的部分 |

## Unicode

如果我们有一个 Unicode 符号，我们可以直接包含它：

```
let xSq = 'x²';

```

或者，我们可以使用 Unicode 代码点来指定 Unicode 字符：

```
let xSq2 = 'x\u{b2}';

```

如果我们有确切的四个十六进制数字，我们可以这样转义：

```
let xSq3 = 'x\u00b2';

```

我们可以使用 `codePointAt` 字符串方法获取代码点：

```
'x\u{b2}'.codePointAt(1)   // 178

```

要使用 `fromCodePoint` 静态方法将代码点转换回字符串：

```
String.fromCodePoint(178)  // "²"

```

如果我们在正则表达式中使用`/u`标志，我们可以搜索 Unicode 字符，这将处理代理对。

## 函数

函数很容易定义，我们只需给它们一个名称，它们接受的参数和一个主体：

```
function add(x, y) {
  return x + y
}

let six = add(10, -4)

```

参数存储在一个隐式变量`arguments`中。我们可以用任意数量的参数调用函数：

```
function add2() {
  let res = 0
  for (let i of arguments) {
    res += i
  }
  return res
}

let five = add2(2, 3)

```

### 默认参数

如果我们想要为参数设置默认值，使用`=`紧跟在参数后面指定它：

```
function addN(x, n=42) {
  return x + n;
}

let forty = addN(-2)
let seven = addN(3, 4)

```

### 变量参数

使用`...` (*rest*)将剩余参数转换为数组：

```
function add_many(...args) {
  // args is an Array
  let result = 0;
  for (const val of args) {
    result += val;
  }
  return result;
}

```

再次，因为 ES6 提供了`arguments`对象，我们也可以创建一个接受可变参数的函数，像这样：

```
function add_many() {
  let result = 0;
  for (const val of arguments) {
    result += val;
  }
  return result;
}

```

### 调用函数

你可以使用`...`来*展开*一个数组成为参数：

```
add_many(...[1, 42., 7])

```

### `bind`方法

函数有一个名为`bind`的方法，允许你设置`this`和任何其他参数。这本质上允许你*部分*函数：

```
function mul(a, b) { return a * b }

let double = mul.bind( undefined, 2 )
let triple = mul.bind( undefined, 3 )

console.log(triple(2))   // 6

```

绑定的第一个参数是传递给`this`的值。其余的是函数的参数。如果你想让回调使用父类的`this`，你可以在函数上调用 bind，传入父类的`this`。或者，你可以使用箭头函数。

### 箭头函数

ES6 引入了匿名*箭头*函数。箭头函数有一些不同之处：

+   隐式返回

+   `this`不会重新绑定

+   不能是生成器

第二个特性使它们适用于回调和处理程序，但不适合方法。我们可以这样写：

```
function add2(val) {
  return val + 2
}

```

如：

```
let add2 = (val) => (val + 2)

```

`=>`被称为*胖箭头*。由于这是一个单行函数，我们可以移除花括号并利用隐式返回。

注意，如果只有一个参数并且内联函数，括号可以被移除：

```
let vals = [1, 2, 3]
console.log(vals.map(v=>v*2))

```

如果我们想要一个多行箭头函数，那么去掉`function`，加上`=>`：

```
function add(x, y) {
  let res = x + y
  return res
}

```

变成：

```
let add = (x,y) => {
  let res = x + y
  return res
}

```

### 尾调用

如果你在函数的最后位置执行递归调用，那就是*尾调用*。ES6 允许你这样做而不会增加堆栈：

```
function fib(n){
  if (n == 0) {
    return 0;
  }
  return n + fib(n-1);
}

```

注意

一些实现可能不支持这个。在 Node 7.7 和 Chrome 56 中，使用`fib(100000)`会失败。

## 类

ES6 引入了`class`，它是围绕使用函数创建对象的语法糖。ES6 类支持原型继承，通过`super`调用父类，实例方法和静态方法：

```
class Bike {
  constructor(wheelSize, gearRatio) {
    this._size = wheelSize;
    this.ratio = gearRatio;
  }

  get size() { return this._size }
  set size(val) { this._size = val }

  gearInches() {
    return this.ratio * this.size;
  }
}

```

注意当你创建一个`class`的新实例时，需要使用`new`：

```
let bike = new Bike(26, 34/13)
bike.gearInches()

```

注意

ES6 中的类不会*提升*。这意味着你不能在定义之前使用类。函数会被提升，你可以在定义它们的代码范围内的任何地方使用它们。

在 ES6 之前，我们只能从函数创建对象：

```
function Bike2(wheelSize, gearRatio) {
  this.size = wheelSize;
  this.ratio = gearRatio;
}

Bike2.prototype.gearInches = function() {
  return this.ratio * this.size
}

let b = new Bike2(27, 33/11);
console.log(b.gearInches());

```

注意

方法是在原型之后添加的，所以实例可以共享该方法。我们可以在函数中定义方法，但那样每个实例都会有自己的副本。

ES6 为此提供了一个更清晰的语法。

### 子类

子类中需要注意的一点是，它们应该调用`super`。因为 ES6 只是语法糖，如果我们不调用`super`，我们将没有原型，而且我们无法创建没有原型的实例。因此，直到调用`super`之前，`this`都是未定义的。如果不调用 super，应该返回`Object.create(...)`。

```
class Tandem extends Bike {
  constructor(wheelSize, rings, cogs) {
    let ratio = rings[0] / cogs;
    super(wheelSize, ratio);
    this.cogs = cogs;
    this.rings = rings;

  }

  shift(ringIdx, cogIdx) {
    this.ratio = this.rings[ringIdx] /
      this.cogs[cogIdx];
  }
}

let tan = new Tandem(26, [42, 36], [24, 20, 15, 11])
tan.shift(1, 2)
console.log(tan.gearInches())

```

### 静态方法

*静态方法*是直接在类上调用的方法，而不是在实例上调用的。

```
class Recumbent extends Bike {
  static isFast() {
    return true;
  }
}

Recumbent.isFast();  // true

rec = new Recumbent(20, 4);
rec.isFast();  // TypeError

```

### 对象字面量

我们还可以使用对象字面量创建实例，尽管实际上这会导致大量的代码重复：

```
 let size = 20;
 let gearRatio = 2;

 let bike = {
   __proto__: protoObj,
   ['__proto__']: otherObj,
   size, // same as `size: size`
   ratio: gearRatio,
   gearInches() {
     return this.size * this.ratio;
   }
  // dynamic properties
  [ 'prop_' + (() => "foo")() ]: "foo"
}

```

## 运算符

赋值

内置运算符

| 运算符 | 描述 |
| --- | --- |
| `=` | 赋值 |
| `+` | 加法，一元加（转换为数字），连接（字符串） |
| `++` | 自增 |
| `-` | 减法，一元否定（转换为数字） |
| `--` | 自减 |
| `*` | 乘法 |
| `/` | 除法 |
| `%` | 取余（模运算） |
| `**` | 幂 |
| `<<` | 左移 |
| `>>` | 右移 |
| `<<<` | 无符号左移 |
| `>>>` | 无符号右移 |
| `&` | 位与操作 |
| `^` | 位异或操作 |
| `&#124;` | 位或操作 |
| `~` | 位非操作 |
| `&&` | 逻辑与 |
| `&#124;&#124;` | 逻辑或操作 |
| `!` | 逻辑非 |
| `,` | 逗号运算符，评估所有操作数并返回最后一个 |
| `delete X` | 删除对象、属性或索引 |
| `typeof` | 返回表示操作数类型的字符串 |
| `void` | 创建一个没有返回值的表达式 |
| `in` | 如果对象中有该属性，则返回布尔值 |
| `instanceof` | 如果对象是某类型的实例，则返回布尔值 |
| `new` | 创建一个类型的新实例 |
| `...` | 将*spread*序列展开为数组或参数 |

## 条件语句

ES6 支持带零个或多个`else if`语句以及可选的结尾`else`语句的`if`语句：

```
let grade = 72;

function letter_grade(grade) {
  if (grade > 90) {
    return 'A';
  }
  else if (grade > 80) {
    return 'B';
  }
  else if (grade > 70) {
    return 'C';
  }
  else {
    return 'D';
  }
}

letter_grade(grade);  // 'C'

```

ES6 支持以下测试：`>`, `>=`, `<`, `<=`, `==`, `!=`, `===`, 和 `!==`。对于布尔运算符，使用`&&`、`||`和`!`分别表示与、或和非。

对于`==`和`!=`，如果操作数具有不同的类型，ES6 会尝试比较其数值，因此：

```
'3' == 3  // true

```

如果这让你感到不适（确实应该），请使用*严格*相等运算符（`===`和`!==`）：

```
'3' === 3  // false

```

### 短路

`and`语句如果求值为 false 将会*短路*：

```
0 && 1/0  // 0

```

类似地，当某些条件求值为 true 时，`or`语句会短路：

```
1 || 1/0  // 1

```

### 三元运算符

ES6 具有三元运算符。不需要编写：

```
let last
if (band == 'Beatles') {
  last = 'Lennon'
}
else {
  last = 'Jones'
}

```

我们可以这样写：

```
let last = (band == 'Beatles) ? 'Lennon' : 'Jones';

```

### Switch

ES6 支持 switch 语句：

```
function strings(inst) {
  switch(inst) {
    case 'guitar':
      return 6;
    case 'violin':
      return 4;
    default:
      return 1;
  }
}

strings('violin');  // 4

```

## 循环

有各种各样的迭代方式：

+   `for ... in` - 遍历对象的属性。这仅遍历具有`[[Enumerable]]`设置为`true`的属性。

+   `for ... of` - 遍历集合的项。任何具有`[Symbol.iterator]`属性的对象都可以使用此方法进行迭代。

+   `forEach`是`Array`对象上的方法。它接受一个回调函数，对数组的每个项都调用该函数。

还有一个`while`循环：

```
let num = 3
while (num > 0) {
  console.log(num)
  num -= 1
}
console.log('Blastoff!')

```

还有一个`do ... while`循环：

```
let num = 3
do {
  console.log(num)
  num -= 1
} while (num > 0)
console.log('Blastoff!')

```

### 迭代

我们可以创建一个知道如何迭代的类。我们必须提供一个 `[Symbol.iterator]` 方法，并且该方法的结果需要有一个 `next` 方法。

这是一个创建迭代类的示例：

```
class Fib {
  constructor() {
    this.val1 = 1;
    this.val2 = 1;
  }

  [Symbol.iterator]() {
    return this;  // something with next
  }

  next() {
    [this.val1, this.val2] = [this.val2, this.val1 + this.val2];
    return {done: false, value: this.val1};
  }
}

```

`next` 的结果应该是一个对象，该对象指示循环是否完成（在 `done` 属性中），并在 `value` 属性中返回迭代项。

在 `for .. of` 循环中使用它：

```
for (var val of new Fib()) {
  console.log(val);
  if (val > 5) {
    break;
  }
}

```

我们也可以使用对象字面量循环：

```
let fib = {
  [Symbol.iterator]() {
    let val1 = 1;
    let val2 = 1;
    return {
      next() {
        [val1, val2] = [val2, val1 + val2];
        return { value: val1, done: false}
      }
    }
  }
}

```

在循环中使用迭代器：

```
for (var val of fib) {
  console.log(val);
  if (val > 5) {
    break;
  }
}

```

## 异常

ES6 允许我们处理异常，如果发生异常的话：

```
try {
   // code missing
}
catch(e) {
  // handle any exception
}

```

如果有 `finally` 语句，则无论是否发生异常，它都会在其他块之后执行：

```
try {
  // code missing
}
catch(e) {
  // handle any exception
}
finally {
  // run after either block
}

```

### 抛出错误

ES6 也允许我们抛出错误：

```
throw new Error("Some error");

```

### 错误类型

有各种内置错误类型：

+   `EvalError` - 在 ES6 中未使用，仅用于向后兼容性

+   `RangeError` - 超出允许值集的值

+   `ReferenceError` - 引用无效引用时出错（`let val = badRef`）

+   `SyntaxError` - 语法不正确时出错（`foo bar`）

+   `TypeError` - 当值的类型不正确时出错（`undefined.junk()`）

+   `URIError` - 当 URI 编码/解码出错时出错（`decodeURI('%2')`）

## 生成器

使用 `function*` 和 `yield` 而不是普通的 `function` 和 `return` 的迭代器是*生成器*。它们在迭代时即时生成值。在 `yield` 语句之后，函数的状态被冻结。

```
let fibGen = {
  [Symbol.iterator]: function*() {
    let val1 = 1;
    let val2 = 2;
    while (true) {
      yield val1;
      [val1, val2] = [val2, val1 + val2];
    }
  }
}

```

使用生成器：

```
for (var val of fibGen) {
    console.log(val);
    if (val > 5) {
        break;
    }
}

```

## 模块

模块是一个 JavaScript 文件。要在其他文件中使用对象，我们需要*导出*对象。在这里，我们创建一个 `fib.js` 文件并导出生成器：

```
// js/fib.js

export let fibGen = {
  [Symbol.iterator]: function*() {
    let val1 = 1;
    let val2 = 2;
    while (true) {
      yield val1;
      [val1, val2] = [val2, val1 + val2];
    }
  }
}

export let takeN = {
  [Symbol.iterator]: function*(seq, n) {
    let count = 0;
    for (let val of seq ) {
      if (count >= n) {
        break;
      }
      yield val;
      count += 1;
    }
  }
}

```

### 使用模块

我们可以使用 `import` 语句加载导出的对象：

```
// js/other.js

import {fibGen, takeN} from "fib.js";

console.log(sum(takeN(fibGen(), 5)));

```

注意

到目前为止，对此功能的支持有限（在 Chrome 60、Edge 38、Firefox 54 及更高版本中通过标志可用）。要在非现代浏览器或节点中使用导入，我们需要使用 Babel 来获得支持：

```
$ npm install --save-dev babel-cli babel-preset-env

```

## 承诺

承诺是允许进行异步编程的对象。它们是回调的替代品。如果你知道一个值可能在未来可用，那么承诺可以表示这一点。

承诺有三种状态：

+   挂起 - 结果尚未准备好

+   已实现 - 结果已准备好

+   已拒绝 - 发生错误

承诺对象上将调用 `then` 方法来将状态移动到已实现或已拒绝。

实现承诺的异步函数允许我们*链接*`then` 和 `catch` 方法：

```
asyncFunction()
.then(function (result) {
  // handle result
  // Fulfilled
})
.catch(function (error) {
  // handle error
  // Rejected
})

```

承诺属性

| 属性 | 描述 |
| --- | --- |
| `Promise.length` | 返回 `1`，构造函数参数的数量 |
| `Promise.name` | 值：`Promise` |
| `Promise.prototype` | `Promise` 的原型 |

承诺方法

| 方法 | 描述 |
| --- | --- |
| `Promise.all(promises)` | 返回一个 `Promise`，在 `promises` 全部实现时返回，或者在其中任何一个拒绝时拒绝。 |
| `Promise.race(promises)` | 返回一个 `Promise`，在任何一个 `promises` 拒绝或实现时返回 |
| `Promise.reject(reason)` | 返回一个以`reason`为参数拒绝的`Promise` |
| `Promise.resolve(value)` | 返回一个以`value`为参数完成的`Promise`。如果`value`有一个`then`方法，它将返回其最终状态 |

Promise 原型方法

| 方法 | 描述 |
| --- | --- |
| `p.p.catch(func)` | 返回具有拒绝处理程序`func`的新`Promise` |
| `p.p.constructor()` | 返回`Promise`函数 |
| `p.p.then(fulfillFn, rejectFn)` | 返回一个`Promise`，成功时调用`fulfillFn(value)`，失败时调用`rejectFn(reason)` |

## 正则表达式

正则表达式允许您在字符串中匹配字符。您可以使用文字语法创建它们。在斜杠之间放置正则表达式。标志可以在末尾指定：

```
let names = 'Paul, George, Ringo, and John'

let re = /(Ringo|Richard)/g;

```

您可以在字符串上使用`match`方法：

```
names.match(re);  //[ 'Ringo', 'Ringo' ]

```

或在正则表达式上使用`exec`方法：

```
re.exec(names)  // [ 'Ringo', 'Ringo' ]

```

如果匹配成功，它将返回一个数组。位置 0 是匹配的部分，其余项目对应于捕获的组。这些用括号指定。您只需计算左括号（除非它们被转义）的数量。索引 1 将是第一个括号的组，索引 2 将是第二个左括号的组，依此类推。

您还可以调用构造函数：

```
let re2 = RegExp('Ringo|Richard', 'g')

```

正则表达式标志

| 标志 | 描述 |
| --- | --- |
| `g` | *全局*匹配，返回所有匹配项，而不仅仅是第一个 |
| `i` | *忽略*大小写匹配 |
| `m` | 将换行符视为`^`和`$`的断点，允��*多行*匹配 |
| `y` | *粘性*匹配，从`r.lastIndex`属性开始查找 |
| `u` | *Unicode*匹配 |

字符类

| 字符 | 描述 |
| --- | --- |
| `\d` | 匹配一个数字（`[0-9]`） |
| `\D` | 匹配一个非数字（`[⁰-9]`） |
| `\w` | 匹配一个*单词*字符（`[0-9a-zA-z]`） |
| `\W` | 匹配一个非单词字符（`[⁰-9a-zA-z]`） |
| `\s` | 匹配一个*空格*字符 |
| `\S` | 匹配一个非空格字符 |
| `\b` | 匹配单词边界 |
| `\B` | 匹配非单词边界 |
| `\t`, `\r`, `\v`, `\f` | 分别匹配一个制表符、回车符、垂直制表符或换行符 |
| `\0` | 匹配一个空字符 |
| `\cCHAR` | 匹配一个控制字符。其中`CHAR`是字符 |
| `\xHH` | 匹配一个具有十六进制代码`HH`的字符 |
| `\uHHHH` | 匹配一个具有十六进制代码`HHHH`的 UTF-16 字符 |
| `\u{HHHH}` | 匹配一个具有十六进制代码`HHHH`的 Unicode 字符（使用`u`标志） |

语法字符

| 字符 | 描述 |
| --- | --- |
| `^` | 匹配行首（请注意字符类中的不同） |
| `$` | 匹配行尾 |
| `a&#124;b` | 匹配`a`或`b` |
| `[abet]` | *字符类*匹配`a`、`b`、`e`或`t`中的一个 |
| `[^abe]` | `^`在字符类中否定匹配。不是`a`、`b`或`e`中的一个 |
| `\[` | *转义*。捕获文字`[`。以下需要转义`^ $ \ . * + ? ( ) [ ] { } &#124;` |
| `.` | 匹配除换行符之外的一个字符 |
| `a?` | 匹配`a`零次或一次（`a{0,1}`）（请注意后面的`*`、`+`、`?`、`}`不同） |
| `a*` | 匹配`a`零次或多次（`a{0,}`） |
| `a+` | 匹配一个或多个 `a`（`a{1,}`） |
| `a{3}` | 匹配三次 `a` |
| `a{3,}` | 匹配三次或更多次 `a` |
| `a{3,5}` | 匹配三到五次 `a` |
| `b.*?n` | `?` 匹配非贪婪。例如从 `banana` 中返回 `ban` 而不是 `banan` |
| `(Paul)` | *捕获* 匹配在组中。捕获的结果将在 `exec` 中的位置 1 |
| `(?:Paul)` | 匹配 `Paul` 但不捕获。结果仅在 `exec` 中的位置 0 |
| `\NUM` | *反向引用* 以匹配先前捕获的组。`/<(\w+)>(.*)<\/\1>/` 将在 1 中捕获 xml 标记名称，在 2 中捕获内容 |
| `Foo(?=script)` | *断言* 仅在后面跟着 `script` 时匹配 `Foo`（不捕获 `script`） |
| `Foo(?!script)` | *断言* 仅当后面不跟着 `script` 时匹配 `Foo` |

正则表达式属性

| 属性 | 描述 |
| --- | --- |
| `RegExp.length` | 值：`2` |
| `RegExp.name` | 值：`RegExp` |
| `RegExp.prototype` | 值：`/(?:)/` |
| `r.p.flags` | 返回正则表达式的标志 |
| `r.p.global` | 如果使用了全局标志 (`g`)，则返回布尔值 |
| `r.p.ignoreCase` | 如果使用了忽略大小写 (`i`) 标志，则返回布尔值 |
| `r.p.multiline` | 如果使用了多行 (`m`) 标志，则返回布尔值 |
| `r.p.source` | 返回正则表达式的字符串值 |
| `r.p.sticky` | 如果使用了粘性 (`y`) 标志，则返回布尔值 |
| `r.p.unicode` | 如果使用了 unicode (`u`) 标志，则返回布尔值 |
| `r.sticky` | 返回指定从哪里开始查找下一个匹配项的整数 |

正则表达式原型方法

| 方法 | 描述 |
| --- | --- |
| `r.p.compile()` | 无用，只是创建一个 `RegExp` |
| `r.p.constructor()` | 返回 `RegExp` 的构造函数 |
| `r.p.exec(s)` | 返回在字符串 `s` 中找到的匹配数组。索引 0 是匹配项，其他项对应捕获括号。更新 `r` 的属性 |
| `r.p.test(s)` | 如果 `r` 与字符串 `s` 匹配，则返回布尔值 |
| `r.p.toString()` | 返回 `r` 的字符串表示形式 |
