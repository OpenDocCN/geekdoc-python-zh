# Python 中的多臂强盗问题

> 原文：<https://www.askpython.com/python/examples/bandit-problem-in-python>

n 臂强盗问题是一个强化学习问题，其中给代理一个有 n 个强盗/臂的老虎机。吃角子老虎机的每个手臂都有不同的获胜机会。拉任何一条手臂要么奖励要么惩罚代理人，即成功或失败。

代理人的目标是一次拉一个强盗/武器，这样操作后获得的总奖励最大化。此外，问题描述指定代理不知道武器成功的可能性。它最终通过试错和价值评估来学习。

本教程将教我们如何利用策略梯度方法，该方法使用 TensorFlow 构建一个基本的神经网络，该网络由与每个可用分支获得老虎机奖金的可能性成比例的权重组成。在这个策略中，代理基于贪婪方法选择机器手臂。

这意味着代理通常会选择具有最高预期值的动作，但它也会随机选择。

通过这种方式，间谍测试了几把枪中的每一把，以便更好地了解它们。当代理采取行动时，例如选择吃角子老虎机的一只手臂，它会得到 1 或-1 的奖励。

* * *

## **用 Python 实现 Bandit 问题**

以下是用 Python 编写的 n 臂/多臂 bandit 问题的简单实现:

对于我们的代码实现，我们选择 n=6(老虎机的 6 个臂)，它们的数量是[2，0，0.2，-2，-1，0.8]。

我们将逐步发现，代理人学习并有效地选择具有最高收益的强盗。

* * *

### **第一步:导入模块**

方法`tf.disable_v2_behavior`(顾名思义)禁用在 **TensorFlow 1.x 和 2.x 之间更改的任何全局行为，并使它们按照 1.x 的意图运行**

```py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```

* * *

### **步骤 2:计算武器奖励**

我们指定我们的土匪在插槽武器阵列。数组的长度存储在`len_slot_arms`中。方法 discovers()从正态分布中创建一个平均值为 0 的随机整数。

武器/强盗数量越少，代理越有可能返回正奖励(1)。

```py
slot_arms = [2,0,0.2,-2,-1,0.8]
len_slot_arms = len(slot_arms)
def findReward(arm):
    result = np.random.randn(1)
    if result > arm:
        #returns a positive reward
        return 1
    else:
        #returns a negative reward
        return -1

```

* * *

### **步骤 3:设置神经代理**

TensorFlow 库的方法`tf.rese_default_graph`清除默认图形堆栈并重置全局默认图形。第 2 行和第 3 行将特定强盗的权重确定为 1，然后进行实际的武器选择。

```py
tf.reset_default_graph()
weights = tf.Variable(tf.ones([len_slot_arms]))
chosen_action = tf.argmax(weights,0)

```

训练由下面的代码处理。它最初向网络提供奖励和指定的动作(arm)。然后，神经网络使用下面所示的算法计算损失。这种损失然后被用于通过更新网络来提高网络性能。

```py
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

Loss = -log(weight for action)*A(Advantage from baseline(here it is 0)).

```

* * *

### **第四步:训练特工并找到最佳武器/强盗**

我们通过做随机活动和获得激励来训练代理。上面的代码启动一个 TensorFlow 网络，然后选择一个随机动作，从其中一个手臂中选择一个奖励。这种激励有助于网络更新，并且也显示在屏幕上。

```py
total_episodes = 1000
total_reward = np.zeros(len_slot_arms) #output reward array
e = 0.1 #chance of taking a random action.
init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)
  i = 0

  while i < total_episodes:
    if np.random.rand(1) < e:
      action = np.random.randint(len_slot_arms)
    else:
      action = sess.run(chosen_action)
    reward = findReward(slot_arms[action])
    _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
    total_reward[action] += reward
    if i % 50 == 0:
      print ("Running reward for the n=6 arms of slot machine: " + str(total_reward))
    i+=1

print ("The agent thinks bandit " + str(np.argmax(ww)+1) + " has highest probability of giving poistive reward")
if np.argmax(ww) == np.argmax(-np.array(slot_arms)):
  print("which is right.")
else:
  print("which is wrong.")

```

* * *

## **结论**

恭喜你！您刚刚学习了如何用 Python 编程语言解决多臂强盗问题。我希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [FizzBuzz 问题——用 Python 实现 FizzBuzz 算法](https://www.askpython.com/python/examples/fizzbuzz-algorithm)
2.  [用 Python 解决梯子问题](https://www.askpython.com/python/examples/ladders-problem)
3.  [使用递归在 Python 中求解 0-1 背包问题](https://www.askpython.com/python/examples/knapsack-problem-recursion)
4.  [在 Python 中解决平铺问题](https://www.askpython.com/python/examples/tiling-problem)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *