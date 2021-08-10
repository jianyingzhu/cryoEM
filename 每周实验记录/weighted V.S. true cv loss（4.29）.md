### weighted l2 reconstruction, random quat ratio = 50%, random.seed(0)
weighted cv loss：weighted l2 reconstruction。

true cv loss：只挑选了正确颗粒做普通l2重构，计算cv loss。(weight选中真值相当于对所有正确颗粒做l2重构并计算cv loss)

normal cv loss：对所有颗粒（50%正确率）做普通l2重构，计算cv loss。

| Protein | Particle numbers | weighted cv loss| true cv loss |
| :-----: | :-----: | :-----: | :-----: | 
| cng | 100 | 25466.82/25488.46/25474.87 | 25318.14 |
| cng | 1K | 25502.10 | 25689.61 |  
| proteasome | 100 | 102359.94 | 100185.14 |
| proteasome | 1K | 102895.34 | 102170.22 |
| fun30 | 100 | 39494.08 | 39300.02 |
| fun30 | 1K | 39866.88 | 40083.74 |

几个观察：
1、weight集中到的单点正确率不高（大约一半）。
2、让weight从正确解（真值）开始迭代，结果依然是收敛到单点。
3、weighted reconstruction过拟合。







