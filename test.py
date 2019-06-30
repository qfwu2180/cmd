# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(221)
# line=ax.plot([0,1],[0,1],'b')
# ax.set_ylim([0,1])
# ax.set_xlim([0,1])
#
# plt.ion()
# plt.show()
# for i in range(5):
#     try:
#         ax.lines.remove(lines[0])
#     except:
#         pass
#     lines = ax.plot([0,1],[0,i/5.0], 'r--')
#     plt.pause(1)

# import tensorflow as tf
# import matplotlib.pyplot as plt
# plt.figure(1) # 第一张图
# plt.subplot(211) # 第一张图中的第一张子图
# plt.plot([1,2,3])
# plt.subplot(212) # 第一张图中的第二张子图
# plt.plot([4,5,6])
# plt.figure(2) # 第二张图
# plt.plot([4,5,6]) # 默认创建子图subplot(111)
# plt.figure(1) # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
# plt.subplot(211) # 令子图subplot(211)成为figure1的当前图
# plt.title('Easy as 1,2,3') # 添加subplot 211 的标题
# plt.show()

# import tensorflow as tf
# import matplotlib.pyplot as plt
# plt.figure(1) # 第一张图
# plt.subplot(211) # 第一张图中的第一张子图
# plt.plot([1,2,3])
# plt.subplot(212) # 第一张图中的第二张子图
# plt.plot([4,5,6])
# plt.figure(2) # 第二张图
# plt.plot([4,5,6]) # 默认创建子图subplot(111)
# plt.figure(1) # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
# plt.subplot(211) # 令子图subplot(211)成为figure1的当前图
# plt.title('Easy as 1,2,3') # 添加subplot 211 的标题
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 100, 0, 1])
plt.ion()

xs = [0, 0]
ys = [1, 1]

for i in range(10):
    y = np.random.random()
    xs[0] = xs[1]
    ys[0] = ys[1]
    xs[1] = i
    ys[1] = y
    plt.plot(xs, ys)
    plt.pause(0.1)
