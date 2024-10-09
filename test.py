import numpy as np
import matplotlib.pyplot as plt

# # 定義曲線函數
# def exponential_curve(x, k=5):
    
#     return (5*(x))**3
#     # return (np.exp(k*((0.3-x))**2))

# # 創建 x 軸數據 (範圍 0 到 1)
# x = np.linspace(0, 1, 100)

# # 計算 y 值
# y_cal = lambda x: exponential_curve(x, k=1) if x <0.3 else 0
# # y = exponential_curve(x, k=5) 
# y = [y_cal(i) for i in x]
# # 繪製曲線
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Exponential-like Curve')
# plt.grid(True)
# plt.show()

steer = -1.25
# if a !=0:
#     print('a')
# else:
#     print('b')
if -0.8<steer < 0.8:
    print(True)

a = 4
b = 3

if b> 4> a:
    print('True123')