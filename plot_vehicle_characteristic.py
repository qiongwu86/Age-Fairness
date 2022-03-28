import matplotlib.pyplot as plt
import numpy as np
from dealData import *

n_max = 10
arrivalMax = 6
leaveMax = 6

ps100_maxAvg = [1.0, 0.9579350933157438, 0.956950311258461, 0.9558579330301898, 0.9545677418679763, 0.9544831363785367, 0.9555091123605012, 0.9561850032540906, 0.9568699636832187, 0.9558499139738066]
ps75_maxAvg = [1.0, 0.945765719310291, 0.94391287023183, 0.9430394690673252, 0.9431316096747797, 0.9430447074243998, 0.9438435459196364, 0.9435177683174727, 0.9424946089570857, 0.94319114269872]
ps100_arrivalAvg = [0.981787141929198, 0.9662844688377332, 0.9511517476071876, 0.9486156174082567, 0.9506811342577268, 0.9494201186010625]
ps75_arrivalAvg = [0.9816084930062882, 0.9582539453229262, 0.9485074500821898, 0.9462437370566401, 0.9479132083722487, 0.9479362499208157]
ps100_leaveAvg = [0.9478842264695527, 0.9505579453624895, 0.954316034812709, 0.9641580155138457, 0.9769913905274719, 0.9787385711104243]
ps75_leaveAvg = [0.9438544675423148, 0.9465807508076309, 0.948874196047707, 0.9578770252167131, 0.975232106228013, 0.9779718326104196]

x_range = [x for x in range(1,n_max+1)]
plt.plot(x_range, ps100_maxAvg, linestyle='--', marker='d', label='ps=1.0')
plt.plot(x_range, ps75_maxAvg, color='red',linestyle='--', marker='p', label='ps=0.75')
plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Max device number')
plt.ylabel('Age fairness utility')
# plt.title('Different max device number rewards cuve \n of complex vehicle scenario')
plt.show()

x_range = [x for x in range(1,arrivalMax+1)]
plt.plot(x_range, ps100_arrivalAvg, linestyle='--', marker='d', label='ps=1.0')
plt.plot(x_range, ps75_arrivalAvg, color='red',linestyle='--', marker='p', label='ps=0.75')
plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Vehicle arrival rate')
plt.ylabel('Age fairness utility')
# plt.title('Different arrival rate rewards cuve \n of complex vehicle scenario.')
plt.show()

x_range = [x for x in range(1,leaveMax+1)]
# plt.plot(x_range, [1]*leaveMax, color='red', label='Absolute fair limit')
plt.plot(x_range, ps100_leaveAvg, linestyle='--', marker='d', label='ps=1.0')
plt.plot(x_range, ps75_leaveAvg, color='red',linestyle='--', marker='p', label='ps=0.75')
plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Vehicle departure rate')
plt.ylabel('Age fairness utility')
# plt.title('Different departure rate rewards cuve \n of complex vehicle scenario')
plt.show()






