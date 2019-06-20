import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[13,1,1,0,2,0,45,23],
     [3,9,6,0,1,0,21,43],
     [0,0,16,2,0,0,65,45],
     [1,10,15,13,0,68,2,23],
     [2,9,20,0,15,0,72,95],
     [3,8,25,0,0,15,76,90],
     [4,7,30,0,15,0,78,85],
     [5,6,35,40,45,50,13,80]]        
df_cm = pd.DataFrame(array, range(8),
                  range(8))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
plt.show()