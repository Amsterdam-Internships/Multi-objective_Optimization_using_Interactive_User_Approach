import matplotlib.pyplot as plt
import pandas as pd 
import os
import seaborn as sns

csv_dir = 'experiments/regret-csvs' # directory where all csv files are saved
output_dir = 'experiments/results/regret-4-100.jpeg' # directory where all plots will be saved

csv_file_1 = 'experiments/regret-csvs/regret-EI_4-100.csv'
csv_file_2 = 'experiments/regret-csvs/regret-TS_4-100.csv'
csv_file_3 = 'experiments/regret-csvs/regret-BLR_4-100.csv'

df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)
df3 = pd.read_csv(csv_file_3)

steps1 = df1['Step']
steps2 = df2['Step']
steps3 = df3['Step']

df1['Average Regret'] = df1['Regret'].apply(lambda x: float(x.strip('[]')))
df2['Average Regret'] = df2['Regret'].apply(lambda x: float(x.strip('[]')))
df3['Average Regret'] = df3['Regret'].apply(lambda x: float(x.strip('[]')))

avge1 = df1.groupby(steps1)['Average Regret'].mean()
avge2 = df2.groupby(steps2)['Average Regret'].mean()
avge3 = df3.groupby(steps3)['Average Regret'].mean()

plt.plot(avge1.index, avge1.values, label='GP-EI', color='orange')
plt.plot(avge2.index, avge2.values, label='GP-TS', color='green')
plt.plot(avge3.index, avge3.values, label='BLR-TS', color='purple')

plt.xlabel('Number of Queries')
plt.ylabel('Average Regret')
plt.title('Average Regret as a function of the Number of Queries')
plt.legend()

plt.ylim(-0.001,0.093)
plt.xlim(0, 100)

# saving the figure
# plt_file = os.path.splitext(csvs)[0] + '.jpeg'
plt.savefig(output_dir, dpi=300)

plt.clf()

print('Finished generation of plots successfully')



