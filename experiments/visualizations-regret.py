import matplotlib.pyplot as plt
import pandas as pd 
import os
import seaborn as sns

csv_dir = 'experiments/regret-csvs' # directory where all csv files are saved
output_dir = 'experiments/results/regret-7-100.jpeg' # directory where all plots will be saved

# # get the list of all csv files in the directory to generate plots together
# csv_files = [c_files for c_files in os.listdir(csv_dir) if c_files.endswith('.csv')]

# # processing each csv file 
# for csvs in csv_files:
#     # reading the csv files
#     data = pd.read_csv(os.path.join(csv_dir, csvs))

#     # extract the columns needed for comparison from the csv files
#     step = data['Step']
#     regret = data['Regret']

#     # the regret values in the dataset are strings so we need to convert them to floating 
#     # point numbers before plotting them
#     regret = [float(value[1:-1]) for value in regret]

#     df = pd.DataFrame({'Step': step, 'Regret': regret})
#     sns.histplot(data=df, x='Step', bins=50, stat='count', alpha=0.8)
#     # stat is the aggregate statistic to compute in each bin. Stat is chosen to be count to show the 
#     # number of observations in each bin
#     plt.xlabel('Steps')
#     plt.ylabel('Count')
#     plt.title('Distribution of Number of Queries as a function of the Regret')
#     # plt.show()

#     # saving the figure
#     plt_file = os.path.splitext(csvs)[0] + '.jpeg'
#     plt.savefig(os.path.join(output_dir, plt_file), format='jpg', dpi=300)

#     plt.clf()
    
# print('Finished generation of plots successfully')

df1 = pd.read_csv('experiments/regret-csvs/regret-EI_7-100.csv')
df2 = pd.read_csv('experiments/regret-csvs/regret-TS_7-100.csv')
df3 = pd.read_csv('experiments/regret-csvs/regret-BLR_7-100.csv')

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

# saving the figure
# plt_file = os.path.splitext(csvs)[0] + '.jpeg'
plt.savefig(output_dir, dpi=300)

plt.clf()

print('Finished generation of plots successfully')



