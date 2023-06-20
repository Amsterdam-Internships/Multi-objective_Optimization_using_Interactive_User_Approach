import matplotlib.pyplot as plt
import pandas as pd 
import os
import seaborn as sns

csv_dir = 'experiments/' # directory where all csv files are saved
output_dir = 'experiments/results' # directory where all plots will be saved

# get the list of all csv files in the directory to generate plots together
csv_files = [c_files for c_files in os.listdir(csv_dir) if c_files.endswith('.csv')]

# processing each csv file 
for csvs in csv_files:
    # reading the csv files
    data = pd.read_csv(os.path.join(csv_dir, csvs))

    # extract the columns needed for comparison from the csv files
    num_queries = data['Number of Queries']
    regret = data['Regret']

    # the regret values in the dataset are strings so we need to convert them to floating 
    # point numbers before plotting them
    regret = [float(value[1:-1]) for value in regret]

    # plotting
    # plt.bar(num_queries, regret)
    # plt.ylabel('Regret')
    # plt.xlabel('Number of Queries')
    # plt.title('Number of Queries vs Regret')
    # plt.grid(True)

    df = pd.DataFrame({'Number of Queries': num_queries, 'Regret': regret})
    sns.histplot(data=df, x='Number of Queries', bins=50, stat='count', alpha=0.8)
    # stat is the aggregate statistic to compute in each bin. Stat is chosen to be count to show the 
    # number of observations in each bin
    plt.xlabel('Number of Queries')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Queries until Convergence')
    # plt.show()


    # saving the figure
    plt_file = os.path.splitext(csvs)[0] + '.jpeg'
    plt.savefig(os.path.join(output_dir, plt_file), format='jpg', dpi=300)

    plt.clf()

    # plt.violinplot(regret, showmedians=True)
    # plt.xlabel('Number of Queries')
    # plt.ylabel('Regret')
    # plt.title('Distribution of Regret by Number of Queries')
    
    # # saving the figure
    # plt_file = os.path.splitext(csvs)[0] + '.jpeg'
    # plt.savefig(os.path.join(output_dir, plt_file), format='jpg', dpi=300)

    # plt.clf()
    
print('Finished generation of plots successfully')


