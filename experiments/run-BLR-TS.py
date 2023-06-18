import papermill as pm

num_iter = 100 # number of times to run the notebook
notebook = 'BLR_TS.ipynb' # replace with correct name and path of notebook

# running the notebook for 100 iterations
for run_notebook in range(num_iter):
    print(f"Run: {run_notebook+1}/{num_iter}")

    try:
        pm.execute_notebook(notebook, 'nul', log_output=False)
    except Exception as e:
        print(f"Error occurred while executing: {str(e)}")
print('All runs successful')
