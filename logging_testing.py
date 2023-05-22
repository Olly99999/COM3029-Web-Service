import pandas as pd

def check_log_file(filename):
    try:
        data = pd.read_csv(filename)
        print(data)
    except FileNotFoundError:
        print(f"{filename} not found.")

check_log_file('interaction_log.csv')
