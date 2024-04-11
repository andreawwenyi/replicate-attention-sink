import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nlls = list()
data = list()
experiment_name = 'window-attention'
with open(f"outputs/{experiment_name}/logs.txt", "r") as f:
    for line in f.readlines():
        data.append({
            "loc": int(re.findall("loc: (.*?),", line)[0]),
            "nll": float(re.findall("nll: (.*?),", line)[0]),
            "experiment": experiment_name
        })
experiment_name = 'sink-with-pos-shift'
with open(f"outputs/{experiment_name}/logs.txt", "r") as f:
    for line in f.readlines():
        data.append({
            "loc": int(re.findall("loc: (.*?),", line)[0]),
            "nll": float(re.findall("nll: (.*?),", line)[0]),
            "experiment": experiment_name
        })

data = pd.DataFrame(data)
data['rolling_nll'] = data['nll'].rolling(10).mean()
g = sns.relplot(data = data[data['loc']<=3000], x='loc', y='rolling_nll', col='experiment', kind='line', height=3, aspect=3)
g.set_ylabels("log(ppl)")
g.set_titles(col_template="{col_name}")

plt.savefig("figure.pdf")


