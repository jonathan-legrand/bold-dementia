# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from data.phenotypes import days_to_onset

date_parser = lambda x: datetime.strptime(x, '%d %m %Y')

phenotypes = pd.read_csv(
    "~/data/Memento/phenotypes.tsv",
    sep="\t",
    encoding="unicode_escape"
)
# %%

# I don't trust date parsing in read_csv
for date_col in {
        "DEMENCE_DAT",
        "INCCONSDAT_D",
        "IRM_M0",
        "IRM_M24",
        "IRM_M48"
    }:
    phenotypes[date_col] = pd.to_datetime(
        phenotypes[date_col],
        format="%d/%m/%Y"
    )
# %%
phenotypes["declared_dementia"] = ~phenotypes.DEMENCE_DAT.isna()
# %%
ax = sns.countplot(phenotypes, x="APOE_geno")
plt.show()

# %%
ax = sns.countplot(phenotypes, x="NIVETUD")
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 5))
sns.swarmplot(
    phenotypes,
    x="DEMENCE_DAT",
    y="SEX",
    hue="SEX",
    ax=ax,
    palette={"Masculin": "tab:blue", "Féminin": "tab:pink"}
)
ax.tick_params(axis='x', labelrotation=45)
plt.title(f"Declared dementia : {phenotypes.declared_dementia.sum()}")
plt.show()

# %%

plt.subplots(figsize=(10, 10))
sns.violinplot(
    phenotypes,
    x="AGE_CONS",
    hue="SEX",
    y="SEX",
    inner=None,
    fill=False,
    bw_method=0.1,
    palette={"Masculin": "tab:blue", "Féminin": "tab:pink"},
)
sns.swarmplot(
    phenotypes,
    x="AGE_CONS",
    hue="declared_dementia",
    y="SEX",
    palette={True: "tab:red", False: "black"},
)

plt.title(f"Memento participants, {phenotypes.declared_dementia.sum()} demented subjects")
plt.show()


# %%
s = (phenotypes.IRM_M24 < phenotypes.DEMENCE_DAT) & (phenotypes.IRM_M48 >= phenotypes.DEMENCE_DAT)
s.sum()
# %%

(phenotypes.DEMENCE_DAT > phenotypes.IRM_M48).sum()
# %%
(phenotypes.IRM_M0 >= phenotypes.DEMENCE_DAT).sum()
# %%
msk = (phenotypes.declared_dementia) & (phenotypes.IRM_M48.isna())
phenotypes[msk].iloc[1, :]
# %%
phenotypes[(~phenotypes.IRM_M48.isna()) & phenotypes.DEMENCE_DAT.isna()]
phenotypes["time_to_onset"] = phenotypes.DEMENCE_DAT - phenotypes.INCCONSDAT_D
phenotypes["days_to_onset"] = days_to_onset(phenotypes.INCCONSDAT_D, phenotypes.DEMENCE_DAT)
# %%
# %%

def timedelta_to_years(td):
    return td.days / 365
    
phenotypes["age_of_onset"] = phenotypes["time_to_onset"].map(timedelta_to_years) + phenotypes.AGE_CONS
# %%


ax = sns.boxplot(phenotypes, x="age_of_onset", fill=None)
sns.swarmplot(phenotypes, x="age_of_onset", color="tab:blue")


# %%

plt.subplots(figsize=(5, 8))

sns.boxplot(
    phenotypes,
    x="age_of_onset",
    y="APOE_geno",
    hue="SEX",
    palette={"Masculin": "tab:blue", "Féminin": "tab:pink"},
    fill=None
)


plt.show()
# %%
plt.subplots(figsize=(5, 8))
sns.swarmplot(
    phenotypes,
    x="age_of_onset",
    y="MCI",
    hue="SEX",
    palette={"Masculin": "tab:blue", "Féminin": "tab:pink"},
)
# %%
plt.subplots(figsize=(5, 8))
sns.swarmplot(
    phenotypes,
    x="age_of_onset",
    y="NIVETUD",
    hue="SEX",
    palette={"Masculin": "tab:blue", "Féminin": "tab:pink"},
)
# %%
ax = sns.histplot(
    phenotypes.time_to_onset.map(lambda x: x.days),
    bins=50
)
plt.axvline(0, 0, 1, color="green", label="MRI")
plt.axvline(365, 0, 1, color="red", label="Blood sampling")
plt.axvline(730, 0, 1, color="green")
plt.axvline(1095, 0, 1, color="red")
plt.axvline(1460, 0, 1, color="green")
plt.axvline(1825, 0, 1, color="red")
plt.legend()
plt.show()

# %%
sns.swarmplot(
    phenotypes,
    x="days_to_onset"
)
