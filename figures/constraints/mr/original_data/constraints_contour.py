# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def density_estimation(m1, m2):
    X, Y = np.mgrid[min(m1) : max(m1) : 100j, min(m2) : max(m2) : 100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


# %%
tab = "NICER+XMM-relative_J0740_RM.txt"
m1, r1 = np.loadtxt(tab, unpack=True, usecols=(0, 4), skiprows=1)
X1, Y1, Z1 = density_estimation(r1, m1)
m2, r2 = np.loadtxt(tab, unpack=True, usecols=(1, 5), skiprows=1)
X2, Y2, Z2 = density_estimation(r2, m2)
fig, ax = plt.subplots()

# Add contour lines
cs1 = plt.contour(X1, Y1, Z1, levels=[0.300])
# ax.clabel(cs1, fontsize=10)
ax.plot(r1, m1, "k.", markersize=2)

cs2 = plt.contour(X2, Y2, Z2, levels=[0.300])
# ax.clabel(cs2, fontsize=10)
ax.plot(r2, m2, "k.", markersize=2)

ax.set_xlim([8, 16])
ax.set_ylim([1, 3])
plt.show()

p1 = cs1.collections[0].get_paths()[0]
coor_p1 = p1.vertices

p2 = cs2.collections[0].get_paths()[0]
coor_p2 = p2.vertices

with open("GW170817_UR.M1R1", "w") as f:
    f.write("{0:^8} {1:^8}\n".format("r1", "m1"))
    f = np.savetxt(f, coor_p1, fmt="%.4e")

with open("GW170817_UR.M2R2", "w") as f:
    f.write("{0:^8} {1:^8}\n".format("r2", "m2"))
    f = np.savetxt(f, coor_p2, fmt="%.4e")

# %%
import plotly.express as px

tab = "J0030_3spot_RM.txt"  #'NICER+XMM-relative_J0740_RM.txt'
df = pd.read_csv(
    tab, sep="\s+"
)  # np.loadtxt(tab, unpack=True, usecols=(1,0), skiprows=6)
# df = df.drop('w',axis=1)
for column in df:
    if df[column].dtype == "float64":
        df[column] = pd.to_numeric(df[column], downcast="float")
# df.info(memory_usage="deep")

data = df.sample(frac=0.0075)

data = data.to_numpy()
data = data.T
r1 = data[0]
m1 = data[1]
X1, Y1, Z1 = density_estimation(r1, m1)

# %%
fig, ax = plt.subplots()
# Add contour lines
cs1 = plt.contour(X1, Y1, Z1, levels=[0.150])
ax.clabel(cs1, fontsize=10)
ax.plot(r1, m1, "k.", markersize=2)

ax.set_xlim([8, 18])
ax.set_ylim([1, 3])
plt.show()

p1 = cs1.collections[0].get_paths()[0]
coor_p1 = p1.vertices
# %%
with open("Miller_PSR_J0030.MR", "w") as f:
    f.write("{0:^8} {1:^8}\n".format("r1", "m1"))
    f = np.savetxt(f, coor_p1, fmt="%.4e")

from scipy import stats


def density_estimation(m1, m2):
    X, Y = np.mgrid[min(m1) : max(m1) : 100j, min(m2) : max(m2) : 100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


path = "/home/lucaslazzari/Dropbox/research/constraints/"
path += "original_data/HESS_J1731-347.txt"
df_hess = pd.read_csv(path, sep=" ")

m1, r1 = (
    df_hess["M"].to_numpy(),
    df_hess["R"].to_numpy(),
)  # np.loadtxt(path, unpack=True, usecols=(0,4), skiprows=1)
X1, Y1, Z1 = density_estimation(r1, m1)
fig, ax = plt.subplots()

# Add contour lines
ax.plot(r1, m1, "k.", markersize=2)
cs1 = plt.contour(X1, Y1, Z1, levels=[0.05], colors="red")

ax.set_xlim([5, 15])
ax.set_ylim([0, 2])
plt.show()

p1 = cs1.collections[0].get_paths()[0]
coor_p1 = p1.vertices

df = pd.DataFrame({"r": coor_p1.T[0], "m": coor_p1.T[1]})
df.to_csv(
    "/home/lucaslazzari/Dropbox/research/constraints/constraints_contours/HESS_J1731-347.csv",
    sep=" ",
    float_format="%.5e",
    index=False,
)
