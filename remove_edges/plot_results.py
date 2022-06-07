import numpy as np
import matplotlib.pyplot as plt


file1 = "output/gcn_Citeseer_remove.txt"
file2 = "output/gat_gcnparam_Citeseer_remove.txt"
model1 = "GCN"
model2 = "GAT"
dataset = "Citeseer"

with open(file1, "r") as f1:
    lines1 = f1.readlines()
with open(file2, "r") as f2:
    lines2 = f2.readlines()


def str_to_arr(line):
    line = line[1:-2]
    line.replace(" ", "")
    line = line.split(",")
    line = np.array(line).astype("float64")
    return np.round(line, 2)


avg_node_deg1 = str_to_arr(lines1[7])
avg_node_deg2 = str_to_arr(lines2[7])
acc1 = str_to_arr(lines1[1])
acc2 = str_to_arr(lines2[1])
ece1 = str_to_arr(lines1[3])
ece2 = str_to_arr(lines2[3])
mece1 = str_to_arr(lines1[5])
mece2 = str_to_arr(lines2[5])

fig1, ax1 = plt.subplots()
ax1.plot(
    avg_node_deg1, acc1, color="cornflowerblue", label=model1,
)
ax1.plot(
    avg_node_deg2, acc2, color="sandybrown", label=model2,
)
ax1.set_xlabel("Average node degree", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
ax1.set_title("{}".format(dataset), fontsize=14)
plt.ylim((50, 90))
ax1.xaxis.set_tick_params(labelsize=13)
ax1.yaxis.set_tick_params(labelsize=13)
plt.legend(loc=0, prop={"size": 14})
plt.show()
fig1.savefig("./output/{}_remove_acc.png".format(dataset))

fig2, ax2 = plt.subplots()
ax2.plot(
    avg_node_deg1, ece1, color="cornflowerblue", label=model1,
)
ax2.plot(
    avg_node_deg2, ece2, color="sandybrown", label=model2,
)
ax2.set_xlabel("Average node degree", fontsize=14)
ax2.set_ylabel("ECE", fontsize=14)
ax2.set_title("{}".format(dataset), fontsize=14)
plt.legend(loc=0, prop={"size": 14})
ax2.xaxis.set_tick_params(labelsize=13)
ax2.yaxis.set_tick_params(labelsize=13)
plt.show()
fig2.savefig("./output/{}_remove_ece.png".format(dataset))

fig3, ax3 = plt.subplots()
ax3.plot(
    avg_node_deg1,
    mece1,
    color="cornflowerblue",
    label=model1,
)
ax3.plot(
    avg_node_deg2, mece2, color="sandybrown", label=model2,
)
ax3.set_xlabel("Average node degree", fontsize=14)
ax3.set_ylabel("MECE", fontsize=14)
ax3.set_title("{}".format(dataset), fontsize=14)
plt.legend(loc=0, prop={"size": 14})
ax3.xaxis.set_tick_params(labelsize=13)
ax3.yaxis.set_tick_params(labelsize=13)
plt.show()
fig3.savefig("./output/{}_remove_mece.png".format(dataset))
