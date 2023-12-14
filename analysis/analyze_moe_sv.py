import torch as th
import matplotlib.pyplot as plt

for weight_idx in range(1, 4):
    print(weight_idx)
    w_sv = th.load(f"layer_moe_svs/w{weight_idx}_svdvals.pt").cpu().detach()
    
    plt.plot(w_sv)
    
    plt.savefig(f"w{weight_idx}_sv_graphed.png")
    plt.clf()
    del w_sv