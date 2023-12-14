import torch as th


full_w = th.empty(32, 8, 14336, 4096)

for i in range(32):
    print(f"loading moe layer {i}")
    moe = th.load(f"layer_{i}_moe.pt")
    for j in range(8):
        full_w[i, j] = moe.experts[j].w3.weight
    
    del moe

full_w = full_w.view(32 * 8, 14336 * 4096).transpose(0, 1)

svdvals = th.linalg.svdvals(full_w)

th.save(svdvals, "w3_svdvals.pt")
