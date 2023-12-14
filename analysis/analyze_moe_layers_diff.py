import torch as th

for i in range(32):
    print(i)
    moe = th.load(f"layer_{i}_moe.pt").cuda()
    w1_sv = th.zeros(8, 8, 4096).cuda()
    w2_sv = th.zeros(8, 8, 4096).cuda()
    w3_sv = th.zeros(8, 8, 4096).cuda()
    for j in range(1, 8):
        for k in range(j):
            print(f"{j} - {k}")
            w1_sv[j, k] = th.linalg.svdvals(moe.experts[j].w1.weight - moe.experts[k].w1.weight)
            w2_sv[j, k] = th.linalg.svdvals(moe.experts[j].w2.weight - moe.experts[k].w2.weight)
            w3_sv[j, k] = th.linalg.svdvals(moe.experts[j].w3.weight - moe.experts[k].w3.weight)
    th.save(w1_sv, f"layer_{i}_moe_w1_sv.pt")
    th.save(w2_sv, f"layer_{i}_moe_w2_sv.pt")
    th.save(w3_sv, f"layer_{i}_moe_w3_sv.pt")