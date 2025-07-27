import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intra_migration import LengthDistDataset
from scipy.stats import entropy
from scipy import interpolate
import scipy.integrate as integrate
import warnings

warnings.filterwarnings("ignore")
step_num = 10
# dataset = LengthDistDataset("/home/twubt/workspace/ROLL/experiments/lunxi_logs/16k/step5_bs64_n4/logs_without_state/unmig_r1/output_lens.log"
# , step_num)
dataset = LengthDistDataset("output_lens_10_unmig.log"
, step_num)
lengths = dataset[0]

def cdf(lengths): 
    cdf = np.cumsum(np.histogram(lengths, bins=len(lengths))[0])/len(lengths)
    # plt.plot(sorted(lengths), cdf)
    cdf_func = interpolate.interp1d(sorted(lengths), cdf)
    def extended_cdf_func(x):
        if x <= min(lengths):
            return 0
        elif x >= max(lengths):
            return 1
        else:
            return cdf_func(x)
    return extended_cdf_func

def hist(lengths): 
    hist_val, bin_edges = np.histogram(lengths, bins=len(lengths))
    hist_val = np.concat([hist_val, [hist_val[-1]]])
    # plt.plot(sorted(lengths), cdf)
    hist_func = interpolate.interp1d(bin_edges, hist_val)
    minl, maxl = min(lengths), max(lengths)
    def extended_hist_func(x):
        if x <= minl:
            return 0
        elif x >= maxl:
            return 0
        else:
            return hist_func(x)
    return extended_hist_func

def inv_cdf(lengths):
    cdf = np.cumsum(np.histogram(lengths, bins=len(lengths))[0])/len(lengths)
    mimi = interpolate.interp1d(cdf, sorted(lengths))
    return mimi


def similarity(cdf1, cdf2, left, right):
    integrand = lambda x: np.abs(cdf1(x) - cdf2(x)) / (max(cdf1(x), cdf2(x)))
    d, _ = integrate.quad(integrand, left, right)
    return 1 - d / (right - left)



similarities_cdf = []
similarities_hist = []

# consider direction of quantity
quant_90 = []
quant_50 = []
quant_25 = []

# # plot cdf or similarity
# for i in range(1,step_num):
#     left = min(np.min(dataset[i]), np.min(dataset[i-1]))
#     right = max(np.max(dataset[i]), np.max(dataset[i-1]))
#     similarities_cdf.append(similarity(cdf(dataset[i]), cdf(dataset[i-1]), left, right))
#     # similarities_hist.append(similarity(hist(dataset[i]), hist(dataset[i-1]), left, right))
# #     inv_cdf1 = inv_cdf(dataset[i])
# #     inv_cdf2 = inv_cdf(dataset[i-1])
# #     quant_90.append(inv_cdf1(0.9)-inv_cdf2(0.9))
# #     quant_50.append(inv_cdf1(0.5)-inv_cdf2(0.5))
# #     quant_25.append(inv_cdf1(0.25)-inv_cdf2(0.25))

#     x = np.linspace(200,16500, 10000)
#     plt.plot(x, [cdf(dataset[i])(y) for y in x], color = "blue", alpha = 0.1*i)


# print("quantity 90", quant_90)
# print("quantity 50", quant_50)
# print("quantity 25", quant_25)


print(similarities_cdf)
## compare of 
# plt.figure(figsize=(6, 2))
# print(np.mean(similarities_hist), np.mean(similarities_cdf))
# plt.plot(np.arange(len(similarities_hist)), similarities_hist, '-v', label='Hist')
# plt.plot(np.arange(len(similarities_cdf)), similarities_cdf, '-o', label='CDF')

# plt.legend()
# plt.xlabel("Step")
# plt.ylabel("Similarity")
# plt.savefig("figures/similarity.png")

# plt.savefig('figures/cdf_sim.png')


## plot the kl-divergence
# kl_divergences = []
# for i in range(1,step_num):
#     kl_divergences.append(entropy(dataset[i],dataset[i-1]))
# print("kl_divergence:", kl_divergences)


# plot two cdfs
x = np.linspace(200,4200, 2000)
plt.plot(x, [cdf(dataset[6])(y) for y in x], color = "blue", alpha = 0.5)
plt.plot(x, [cdf(dataset[7])(y) for y in x], color = "blue", alpha = 1)
plt.savefig("figures/cdf_6_7_4k.png")


# # plot the change of histogram 
# for i in range(step_num):
#     if i % 2 == 1:
#         color_name = "blue"
#     else:
#         color_name = "yellow"
#     plt.hist(dataset[i], bins = 50, alpha = (0.8/step_num) * (i+1), color = color_name ) 

# plt.savefig('figures/qaq.png')


