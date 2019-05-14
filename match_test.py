from __future__ import print_function
import os
import random
from tqdm import trange
import torch
import torch.nn.parallel
from torch.autograd import Variable
import json
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import models.curve_compare_mode as ccm
from skyline_dataloader import *


def eval_topN(net, path, device):
    net.eval()

    results_save = []

    with torch.no_grad():
        filenames = os.listdir(path)[:200]
        acc_top1 = 0
        acc_top5 = 0
        acc_top10 = 0
        acc_top20 = 0
        acc_top50 = 0
        acc_top100 = 0
        acc_top200 = 0

        count = 1
        for _ in range(count):
            for _ in trange(100):
                index_target = random.randint(0, len(filenames) - 1)
                file0 = read_data_row(os.path.join(path, filenames[index_target]), 1).strip().split(" ")
                line0 = np.array(list(map(int, file0[0].split(','))))
                dist = []
                result_item = {}
                for f in filenames:
                    file1 = read_data_row(os.path.join(path, f), 1).strip().split(" ")
                    line1 = np.array(list(map(int, file1[1].split(','))))
                    length = len(line1)
                    line = np.hstack((line0, line1))
                    line_min, line_max = line.min(), line.max()
                    line = (line - line_min) / (line_max - line_min)
                    line1 = torch.from_numpy(line[:length].reshape(1, 1, -1))
                    line2 = torch.from_numpy(line[length:].reshape(1, 1, -1))

                    output = net(line1.float().to(device), line2.float().to(device))
                    dist.append(torch.sigmoid(output)[0][1].item())

                dist = pd.Series(np.array(dist)).sort_values(ascending=False)

                flag_tp1 = False
                if index_target in dist[:1].index.tolist():
                    acc_top1 += 1
                    flag_tp1 = True
                if index_target in dist[:5].index.tolist():
                    acc_top5 += 1
                if index_target in dist[:10].index.tolist():
                    acc_top10 += 1
                if index_target in dist[:20].index.tolist():
                    acc_top20 += 1
                if index_target in dist[:50].index.tolist():
                    acc_top50 += 1
                if index_target in dist[:100].index.tolist():
                    acc_top100 += 1
                if index_target in dist[:200].index.tolist():
                    acc_top200 += 1

                result_item["filenames"] = filenames
                result_item["target"] = index_target
                result_item["dist"] = dist.index.tolist()
                result_item["status"] = "success" if flag_tp1 else "failed"
                results_save.append(result_item)

            with open("./data/result.json", 'w') as f:
                json.dump(results_save, f)
                print("保存json文件完成....")


        acc_top1 /= 100.0 * count
        acc_top5 /= 100.0 * count
        acc_top10 /= 100.0 * count
        acc_top20 /= 100.0 * count
        acc_top50 /= 100.0 * count
        acc_top100 /= 100.0 * count
        acc_top200 /= 100.0 * count
        acc_mean = (acc_top1 + acc_top5 + acc_top10 + acc_top20 + acc_top50 + acc_top100 + acc_top200) / 7
        sf = ' acc_top1: %.3f \n acc_top5: %.3f \n acc_top10: %.3f \n acc_top20: %.3f \n acc_top50: %.3f \n acc_top100: %.3f \n acc_top200: %.3f \n acc_mean: %.3f'
        print(sf% (
        acc_top1,
        acc_top5,
        acc_top10,
        acc_top20,
        acc_top50,
        acc_top100,
        acc_top200,
        acc_mean))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ccm.Curve_Compare()
    model.load_state_dict(torch.load("./checkpoint/new/2019-05-09_best.pth"))
    model.to(device)

    eval_topN(model, "./data/test", device)