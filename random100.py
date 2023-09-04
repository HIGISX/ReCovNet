import pandas as pd
import numpy as np
import geopandas as gpd
import torch
import os

num_sample = 128000
M = 100
r = 2000
p = 20

uses = 'train'
#uses = 'valid'
# uses = 'test'

ls = gpd.read_file(r".\data\real\nyc_spatial_units_wspeed.shp")
ls['POINT_X'] = ls.geometry.x
ls['POINT_Y'] = ls.geometry.y
ls['speed_pct_freeflow_rev'] = ls.pct_from_f * (-1)
sitedf = pd.read_csv(r".\data\real\billboards_nyc.csv")

def Normalization(x, y):
    max_x, max_y = np.max(x), np.max(y)
    min_x, min_y = np.min(x), np.min(y)
    S_x = (max_x-min_x)
    S_y = (max_y-min_y)
    S = max(S_x, S_y)
    new_x, new_y = (x-min_x)/S, (y-min_y)/S
    data_xy = np.vstack((new_x, new_y))
    Data = data_xy.T
    return new_x, new_y, S


ls_X = np.array(ls['POINT_X'])
ls_Y = np.array(ls['POINT_Y'])
bbs_X = np.array(sitedf['POINT_X'])
bbs_Y = np.array(sitedf['POINT_Y'])
X = np.concatenate([ls_X, bbs_X])
Y = np.concatenate([ls_Y, bbs_Y])
NORM_X, NORM_Y, S = Normalization(X, Y)
ls['NORM_X'] = NORM_X[:len(ls)]
ls['NORM_Y'] = NORM_Y[:len(ls)]
sitedf['NORM_X'] = NORM_X[len(ls):]
sitedf['NORM_Y'] = NORM_Y[len(ls):]



def gen_real_data(num_sample, M, p ,r):
    real_datasets = []
    for i in range(num_sample):
        index = np.random.choice(len(sitedf), M)
        selected_sites_list = sitedf.iloc[index]
        real_data = {}
        real_data["users"] = torch.tensor(np.array(ls[['NORM_X', 'NORM_Y']])).to(torch.float32)
        real_data["facilities"] = torch.tensor(np.array(selected_sites_list[['NORM_X', 'NORM_Y']])).to(torch.float32)
        real_data['demand'] = torch.tensor(np.array(ls['speed_pct_freeflow_rev'])).to(torch.float32)
        real_data["p"] = p
        real_data["r"] = r/S
        real_datasets.append(real_data)
        print(i)
#        print(real_datasets)
    return real_datasets

def generate_realdata(num_sample, n_facilities, p, r):
    filename = os.path.join(r".\data\MCLP\MCLP_" + str(r) + "_" + str(p), f"MCLP_100_" + str(p) + "_"
                            + uses + "_Normalization"+".pkl")
    dataset = random100.gen_real_data(num_sample, n_facilities, p, r)
    data_utils.save_dataset(dataset, filename)
    print(filename)
generate_realdata(num_sample, M, p, r)



