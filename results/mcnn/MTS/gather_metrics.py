# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

def gather_metrics(root_dir):
    all_metrics = []
    metrics_for_fold = []
    for subdir, dirs, files in os.walk(root_dir):
        # print("subdir",subdir, "files", files)
        for file in files:
            if file == 'df_metrics.csv':
                file_path = os.path.join(subdir, file)
                # 从文件路径中提取配置信息
                config_info = subdir.replace(root_dir, '').strip(os.path.sep).split(os.path.sep)
                # 读取CSV文件
                df = pd.read_csv(file_path)
                # 添加配置信息作为新的列
                for i, info in enumerate(config_info):
                    if info == "hyper_param_search":
                        continue
                    if 'pool' in info:
                        df['pool_factor'] = int(info.split('_')[-1])
                    if 'filter' in info:
                        df['filter_factor'] = info.split('_')[-1]
                        # print(info," info.split('_')[-1]: ",info.split('_')[-1])
                    if 'fold' in info:
                        df['fold'] = int(info.split('_')[-1])
                if 'fold' in subdir.strip(os.path.sep).split(os.path.sep)[-1]:
                    metrics_for_fold.append(df)
                all_metrics.append(df)

    # 合并所有的DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)
    final_fold_metric = pd.concat(metrics_for_fold, ignore_index=True)
    # 保存到新的CSV文件
    final_df.to_csv(os.path.join(root_dir, 'all_metrics_combined.csv'), index=False)
    final_fold_metric.to_csv(os.path.join(root_dir, 'fold_metrics_combined.csv'), index=False)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <root_directory_path>")
        sys.exit(1)

    root_directory = sys.argv[1]
    gather_metrics(root_directory)
