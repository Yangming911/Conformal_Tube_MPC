import pandas as pd
import os

def convert_citr(num):
    df_ped = pd.read_csv(f'vci-dataset-citr/vci-dataset-citr-master/data/trajectories_filtered/vci_lat_bi/bidirection_normal_driving_{num:02d}_traj_ped_filtered.csv')
    df_veh = pd.read_csv(f'vci-dataset-citr/vci-dataset-citr-master/data/trajectories_filtered/vci_lat_bi/bidirection_normal_driving_{num:02d}_traj_veh_filtered.csv')

    # df_ped = pd.read_csv(f'vci-dataset-citr/vci-dataset-citr-master/data/trajectories_filtered/vci_lat_uni/unidirection_normal_driving_{num:02d}_traj_ped_filtered.csv')
    # df_veh = pd.read_csv(f'vci-dataset-citr/vci-dataset-citr-master/data/trajectories_filtered/vci_lat_uni/unidirection_normal_driving_{num:02d}_traj_veh_filtered.csv')

    df_merged = pd.merge(df_ped, df_veh, on=['frame'], suffixes=['_ped','_veh'])
    df_new = pd.DataFrame(columns=["u0","u1","u2","u3","u4","u5","u6","u7","u8","u9","p_veh0_x","p_veh0_y","p_ped0_x","p_ped0_y","p_ped1_x","p_ped1_y","p_ped2_x","p_ped2_y","p_ped3_x","p_ped3_y","p_ped4_x","p_ped4_y","p_ped5_x","p_ped5_y","p_ped6_x","p_ped6_y","p_ped7_x","p_ped7_y","p_ped8_x","p_ped8_y","p_ped9_x","p_ped9_y","p_ped10_x","p_ped10_y"])

    for idx,data in df_merged.groupby('id_ped'):
        if len(data) < 10:
            continue
        data = data.sort_values(by='frame')
        data = data.reset_index(drop=True)
        for i in range(len(data)-10):
            df_new.loc[len(df_new)] = [
                data['vel_est'].iloc[i],
                data['vel_est'].iloc[i+1],
                data['vel_est'].iloc[i+2],
                data['vel_est'].iloc[i+3],
                data['vel_est'].iloc[i+4],
                data['vel_est'].iloc[i+5],
                data['vel_est'].iloc[i+6],
                data['vel_est'].iloc[i+7],
                data['vel_est'].iloc[i+8],
                data['vel_est'].iloc[i+9],
                data['x_est_veh'].iloc[i],
                data['y_est_veh'].iloc[i],
                data['x_est_ped'].iloc[i],
                data['y_est_ped'].iloc[i],
                data['x_est_ped'].iloc[i+1],
                data['y_est_ped'].iloc[i+1],
                data['x_est_ped'].iloc[i+2],
                data['y_est_ped'].iloc[i+2],
                data['x_est_ped'].iloc[i+3],
                data['y_est_ped'].iloc[i+3],
                data['x_est_ped'].iloc[i+4],
                data['y_est_ped'].iloc[i+4],
                data['x_est_ped'].iloc[i+5],
                data['y_est_ped'].iloc[i+5],
                data['x_est_ped'].iloc[i+6],
                data['y_est_ped'].iloc[i+6],
                data['x_est_ped'].iloc[i+7],
                data['y_est_ped'].iloc[i+7],
                data['x_est_ped'].iloc[i+8],
                data['y_est_ped'].iloc[i+8],
                data['x_est_ped'].iloc[i+9],
                data['y_est_ped'].iloc[i+9],
                data['x_est_ped'].iloc[i+10],
                data['y_est_ped'].iloc[i+10]]

    df_new.to_csv(f'assets/citr_data/citr_bi_{num:02d}.csv', index=False)

if __name__ == '__main__':
    # for num in range(1, 11):
    #     convert_citr(num)

    # # 将多个csv文件合并为一个
    # all_dfs = []
    # for num in range(1, 11):
    #     df = pd.read_csv(f'assets/citr_data/citr_bi_{num:02d}.csv')
    #     all_dfs.append(df)
    # for num in range(1, 9):
    #     df = pd.read_csv(f'assets/citr_data/citr_uni_{num:02d}.csv')
    #     all_dfs.append(df)
    # df_merged = pd.concat(all_dfs, ignore_index=True)
    # df_merged.to_csv('assets/citr_data/citr_all.csv', index=False)
    # # random选取一半作为train，一半作为conformal grid
    # df_train = df_merged.sample(frac=0.5, random_state=42)
    # df_conformal_grid = df_merged.drop(df_train.index)
    # df_train.to_csv('assets/citr_data/citr_train.csv', index=False)
    # df_conformal_grid.to_csv('assets/citr_data/citr_conformal_grid.csv', index=False)

    df_real = pd.read_csv('assets/citr_data/citr_train.csv')
    df_sim = pd.read_csv('assets/control_sequences.csv')
    df_all = pd.concat([df_real, df_sim], ignore_index=True)
    df_all.to_csv('assets/real_sim_data.csv', index=False)