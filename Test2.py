import pickle

if __name__ == '__main__':
    file_path5 = './Sorted_nodes_data/sage_sorted_nodes.pkl'
    with open(file_path5, 'rb') as file:
        data_sage = pickle.load(file)
    print(data_sage)





    # file_path4 = './Sorted_nodes_data/sage_sorted_nodes.pkl'
    # with open(file_path4, 'rb') as file:
    #     data_sage = pickle.load(file)
    #
    #
    #
    # file_path3 = './Sorted_nodes_data/ed_sorted_nodes.pkl'
    # with open(file_path3, 'rb') as file:
    #     data_ed = pickle.load(file)
    #
    # file_path1 = './Sorted_nodes_data/f-e_sorted_nodes.pkl'
    # with open(file_path1, 'rb') as file:
    #     data_fe = pickle.load(file)
    #
    # file_path2 = './Sorted_nodes_data/prgc_sorted_nodes.pkl'
    # with open(file_path2, 'rb') as file:
    #     data_prgc = pickle.load(file)
    #
    # data_list = [data_fe, data_prgc,data_ed,data_sage]
    #
    # # 初始化一个空字典
    # merged_data = {}
    #
    # # 遍历所有数据
    # for data in data_list:
    #     for network, methods in data.items():
    #         if network not in merged_data:
    #             merged_data[network] = {}
    #         for method, nodes in methods.items():
    #             merged_data[network][method] = nodes
    #
    # print(merged_data)
    # new_data = {network: {'MGNN-AL': nodes} for network, nodes in data.items()}
    #
    # print(new_data)
    # with open('./Sorted_nodes_data/[f-e,prgc,ed,sage]_sorted_nodes.pkl','wb') as f:
    #     pickle.dump(merged_data, f)