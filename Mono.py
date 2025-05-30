import pickle
import pandas as pd
from E_D import GLSTM
from Utils import LSTMModel
def Mr(dic):
    '''
    两种dic
    1.[{},{}],有些算法是一块块拼接的，用上边算
    2.{}，普通的用这样算
    '''
    N = 0
    m = 0

    summary = {}
    for i in dic:   #key
        N += 1
        summary.setdefault(dic[i],list()).append(i)
    for i in summary:
            Nr = len(summary[i])
            m += Nr*(Nr-1)
    m = (1 - m/(N*(N-1))) ** 2
    return m


if __name__ == '__main__':
    with open('kendall[0.5-1.5].pkl', 'rb') as f:   #{network1:{1:{method1,method2}}}
        data = pickle.load(f)

    # for k,v in data.items():
    #     for v1,v2 in v.items():
    #         print(k,v1,v2)
    #     print('-----------------------------------------------')

    # Result = {}
    # for networkname,t in data.items():
    #     method_result = {}
    #     for t1,method_dict in t.items():
    #         for method,sorted_dict in method_dict.items():
    #             method_result[method] = round(Mr(sorted_dict),4)
    #
    #     Result[networkname] = method_result
    #
    # for name,r in Result.items():
    #     print(name,r)

    # PATH='./dataset/real_multiplex_networks/MNdata/cKM-Physicians-Innovation_multiplex.edges'
    # nodes_num=246
    # network_name='./dataset/real-influence/MN_SIR_' + '1beitac/' + 'cKM-Physicians-Innovation_multiplex' + '.txt'
    # glstm_k, glstm_dict = GLSTM(PATH, nodes_num, network_name)
    #
    # print(glstm_k)

    # 整理成一个列表，方便转DataFrame
    rows = []
    for network, t_dict in data.items():
        for t, methods in t_dict.items():
            row = {"Network": network, "t": t}
            row.update(methods)
            rows.append(row)

    # 转成DataFrame
    df = pd.DataFrame(rows)

    # 保存到Excel
    df.to_excel("ken)result.xlsx", index=False)