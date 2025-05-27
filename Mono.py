import pickle
import pandas as pd

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
    with open('node_ranking_dict.pkl', 'rb') as f:   #{network1:{1:{method1,method2}}}
        data = pickle.load(f)
    Result = {}
    for networkname,t in data.items():
        method_result = {}
        for t1,method_dict in t.items():
            for method,sorted_dict in method_dict.items():
                method_result[method] = round(Mr(sorted_dict),4)

        Result[networkname] = method_result

    for name,r in Result.items():
        print(name,r)





    # # 转换成 DataFrame
    # df = pd.DataFrame.from_dict(Result, orient='index')
    #
    # # 重置索引，把 Network 变成第一列
    # df.index.name = 'Network'
    # df.reset_index(inplace=True)
    #
    # # 写入 Excel
    # df.to_excel("mono_result.xlsx", index=False)
