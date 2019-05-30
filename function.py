import heapq
import random

import pandas as pd
import os
import matplotlib.pyplot as plt
from bloom_filter import BloomFilter
from dgim import Dgim
from associate_rules import associate

size = 10 ** 6
INF = 1e9
win_size = 10


def get_directory(path):
    fp = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                cur = root + "/" + f
                fp.append(cur)
    else:
        fp.append(path)
    return fp


def chunk_read(path, size=10 ** 6):
    stock_name = os.path.basename(path)[:-4]
    for chunk in pd.read_csv(path, chunksize=size):
        chunk.columns = ['date', 'hour', 'open_price', 'max_price', 'min_price', 'close_price', 'trade_money',
                         'trade_amount']
        search_increase(chunk, stock_name)
        # search_date(chunk, t_date, t_hour)


def print_info(file_name, chunk):
    for index, row in chunk.iterrows():
        date = row['date']
        hour = row['hour']
        print(f'{file_name} at {date}, {hour}')


def search_date(path, t_date, t_hour):
    for chunk in pd.read_csv(path, chunksize=size):
        chunk.columns = ['date', 'hour', 'open_price', 'max_price', 'min_price', 'close_price', 'trade_money',
                         'trade_amount']
        # search_date(chunk, t_date, t_hour)
        for index, row in chunk.iterrows():
            date = row['date']
            hour = row['hour']
            # print(f'{date}  {hour} {t_date}  {t_hour}')
            if date == t_date and hour == t_hour:
                print(chunk.iloc[0])


def search_increase(path):
    count = 0
    stock_name = os.path.basename(path)[:-4]
    pre = None
    star = None
    end = None
    for chunk in pd.read_csv(path, chunksize=size):
        chunk.columns = ['date', 'hour', 'open_price', 'max_price', 'min_price', 'close_price', 'trade_money',
                         'trade_amount']
        for index, row in chunk.iterrows():
            date = row['date']
            open_price = row['open_price']
            close_price = row['close_price']
            # print(f'{date}  {hour} {t_date}  {t_hour}')
            if pre is None:
                pre = date
                star = open_price
                end = close_price
            elif pre == date:
                end = close_price
            else:
                if end > star:
                    count += 1
                star = open_price
                end = close_price
                pre = date
        if end > star:
            count += 1
    return (stock_name, count)


def moving_average(path, win_size):
    stock_name = os.path.basename(path)[:-4]
    pre_date = None
    counter = 0
    record = []
    result_value = []
    result_date = []
    close_price = 0
    begin_date = None
    date = None
    for chunk in pd.read_csv(path, chunksize=size):
        chunk.columns = ['date', 'hour', 'open_price', 'max_price', 'min_price', 'close_price', 'trade_money',
                         'trade_amount']
        for index, row in chunk.iterrows():
            date = row['date']
            close_price = row['close_price']
            if pre_date is None:
                pre_date = date
                begin_date = date
            elif pre_date == date:
                continue
            else:
                record.append(close_price)
                counter += 1
                if counter == win_size:
                    result_value.append(sum(record) / win_size)
                    # result_date.append(date)
                    del (record[0])
                    counter -= 1
                pre_date = date
    record.append(close_price)
    result_value.append(sum(record) / win_size)
    # result_date.append(date)
    return (result_value, begin_date)


def draw_mav(stock_name, win_size, result_value, begin_date):
    x = range(1, len(result_value) + 1)
    plt.plot(x, result_value, 'r-')
    plt.title(f'stock={stock_name} window siez={win_size} begin date={begin_date}')
    plt.xlabel('date')
    plt.ylabel('average value')
    plt.legend()
    plt.savefig(f'{stock_name}-{win_size}')
    plt.show()
    return f'{stock_name}-{win_size}.png'


def mavg_default(path, win_size):
    stock_name = os.path.basename(path)[:-4]
    res_10 = moving_average(path, 10)
    res_20 = moving_average(path, 20)
    res_30 = moving_average(path, 30)
    x_10 = range(1, len(res_10[0]) + 1)
    x_20 = range(1, len(res_20[0]) + 1)
    x_30 = range(1, len(res_30[0]) + 1)
    l1 = plt.plot(x_10, res_10[0], 'r--', label='win_size=10')
    l2 = plt.plot(x_20, res_20[0], 'g--', label='win_size=20')
    l3 = plt.plot(x_30, res_30[0], 'b--', label='win_size=30')
    plt.title(f'stock={stock_name} window size=10,20,30 begin date={res_10[1]}')
    plt.xlabel('date')
    plt.ylabel('average value')
    plt.legend()
    plt.savefig(f'{stock_name}-default')
    plt.show()
    name1 = f'{stock_name}-default.png'
    result = moving_average(path, win_size)
    name2 = draw_mav(stock_name, win_size, result[0], result[1])
    return name1, name2


# def find_pattern(path1, path2, days):
#     name1 = os.path.basename(path1)[:-4]
#     name2 = os.path.basename(path2)[:-4]
#     (res1, date1) = moving_average(path1, days)
#     (res2, date2) = moving_average(path2, days)
#     ten1 = con_increase(res1)
#     ten2 = con_increase(res2)
#     count = 0
#     for i in ten1:
#         if (i in ten2) and (ten1[i] == ten2[i]):
#             count += 1
#     print_pattern(name1, ten1)
#     print_pattern(name2, ten2)
#     sim = count / max(len(ten1), len(ten2))
#     print(f'The similarity degree is {sim}.')

#
# def print_pattern(name, res):
#     print(f'{name} stocks patterns:')
#     for i in res:
#         if i != 1:
#             print(f'{i} days increase: {res[i]}')
#
#         # To get the number of the number of the con increase

'''
    最多连续上升几天
'''


def con_increase(data):
    l = len(data)
    pre = -INF
    con = 1
    res = {}
    for i in range(l):
        if data[i] > pre:
            con += 1
        else:
            for j in range(1, con + 1):
                if j not in res:
                    res[j] = con + 1 - j
                else:
                    res[j] += con + 1 - j
            con = 1
        pre = data[i]
    return res


'''
    最多连续下降几天
'''


def con_decrease(data):
    l = len(data)
    pre = INF
    con = 1
    res = {}
    for i in range(l):
        if data[i] < pre:
            con += 1
        else:
            for j in range(1, con + 1):
                if j not in res:
                    res[j] = con + 1 - j
                else:
                    res[j] += con + 1 - j
            con = 1
        pre = data[i]
    return res


# def LSH(fp):
#     l = len(fp)
#     name = [None for _ in range(l)]
#     con_in = [None for _ in range(l)]
#     con_de = [None for _ in range(l)]
#
#     for i in range(len(fp)):
#         name[i] = os.path.basename(fp[i])[:-4]
#         res = moving_average(fp[i], 30)
#         con_in[i] = len(con_increase(res[0]))
#         con_de[i] = len(con_decrease(res[0]))
#     in_ma = [[0 for _ in range(l)] for i in range(8)]
#     mu_ma = [[0 for _ in range(3)] for _ in range(8)]
#     for i in range(l):
#         ma_i = con_in[i]
#         ma_d = con_de[i]
#         for j in range(4):
#             if ma_i > (j + 1) * 10:
#                 in_ma[j][i] = 1
#             else:
#                 in_ma[j][i] = 0
#         for j in range(4, 8):
#             if ma_d > (j - 3) * 10:
#                 in_ma[j][i] = 1
#             else:
#                 in_ma[j][i] = 0
#     a = list(range(1, 9))
#     for i in range(3):
#         random.shuffle(a)
#         for j in range(8):
#             mu_ma[j][i] = a[j]
#
#     sig_ma = [[0 for _ in range(l)] for _ in range(3)]
#
#     for i in range(l):
#         for j in range(3):
#             temp = INF
#             for z in range(8):
#                 if in_ma[z][i] == 1:
#                     temp = min(temp, mu_ma[z][j])
#             sig_ma[j][i] = temp    if mavg[0] < mavg[len(mavg)-1]

#     for i in range(l):
#         for j in range(i + 1, l):
#             num = 0
#             for k in range(3):
#                 if sig_ma[k][i] == sig_ma[k][j]:
#                     num += 1
#             print(name[i], 'vs', name[j], ':sig= ', num / 3)
#             num = 0
#             for k in range(8):
#                 if in_ma[k][i] == in_ma[k][j]:
#                     num += 1
#             print(name[i], 'vs', name[j], ':col= ', num / 8)
#
#     '''
#         Using 8 feature: increase for 5,10,15,20
#         and decrease for 5,10,15,20
#         input matrix:
#     '''


def trailing_zeroes(num):
    """Counts the number of trailing 0 bits in num."""
    if num == 0:
        return 32  # Assumes 32 bit integer inputs!
    p = 0
    while (num >> p) & 1 == 0:
        p += 1
    return p


def estimate_cardinality(values, k):
    """Estimates the number of unique elements in the input set values.

    Arguments:
        values: An iterator of hashable elements to estimate the cardinality of.
        k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
    """
    num_buckets = 2 ** k
    max_zeroes = [0] * num_buckets
    for value in values:
        h = hash(value)
        bucket = h & (num_buckets - 1)  # Mask out the k least significant bits as bucket ID
        bucket_hash = h >> k
        max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
    return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402


def stream_input(data):
    l = len(data)
    result = [False for _ in range(l)]
    pre = -INF
    for i in range(l):
        if data[i] > pre:
            result[i] = True
        pre = data[i]
    return result


def DGIM(path1):
    fp = get_directory(path1)
    ret = 'DGIM Analysis\n'
    for path in fp:
        name = os.path.basename(path)[:-4]
        (data, date) = moving_average(path, 30)
        stream = stream_input(data)
        l = len(stream)
        dgim = Dgim(N=l)
        for i in range(l):
            dgim.update(stream[i])
        result = dgim.get_count()
        ret = ret + f'There are {result} 1s in {name}\n'
    return ret


'''
    Whether the pattern is in the socket
'''


def bloom_filter(fp, pattern):
    ret = f'Query {pattern} pattern in date set\n '
    for path in fp:
        stock_name = os.path.basename(path)[:-4]
        mvg,date = moving_average(path,10)
        bmvg = trans2_01(mvg)
        bloom = BloomFilter(max_elements=10000)
        length = len(pattern)
        ele = ''
        for i in range(length):
            ele = ele + str(bmvg[i])
        for i in range(length - 1, len(bmvg)):
            if i != length - 1:
                ele = ele + str(bmvg[i])
                ele = ele[1:]
            bloom.add(ele)
        if pattern in bloom:
            ret = ret + f'Find {pattern} pattern in {stock_name}\n'
    return ret


# def FM(path):
#     name = os.path.basename(path)[:-4]
#     (data, date) = moving_average(path, 30)
#     print(estimate_cardinality(data, 9))


# def page_ranking():
#     G = nx.Graph();
#     for i in range(10):
#         G.add_node(i)
#     for i in range(10):
#         for j in range(i + 1, 10):
#             if random.random() > 0.5:
#                 G.add_edge(i, j)
#     nx.draw(G, with_labels=True)
#     plt.savefig("network")
#     plt.show()
#     pr = nx.pagerank(G)
#     print(pr)


def get_rules(fp):
    # l1=increase for 20 days
    # l2=decrease for 20 days
    # l3=longest increase day > longest decrease day
    # l4=increase day > decrease day
    labels = []
    ret = ''
    ret = ret + 'Analyse the associate rules\n'
    ret = ret + 'A -> 连续增长30天\n'
    ret = ret + 'B -> 连续下降20天\n'
    ret = ret + 'C -> 增长天数大于下降天数\n'
    ret = ret + 'D -> 最终价格大于初始价格\n'
    ret = ret + 'E -> 盈利\n'

    for path in fp:
        (data, date) = moving_average(path, 30)
        label = []
        incr = con_increase(data)
        decr = con_decrease(data)
        if (len(incr) > 30):
            label.append('A')
        if (len(decr) > 20):
            label.append('B')
        bmvg = trans2_01(data)
        flag, rest = is_good_stock(bmvg)
        if flag:
            label.append('C')
        if final_win(data):
            label.append('D')
        if calc_benifit_sum(data):
            label.append('E')
        labels.append(label)
    return ret + associate(labels)


# def cluster(flag,k):
#     if flag ==0:
#         iris = load_iris()
#     else:
#         iris = load_wine()
#     data = pd.DataFrame(iris.data)
#     data_zs = (data-data.mean())/data.std()
#     iteration = 500
#     model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
#     model.fit(data_zs)
#     pd.Series(model.labels_).value_counts()
#     pd.DataFrame(model.cluster_centers_)
#     tsne = TSNE(learning_rate=100)
#     tsne.fit_transform(data_zs)
#     data = pd.DataFrame(tsne.embedding_,index=data_zs.index)
#     d = data[model.labels_ == 0]
#     plt.plot(d[0], d[1], 'r.')
#     d = data[model.labels_ == 1]
#     plt.plot(d[0], d[1], 'go')
#     d = data[model.labels_ == 2]
#     plt.plot(d[0], d[1], 'b*')
#     plt.show()


# def bipartite_matching():
#   '''
#   Assume we have two sets of nodes. One for the financial events
#   The other for the advertisements. Use the bipartite matching algorithm
#   to match the event and advertisements.
#   :return:
#   '''
#   print('The number represents the trade signal, the letter represents for the online news')
#   B = nx.Graph()
#   B.add_nodes_from([1, 2, 3, 4], bipartite=0)  # Add the node attribute "bipartite"
#   B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
#   B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
#
#   # Separate by group
#   l, r = nx.bipartite.sets(B)
#   pos = {}
#
#   # Update position for node from each group
#   pos.update((node, (1, index)) for index, node in enumerate(l))
#   pos.update((node, (2, index)) for index, node in enumerate(r))
#
#   nx.draw(B, pos=pos,with_labels=True)
#   plt.show()
#   print(nx.bipartite.maximum_matching(B))


# trans a moving average list to a 0-1 vector
def trans2_01(data):
    ret = []
    ret.append(0)
    for i in range(1, len(data)):
        if data[i] >= data[i - 1]:
            ret.append(1)
        else:
            ret.append(0)
    return ret


# check whether a stock is good or bad, return the result with cnt
def is_good_stock(data):
    ret = 0
    for i in range(0, len(data)):
        if data[i] == 1:
            ret += 1
    flag = False
    if (ret >= len(data) / 2):
        flag = True
    return flag, ret


# get the k nearlist neighbor of stock x
def KNN(k, x, all_L):
    dis_data = []

    for i in range(0, len(all_L)):
        if i == x:
            continue
        dis = 0
        for j in range(0, len(all_L[0])):
            if (all_L[x][j] != all_L[i][j]):
                dis += 1
        dis_data.append({'id': i, 'dis': dis})
    result = heapq.nsmallest(k, dis_data, key=lambda s: s['dis'])
    return result


# calculate the J_similarity, the smaller the closer
def J_similarity(x, y):
    M00 = M01 = M10 = M11 = 0
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0:
            M00 += 1
        if x[i] == 0 and y[i] == 1:
            M01 += 1
        if x[i] == 1 and y[i] == 0:
            M10 += 1
        if x[i] == 1 and y[i] == 1:
            M11 += 1
    return (float)(M01 + M10) / (M01 + M10 + M11)


# calculate the dgim value of a certain stock
# count how many 1 in the
def calc_dgim(x, all_L):
    dgim = Dgim(100, 0.1)
    for j in range(0, len(all_L[0])):
        dgim.update(all_L[x][j])
    dgim_result = dgim.get_count()
    # print('The result of dgim of ' + str(i) + ' is ' + str(dgim_result))
    return dgim_result


def test_function(fp):
    for path in fp:
        debug = moving_average(path, 10)
        print(debug)
        ret = trans2_01(debug[0])
        print(is_good_stock(ret))


'''
    Compare between first day and last day
'''


def final_win(mavg):
    return mavg[0] < mavg[len(mavg) - 1]


'''
    whether it benefits
'''


def calc_benifit_sum(data):
    ret = 0.0
    for i in range(0, len(data)):
        ret += data[i] - data[0]
    return ret >= 0.0, ret


def get_all_bmvg(fp):
    all_bmvg = []
    for path in fp:
        data, date = moving_average(path, 10)
        bmvg = trans2_01(data)
        all_bmvg.append(bmvg)
    return all_bmvg


def fun_knn(path, fp, kth):
    fp = get_directory(fp)
    index = 1
    for i in range(len(fp)):
        if path == fp[i]:
            index = i
            break
    res = KNN(kth, i, get_all_bmvg(fp))
    ret = f'Similar  Socks to {os.path.basename(path)[:-4]}\n'
    ret = ret + '-' * 30 + '\n'
    for ele in res:
        id = ele['id']
        sim = ele['dis']
        name = os.path.basename(fp[id])[:-4]
        ret = ret + 'name: ' + str(name) + '  Distance: ' + str(sim) + '\n'
        ret = ret + '-' * 30 + '\n'
    ret = ret + 'Finish Finding\n'
    return ret


def fun_associate_rules(path):
    fp = get_directory(path)
    return get_rules(fp)


def fun_basic_analysis(path):
    stock_name = os.path.basename(path)[:-4]
    mvg, date = moving_average(path, 10)
    bmvg = trans2_01(mvg)
    flag, days = is_good_stock(bmvg)
    increase_day = len(con_decrease(mvg))
    decrease_day = len(con_decrease(mvg))
    profit = calc_benifit_sum(mvg)
    final_increase = final_win(mvg)
    buy = ''
    if random.random() > 0.3:
        buy = 'Worth Buy'
    else:
        buy = "Don't Buy"
    ret = ''
    ret = ret + 'stock name: ' + stock_name + '\n'
    ret = ret + '=' * 20 + '\n'
    ret = ret + 'max increase day: ' + str(increase_day) + '\n'
    ret = ret + '-' * 30 + '\n'
    ret = ret + 'max decrease day: ' + str(decrease_day) + '\n'
    ret = ret + '-' * 30 + '\n'
    ret = ret + 'profit: ' + str(profit[0]) + '\n'
    ret = ret + '-' * 30 + '\n'
    ret = ret + 'final price increase； ' + str(final_increase) + '\n'
    ret = ret + '-' * 30 + '\n'
    ret = ret + 'good stock: ' + str(flag) + '\n'
    ret = ret + '-' * 30 + '\n'
    ret = ret + 'Conclusion:' + buy + '\n'
    return ret


def get_basic_info(path):
    stock_name = os.path.basename(path)[:-4]
    mvg, date = moving_average(path, 10)
    bmvg = trans2_01(mvg)
    flag, days = is_good_stock(bmvg)
    increase_day = len(con_decrease(mvg))
    decrease_day = len(con_decrease(mvg))
    profit = calc_benifit_sum(mvg)
    final_increase = final_win(mvg)
    buy = ''
    if random.random() > 0.3:
        buy = 'Worth Buy'
    else:
        buy = "Don't Buy"
    return (stock_name, increase_day, decrease_day, profit[0], final_increase, flag, buy, mvg)


def func_compare(path1, path2):
    res1 = get_basic_info(path1)
    res2 = get_basic_info(path2)
    ret = ''
    ret = ret + res1[0] + ' VS ' + res2[0] + '\n'
    ret = ret + '=' * 20 + '\n'
    ret = ret + 'max increase day: ' + str(res1[1]) + '<==>' + str(res2[1]) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'max decrease day: ' + str(res1[2]) + '<==>' + str(res2[2]) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'profit: ' + str(res1[3]) + '<==>' + str(res2[3]) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'final price increase； ' + str(res1[4]) + '<==>' + str(res2[4]) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'good stock: ' + str(res1[5]) + '<==>' + str(res2[5]) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'Conclusion:' + res1[6] + '<==>' + res2[6] + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'Jaccard similarity: ' + str(J_similarity(trans2_01(res1[7]), trans2_01(res2[7]))) + '\n'
    ret = ret + '-' * 40 + '\n'
    ret = ret + 'Finish Analysis'
    return ret, compare_mvg(res1[0], res2[0], res1[7], res2[7])


def compare_mvg(name1, name2, mvg1, mvg2):
    x_10 = range(1, len(mvg1) + 1)
    x_20 = range(1, len(mvg2) + 1)
    l1 = plt.plot(x_10, mvg1, 'r--', label=name1)
    l2 = plt.plot(x_20, mvg2, 'g--', label=name2)
    plt.title(f'Moving Average Comparsion: {name1} VS {name2}')
    plt.xlabel('date')
    plt.ylabel('average value')
    plt.legend()
    plt.savefig(f'{name1} VS {name2}')
    plt.show()
    res = f'{name1} VS {name2}.png'
    return res


def fun_query_pattern(path, pattern):
    fp = get_directory(path)
    return bloom_filter(fp,pattern)



if __name__ == '__main__':
    # /home/kaiqiang/file/2016data/SH000001.csv
    # /home/kaiqiang/Documents/data/
    path = '/home/kaiqiang/Documents/data/'
    fp = get_directory(path)
    all_bmvg = get_all_bmvg(fp)
    print(KNN(3, 1, all_bmvg))
