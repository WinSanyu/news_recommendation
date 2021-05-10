# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# 节约内存的一个标配函数
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path=r'../user_data/tianchi_small/'):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
    """
    all_click = pd.read_csv(data_path + 'small_click_log.csv')

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path=r'../tcdata/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df

def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp')
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict

# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)
    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]
    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df

# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    
    return item_type_dict, item_words_dict, item_created_time_dict

def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if(i == j):
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击    
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将得到的相似性矩阵保存到本地
    # pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_

def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全        
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
                
            item_rank.setdefault(j, 0)
            item_rank[j] +=  wij
    
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank

def item_based_recommend_thread(args):
    user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click = args
    return item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click)

def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    
    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))
    
    return user_activate_degree_dict

# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵
        
        思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)
    
    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])   
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)
    
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])
    
    # 将得到的相似性矩阵保存到本地
    # pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))

    return u2u_sim_

# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num, 
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]    #  [(item1, time1), (item2, time2)..]
    user_hist_items = set([i for i, t in user_item_time_list])   # 存在一个用户与某篇文章的多次交互， 这里得去重
    
    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)
            
            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0
            
            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                # if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                #     content_weight += emb_i2i_sim[i][j]
                # if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                #     content_weight += emb_i2i_sim[j][i]
                
                # 创建时间差权重
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                
            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv
        
    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items(): # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100 # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break
        
    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]    
    
    return items_rank

# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        recall_num=user_num*topk
        nmrr=0

        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
                ind=tmp_recall_items.index(last_click_item_dict[user])+1
                nmrr+=round(1.0 / ind, 5)

            
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        precision = round(hit_num * 1.0 / recall_num, 5)
        recall = round(hit_num * 1.0 / user_num, 5)
        if hit_rate==0:
            f1=0
        else:
            f1 = round(recall * precision * 2.0 / ( recall + precision ), 5)
        mrr = round(nmrr * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num,'precision:',precision,'recall:',recall,'f1:',f1,'mrr:',mrr)

def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}
    
    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
            
        return norm_sorted_item_list
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]
        
        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score  
    
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    # pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank

if __name__ == '__main__':
    # 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
    user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                                'usercf_sim_usercf_recall': {},
                                # 'embedding_sim_item_recall': {},
                                # 'youtubednn_recall': {},
                                # 'youtubednn_usercf_recall': {}, 
                                # 'cold_start_recall': {}
                                }
    recall_strategy_dict = {'itemcf_sim_itemcf_recall': True,
                            'usercf_sim_usercf_recall': True,
                            # 'embedding_sim_item_recall': 1.0,
                            # 'youtubednn_recall': 1.0,
                            # 'youtubednn_usercf_recall': 1.0, 
                            # 'cold_start_recall': 1.0
                            }
    all_click_df = get_all_click_sample()
    all_click_df = reduce_mem(all_click_df)
    # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
    # 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
    all_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    item_info_df = get_item_info_df('../tcdata/')
    # 获取文章的属性信息，保存成字典的形式方便查询
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)
    if recall_strategy_dict['itemcf_sim_itemcf_recall'] == True:
        i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)
    # 定义
    user_recall_items_dict = collections.defaultdict(dict)
    # 获取 用户 - 文章 - 点击时间的字典*
    user_item_time_dict = get_user_item_time(all_click_df)
    # 去取文章相似度
    # i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
    # 相似文章的数量
    sim_item_topk = 10
    # 召回文章数量
    recall_item_num = 10
    # 用户热度补全
    item_topk_click = get_item_topk_click(all_click_df, k=0)
    
    if recall_strategy_dict['itemcf_sim_itemcf_recall'] == True:
        len_it = len(all_click_df['user_id'].unique())
        it = zip(all_click_df['user_id'].unique(),[user_item_time_dict]*len_it,[i2i_sim]*len_it,[sim_item_topk]*len_it,[recall_item_num]*len_it,[item_topk_click]*len_it)
        from multiprocessing import Pool as ThreadPool
        with ThreadPool(4) as p:
            res = p.map(item_based_recommend_thread, it)
        for i, user in enumerate(all_click_df['user_id'].unique()):
            user_recall_items_dict[user] = res[i]
        if 0:
            for user in tqdm(all_click_df['user_id'].unique()):
                user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                                    sim_item_topk, recall_item_num, item_topk_click)
        user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict

    # 由于usercf计算时候太耗费内存了，这里就不直接运行了
    # 如果是采样的话，是可以运行的
    if recall_strategy_dict['usercf_sim_usercf_recall'] == True:
        user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
        u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)

    sim_user_topk = 10
    emb_i2i_sim = {}
    if recall_strategy_dict['usercf_sim_usercf_recall'] == True:
        for user in tqdm(all_click_df['user_id'].unique()):
            user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk,
                                                                recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
        user_multi_recall_dict['usercf_sim_usercf_recall'] = user_recall_items_dict
    
    weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
                    'usercf_sim_usercf_recall': 1.0,
                    # 'embedding_sim_item_recall': 1.0,
                    # 'youtubednn_recall': 1.0,
                    # 'youtubednn_usercf_recall': 1.0, 
                    # 'cold_start_recall': 1.0
                    }
    
    # 最终合并之后每个用户召回150个商品进行排序
    final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)

    metrics_recall(final_recall_items_dict_rank, trn_last_click_df, topk=10)

    # 将字典的形式转换成df
    # user_item_score_list = []

    # for user, items in tqdm(user_recall_items_dict.items()):
    #     for item, score in items:
    #         user_item_score_list.append([user, item, score])

    # recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
