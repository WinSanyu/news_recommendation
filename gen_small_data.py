import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./data')
opt = parser.parse_args()

def get_pd(pd_filename):
    pd_file = pd.read_csv(pd_filename)
    df = pd.DataFrame(pd_file)
    return df

def get_train_click_log():
    return get_pd(os.path.join(opt.data_path, "train_click_log.csv"))

def get_articles():
    return get_pd(os.path.join(opt.data_path, "articles.csv"))

def get_articles_emb():
    return get_pd(os.path.join(opt.data_path, "articles_emb.csv"))

def get_small_click_log():
    return get_pd('./gen_data/small_click_log.csv')

def get_small_emb():
    emb = get_articles_emb()
    pd_small_click_log = get_small_click_log()

    res = None
    for art_id in pd_small_click_log['click_article_id']:
        tmp = emb[emb['article_id'] == art_id]
        if res is None:
            res = tmp
        else:
            res = pd.concat([res, tmp])
    return res

def get_small_art():
    art = get_articles()
    pd_small_click_log = get_small_click_log()
    res = None
    for art_id in pd_small_click_log['click_article_id']:
        tmp = art[art['article_id'] == art_id]
        if res is None:
            res = tmp
        else:
            res = pd.concat([res, tmp])
    return res

def save_csv(saved_pd, name):
    saved_pd.to_csv('./gen_data/{}.csv'.format(name), index=False)

small_emb = get_small_emb()
small_art = get_small_art()
save_csv(small_emb, 'small_emb')
save_csv(small_art, 'small_art')