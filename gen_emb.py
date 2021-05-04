import pandas as pd

from load_data import get_articles_emb, get_pd, get_articles

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