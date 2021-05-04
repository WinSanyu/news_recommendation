import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_click_log", type=str, default="./data/train_click_log.csv")
parser.add_argument("--articles", type=str, default="./data/articles.csv")
parser.add_argument("--articles_emb", type=str, default="./data/articles_emb.csv")

opt = parser.parse_args()

def get_pd(pd_filename):
    pd_file = pd.read_csv(pd_filename)
    df = pd.DataFrame(pd_file)
    return df

def get_train_click_log():
    return get_pd(opt.train_click_log)

def get_articles():
    return get_pd(opt.articles)

def get_articles_emb():
    return get_pd(opt.articles_emb)