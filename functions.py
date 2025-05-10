import os
import re
import email
from collections import Counter
from email import policy
from email.parser import BytesParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
import joblib


# 定义解析邮件的函数
def parse_email(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    email_data = {
        "From": msg["From"],
        "To": msg["To"],
        "Cc": msg["Cc"],
        "Bcc": msg["Bcc"],
        "Date": msg["Date"],
        "Subject": msg["Subject"],
        "Body": msg.get_body(preferencelist=('plain', 'html')).get_content(),
        "Attachments": [part.get_filename() for part in msg.iter_attachments()]
    }

    return email_data


# 定义处理目录的函数
def process_directory(directory_path):
    emails = []

    # 首先获取所有文件的总数用于进度条
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_'):  # 注意：这里您使用的是endswith('_')，可能需要检查是否正确
                all_files.append(os.path.join(root, file))

    # 使用tqdm创建进度条
    for file_path in tqdm(all_files, desc="处理邮件文件"):
        try:
            email_data = parse_email(file_path)
            emails.append(email_data)
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")
            continue

    return pd.DataFrame(emails)


def read_email_csv(directory_path):
    emails = []
    all_files = []

    # 正确使用os.scandir()
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.csv'):
                all_files.append(entry.path)

    for file_path in tqdm(all_files, desc="读取邮件csv文件"):
        try:
            email_data = pd.read_csv(file_path)
            emails.append(email_data)
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")
            continue

    if emails:  # 确保列表不为空
        df = pd.concat(emails, ignore_index=True)
        return df.where(df.notnull(), None)
    else:
        return pd.DataFrame()  # 如果没有找到文件，返回空DataFrame

def is_poi_involved(email_list, poi_set):
    """判断邮箱列表中是否有 POI 成员"""
    return any(email in poi_set for email in email_list)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def clean_email(email_field):
    # 简单的邮箱验证正则（用于排除格式异常的脏数据）
    EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

    if pd.isna(email_field):
        return []

    # 如果已经是列表，则直接处理
    if isinstance(email_field, list):
        emails = email_field
    else:
        # 尝试按逗号或分号拆分（考虑不同邮件系统）
        emails = re.split(r'[;,]', str(email_field))

    # 标准清理流程：strip、lower、去空、去重、正则验证
    cleaned = []
    for e in emails:
        e_clean = e.strip().lower()
        if e_clean and EMAIL_REGEX.match(e_clean):
            cleaned.append(e_clean)
    return list(set(cleaned))  # 去重


from pyvis.network import Network
import networkx as nx
import community as community_louvain  # Louvain 社区检测


def build_weighted_poi_network(email_df, poi_set, min_weight=3, html_name='poi_network_with_community.html'):
    """
    构建加权POI通信网络并用PyVis可视化，自动进行社区检测并上色。

    参数:
    - email_df: 邮件数据的DataFrame
    - poi_set: 包含POI邮箱的set
    - min_weight: 最小通信次数阈值（用于边过滤）
    - html_name: 输出HTML文件名
    """

    # Step 1: 构建加权图
    G_weighted = nx.Graph()

    for _, row in email_df.iterrows():
        sender = row['From_clean'][0] if row['From_clean'] else 'unknown'
        recipients = row['To_clean'] + row['Cc_clean'] + row['Bcc_clean']

        if sender not in poi_set:
            continue

        for recipient in recipients:
            if recipient in poi_set and recipient != sender:
                if G_weighted.has_edge(sender, recipient):
                    G_weighted[sender][recipient]['weight'] += 1
                else:
                    G_weighted.add_edge(sender, recipient, weight=1)

    # Step 2: 过滤边
    G_filtered = nx.Graph(
        [(u, v, d) for u, v, d in G_weighted.edges(data=True) if d['weight'] >= min_weight]
    )

    # Step 3: 社区检测
    partition = community_louvain.best_partition(G_filtered)

    # Step 4: 可视化
    net = Network(height='750px', width='100%', notebook=True, cdn_resources='remote')
    net.from_nx(G_filtered)

    # 设置节点颜色：基于社区编号
    for node in net.nodes:
        community_id = partition.get(node['id'], 0)
        hue = (community_id * 45) % 360
        node['color'] = f'hsl({hue}, 70%, 70%)'

    # 设置边样式：宽度表达联系频繁程度
    for edge in net.edges:
        weight = edge.get('value', 1)
        edge['width'] = weight
        edge['color'] = f'rgba(0,0,0,{min(1.0, weight / 10)})'

    net.show_buttons(filter_=['physics'])
    net.toggle_physics(True)
    net.show(html_name)

def get_centrality(sender_list, centrality_df):
    sender = sender_list[0] if sender_list else None
    if sender in centrality_df.index:
        return centrality_df.loc[sender].values  # 返回 degree, betweenness, eigenvector
    else:
        return np.zeros(3)