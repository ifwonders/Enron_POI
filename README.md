
# Enron Dataset POI 分析与欺诈邮件检测

本项目基于 [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) 数据集，旨在识别与安然公司财务欺诈案相关的关键人员（POI, Person of Interest），并通过通信网络分析与文本特征建模，构建一个邮件欺诈检测模型。

##  项目背景

在 Enron Dataset 的背景下，POI 通常指：

- 被司法部门确认为参与欺诈的高管和员工；
- 在调查过程中被重点审查的人员；
- 邮件往来中表现出异常通信模式的关键节点人物。

该数据集中包含约 146 名 POI，包括 CEO Jeffrey Skilling、CFO Andrew Fastow、董事会成员以及财务和交易部门的关键人员。

---

##  数据准备与预处理

1. 解压原始数据集：

   - `EnronEmails.zip` → 解压为 `EnronEmails` 文件夹 → 加载邮件为 `email_df`
   - `poi_email_addresses.csv` → 加载为 `poi_emails`
   - `poi_names.txt` → 加载为 `poi_names`

2. 对邮件数据进行结构化处理，提取字段如：`From`, `To`, `Subject`, `Date`, `Body` 等。

---

## 通信网络分析

### 1. POI 标记逻辑

- 判断邮件的发送人、接收人、抄送人、密送人是否为 POI。
- 提取 POI 通信记录 `poi_communications`，其中：
  - 发送人是 POI；
  - 且收件人 / 抄送人 / 密送人中至少有一人是 POI。

### 2. 构建 POI 通信图网络

- 节点为 POI；
- 边表示邮件通信频次或联系强度。

### 3. 中心性指标计算

- 计算每个 POI 的网络中心性指标（如 Degree Centrality、Betweenness Centrality 等）；
- 输出中心性排序结果，用于识别关键人物。

---

##  邮件内容分析

### 1. 邮件结构特征提取

为每封邮件增加以下字段：

- 正文长度、标题长度；
- 收件人数量、是否包含附件；
- 发送日期、时间、星期几、是否为周末等时间特征。

### 2. 邮件文本特征提取

- 使用 TF-IDF 方法从正文中提取关键词特征，形成 `tfidf_features`。

### 3. 可疑行为特征

- 是否包含可疑关键词；
- 是否非工作时间发送；
- 是否大量密送（BCC）；

### 4. 网络结构特征

- 每封邮件关联发送人、收件人的 POI 中心性特征。

---

##  欺诈检测模型

### 1. 训练数据构建

- 将以上所有结构、文本、网络特征合并为特征矩阵 `X`；
- 标签 `y` 表示该邮件是否与 POI 有关。

### 2. 模型训练与评估

- 使用 `RandomForestClassifier` 构建分类模型；
- 对模型进行评估（如准确率、召回率等）；
- 提取并输出最重要的特征。

### 3. 应用与风险评估

- 为每封邮件生成一个“欺诈风险分数”；
- 输出 POI 邮件活动摘要报告。

---

##  输出结果

- POI 通信网络图及中心性排序；
- 邮件欺诈风险评分表；
- 重要特征排名；

---
