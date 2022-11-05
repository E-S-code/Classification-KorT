#%%
from re import T
import pandas as pd
df = pd.read_csv("datafiles\KvsT.csv") #csvを読み込んでデータフレームに変換
df.head(3) #先頭の3行だけ

# %%
df["身長"] #身長列だけを参照

# %%
col = ["身長", "体重"]
df[col] #身長列と体重のみを抜き出す


# %%
type(df["派閥"])

# %%
xcol = ["身長", "体重", "年代"] #特徴量データだけxcolにぶちこむ
x = df[xcol]

#%%
t = df["派閥"] #正解データをtにぶちこむ
t

#%%
from sklearn import tree

#%%
#モデルの準備（未学習）
model = tree.DecisionTreeClassifier(random_state = 0)

#学習の実行（x、tは事前に定義済みの特徴量と正解ラベル）
model.fit(x, t)


# %%
#身長170、体重70、年齢20のデータを新しく2次元リストで作成
taro = [[170, 70, 20]]

#taroがどっちに分類されるか予測
model.predict(taro)

#%%
matsuda = [172, 65, 20]
asagi = [158, 48, 20]
new_data = [matsuda, asagi] #松田と浅木のデータを二次元リスト化

model.predict(new_data) #二人のデータを一括で予測

#%%
#己
sugiyama = [[165,56,20]]
model.predict(sugiyama)

# %%
#データフレーム型でモデルを作ったのにリスト型で渡しているからワーニングが出るのの対処
taro = [[170, 70, 20]]
taro = pd.DataFrame(taro, columns = ["身長", "体重", "年代"])
model.predict(taro)

# %%
#木の画像を出す
from sklearn.tree import plot_tree
plot_tree(model)

# %%
