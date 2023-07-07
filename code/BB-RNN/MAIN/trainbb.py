# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
#from dataset import ptb
from simple_rnnbb import SimpleRnnbb


# ハイパーパラメータの設定
batch_size = 10
input_size = 100 #sbarの次元数
output_size = 20
hidden_size = 10  # RNNの隠れ状態ベクトルの要素数
#time_size = 5  # RNNを展開するサイズ
lr = 0.1
max_epoch = 100

# 学習データの読み込み
xs = corpus[:-1]  # 入力
ts = corpus[1:]  # 出力（教師ラベル）play

# モデルの生成
#model = SimpleRnnbb(output_size, input_size, hidden_size)
#optimizer = SGD(lr)
#trainer = RnnlmTrainer(model, optimizer)

#trainer.fit(xs, ts, max_epoch, batch_size)
#trainer.plot()
