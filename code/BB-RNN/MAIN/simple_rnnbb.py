# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
#from common.time_layers_pred import *
from common.time_layers import *
import pickle


class SimpleRnnbb:
    def __init__(self, output_size, input_size, hidden_size):
        V, D, H = output_size, input_size, hidden_size
        rn = np.random.randn
        # 重みの初期化
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        '''
        rnn_Wx = rnn_Wx.tolist()
        rnn_Wh = rnn_Wh.tolist()
        rnn_b = rnn_b.tolist()
        affine_W = affine_W.tolist()
        affine_b = affine_b.tolist()
        '''
        # レイヤの生成
        self.layers = [
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

class Pred_simpleRnnbb:
    def __init__(self, output_size, input_size, hidden_size, params):
        V, D, H = output_size, input_size, hidden_size
        #rn = np.random.randn
        # 重みの初期化
        ''''''
        #rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        #rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        #rnn_b = np.zeros(H).astype('f')
        #affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        #affine_b = np.zeros(V).astype('f')

        rnn_Wx = params[0]
        rnn_Wh = params[1]
        rnn_b = params[2]
        affine_W = params[3]
        affine_b = params[4]
        '''
        rnn_Wx = rnn_Wx.tolist()
        rnn_Wh = rnn_Wh.tolist()
        rnn_b = rnn_b.tolist()
        affine_W = affine_W.tolist()
        affine_b = affine_b.tolist()
        '''
        # レイヤの生成
        self.layers = [
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        ys = self.loss_layer.forward(xs, ts)
        return ys
    
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    '''
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)'''