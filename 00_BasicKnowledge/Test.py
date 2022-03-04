import torch
from torch import nn
from d2l import torch as d2l

n_train,n_test,num_inputs,batch_size = 20,100,200,5

true_w,true_b = torch.ones((num_inputs,1)) * 100,0.05

train_data = d2l.synthetic_data(true_w,true_b,n_train)
train_iter = d2l.load_array(train_data,batch_size)

test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)