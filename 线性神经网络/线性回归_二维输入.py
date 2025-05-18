import torch
import torch.nn as nn
import matplotlib.pylab as plt

x=torch.tensor([[1.0,2.0],
                [2.0,1.0],
                [3.0,3.0],
                [4.0,2.0],
                [5.0,1.0]],dtype=torch.float32)

y=torch.tensor([[3*1+2*2+1],
                [3*2+2*1+1],
                [3*3+2*3+1],
                [3*4+2*2+1],
                [3*5+2*1+1]],dtype=torch.float32)

model=nn.Linear(in_features=2,out_features=1)

loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(5000):

    y_pred=model(x)
    loss=loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch%100==0:
        print(f"Epoch{epoch},Loss:{loss.item():.4f}")

w,b=model.weight.data,model.bias.data
print(f"Learned weight:{w}")
print(f"Learned bias:{b}")


