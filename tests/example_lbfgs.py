import torch

f = lambda x: (torch.sin(x)**2/3).sum()
X = torch.zeros(1000)
X[:900] = torch.randn(900).requires_grad_(True)

XX = X[:900].clone().detach().requires_grad_(True)
YY = X[900:].clone().detach().requires_grad_(True)

optimizer_f = torch.optim.LBFGS(
        [X],
        max_eval=15,
        max_iter=10,
        line_search_fn="strong_wolfe",
        history_size=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )
    
 
def closure():
    optimizer_f.zero_grad()

    L = f(X)
    print("loss", L.detach().cpu().numpy())
    L.backward()
    return L 


for i in range(10):
    print("it ", i, ": ", end="")
    optimizer_f.step(closure)
    
 

ff = lambda x,y: (torch.sin(x)**2/3).sum() + (torch.sin(y)**2/3).sum()

optimizer_ff = torch.optim.LBFGS(
        [XX, YY],
        max_eval=15,
        max_iter=10,
        line_search_fn="strong_wolfe",
        history_size=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )
    
 
def closure():
    optimizer_ff.zero_grad()

    L = ff(XX, YY)
    print("loss", L.detach().cpu().numpy())
    L.backward()
    return L 


for i in range(10):
    print("it ", i, ": ", end="")
    optimizer_ff.step(closure)
