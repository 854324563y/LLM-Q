import torch
import numpy as np

m = torch.randn(10, 10).numpy()
print(m)

es, us = np.linalg.eig(m)

m_ = us @ np.diag(es) @ us.T
print(m_)

# calculate error
error = np.sum((m - m_)**2)/np.sum(m**2)
print(error)

