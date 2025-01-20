from datautils import get_loaders
import torch
cache_dir = './cache'

model = './quant/Llama-2-7b-chat-hf-w4a4/'
net = model.split('/')[-1]
model_family = net.split('-')[0]

dataset = 'c4'
seed = 2
seqlen = 2048

cache_testloader = f'{cache_dir}/testloader_{model_family}_{dataset}_all.cache'
cache_trainloader = f'{cache_dir}/trainloader_{model_family}_{dataset}_all.cache'
dataloader, testloader = get_loaders(
    dataset,
    seed=seed,
    model=model,
    seqlen=seqlen,
)
torch.save(dataloader, cache_trainloader)
torch.save(testloader, cache_testloader)


# python -m my_tests.get_dataset.py