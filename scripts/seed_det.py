import random, hashlib, sys
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
seed = int(sys.argv[1])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
g = torch.Generator().manual_seed(seed)
model = resnet18(weights="IMAGENET1K_V1"); model.fc = nn.Linear(model.fc.in_features, 1)
ds = TensorDataset(torch.arange(1307))
order = [int(i) for epoch in range(3) for (b,) in DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, generator=g) for i in b]
print("seed", seed, "fc", hashlib.sha256(model.fc.weight.detach().numpy().tobytes()).hexdigest()[:12],
      "order(3 epochs)", hashlib.sha256(str(order).encode()).hexdigest()[:12], "first8", order[:8])
