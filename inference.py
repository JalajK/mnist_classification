import torch

from utils import Net, test_loader

model = Net()
network_state_dict = torch.load("model.pth")
model.load_state_dict(network_state_dict)


sample = next(iter(test_loader))
imgs, nums = sample
actual_number = nums[:10].numpy()

test_output = model(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f"Prediction number: {pred_y}")
print(f"Actual number: {actual_number}")
