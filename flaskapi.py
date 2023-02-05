import torch
from flask import Flask, jsonify

app = Flask(__name__)

from utils import Net, test_loader

model = Net()
network_state_dict = torch.load("model.pth")
model.load_state_dict(network_state_dict)


@app.route("/predict", methods=["POST"])
def predict():
    sample = next(iter(test_loader))
    imgs, nums = sample
    actual_number = nums[:10].numpy()

    test_output = model(imgs[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    return jsonify({"Predictions": str(pred_y), "actuals": str(actual_number)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
