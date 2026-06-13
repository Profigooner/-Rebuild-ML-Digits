import json
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
class network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases= [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights=[np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    def feed_forward(self, x):
        # Ensure column-vector shape for matrix math.
        a = np.asarray(x).reshape(-1, 1)
        # Apply affine + sigmoid layer by layer.
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feed_forward_with_activations(self, x):
        a = np.asarray(x).reshape(-1, 1)
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
            activations.append(a)
        return activations

    def to_dict(self):
        return {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        net = cls(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net

if __name__ == "__main__":
    net1 = network([3, 4, 8, 1])
    print(net1.feed_forward([1, 2, 3]))
