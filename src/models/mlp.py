class MLPModel(nn.Module):
    def __init__(self, layers):
        super(MLPModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        module = []
        input_dim = layers[0]
        for n in layers[1:-1]:
            module.append(nn.Linear(input_dim, n))
            module.append(nn.ReLU())
            input_dim = n
        module.append(nn.Linear(input_dim, layers[-1]))
        self.net = nn.Sequential(*module).to(self.device)
        print(f"Using device: {self.device}")

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return self.device