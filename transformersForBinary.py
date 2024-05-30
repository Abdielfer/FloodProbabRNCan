import torch
import torch.nn as nn
import torch.nn.functional as F
import myServices as ms


# Define the model
class TransformerClassifier(nn.Module):
    def __init__(self, n_features, outDimention:int=1):
        super(TransformerClassifier, self).__init__()
        self.Features = n_features
        self.transformer = nn.MultiheadAttention(n_features,n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(),
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(),
            nn.Linear(n_features, int(n_features/2)),
            nn.LeakyReLU(),
            nn.Linear(int(n_features/2), outDimention)
        )
        self.a = nn.Parameter(torch.tensor([0.001]))
        self.b = nn.Parameter(torch.tensor([0.999]))

    def forward(self, x):
        att_out,_ = self.transformer(x,x,x)
        ## Attention OutPut flattenned
        att_out = att_out.view(-1,self.Features)
        mlp_out = self.mlp(att_out)
        prob = torch.sigmoid(self.b * (mlp_out- self.a))
        return prob

class TransformerModel_1(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=2, num_heads=4):
        super(TransformerModel_1, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)
        self.a = nn.Parameter(torch.tensor([0.001]))
        self.b = nn.Parameter(torch.tensor([0.999]))
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[-1])
        prob = torch.sigmoid(self.b*x- self.a)
        return prob
    

def main():
    modelPath = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\MLP_Models\2404081916.pkl'
    model = ms.loadModel(modelPath)
    


    
if __name__ == "__main__":
    with ms.timeit():
        main()
       