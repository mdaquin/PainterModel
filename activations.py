from model import PainterDataset, PainterModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

class PainterActivations:
    def __init__(self, layer, max_length=128, batch_size=128, aggregation="cls", traintest = "both", device="cpu"):
        self.device = device
        self.layer = layer
        MODEL_NAME = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if traintest == "both": dataset = PainterDataset(["data/test.csv", "data/train.csv"], tokenizer, max_length, returnIDs=True)
        elif traintest == "train": dataset = PainterDataset("data/train.csv", tokenizer, max_length, returnIDs=True)
        elif traintest == "test": dataset = PainterDataset("data/test.csv", tokenizer, max_length, returnIDs=True)
        else: raise ValueError('traintest should be "train", "test", or "both"')
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = PainterModel(MODEL_NAME, n_classes=2, dropout=0.3)
        self.model = self.model.to(device)
        self.model.eval()
        self.aggregation = "cls"

        self.llayers = []
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                if type(output) != torch.Tensor: self.activation[name] = output
                else: 
                    self.activation[name] = output.cpu().detach()
                    # print(name, activation[name].shape)
            return hook
        def rec_reg_hook(mo, prev="", lev=0):
            for k in mo.__dict__["_modules"]:
                name = prev+"."+k if prev != "" else k
                nmo = getattr(mo,k)
                if name == self.layer: 
                    nmo.register_forward_hook(get_activation(name))
                    print("--"+"--"*lev, "hook added for",name)
                self.llayers.append(name)
                rec_reg_hook(nmo, prev=name, lev=lev+1)
        rec_reg_hook(self.model)
    
    def __iter__(self): 
        self.loader.__iter__()
        self.counter = 0
        return self

    def __len__(self): return len(self.loader)

    def __next__(self):
        if self.counter is not None and self.counter >= len(self): raise StopIteration()
        self.counter += 1
        batch = next(iter(self.loader))
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label']
        IDs = batch['ID']
        self.activation = {}
        acts={}
        with torch.no_grad():
            preds = self.model(input_ids=input_ids, attention_mask=attention_mask)
        for layer in self.activation:
            act = self.activation[layer] 
            if type(act) == tuple: 
                if self.aggregation == "cls": acts[layer] = act[0].cpu()[:,0:,]
            elif str(type(act)) == "<class 'transformers.modeling_outputs.BaseModelOutput'>":
                if self.aggregation == "cls": acts[layer] = act.last_hidden_state.cpu()[:,0,:]
            else: 
                if self.aggregation == "cls":  
                    if len(act.shape) == 3: acts[layer] = act.cpu()[:,0,:]
                    else: acts[layer] = act.cpu()
        return IDs, acts, labels, preds

if __name__ == "__main__": 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    acti = PainterActivations("bert.transformer", device=device)
    for i,layer in enumerate(acti.llayers):
        print(i,"-",layer)
    idx = int(input("For which layer do you which to save activations? "))
    acti = PainterActivations(acti.llayers[idx], device=device)
    tosave = {"ids": [], "acts": {}}
    tosave["acts"][acti.llayers[idx]] = []
    for ids, acts, label, preds in tqdm(acti, f"Extracting activations for {acti.llayers[idx]}"):
       tosave["ids"]+=ids
       for layer in acts: tosave["acts"][layer]+=acts
    torch.save(tosave, f"activations/{acti.llayers[idx]}.pth")
       
        
