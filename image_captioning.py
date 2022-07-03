# -*- coding: utf-8 -*-
"""Image captioning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10bskcAOGXZ5k7ofI5tomNerT9dPyrcyE
"""

import os
import pandas as pd
import spacy
import torch
from torch.nn .utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0 : "<PAD>" , 1 :"<SOS>" , 2 : "<EOS>", 3 : "<UNK>"}
        self.stoi = {"<PAD>" : 0 , "<SOS>" :1 ,"<EOS>" : 2 , "<UNK>" :3 }
        self.freq_threshold = freq_threshold
    def __len__(self):
        return len(self.itos)
    @staticmethod
    def tokenizer_eng(text):
        return[tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    def build_vocabulary(self,sentence_list):
        frequencies ={}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] +=1

                if (frequencies[word] == self.freq_threshold):
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx +=1
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        return[
               self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
               for token in tokenized_text
        ]                

class FlickrDataset(Dataset):
    def __init__(self,root_dir , captions_file ,transform =None , freq_threshold = 5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir , img_id)).convert("RGB")
        if self.transforms is not None:
            img = self.transform(img)
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return img, torch.tensor(numerical)  

class MyCollate:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
     
    def __cal__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets , batch_first = False , padding_value = self.pad_idx)
        return imgs,targets

def get_loader(
    root_folder, annotation_file , transform, batch_size = 32 ,shuffle = True, pin_memory = True,
):
    dataset = FlickrDataset(root_folder,annotation_file , transform = transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset,batch_size = batch_size  , shuffle = shuffle, pin_memory = pin_memory ,collate_fn = MyCollate(pad_idx = pad_idx))
    return loader
    
def main():  
    transform = transforms.Compose(
        [
         transforms.Resize((224,224)),
         transforms.ToTensor(),
        ]
    )  
    dataloader = get_loader("/content/drive/MyDrive/Colab Notebooks/archive (1)/flickr8k/images" , annotation_file = "/content/drive/MyDrive/Colab Notebooks/archive (1)/flickr8k/captions.txt",transform = transform)

    for idx,(img,captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN,self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True , aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu = nn.ReLu()
        self.dropout = nn.Dropout(0.5)
    def forward(self,images):
        features = self.inception(images)
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        return self.dropout(self.relu(features))
class DecoderRNN(nn.Module):
    def __init__(self, embed_size , hidden_size, vocab_size , num_layers):
        super(DecoderRNN , self).__init__()
        self.embed = nn.Embedding(vocab_size , embed_size)
        self.lstm = nn.LSTM(embed_size , hidden_size , num_layers)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)
    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)    
        hiddens,_ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module) :
    def __init__(self, embed_size , hidden_size, vocab_size , num_layers):
        super(DecoderRNN , self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size , hidden_size, vocab_size , num_layers) 
    def forward(self,images,captions):
        features = self.encoderCNN(images)
        outputs  = self.decoderRNN(features,captions)  
    def caption_image(self,image,vocabulary,max_length = 50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            for _ in ranage(50):
                hiddens,states = self.decoderRnn.lstm(x,states)
                output = self.decoderRNN.linear(hiddens.unsqueeze(0))
                predicted = ouput.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                if vocabulary.itos[predicted.item()]== "<EOS>":
                    break  
        return [vocabulary.itos[idx] for idx in result_caption]

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def train():
    transform = transforms.Compose(
       [transforms.Resize([356, 356]),
        transforms.Randomcrop((299,299)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5,0.5,0.5], std= [0.5,0.5, 0.5])
       ])
    train_loader = get_loader(
        root_folder = "/content/drive/MyDrive/Colab Notebooks/archive (1)/flickr8k/images",
        annotation_file= "/content/drive/MyDrive/Colab Notebooks/archive (1)/flickr8k/captions.txt",
        transform = tranform
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True


    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    model = CNNtoRNN(embed_size, hidden_size, vocab_size,num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    
    model.train()
    
    for epoch in range(num_epochs):
        for idx,(imgs,captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs,captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]),captions.reshape(-1))
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            print("Step: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}%".format(i+1, len(dataset), epoch_loss/(i+1), 100*epoch_acc/(i+1)), end = '\r')
if __name__ == "main" :
    train()