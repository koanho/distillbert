from transformers import BertForMaskedLM, BertConfig, AdamW, BertForNextSentencePrediction
from tqdm import tqdm 
import gc
import torch
import torch.nn as nn
import numpy as np
import os
import load_data
import torch.nn.functional as F



def distillation_loss(teacher_logits, t_h, student_logits, s_h, mlm_loss, criterion, device, num_hidden = 1, t = 1):
    loss = torch.tensor(0)
    t_i = F.softmax(teacher_logits / t, dim = -1)
    log_s_i = F.log_softmax(student_logits / t, dim = -1)
    loss = torch.mean(- t_i * log_s_i)
    loss += torch.mean(mlm_loss)
    for i in range(num_hidden):
        y = torch.ones(t_h[i*2].shape[0], t_h[i*2].shape[1]).to(device)
        loss += criterion(t_h[i*2].view(-1, t_h[i*2].shape[-1]), s_h[i].view(-1, s_h[i].shape[-1]), y.view(-1))
        
    return loss
    

def load_student(device, pretrained=False, teacher_model=None):
    vocab_size = 30522
    student_hddn_dim = 768
    student_num_hddn_layers =  6
    student_num_attn_heads = 12
    student_ff_dim = 3072
    student_embd_dim = 768
    
    student_config = BertConfig(
        hidden_size=student_hddn_dim,
        num_hidden_layers=student_num_hddn_layers,
        num_attention_heads=student_num_attn_heads,
        intermediate_size=student_ff_dim,
    )
    
    if pretrained:
        student_model = BertForNextSentencePrediction(student_config)
        student_model.load_state_dict(torch.load('/home/nas3_userJ/koanholee/Bert/bert_models/student_wikitext.pt'), strict = False)
    else :
        student_model = BertForMaskedLM(student_config)
        ### initialization using pretrained_bert weight ###
        student_model.bert.embeddings = teacher_model.bert.embeddings
        for i in range(0, len(teacher_model.bert.encoder.layer)//2):
            student_model.bert.encoder.layer[i] = teacher_model.bert.encoder.layer[i*2]

    student_model = nn.DataParallel(student_model)
    student_model = student_model.to(device)

    return student_model



def train_control_wiki(device):
    vocab_size = 30522
    student_hddn_dim = 768
    student_num_hddn_layers =  6
    student_num_attn_heads = 12
    student_ff_dim = 3072
    student_embd_dim = 768
    
    student_config = BertConfig(
        hidden_size=student_hddn_dim,
        num_hidden_layers=student_num_hddn_layers,
        num_attention_heads=student_num_attn_heads,
        intermediate_size=student_ff_dim,
    )
    control_model = BertForMaskedLM(student_config)
    control_model = nn.DataParallel(control_model)
    control_model = control_model.to(device)
    wiki = load_data.wiki()

    gc.collect()
    lr = 1e-4
    batch_size = 30
    optimizer = AdamW(control_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=20
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        wiki["train"].shuffle()
        t = tqdm(range(0,len(wiki["train"]),batch_size))
        accuracies = []
        losses = []
        control_model.train()
        for i in t:
            optimizer.zero_grad()
            data = wiki["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            masked_lm_loss, prediction_scores = control_model(**data)
            loss = torch.sum(masked_lm_loss)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        control_model.eval()
        t_val = tqdm(range(0,len(wiki["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = wiki["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                masked_lm_loss, prediction_scores = control_model(**data)
                accuracy = torch.eq(prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(control_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/control_wikitext.pt')


def train_control_mrpc(device):
    vocab_size = 30522
    student_hddn_dim = 768
    student_num_hddn_layers =  6
    student_num_attn_heads = 12
    student_ff_dim = 3072
    student_embd_dim = 768
    
    student_config = BertConfig(
        hidden_size=student_hddn_dim,
        num_hidden_layers=student_num_hddn_layers,
        num_attention_heads=student_num_attn_heads,
        intermediate_size=student_ff_dim,
    )
    control_model = BertForNextSentencePrediction(student_config)
    control_model.load_state_dict(torch.load('/home/nas3_userJ/koanholee/Bert/bert_models/control_wikitext.pt'), strict = False)
    control_model = nn.DataParallel(control_model)
    control_model = control_model.to(device)

    mrpc = load_data.mrpc()

    gc.collect()
    lr = 2e-5
    batch_size = 3
    optimizer = AdamW(control_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=10
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        mrpc["train"].shuffle()
        t = tqdm(range(0,len(mrpc["train"]),batch_size))
        accuracies = []
        losses = []
        control_model.train()
        for i in t:
            optimizer.zero_grad()
            data = mrpc["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            prediction_scores = control_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
            loss = criterion(prediction_scores, data['labels']).to(device)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        control_model.eval()
        t_val = tqdm(range(0,len(mrpc["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = mrpc["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                prediction_scores = control_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
                accuracy = torch.eq(prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
                del data
                torch.cuda.empty_cache()
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(control_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/control_mrpc.pt')


def train_teacher_wiki(device):
    wiki = load_data.wiki()
    teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = teacher_model.to(device)
    gc.collect()
    lr = 1e-4
    batch_size = 60
    optimizer = AdamW(teacher_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=20
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        wiki["train"].shuffle()
        t = tqdm(range(0,len(wiki["train"]),batch_size))
        accuracies = []
        losses = []
        teacher_model.train()
        for i in t:
            optimizer.zero_grad()
            data = wiki["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            masked_lm_loss, prediction_scores = teacher_model(**data)
            loss = torch.sum(masked_lm_loss)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        teacher_model.eval()
        t_val = tqdm(range(0,len(wiki["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = wiki["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                masked_lm_loss, prediction_scores = teacher_model(**data)
                accuracy = torch.eq(prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(teacher_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/teacher_wikitext.pt')


def train_student_wiki(device):
    gc.collect()

    ##load data
    wiki = load_data.wiki()

    ##load modelsx
    teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    teacher_model.load_state_dict(torch.load('/home/nas3_userJ/koanholee/Bert/bert_models/teacher_wikitext.pt'))

    student_model = load_student(device, False, teacher_model)

    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = teacher_model.to(device)

    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    
    lr = 2e-4
    batch_size = 30
    optimizer = AdamW(student_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=20
    cosine_embedding_loss = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        wiki["train"].shuffle()
        t = tqdm(range(0,len(wiki["train"]),batch_size))
        accuracies = []
        losses = []
        teacher_model.train()
        for i in t:
            optimizer.zero_grad()
            
            ## data to GPU
            data = wiki["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            
            _, teacher_prediction_scores, teacher_hidden_states = teacher_model(**data, output_hidden_states= True)
            masked_lm_loss, student_prediction_scores, student_hidden_states = student_model(**data, output_hidden_states = True)
        
            loss = distillation_loss(teacher_prediction_scores, teacher_hidden_states, student_prediction_scores, student_hidden_states, masked_lm_loss, cosine_embedding_loss, device, num_hidden = 6, t = 10).to(device)

            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(student_prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        teacher_model.eval()
        t_val = tqdm(range(0,len(wiki["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = wiki["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                masked_lm_loss, prediction_scores = student_model(**data)
                accuracy = torch.eq(prediction_scores.argmax(dim=2,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
                del data
                torch.cuda.empty_cache()
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(student_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/student_wikitext.pt')


def train_mrpc(device):
    mrpc = load_data.mrpc()
    student_model = load_student(device, True)

    gc.collect()
    lr = 2e-5
    batch_size = 3
    optimizer = AdamW(student_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=10
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        mrpc["train"].shuffle()
        t = tqdm(range(0,len(mrpc["train"]),batch_size))
        accuracies = []
        losses = []
        student_model.train()
        for i in t:
            optimizer.zero_grad()
            data = mrpc["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            student_prediction_scores = student_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
            loss = criterion(student_prediction_scores, data['labels']).to(device)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(student_prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        student_model.eval()
        t_val = tqdm(range(0,len(mrpc["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = mrpc["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                prediction_scores = student_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
                accuracy = torch.eq(prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
                del data
                torch.cuda.empty_cache()
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(student_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/student_mrpc.pt')

def train_teacher_mrpc(device):
    mrpc = load_data.mrpc()
    gc.collect()
    
    teacher_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    teacher_model.load_state_dict(torch.load('/home/nas3_userJ/koanholee/Bert/bert_models/teacher_wikitext.pt'), strict = False)
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = teacher_model.to(device)
    
    lr = 2e-5
    batch_size = 5
    optimizer = AdamW(teacher_model.parameters(), lr = lr, weight_decay=0.01)
    epochs=10
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        mrpc["train"].shuffle()
        t = tqdm(range(0,len(mrpc["train"]),batch_size))
        accuracies = []
        losses = []
        teacher_model.train()
        for i in t:
            optimizer.zero_grad()
            data = mrpc["train"][i:i+batch_size]
            data = {k: torch.tensor(v).to(device) for k,v in data.items()}
            teacher_prediction_scores = teacher_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
            loss = criterion(teacher_prediction_scores, data['labels']).to(device)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(teacher_prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
            accuracies.append(accuracy.detach().cpu().numpy())
            loss = np.around(np.mean(losses[-100:]),5)
            accuracy = np.around(np.mean(accuracies[-100:]),3)
            t.set_description("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Accuracy: "+str(accuracy))
    
        val_loss = []
        val_accuracy = []
        teacher_model.eval()
        t_val = tqdm(range(0,len(mrpc["validation"]),batch_size))
        with torch.no_grad():
            for i in t_val:
                data = mrpc["validation"][i:i+batch_size]
                data = {k: torch.tensor(v).to(device) for k,v in data.items()}
                teacher_prediction_scores = teacher_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'])[0]
                accuracy = torch.eq(teacher_prediction_scores.argmax(dim=1,keepdim=False).float(),data['labels']).float().mean()
                val_accuracy.append(accuracy.detach().cpu().numpy())
                accuracy = np.around(np.mean(val_accuracy),3)
                t_val.set_description("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))
                del data
                torch.cuda.empty_cache()
        print("Epoch: "+str(epoch)+" val_Accuracy: "+str(accuracy))

    torch.save(teacher_model.module.state_dict(),'/home/nas3_userJ/koanholee/Bert/bert_models/teacher_mrpc.pt')



def train_control(device):
    print('#'*10 , "train control wiki", '#' * 10)
    train_control_wiki(device)
    print('#'*10 , "train control mrpc", '#' * 10)
    train_control_mrpc(device)


def main():
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,6'
    
    #print("#"*10, "Training Teacher", "#"*10)
    #train_teacher_wiki(device)

    #print("#"*10, "Training Student", "#"*10)
    #train_student_wiki(device)

    #print("#"*10, "Training downstream Student MRPC", "#"*10)
    ## downstream task MRPC ###
    #train_mrpc(device)

    #print("#"*10, "Training downstream teacher MRPC", "#"*10)
    #train_teacher_mrpc(device)

    train_control(device)

if __name__ == '__main__':
    main()