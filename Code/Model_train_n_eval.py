# Basic lib
import os
import re
import string
import emoji
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Transformers lib
import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn 
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

# Sklearn 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Dict for text preprocessing
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

# Create empty list for metrics 
train_loss_values = []
train_loss_avg = []
validation_loss_values = []
validation_loss_avg = []
train_accuracy_values = []
validation_accuracy_values = []
learning_rates = []


#Positive:
###Group1 - Joy & Vitality: ["joy", "excitement", "amusement", "happiness", 'fun']
###Group2 - Love & Gratitude: ["optimism", "love", "approval", "gratitude"]
###Group3 - Warm human relationship: ["caring", "admiration", "pride"]

#Negative:
###Group4 - Sadness & disappointment: ["sadness", "grief", "disappointment"]
###Group5 - Anxiety & Fear: ["fear", "nervousness", "remorse", "worry"]
###Group6 - Anger & dissatisfaction: ["anger", "annoyance", "disgust", "disapproval", "hate"]

#Other emotions:
###Group7 - Confusion & Curiosity: ["confusion", "curiosity", "realization", "surprise"]
###Group8 - embarrassment & Relief: ["embarrassment", "relief"]
###Group9 - Passion & desire: ["desire", "Enthusiasm"]


label_names = ['1','2','3','4','5','6','7','8','9']
label_names_ = ['0','1','2','3','4','5','6','7','8','9']

labels_w_Neutral = ["Neutral", "Joy & Vitality", "Love & Gratitude", "Warm Relationships",
          "Sadness & Disappointment", "Anxiety & Fear", "Anger & Dissatisfaction",
          "Confusion & Curiosity", "Embarrassment & Relief", "Passion and Desire"]

labels_wo_Neutral = ["Joy & Vitality", "Love & Gratitude", "Warm Relationships",
          "Sadness & Disappointment", "Anxiety & Fear", "Anger & Dissatisfaction",
          "Confusion & Curiosity", "Embarrassment & Relief", "Passion and Desire"]

Full_28_labels = ["admiration",
                "amusement",        
                "anger",         
                "annoyance",         
                "approval",          
                "caring",            
                "confusion",         
                "curiosity",         
                "desire",            
                "disappointment",    
                "disapproval",       
                "disgust",           
                "embarrassment",     
                "excitement",        
                "fear",              
                "gratitude",         
                "grief",             
                "joy",               
                "love",             
                "nervousness",       
                "optimism",
                "pride",
                "realization",
                "relief",
                "remorse",
                "sadness",
                "surprise",
                "neutral"
                ]

# Parameters Setup

# Path Setup
DATAFRAME_PATH = "Dataset/Dataset1_9.csv"
SAVE_MODEL_NAME = "BERT_CNN_MODEL_W_Corr.pth"
CORR_PATH = "Dataset/corr_rels3_10.csv"

SPLIT_RATIO = 0.1 # Train 90% Test 10%

device = None

# DataLoader Parameters
MAX_LEN = 256
BATCH_SIZE = 16

# Train Parameters
EPOCHS = 2
LEARNING_RATE = 5e-5
DROPOUT = 0.1
NUM_LABELS = 9
WARMING_UP = 0.3
WEIGHT_DECAY = 1e-6
MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL)


# Define BERT Class : BERT + CNN Layer
class BERT_CNN_Class(torch.nn.Module):
    def __init__(self, num_labels, dropout, correlation, corr_rels):
        super(BERT_CNN_Class, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1d = torch.nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.l2 = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(256, num_labels)
        self.correlation = correlation 
        self.num_labels = num_labels
        self.corr_rels = corr_rels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs.last_hidden_state # Use hidden state  
        sequence_output = sequence_output.permute(0, 2, 1)  # (batch_size, hidden_size, seq_length)
        conv_output = self.conv1d(sequence_output)  # (batch_size, out_channels, seq_length)
        conv_output = F.relu(conv_output)  # ReLU Function
        pooled_output = F.max_pool1d(conv_output, conv_output.shape[2]).squeeze(2)  # max pooling
        logits = self.fc(self.l2(pooled_output))

        prob = torch.sigmoid(logits)
        per_example_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss = per_example_loss.mean()

        probs_exp = prob.unsqueeze(1)
        m = probs_exp.repeat(1, self.num_labels, 1)
        probs_exp_t = probs_exp.transpose(1, 2)
        
        dists = torch.square(m - probs_exp_t)
        dists = dists.transpose(1, 2)
        
        # Correlation-based regularization
        corr_reg = self.correlation * torch.mean(dists * self.corr_rels)

        loss += corr_reg

        return logits, loss
    


# BERT + Fully Connected Layer
class BERTClass(torch.nn.Module):
    def __init__(self, num_labels, dropout):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.fc(self.l2(pooled_output))
        return logits


# BERT + RNN Layer
class BERT_RNN_Class(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, bidirectional=True):
        super(BERT_RNN_Class, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # Reshape pooled output to match the input shape of the RNN
        pooled_output = pooled_output.unsqueeze(1)
        rnn_output, _ = self.rnn(pooled_output)
        # Get the last hidden state of the RNN output
        last_hidden_state = rnn_output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits


# Text Preprocessing Functions

def clean_text(text):
    """Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''    
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    #Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_special_chars(text, punct, mapping):
    #Cleans special characters present   
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def remove_space(text):
    '''Removes awkward spaces'''   
    #Removes awkward spaces 
    text = text.strip()
    text = text.split()
    return " ".join(text)

"""

"Hi how are  you?"  => ["Hi", "how", "are", "you"]  

"""

# Text Preprocessing Pipeline
def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_contractions(text, contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = remove_space(text)
    return text


# Tokenize Texts
def prepare_data(texts, tokenizer, max_len):
    inputs = [tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True
            )for text in texts]
    input_ids = [item['input_ids'] for item in inputs]
    attention_mask = [item['attention_mask'] for item in inputs]
    token_type_ids = [item['token_type_ids'] for item in inputs]

    return input_ids, attention_mask, token_type_ids


# Convert data to 'torch' and make them to DataLoader
def create_dataloader(input_ids, attention_mask, token_type_ids, labels, batch_size, device, train):

    input_ids= torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    if train:
        data = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
        # If train is True, use RandomSampler
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    else: 
        data = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
        dataloader = DataLoader(data, batch_size=batch_size, pin_memory=False)

    return dataloader

def loss_fn(outputs, labels):
    return torch.nn.BCEWithLogitsLoss()(outputs, labels) # Loss Function for Multi-Label Classification

def dataloader_pipeline(df):
    # Spliting Data
    test_df = df.sample(frac=SPLIT_RATIO) # Selects 10% of the data randomly
    train_df = df.drop(test_df.index)

    # Apply Text Preprocessing Pipeline
    train_texts = train_df['text'].apply(text_preprocessing_pipeline)
    test_texts = test_df['text'].apply(text_preprocessing_pipeline)

    # Convert Labels to List
    train_labels = train_df.iloc[:,1:].values.tolist()
    test_labels = test_df.iloc[:, 1:].values.tolist()

    # Convert Labels to Torch
    train_labels = torch.stack([torch.tensor(label) for label in train_labels])
    test_labels = torch.stack([torch.tensor(label) for label in test_labels])

    # Make DataLoaders
    input_ids, attention_mask, token_type_ids = prepare_data(train_texts, tokenizer, max_len=MAX_LEN)
    train_dataloader = create_dataloader(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids, labels=train_labels,
                                        batch_size=BATCH_SIZE, device=device, train=True)
    input_ids, attention_mask, token_type_ids = prepare_data(test_texts, tokenizer, max_len=MAX_LEN)
    test_dataloader = create_dataloader(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids, labels=test_labels,
                                        batch_size=BATCH_SIZE, device=device, train=False)

    return train_dataloader, test_dataloader

# Get Optimizer ans Scheduler
# Optimizer : AmamW
# Scheduler : linear scheduler with warmup
def get_opti(model, train_dataloader):
    num_training_steps = len(train_dataloader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMING_UP)  # Define Warming Up Ratio

    optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
        )
    return optimizer, scheduler


# Train & Validation Function
def train_validation(epoch, model, train_dataloader, test_dataloader, optimizer, scheduler):
    model.train() # Set model to train mode
    print(f'Epoch:  {epoch}')
    
    # Initialize some variables
    total_loss = 0
    total_correct = 0
    total_elements = 0
    batch_count, batch_loss = 0, 0

    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*70)

    t0_epoch, t0_batch = time.time(), time.time()

    for _,data in enumerate(train_dataloader, 0):
        batch_count += 1
        lr = scheduler.get_last_lr()[0] # Get Current Learning Rate

        # Get 4 variables from train dataloader
        input_ids, attention_mask, token_type_ids, labels = data
        
        # Move these variables to device
        input_ids = input_ids.to(device, dtype = torch.long)
        attention_mask = attention_mask.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        labels = labels.to(device, dtype = torch.float)

        # Get Outputs(logits)
        outputs = model(input_ids, attention_mask, token_type_ids)

        # Get Current Loss value 
        loss = loss_fn(outputs, labels=labels)
        batch_loss += loss.item()
        total_loss += loss.item()
    
        # Print States every 20 steps
        if (_ % 20 == 0 and _ != 0) or (_ == len(train_dataloader) - 1):            
            time_elapsed = time.time() - t0_batch
            print(f"{epoch + 1:^7} | {_:^7} | {batch_loss / batch_count:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
            batch_loss, batch_count = 0, 0
            t0_batch = time.time()

        # Append learning rate every 50 steps
        if _%50 == 0:
            learning_rates.append(lr)

        # Append loss value every 50 steps
        if _%50 == 0:
            train_loss_values.append(loss.item())
            
        loss.backward() # Perform backward
        optimizer.step()  # Perform optimization step
        scheduler.step()  # Update learning rate scheduler
        optimizer.zero_grad()  # Reset gradients

        # Apply sigmoid function to convert predictions to probabilities and set prediction labels based on threshold 0.5
        predictions = torch.sigmoid(outputs) > 0.5

        total_correct += torch.sum(predictions == labels).item()  # Accumulate the number of correct predictions
        total_elements += torch.numel(labels)  # Count the total number of labels

    train_acc = total_correct / total_elements  # Accuracy
    train_loss = total_loss / len(train_dataloader)  # Average loss
    train_loss_avg.append(train_loss)
    train_accuracy_values.append(train_acc)

    print("-" * 70)
    # =======================================
    #               Evaluation
    # =======================================
    model.eval()
    total_loss = 0
    total_correct = 0
    total_elements = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader, 0):
            input_ids, attention_mask, token_type_ids, labels = data
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(input_ids, attention_mask, token_type_ids)

            # Calculate the loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            if _ % 50 == 0:
                validation_loss_values.append(loss.item())

            # Select prediction labels based on the highest probability
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            # Calculate accuracy
            total_correct += torch.sum(preds == labels).item()
            total_elements += torch.numel(labels)

    # Continue with the rest of your code

    test_loss = total_loss / len(test_dataloader)
    test_acc = total_correct / total_elements  # For multi-label problems, divide by the number of classes
    validation_loss_avg.append(test_loss)
    validation_accuracy_values.append(test_acc)
    time_elapsed = time.time() - t0_epoch
    print(f"{epoch + 1:^7} | {'-':^7} | {train_loss:^12.6f} | {test_loss:^10.6f} | {test_acc:^9.2f} | {time_elapsed:^9.2f}")
    print("-" * 70)
    print("\n")

    print("Training complete!")
    return train_loss, train_acc, test_loss, test_acc


# Train & Validation Function
def _train_validation(epoch, model, train_dataloader, test_dataloader, optimizer, scheduler):
    model.train() # Set model to train mode
    print(f'Epoch:  {epoch}')
    
    # Initialize some variables
    total_loss = 0
    total_correct = 0
    total_elements = 0
    batch_count, batch_loss = 0, 0

    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*70)

    t0_epoch, t0_batch = time.time(), time.time()

    for _,data in enumerate(train_dataloader, 0):
        batch_count += 1
        lr = scheduler.get_last_lr()[0] # Get Current Learning Rate

        # Get 4 variables from train dataloader
        input_ids, attention_mask, token_type_ids, labels = data
        
        # Move these variables to device
        input_ids = input_ids.to(device, dtype = torch.long)
        attention_mask = attention_mask.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        labels = labels.to(device, dtype = torch.float)

        # Get Outputs(logits)
        outputs = model(input_ids, attention_mask, token_type_ids, labels)
        
        loss = outputs[1]
        # Get Current Loss value 
        batch_loss += loss.item()
        total_loss += loss.item()
    
        # Print States every 20 steps
        if (_ % 20 == 0 and _ != 0) or (_ == len(train_dataloader) - 1):            
            time_elapsed = time.time() - t0_batch
            print(f"{epoch + 1:^7} | {_:^7} | {batch_loss / batch_count:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
            batch_loss, batch_count = 0, 0
            t0_batch = time.time()

        # Append learning rate every 50 steps
        if _%50 == 0:
            learning_rates.append(lr)

        # Append loss value every 50 steps
        if _%50 == 0:
            train_loss_values.append(loss.item())
            
        loss.backward() # Perform backward
        optimizer.step()  # Perform optimization step
        scheduler.step()  # Update learning rate scheduler
        optimizer.zero_grad()  # Reset gradients

        # Apply sigmoid function to convert predictions to probabilities and set prediction labels based on threshold 0.5
        predictions = torch.sigmoid(outputs[0]) > 0.5

        total_correct += torch.sum(predictions == labels).item()  # Accumulate the number of correct predictions
        total_elements += torch.numel(labels)  # Count the total number of labels

    train_acc = total_correct / total_elements  # Accuracy
    train_loss = total_loss / len(train_dataloader)  # Average loss
    train_loss_avg.append(train_loss)
    train_accuracy_values.append(train_acc)

    print("-" * 70)
    # =======================================
    #               Evaluation
    # =======================================
    model.eval()
    total_loss = 0
    total_correct = 0
    total_elements = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader, 0):
            input_ids, attention_mask, token_type_ids, labels = data
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(input_ids, attention_mask, token_type_ids, labels)

            # Calculate the loss
            loss = outputs[1]

            total_loss += loss.item()
            if _ % 50 == 0:
                validation_loss_values.append(loss.item())

            # Select prediction labels based on the highest probability
            probs = torch.sigmoid(outputs[0])
            preds = (probs > 0.5).int()

            # Calculate accuracy
            total_correct += torch.sum(preds == labels).item()
            total_elements += torch.numel(labels)

    # Continue with the rest of your code

    test_loss = total_loss / len(test_dataloader)
    test_acc = total_correct / total_elements  # For multi-label problems, divide by the number of classes
    validation_loss_avg.append(test_loss)
    validation_accuracy_values.append(test_acc)
    time_elapsed = time.time() - t0_epoch
    print(f"{epoch + 1:^7} | {'-':^7} | {train_loss:^12.6f} | {test_loss:^10.6f} | {test_acc:^9.2f} | {time_elapsed:^9.2f}")
    print("-" * 70)
    print("\n")

    print("Training complete!")
    return train_loss, train_acc, test_loss, test_acc

       

def get_params():
    print(f'BATCH SIZE: {BATCH_SIZE} \n EPOCHS: {EPOCHS} \n LEARNING RATE: {LEARNING_RATE} \n DROPOUT : {DROPOUT} \n NUM LABELS: {NUM_LABELS} \n WARMING UP: {WARMING_UP} \n WEIGHT DECAY: {WEIGHT_DECAY} \n MODEL: {MODEL}')

def classification_report_for_each_label(true_labels, pred_labels, label_names):
    for i, label in enumerate(label_names):
        print(f"Label: {label}")
        print("0 - Accuracy: ", accuracy_score(true_labels[:, i], pred_labels[:, i]))
        print("0 - Precision: ", precision_score(true_labels[:, i], pred_labels[:, i], pos_label=0))
        print("0 - Recall: ", recall_score(true_labels[:, i], pred_labels[:, i], pos_label=0))
        print("0 - F1-Score: ", f1_score(true_labels[:, i], pred_labels[:, i], pos_label=0))
        print("1 - Accuracy: ", accuracy_score(true_labels[:, i], pred_labels[:, i]))
        print("1 - Precision: ", precision_score(true_labels[:, i], pred_labels[:, i], pos_label=1))
        print("1 - Recall: ", recall_score(true_labels[:, i], pred_labels[:, i], pos_label=1))
        print("1 - F1-Score: ", f1_score(true_labels[:, i], pred_labels[:, i], pos_label=1))
        print()

def get_report(model, test_dataloader, label_names):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for _, data in enumerate(test_dataloader, 0):
            input_ids, attention_mask, token_type_ids, labels = data
            input_ids = input_ids.to(device, dtype = torch.long)
            attention_mask = attention_mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            labels = labels.to(device, dtype = torch.float)

            outputs = model(input_ids, attention_mask, token_type_ids, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())

    # List를 numpy 배열로 변환
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    # 각 레이블에 대한 지표 계산
    for i, label in enumerate(label_names):
        print(f"Label: {label}")
        print("Accuracy: ", accuracy_score(true_labels[:, i], pred_labels[:, i]))
        print("Precision: ", precision_score(true_labels[:, i], pred_labels[:, i]))
        print("Recall: ", recall_score(true_labels[:, i], pred_labels[:, i]))
        print("F1-Score: ", f1_score(true_labels[:, i], pred_labels[:, i]))
        print()

    print(classification_report(true_labels, pred_labels, target_names=label_names))
    classification_report_for_each_label(true_labels, pred_labels, label_names)

def get_plots():
    try:
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_avg, label='Train Loss')
        plt.plot(validation_loss_values, label='Validation Loss')
        plt.title('Train vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_values, label='Train Loss')
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(validation_loss_values, label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(validation_loss_avg, label='Validation Loss')
        plt.title('Validation Average Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracy_values, label='Train Accuracy')
        plt.plot(validation_accuracy_values, label='Validation Accuracy')
        plt.title('Train vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Learning rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(learning_rates, label='Learning Rate')
        plt.title('Learning Rate over time')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.show()
    except:
        raise RuntimeError("Train the model first.")


def get_predict(sentence, Tokenizer, model, num_labels):
    device = torch.device('cpu')
    model.to(device)
    # Preprocess the input sentence
    tokenizer = Tokenizer
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, padding='longest', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    
    if num_labels == 9:
        labels = labels_wo_Neutral
        # Make the prediction
        with torch.no_grad():
            try:
                model.eval()
                logits = model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                # Convert probabilities to labels
                sorted_probs_idx = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
                top_3_labels = [labels[idx] for idx in sorted_probs_idx[:3]]
                top_3_probs = [str(round(probabilities[idx]*100))+'%' for idx in sorted_probs_idx[:3]]
                
                return top_3_labels, top_3_probs
            except:

                raise ValueError("Model and num_labels didn't match.")


    elif num_labels == 10:
        labels = labels_w_Neutral
        # Make the prediction
        with torch.no_grad():
            try: 
                model.eval()
                logits = model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                # Convert probabilities to labels
                sorted_probs_idx = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
                top_3_labels = [labels[idx] for idx in sorted_probs_idx[:3]]
                top_3_probs = [str(round(probabilities[idx]*100))+'%' for idx in sorted_probs_idx[:3]]
                
                return top_3_labels, top_3_probs
            except:
                raise ValueError("Model and num_labels didn't match.")
    elif num_labels == 28:
        labels = Full_28_labels
        # Make the prediction
        with torch.no_grad():
            try:
                model.eval()
                logits = model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                # Convert probabilities to labels
                sorted_probs_idx = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
                top_3_labels = [labels[idx] for idx in sorted_probs_idx[:3]]
                top_3_probs = [str(round(probabilities[idx]*100))+'%' for idx in sorted_probs_idx[:3]]
    
                return top_3_labels, top_3_probs
            except:
                raise ValueError(f"{model} and {num_labels} didn't match.")
    else:
        raise ValueError("Model and num_labels didn't match.")
    


def main():
    global device  # Device as a global variable

    try:
        device = torch.device("mps")  # MPS is for macOS
    except Exception as e:
        pass

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Selected device:", device)

    df = pd.read_csv(DATAFRAME_PATH)
    train_dataloader, test_dataloader = dataloader_pipeline(df=df)
    corr = pd.read_csv(CORR_PATH)
    corr_rels = torch.tensor(corr.values, dtype=torch.float).to(device)
    model = BERTClass(num_labels=NUM_LABELS, dropout=DROPOUT)

    #model = BERT_CNN_Class(num_labels=NUM_LABELS, dropout=DROPOUT, correlation=1, corr_rels=corr_rels)
    model.to(device)

    optimizer, scheduler = get_opti(model=model, train_dataloader=train_dataloader)

    for epoch in range(EPOCHS):
        train_validation(epoch=epoch, model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer, scheduler=scheduler,
                        )
    # Save the model after epoch

    torch.save(model.state_dict(), SAVE_MODEL_NAME)

    get_report(model=model, test_dataloader=test_dataloader, label_names=label_names)
    get_plots()

    sentence = text_preprocessing_pipeline(str(input("Enter the sentence: ")))
    device = torch.device('cpu')
    top_3_labels, top_3_probs = get_predict(sentence=sentence, Tokenizer=tokenizer, model=model, num_labels=9)

    print("Predicted labels and probabilities:")
    for label, prob in zip(top_3_labels, top_3_probs):
        print(f"Label: {label}, Probability: {prob}")


if __name__ =='__main__':
    main()
