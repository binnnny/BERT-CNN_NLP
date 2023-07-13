from transformers import pipeline, BertModel, BertTokenizer
from openai import ChatCompletion
import os
import openai
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import re
import emoji
from bs4 import BeautifulSoup
import string
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# OpenAI GPT-3 Key
openai.api_key = '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'


class BERT_CNN_Class(torch.nn.Module):
    def __init__(self, num_labels, dropout):
        super(BERT_CNN_Class, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1d = torch.nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.l2 = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs.last_hidden_state  
        sequence_output = sequence_output.permute(0, 2, 1)  # (batch_size, hidden_size, seq_length)
        conv_output = self.conv1d(sequence_output)  # (batch_size, out_channels, seq_length)
        conv_output = F.relu(conv_output)  # ReLU 활성화 함수 적용
        pooled_output = F.max_pool1d(conv_output, conv_output.shape[2]).squeeze(2)  # max pooling 적용
        logits = self.fc(self.l2(pooled_output))
        return logits


labels = ["Joy & Vitality", "Love & Gratitude", "Warm Relationships",
          "Sadness & Disappointment", "Anxiety & Fear", "Anger & Dissatisfaction",
          "Confusion & Curiosity", "Embarrassment & Relief", "Passion and Desire"]



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



def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
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
    '''Cleans special characters present(if any)'''   
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

def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_contractions(text, contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = remove_space(text)
    return text


def _predict(sentence, tokenizer, model):
    device = torch.device('cpu')
    model.to(device)
    # Preprocess the input sentence
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, padding='longest', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    
    # Make the prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # Convert probabilities to labels
    sorted_probs_idx = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
    labelss = labels[sorted_probs_idx[0]]
    return labelss




class SentimentAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_sentiment(self, text):
        device = torch.device('cpu')
        self.model.to(device)
        # Preprocess the input sentence
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='longest', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        # Make the prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids)
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

        # Convert probabilities to labels
        sorted_probs_idx = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
        labelss = labels[sorted_probs_idx[0]]  # Assuming labels is defined within the class

        return labelss

    def create_chat(self, sentiment, text):
        chat_models = "gpt-3.5-turbo"
        messages = []
        # Chat setup depending on sentiment
        if sentiment == "Joy & Vitality":
            messages = [
                {"role": "system", "content": "You are a joyful assistant, always encouraging positivity."},
                {"role": "user", "content": "I'm feeling full of life today!"},
            ]
        elif sentiment == "Love & Gratitude":
            messages = [
                {"role": "system", "content": "You are a loving assistant, always appreciating the good in people."},
                {"role": "user", "content": "I'm feeling thankful and love today!"},
            ]
        elif sentiment == "Warm Relationships":
            messages = [
                {"role": "system", "content": "You are a friendly assistant, always fostering connections."},
                {"role": "user", "content": "I'm feeling connected to my friends today!"},
            ]
        elif sentiment == "Sadness & Disappointment":
            messages = [
                {"role": "system", "content": "You are a sympathetic assistant, providing comfort in times of distress."},
                {"role": "user", "content": "I'm feeling sad and grief today."},
            ]
        elif sentiment == "Anxiety & Fear":
            messages = [
                {"role": "system", "content": "You are a supportive assistant, providing reassurance during times of anxiety."},
                {"role": "user", "content": "I'm feeling a fear and anxious today."},
            ]
        elif sentiment == "Anger & Dissatisfaction":
            messages = [
                {"role": "system", "content": "You are a calming assistant, helping to diffuse negative feelings."},
                {"role": "user", "content": "I'm feeling anger and disapproval and hate today."},
            ]
        elif sentiment == "Confusion & Curiosity":
            messages = [
                {"role": "system", "content": "You are a helpful assistant, providing clarity and encouraging curiosity."},
                {"role": "user", "content": "I'm feeling confused today."},
            ]
        elif sentiment == "Embarrassment & Relief":
            messages = [
                {"role": "system", "content": "You are an understanding assistant, providing comfort in awkward situations."},
                {"role": "user", "content": "I'm feeling embarrassed today."},
            ]
        elif sentiment == "Passion and Desire":
            messages = [
                {"role": "system", "content": "You are an enthusiastic assistant, always supporting people's passions."},
                {"role": "user", "content": "I'm feeling very passionate today!"},
            ]

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
        messages.append({'role':'user', 'content': text})
        # Initiate chat and get the response
        response = ChatCompletion.create(
          model=chat_models,
          messages=messages
        )
        assistant_message = response['choices'][0]['message']['content']
        print("\n------------------------")
        print("Assistant: ", assistant_message)
        print("------------------------")
        messages.append({'role':'assistant', 'content': assistant_message})
        # Continue the chat based on user's input
        while True:
            print("\n------------------------")
            user_message = input("You: ")
            print("------------------------")
            messages.append({'role':'user', 'content': user_message})
            response = ChatCompletion.create(
                model=chat_models,
                messages=messages
            )
            assistant_message = response['choices'][0]['message']['content']
            print("\n------------------------")
            print("Assistant: ", assistant_message)
            print("------------------------")
            messages.append({'role':'assistant', 'content': assistant_message})

# Create an instance of SentimentAnalyzer


import tkinter as tk
from tkinter import ttk
from openai import ChatCompletion  # 필요한 라이브러리 import

class ChatWindow(tk.Tk):
    def __init__(self, sentiment_analyzer):
        super().__init__()

        self.sentiment_analyzer = sentiment_analyzer

        self.title("Chat with GPT")
        
        self.chat_transcript_area = tk.Text(self, wrap='word', height=15)
        self.chat_transcript_area.grid(column=0, row=0, columnspan=2, sticky='nsew')

        self.message_area = ttk.Entry(self, width=100)
        self.message_area.grid(column=0, row=1, sticky='nsew')

        self.send_button = ttk.Button(self, text='Send', command=self.send_message)
        self.send_button.grid(column=1, row=1, sticky='nsew')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def send_message(self):
        message = self.message_area.get()
        self.chat_transcript_area.insert('end', 'You: {}\n'.format(message))
        self.chat_transcript_area.see('end')
        
        sentiment = self.sentiment_analyzer.predict(message)  # sentiment analysis
        self.sentiment_analyzer.create_chat(sentiment, message)  # Chat creation
        response = self.sentiment_analyzer.assistant_message  # get assistant message
        self.chat_transcript_area.insert('end', 'GPT: {}\n'.format(response))
        self.chat_transcript_area.see('end')
        
        self.message_area.delete(0, 'end')




def main():
    model = BERT_CNN_Class(num_labels=9, dropout=0.1)

    model.load_state_dict(torch.load('Model/BERT_CNN_Model_9.pth'))

    model.eval()

    device = torch.device('cpu')
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    model = model.to(device)
    sentence = text_preprocessing_pipeline(str(input("Enter any sentence: ")))


    sentiment = _predict(sentence=sentence, tokenizer=tokenizer, model=model)


    print("Predicted labels and probabilities:")
    
    print(f"Label: {sentiment}")
    text = input("Enter your journal entry: ")

    # Create an instance of SentimentAnalyzer
    analyzer = SentimentAnalyzer(model, tokenizer)
    emotion = analyzer.get_sentiment(text=text_preprocessing_pipeline(text))


    # Generate feedback based on the sentiment
    analyzer.create_chat(emotion, text=text)


if __name__ == '__main__':
    main()
