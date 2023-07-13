import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import re
import emoji
from bs4 import BeautifulSoup
import string
import tkinter as tk
import tkinter as tk
import tkinter as tk
from pathlib import Path




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


labels = ["Joy & Vitality", "Love & Gratitude", "Warm Relationships",
          "Sadness & Disappointment", "Anxiety & Fear", "Anger & Dissatisfaction",
          "Confusion & Curiosity", "Embarrassment & Relief", "Passion and Desire"]


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
    top_3_labels = [labels[idx] for idx in sorted_probs_idx[:3]]
    top_3_probs = [probabilities[idx] for idx in sorted_probs_idx[:3]]
    
    return top_3_labels, top_3_probs

ASSETS_PATH = Path("Code/assets/")



def main():
    window = tk.Tk()

    # Set window properties
    window.title("Tkinter Designer")
    window.geometry("862x519")
    window.configure(bg="#3A7FF6")

    canvas = tk.Canvas(
        window, bg="#3A7FF6", height=519, width=862,
        bd=0, highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)
    canvas.create_rectangle(431, 0, 431 + 431, 0 + 519, fill="#FCFCFC", outline="")

    # Description of the model
    model_info = """This application performs emotion analysis using BERT and CNN. 
                    It analyses given text and identifies the primary emotions 
                    and predicts the probabilities."""

    # Position of the label: x=20, y=50
    # Font size: 16 (the number after the font name in the tuple)
    # Font style: Normal (you can make it bold by changing "Helvetica" to "Helvetica-Bold")
    # Text color: white
    # Background color: same as the window's background color ("#3A7FF6")
    info_label = tk.Label(window, text=model_info, bg="#3A7FF6", fg="white", wraplength=400, font=("Helvetica", 16))
    info_label.place(x=20, y=50)

    # Load text box background
    text_box_bg = tk.PhotoImage(file=ASSETS_PATH / "TextBox_Bg.png")

 # Load text box background
    title_entry_img = canvas.create_image(650.5, 107.5, image=text_box_bg)

    title_entry = tk.Entry(bd=0, bg="#F6F7F9", fg="#000716",  highlightthickness=0)
    title_entry.place(x=490.0, y=77+25, width=321.0, height=35)

    # Increase the size of the content entry
    content_entry = tk.Text(bd=0, bg="#F6F7F9", fg="#000716",  highlightthickness=0)
    content_entry.place(x=490.0, y=158+25, width=321.0, height=105)

    model = BERT_CNN_Class(num_labels=9, dropout=0.1)
    model.load_state_dict(torch.load('Model/BERT_CNN_Model_9.pth'))
    model.eval()
    device = torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = model.to(device)

    def on_click():
        sentence = text_preprocessing_pipeline(content_entry.get("1.0", 'end-1c'))

        top_3_labels, top_3_probs = _predict(sentence=sentence, tokenizer=tokenizer, model=model)

        result = "\n".join(f"{label} : {prob:.4f}" for label, prob in zip(top_3_labels, top_3_probs))
        result_label['text'] = result

    canvas.create_text(
        490.0, 96.0, text="Diary Title", fill="#515486",
        font=("Arial-BoldMT", int(13.0)), anchor="w")
    canvas.create_text(
        490.0, 174.5, text="Diary Content", fill="#515486",
        font=("Arial-BoldMT", int(13.0)), anchor="w")

    # Button
    btn_img = tk.PhotoImage(file=ASSETS_PATH / "generate.png")
    button = tk.Button(
        image=btn_img, borderwidth=0, highlightthickness=0,
        command=on_click, relief="flat")
    button.place(x=557, y=350, width=180, height=55)

    # Result label with a decorative background
    result_bg_img = tk.PhotoImage(file=ASSETS_PATH / "result_bg.png")
    result_label_bg = canvas.create_image(650.5, 480.0, image=result_bg_img)
    result_label = tk.Label(
        text="", bg="#F6F7F9", fg="#000716", justify="center",
        font=("Arial", int(13.0)), relief="flat", wraplength=300)
    result_label.place(x=500, y=430, width=300, height=80)

    window.resizable(False, False)
    window.mainloop()

if __name__ == "__main__":
    main()


