import optuna
from transformers import BertForSequenceClassification
import torch
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 2, 8)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(device, "MPS Device!")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_sampler = RandomSampler(train_data)
    validation_sampler = SequentialSampler(validation_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    for epoch in range(epochs):
        torch.mps.empty_cache()
        print("model trainning!. epoch :", epoch)
        model.train()
        for step, batch in enumerate(train_dataloader):
            print('step :', step)
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=(b_input_ids > 0), labels=b_labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

        model.eval()
        eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=(b_input_ids > 0), labels=b_labels)
                eval_loss += outputs[0].item()

        eval_loss /= len(validation_dataloader)

    return eval_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
