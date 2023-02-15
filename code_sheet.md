###### Code to check GPU

``` py
# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
```

``` py
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```
``` py
#check the ongoing process on GPU
!nvidia-smi
```


###### Create Custom Dataset from text using PyTorch

``` py
#import the required libraries

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

#load the dataset
df = pd.read_csv('dataset.csv')
df.head()

#convert negative to 0 and positive to 1
df['label'] = df['sentiment'].map({'negative':0, 'positive':1})

#re-create the dataset keep the target columns only
df = df[['text', 'label']]

#create two new dataset based on features and labels
X = df['text']
y = df['label']

# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(X, y, random_state=2018, test_size=0.3, stratify=y)
# split validation dataset into validation and test sets
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)

#import the tokenizer
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

#custom dataset class
class CustomDataset(Dataset):
    def __init__(self, review, targets, tokenizer: BertTokenizer, max_len: int 128):
        self.review = review
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        target = self.targets[item]
        
        encoder = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        return {
            'review_text': review,
            'input_ids': encoder['input_ids'].flatten(),
            'attention_mask': encoder['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
        
#initialize the customdataset
dataset = CustomDataset(train_text, train_labels, tokenizer, max_len)
print(len(dataset))
dataset[0]
for item in dataset:
    print(item['input_ids'][0:10])
    print(item['attention_mask'][0:10])
    print(item['targets'])
    break
    
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)   
  
#get value from dataset
data = dataset[0]
data.keys()
print(data['review_text'])
print(data['input_ids'])
print(data['attention_mask'])
print(data['targets'])
```


``` py
#Model declearation
class LogisticRegression(nn.Module):
  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)
    
  def forward(self, x):
    y_predicted = torch.Relu(self.linear(x))
    return y_predicted

model = LogisticRegression(n_features)
model

#Training Loop
def train_val(model, criterion, optimizer, train_loader, val_loader, num_epochs):
  for epoch in range(num_epochs):
    total_loss = 0.0
    valid_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      predict = model(inputs)
      loss = criterion(predict, labels)
      loss.backward()
      optimizer.step()
      total_loss +=loss.item()
    avg_training_loss = total_loss / len(train_loader)
    
    model.eval()
    with torch.no_grad():
      for inputs, labels in val_loader:
        predict = model(inputs)
        loss = criterion(predict, labels)
        total_loss +=loss.item()
      avg_val_loss = total_loss / len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_training_loss:.4f}, validation Loss: {avg_val_loss:.4f}')


from torch.optim.optimizer import Optimizer
num_epochs = 1000

#Train the model for a specified number of epochs
train_val(model, criterion, optimizer, train_loader, val_loader, num_epochs)


```
