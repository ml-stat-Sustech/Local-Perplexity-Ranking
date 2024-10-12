import pandas as pd
from sklearn.model_selection import train_test_split



class GenerationTasksDatasets:
    def __init__(self, dataset_type, task, noise_type, noise_ratio, seed):
        self.dataset_type = dataset_type
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.task = task
        self.seed = seed

        raw_train_dataset = pd.read_csv("sciq_train.csv", header=0)
        raw_test_dataset = pd.read_csv("sciq_test.csv", header=0)

        
        if self.dataset_type == 'train':
            clean_ds, noise_ds = train_test_split(raw_train_dataset, test_size=noise_ratio, shuffle=True, random_state=self.seed)
            if noise_type == 'relevant':
                self.text_total = sum([clean_ds['text'].tolist(), noise_ds['text'].tolist()], [])
                self.label_total = sum([clean_ds['correct'].tolist(), noise_ds['relevant'].tolist()], [])

            elif noise_type == 'irrelevant':
                self.text_total = sum([clean_ds['text'].tolist(), noise_ds['text'].tolist()], [])
                self.label_total = sum([clean_ds['correct'].tolist(), noise_ds['irrelevant'].tolist()], [])


            elif noise_type == 'real':
                self.text_total = sum([clean_ds['text'].tolist(), noise_ds['text'].tolist()], [])
                self.label_total = sum([clean_ds['correct'].tolist(), noise_ds['correct'].tolist()], [])

            
            else:
                print("ERROR NOISE TYPE")
        
        else:
            self.text_total = raw_test_dataset['text']
            self.label_total = raw_test_dataset['label']

        


    
    def __getitem__(self, index):
        text, target, = self.text_total[index], self.label_total[index]

        return text, target, index

    def __len__(self):
        return len(self.text_total)