import pandas as pd
import glob
import csv

class Brat:
    
    def __init__(self,config):

        self.config = config
    
    def read_ann_files(self):

        path = config["files_path"]
        all_files = glob.glob(path + "/*.ann")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename,header=0,sep='\t',names=['ann','micro-event','text_span'],encoding='utf-8')
            li.append(df)

        ann_files = pd.concat(li,axis=0)
        return ann_files

    def read_txt_files(self):
        path = config["files_path"]
        all_files = glob.glob(path + "/*.txt")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename,header=0,sep='\t',names=['sentences'], encoding='utf-8',quoting=3)
            li.append(df)

        text_files = pd.concat(li , axis=0)

        return text_files

    def preprocess_sentifm(self,config):
        ann_files = self.read_ann_files()
        text_files = self.read_txt_files()
        ann_file_macro_events = ann_files.dropna()
        ann_file_macro_events['micro-event'] = ann_file_macro_events['micro-event'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()
        ann_file_macro_events = ann_file_macro_events[ann_file_macro_events['micro-event']!='Company']
        text_spans = ann_file_macro_events["text_span"]

        text_spans = text_spans.str.strip()
        sentences = text_files['sentences'].str.strip()
        output = []
        for text_span in text_spans:
            for sent in sentences:
                if text_span in sent:
                    out = sent
                    continue
            output.append(out)
        ann_file_macro_events['sentences'] = output
        ann_file_macro_events = ann_file_macro_events.reset_index()
        ann_file_macro_events = ann_file_macro_events[['ann','sentences','micro-event','text_span']]
        ann_file_macro_events.to_csv(config["output"]+"ann_file_macro_events.tsv",sep='\t')
        return ann_file_macro_events

if __name__ == '__main__':

    config = {
        "files_path": '/ann/txt/files/input/path/',
        "output":'/processed/dump/output/path/'
    }
    brat = Brat(config)
    output = brat.preprocess_sentifm(config)
    
