import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Dataset is read with all columns and printed information about dataset.
df = pd.read_csv("consumer_complaints.csv")
print("\nDATAFRAME INFORMATION:\n")
print(df.info())

#I need just two cloumns(product and consumer_complaint_narrative)
fields= ['product','consumer_complaint_narrative'] 
df=pd.read_csv('consumer_complaints.csv', usecols=fields)
df = df[pd.notnull(df['consumer_complaint_narrative'])]

#I joined two related columns and two columns have few row.
for i in range(df.shape[0]):
    if (df.iloc[i]=="Prepaid card").any():
        df.iloc[i]="Credit card"
    if (df.iloc[i]=="Virtual currency").any():
        df.iloc[i]="Other financial service"  

#After joining columns step
print("\nPRODUCT VALUE COUNT:\n")
print(df["product"].value_counts())
print("\nDATAFRAME HEAD:\n")
print(df.head())

df = shuffle(df)
train , test = train_test_split(df,test_size=0.1, random_state = 42)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)