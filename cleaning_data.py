import pandas as pd

df = pd.read_csv('csv_files/Cleaned-Data.csv')

df.columns = ['Fever','Tiredness','Dry-Cough','Difficulty-in-Breathing','Sore-Throat','None_Sympton','Pains','Nasal-Congestion','Runny-Nose','Diarrhea','None_Experiencing','Age_0-9','Age_10-19','Age_20','Age_25','Age_60','Gender_Female','Gender_Male','Gender_Transgender','Severity_Mild','Severity_Moderate','Severity_None','Severity_Severe','Contact_Dont-Know','Contact_No','Contact_Yes','Country']
df = df[df.Country != "Italy"]
df = df[df.Country != "Other"]
df = df[df.Country != "Other-EUR"]
df = df[df.Country != "UAE"]
df = df[df.Country != "Germany"]
df = df[df.Country != "Spain"]
# df = df[df.Country != "France"]
df = df[df.Country != "Republic of Korean"]
df = df[df.Country != "Iran"]
df = df[df.Country != "China"]
df = df[df.Gender_Transgender != 1]
df = df[df.Age_20 != 1]
df = df[df.Age_25 != 1]
df = df[df.Age_60 != 1]
df.loc[(df.Severity_Moderate == 1), 'Severity_Mild'] = 2
df.loc[(df.Severity_Severe == 1), 'Severity_Mild'] = 3


df.drop(columns=['Age_0-9','Age_20','Age_25','Age_60','Gender_Transgender','None_Sympton','None_Experiencing','Contact_Dont-Know','Contact_No','Contact_Yes','Severity_Moderate','Severity_Severe','Severity_None','Gender_Male'], axis=1, inplace=True)

df.columns = ['Fever','Tiredness','Dry-Cough','Difficulty-in-Breathing','Sore-Throat','Pains','Nasal-Congestion','Runny-Nose','Diarrhea','Age','Gender','Severity','Country']
print(df.info())
print(df)

df.to_csv(r'csv_files/France.csv', index=False, header=True)
