df2=df#to add noise in the dataset
for i in range(10):
    df2=df2.append(pd.Series([random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10),random.choice([2,4])], index=df2.columns ), ignore_index=True)
df2.to_csv('noise_added_another.csv')