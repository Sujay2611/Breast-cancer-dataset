colnames= ['id','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','target_class']
df = pd.read_csv("breast-cancer.csv",names = colnames, header= None)
df
df2=df.drop(['id'], axis=1)
df4=df2
df5=df4






import math
s=0
k=0
for index,row in df.bare_nuclei.iteritems():
    if(row!='?'):
        s=s+int(row)
        k+=1
p=round(s/k)

df1=df['bare_nuclei'].replace('?',p)
df
df.to_csv('clean_ml.csv')
df4=pd.read_csv('clean_ml.csv')