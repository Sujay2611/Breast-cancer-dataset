#KNN implementation
from random import randrange
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def test_split(column_number, value, dataset):
    left_array=[]
    right_array = []
    for row in dataset:
        if row[column_number] < value:
            left_array.append(row)
        else:
            right_array.append(row)
    return left_array, right_array
def gini_index_value(groups, target_values):
    n = float(sum([len(group) for group in groups]))
    gini_val = 0
    for each_group in groups:
        if len(each_group)!=0:
            score=0
            for cls in target_values:
                p=len([g[-1] for g in each_group if g[-1] == cls])/(len(each_group))
                score+=p*p
            gini_val +=(1-score)*len(each_group)/n
    return gini_val
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for index in range(1,9):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index_value(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index':best_index, 'value':best_value, 'groups':best_groups}
def leaf(group):
    target = [row[-1] for row in group]
    return max(set(target), key=target.count)
def split_tree(root_node, maximum_depth, min_size, depth):
    l, r = root_node['groups']
    del(root_node['groups'])
    if not l or not r:
        root_node['l'] = root_node['r'] = leaf(l + r)
        return
    if maximum_depth<=depth :
        root_node['l'], root_node['r'] = leaf(l), leaf(r)
        return
    if min_size>=len(l) :
        root_node['l'] = leaf(l)
    else:
        root_node['l'] = get_split(l)
        split_tree(root_node['l'], maximum_depth, min_size, depth+1)
    if min_size>=len(r):
        root_node['r'] = leaf(r)
    else:
        root_node['r'] = get_split(r)
        split_tree(root_node['r'], maximum_depth, min_size, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root_node = get_split(train)
    split_tree(root_node, max_depth, min_size, 1)
    return root_node

# Make a prediction with a decision tree
def predict_recursive(root_node, each_row):
    if each_row[root_node['index']] < node['value']:
        if isinstance(root_node['l'], dict):
            return predict_recursive(root_node['l'], each_row)
        else:
            return root_node['l']
    else:
        if isinstance(root_node['r'], dict):
            return predict_recursive(root_node['r'], each_row)
        else:
            return root_node['r']

# Classification and Regression Tree Algorithm
def decision_tree_classifier(train, test, max_depth, min_size):
    tree_built = build_tree(train, max_depth, min_size)
    print_the_tree(tree_built)
    total_predicted_values = []
    for row in test:
        total_predicted = predict(tree_built, row)
        total_predicted_values.append(total_predicted)
    return(total_predicted_values)
def print_the_tree(node_root, depth=0):
    if isinstance(node_root, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node_root['index']+1), node_root['value'])))
        printtree(node_root['l'], depth+1)
        printtree(node_root['r'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node_root)))
def ReadingDataset(fd):
    df=pd.read_csv(fd)
    data=df.values.tolist()
    return data

def Train_Test(data,s_r,train,test):
    for x in range(len(data)):
        for y in range(0,9):
            data[x][y] = int(data[x][y])
        if random.random()>s_r:
            test.append(data[x])
        else:
            train.append(data[x])
            
def accuracy_in_percentage(actual_data,predicted_values):
    count=0
    for val in range(len(actual_data)):
        if actual_data[val]==predicted_values[val]:
            count+=1
    value=count/float(len(actual_data))*100
    return value

#CROSS VALIDATING AN ALGORITHM
def evaluate_an_algorithm(dataset, algorithm, *args):
    six_folds = cross_validation_part(dataset)
    score=calculate_scores(six_folds,algorithm,*args)
    return score
def calculate_scores(folds,algorithm,*args):
    total_scores=[]
    for fold_data in folds:
        train = list(folds)
        train.remove(fold_data)
        train = sum(train, [])
        test = list()
        for row in fold_data:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train, test, *args)
        actual = [row[-1] for row in fold_data]
        accuracy = accuracy_in_percentage(actual, predicted)
        total_scores.append(accuracy)
    return sum(total_scores)/6
def manhattan_distance(ins1,ins2,features):
    distance = 0
    for x in range(2,features-1):
        #print(ins1[x])
        distance += abs(ins1[x] - ins2[x])

    return distance
def cosine_similarity(a,b,columns):
    distance1=0
    distance2=0
    distance3=0
    for x in range(2,columns-1):
        distance1+=(ins1[x]*ins2[x])
        distance2+=(ins1[x]*ins1[x])
        distance3+=(ins2[x]*ins2[x])
    b=math.sqrt(distance2)
    c=math.sqrt(distance3)
    return 1-(distance1/(b*c))

#KNN PART
def getNeighbour_members(training_data,test_ins,k):
    distances = []
    features = len(test_ins) - 1
    for x in range(len(training_data)):
        distance = euclidean_distance(test_ins,training_data[x], features)
        distances.append((training_data[x],distance))
    distances.sort(key = lambda x:(x[1]))
    neighbours = []
    for x in range(k):                      
        neighbours.append(distances[x][0])

    return neighbours



def euclidean_distance(test_row,train_row,columns):
    distance= 0
    for x in range(columns-1):
        distance += pow(test_row[x] - train_row[x],2)
    return math.sqrt(distance)    
def cross_validation_part(dataset):
    data_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / 6)
    for _ in range(6):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        data_split.append(fold)
    return data_split


def make_a_prediction(training_data,test_data,predictions,k):
    for x in range(len(test_data)):                       
        neighbours = getNeighbour_members(training_data, test_data[x],k)
        prediction = getClassvalue(neighbours)
        predictions.append(prediction) 
        
def getClassvalue(neighbors_data):
    count_benign=0
    count_malignant=0
    for x in range(len(neighbors_data)):       # iterate through each neighbour
        last_value = neighbors_data[x][-1]
        if(last_value==2):
            count_benign+=1
        else:
            count_malignant+=1
    if(count_benign>count_malignant):
        return 2
    else:
        return 4
    
def KNN_algo(training_data,test_data,x):
    predictions = []    
    make_a_prediction(training_data,test_data,predictions,x)
    return predictions


training_data = []
test_data = []
filename = "clean_ml_2.csv"#GIVE THE DATASET ACCORDING AFTER CLEANING
split_ratio = 0.76
df=pd.read_csv(filename)
data=df.values.tolist()
#data=ReadingDataset(filename)
scores=[]
x=int(input("WHICH ALGORITHM DECTREE1 or KNN(2)"))
if(x==1):
    Train_Test(data,split_ratio,training_data,test_data)
    scores = evaluate_an_algorithm(training_data, decision_tree_classifier,6,10)
    print(scores)
else:
    training_data = []
    test_data = []
    Train_Test(data,split_ratio,training_data,test_data)
    filename = "noise_added.csv"
    split_ratio = 0.75
    accuracy_scores=[]
    ival=[]
    for i in range(20,30):
        scores = evaluate_algorithm(training_data, KNN_algo, i)
        print(scores)
        accuracy_scores.append(scores)
        ival.append(i)
    #print('Scores: %s' % scores)
    plt.plot(ival,accuracy_scores)
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    #plt.legend()
    plt.show()
    print(ival)
    print("KNN")
    print('Scores: %s' % scores)
    print(accuracy_scores)



