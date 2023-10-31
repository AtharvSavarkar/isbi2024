def custom_dataset(classes=[0,1],path="./data/", fraction=1.0):
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load(path+"APTOS_5.npz")
    custom_data = {'train_images':[], 'train_labels':[], 'test_images':[], 'test_labels':[]}

    # Creating Train Set
    idx = np.where(np.isin(data["train_labels"], classes))[0]
    custom_data['train_labels'] = [classes.index(i) for i in data["train_labels"][idx]]
    custom_data['train_images'] = data["train_images"][idx]

    # # Creating Val Set
    # idx = np.where(np.isin(data["val_labels"], classes))[0]
    # custom_data['val_labels'] = [classes.index(i) for i in data["val_labels"][idx]]
    # custom_data['val_images'] = data["val_images"][idx]
    
    # Creating Test Set
    idx = np.where(np.isin(data["test_labels"], classes))[0]
    custom_data['test_labels'] = [classes.index(i) for i in data["test_labels"][idx]]
    custom_data['test_images'] = data["test_images"][idx]

    x = custom_data['train_images']
    y = custom_data['train_labels']

    # fraction=0 means rem_images=train_images
    # fraction=1 means classes having below avg frequencies are absent from rem_images, present only in x_train 
    # train,test,val_images are trimmed based on 'classes', nothing else
    img_per_class = int(fraction*len(y)/len(classes))

    x_train = []
    y_train = []
    indices = []
    
    for j in range(len(classes)):
        for i in range(len(y)):
            if y_train.count(j) < img_per_class and y[i] == j: #if label belongs to class & count less than threshold
                x_train.append(x[i])
                y_train.append(y[i])
                indices.append(i)

    custom_data['rem_images'] = np.delete(x, indices, 0)
    custom_data['rem_labels'] = np.delete(y, indices)
    
    # custom_data['rem_images'] = []
    # custom_data['rem_labels'] = []

    # remx = np.delete(x, indices, 0) #deleting all the indices from x & y at the end of loop
    # remy = np.delete(y, indices)

    # for j in range(len(classes)):
    #     for i in range(len(remy)):
    #         if  custom_data['rem_labels'].count(j) < 800 and remy[i] == j:
    #             custom_data['rem_images'].append(remx[i])
    #             custom_data['rem_labels'].append(remy[i])


    # Custom_data has images from only selected classes; x_train & y_train have 
    return custom_data, np.array(x_train),np.array(y_train)

def custom_random_dataset(classes=[0,1],path="./data/", fraction=1.0):
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load(path+"APTOS_5.npz")
    custom_data = {'train_images':[], 'train_labels':[], 'test_images':[], 'test_labels':[]}

    # Creating Train Set
    idx = np.where(np.isin(data["train_labels"], classes))[0]
    custom_data['train_labels'] = [classes.index(i) for i in data["train_labels"][idx]]
    custom_data['train_images'] = data["train_images"][idx]

    # # Creating Val Set
    # idx = np.where(np.isin(data["val_labels"], classes))[0]
    # custom_data['val_labels'] = [classes.index(i) for i in data["val_labels"][idx]]
    # custom_data['val_images'] = data["val_images"][idx]
    
    # Creating Test Set
    idx = np.where(np.isin(data["test_labels"], classes))[0]
    custom_data['test_labels'] = [classes.index(i) for i in data["test_labels"][idx]]
    custom_data['test_images'] = data["test_images"][idx]

    x = custom_data['train_images']
    y = custom_data['train_labels']

    # fraction=0 means rem_images=train_images
    # fraction=1 means classes having below avg frequencies are absent from rem_images, present only in x_train 
    # train,test,val_images are trimmed based on 'classes', nothing else
    img_per_class = int(fraction*len(y)/len(classes))

    x_train, y_train, indices = random_seed(x, y, classes, img_per_class)

    custom_data['rem_images'] = np.delete(x, indices, 0)
    custom_data['rem_labels'] = np.delete(y, indices)
    
    # custom_data['rem_images'] = []
    # custom_data['rem_labels'] = []

    # remx = np.delete(x, indices, 0) #deleting all the indices from x & y at the end of loop
    # remy = np.delete(y, indices)

    # for j in range(len(classes)):
    #     for i in range(len(remy)):
    #         if  custom_data['rem_labels'].count(j) < 800 and remy[i] == j:
    #             custom_data['rem_images'].append(remx[i])
    #             custom_data['rem_labels'].append(remy[i])


    # Custom_data has images from only selected classes; x_train & y_train have 
    return custom_data, np.array(x_train),np.array(y_train)

def custom_random_seeds(classes=[0,1],path="./data/", fraction=0.05, n_seeds=7):
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load(path)
    custom_data = {'train_images':[], 'train_labels':[], 'val_images':[],'val_labels':[], 'test_images':[], 'test_labels':[]}

    # Creating Train Set
    idx = np.where(np.isin(data["train_labels"], classes))[0]
    custom_data['train_labels'] = [classes.index(i) for i in data["train_labels"][idx]]
    custom_data['train_images'] = data["train_images"][idx]

    # # Creating Val Set
    # idx = np.where(np.isin(data["val_labels"], classes))[0]
    # custom_data['val_labels'] = [classes.index(i) for i in data["val_labels"][idx]]
    # custom_data['val_images'] = data["val_images"][idx]
    
    # Creating Test Set
    idx = np.where(np.isin(data["test_labels"], classes))[0]
    custom_data['test_labels'] = [classes.index(i) for i in data["test_labels"][idx]]
    custom_data['test_images'] = data["test_images"][idx]
    
    custom_data['rem_images'] = custom_data['train_images'].copy()
    custom_data['rem_labels'] = custom_data['train_labels'].copy()

    # fraction=0 means rem_images=train_images
    # fraction=1 means classes having below avg frequencies are absent from rem_images, present only in x_train 
    # train,test,val_images are trimmed based on 'classes', nothing else
    img_per_class = int(fraction*len(custom_data['rem_labels'])/len(classes)/n_seeds)

    seeds_x = []
    seeds_y = []

    for i in range(n_seeds):
        x_train, y_train, indices = random_seed(custom_data['rem_images'],custom_data['rem_labels'],classes,    img_per_class)
        seeds_x.append(x_train)
        seeds_y.append(y_train)
        custom_data['rem_images'] = np.delete(custom_data['rem_images'], indices, 0)
        custom_data['rem_labels'] = np.delete(custom_data['rem_labels'], indices)

    # remx = np.delete(x, indices, 0) #deleting all the indices from x & y at the end of loop
    # remy = np.delete(y, indices)

    # for j in range(len(classes)):
    #     for i in range(len(remy)):
    #         if  custom_data['rem_labels'].count(j) < 800 and remy[i] == j:
    #             custom_data['rem_images'].append(remx[i])
    #             custom_data['rem_labels'].append(remy[i])


    # Custom_data has images from only selected classes; x_train & y_train have 
    return custom_data, np.array(seeds_x),np.array(seeds_y)


def random_seed(x,y,classes, img_per_class):
    import numpy as np
    x_train = []
    y_train = []
    indices = []
    
    for j in range(len(classes)):
        xtj = []
        ytj = []
        ij = []
        for i in range(len(y)):
            if y[i] == j: #if label belongs to class & count less than threshold
                xtj.append(x[i])
                ytj.append(j)
                ij.append(i)
        x_train.append(xtj)
        y_train.append(ytj)
        indices.append(ij)
    
    x_seed = np.array([])
    y_seed = np.array([])
    index_seed = np.array([])

    for j in range(len(classes)):
        x_train_j = np.array(x_train[j])
        y_train_j = np.array(y_train[j])
        index_j = np.array(indices[j])
        
        # index = np.random.choice(x_train_j.shape[0], img_per_class, replace=False)
        index = np.random.choice(x_train_j.shape[0], img_per_class, replace=True)
        if j==0:
            x_seed = x_train_j[index]
            y_seed = y_train_j[index]
            index_seed = index_j[index]
        else:
            x_seed = np.append(x_seed,x_train_j[index], axis=0)
            y_seed = np.append(y_seed,y_train_j[index], axis=0)
            index_seed = np.append(index_seed,index_j[index], axis=0)

    perm = np.random.permutation(len(x_seed))
    x_seed = x_seed[perm]
    y_seed = y_seed[perm]
    index_seed = index_seed[perm]

    return x_seed, y_seed, index_seed


# -----------------------------------------------
def train_lf(x,y):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    import numpy as np

    x = np.reshape(x, (x.shape[0], -1))
    # print(set(y))
    svm=SVC(probability=True)
    rf=RandomForestClassifier()
    knn=KNeighborsClassifier()
    dtc=DecisionTreeClassifier()
    nb=GaussianNB()
    lr=LogisticRegression()

    svm.fit(x, y)
    rf.fit(x,y)
    knn.fit(x,y)
    dtc.fit(x,y)
    nb.fit(x,y)
    lr.fit(x,y)

    return svm,rf,knn,dtc,nb,lr
# -----------------------------------------------------
def train_all_LF(x,y,num_cls,path,fraction):
    import pickle
    import copy 
    import os
    for main_cls in range(0,num_cls):
        y_custom = copy.copy(y)
        y_custom[y != main_cls] = -1 #Re-labelling all classes except main_cls to 0
        svm,rf,knn,dtc,nb,lr = train_lf(x,y_custom)
        if not os.path.exists(path):
            os.mkdir(path)
        pickle.dump(svm, open(path+'/'+str(main_cls)+'_svm.pkl', 'wb'))
        pickle.dump(rf, open(path+'/'+str(main_cls)+'_rf.pkl', 'wb'))
        pickle.dump(knn, open(path+'/'+str(main_cls)+'_knn.pkl', 'wb'))
        pickle.dump(dtc, open(path+'/'+str(main_cls)+'_dtc.pkl', 'wb'))
        pickle.dump(nb, open(path+'/'+str(main_cls)+'_nb.pkl', 'wb'))
        pickle.dump(lr, open(path+'/'+str(main_cls)+'_lr.pkl', 'wb'))
        print("Trained & Saved 6 models")

def binary_model_LF(x,y,num_cls,path,fraction):
    import pickle
    import copy 
    import os
    for main_cls in range(0,num_cls):
        for other_class in range(0,num_cls):
            if other_class!= main_cls:
                y_custom = copy.copy(y)
                x_custom = copy.copy(x)
                filters = (y_custom==main_cls) | (y_custom==other_class)
                y_custom = y_custom[filters]
                y_custom[y_custom != main_cls] = -1
                x_custom = x_custom[filters]
                svm,rf,knn,dtc,nb,lr = train_lf(x_custom,y_custom)
                if not os.path.exists(path):
                    os.mkdir(path)
                pickle.dump(svm, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_svm.pkl', 'wb'))
                pickle.dump(rf, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_rf.pkl', 'wb'))
                pickle.dump(knn, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_knn.pkl', 'wb'))
                pickle.dump(dtc, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_dtc.pkl', 'wb'))
                # pickle.dump(nb, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_nb.pkl', 'wb'))
                pickle.dump(lr, open(path+'/'+str(main_cls)+'_'+str(other_class)+'_lr.pkl', 'wb'))
                print("Trained & Saved 5 models")

def create_cnn(num_classes = 3):
    import numpy as np
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
    from keras.models import Sequential
    from keras import Input
    import tensorflow as tf
    import os
    import random as rn
    #disable warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    #Seeding model weights
    SEED = 10
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(42)
    rn.seed(SEED)
    
    # Model
    model = Sequential(
        [
            Input(shape=(28, 28, 3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model