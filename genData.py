#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.utils import np_utils 
from keras.models import model_from_json


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_max(a, w=3) :
    ret = np.zeros(a.shape)
    if w <= len(a):
        for n in range(len(a)):
            if n < w:
                ret[n] = a[n]
            else:
                ret[n] = np.max(a[n-w:n])
        return ret
    else:
        return(None)



def load_model(file):
    
    # load json and create model
    json_file = open(file+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file+'.h5')
    print("Loaded model from disk")
    return loaded_model


def randomize(arr):
    """
    input:
        arr: a numpy array with zeros and ones
    output:
        arr2: a numpy array with replacements:
                0 --> random between 0 and 10
                1 --> random between 30 and 100
    """
    arr2 = np.zeros(arr.shape )
    for n,i in enumerate(arr):
        arr2[n] = np.random.rand() * 10 + i * (30 + np.random.rand() * 60)
    return arr2

def rerange(a, rang):
    """
    inputs:
        a (numpy array): array to be reranged.
        rang (int): new range
    output:
        a numpy array with number in the interval 0 < number < rang
    """
    a += -a.min()
    return a/a.max() * rang

def rerange2(a, rang = 100):
    """
    inputs:
        a (numpy array): array to be reranged.
        rang (int): new range
    output:
        a numpy array with number in the interval 0 < number < rang
    """
    a += -a.min()
    return a/a.max() * rang


def randn2(d1, floor, rang):
    """
    Creates a 'normal curve' 1D random array
    inputs:
        d1 (int):
            desired output array size.
        floor:
            minimum value of output array.
        rang: interval ou output array.
    output:
        a 'normal curve' 1D array with interval floor--rang
    """
    return rerange(np.random.randn(1, d1), rang)  + floor


def rand2(d1, floor, rang):
    """
    Creates a  1D random array
    inputs:
        d1 (int):
            desired output array size.
        floor:
            minimum value of output array.
        rang: interval ou output array.
    output:
        a 1D array with interval floor--rang
    """
    return rerange(np.random.rand(1, d1), rang)  + floor


def create_randn_day(cat, f1 =0, f2 =30, r1=10, r2=70):
    """
    Cria um vetor (1,24), sendo random(0,r1) - inativo e random(f2,r2+f2)-ativo. De acordo com o perfil de atividade do dia
    Input:
        cat (string): categoria ( do dia) - pode ser:
            h24: 24 horas
            com: comercial (das 8 as 18)
            ext: comercial extendido (das 8 as 22)
            man: manhã (das 8 as 12)
            emp: vazio/empty (nada)
            not: noturno - (das 8 as 4)
    """
    if cat=='h24':
        return randn2(24, f2, r2)
    elif cat=='com':
        return concat( (randn2(8, f1, r1), randn2(10,f2, r2) , randn2(6, f1, r1) ) , axis = 1 )
    elif cat=='ext':
        return concat( (randn2(8, f1, r1), randn2(14, f2, r2), randn2(2, f1, r1)) , axis = 1 )
    elif cat=='man':
        return concat( ( randn2(8, f1, r1), randn2(4, f2, r2), randn2(12, f1, r1)) , axis = 1  )
    elif cat=='emp':
        return randn2(24, f1, r1)  
    elif cat=='not':
        return concat( ( randn2(4, f2, r2), randn2(4, f1,r1), randn2(16, f2, r2))  , axis = 1 )
    else:
        print('Please specify a category: h24, com, ext, man, emp')

def create_rand_day(cat, f1 =0, f2 =30, r1=10, r2=70):
    """
    Cria um vetor (1,24), sendo random(0,r1) - inativo e random(f2,r2+f2)-ativo. De acordo com o perfil de atividade do dia
    Input:
        cat (string): categoria ( do dia) - pode ser:
            h24: 24 horas
            com: comerical (das 8 as 18)
            ext: comercial extendido (das 8 as 22)
            man: manhã (das 8 as 12)
            emp: vazio/empty (nada)
            not: noturno - (das 8 as 4)
    """
    if cat=='h24':
        return rand2(24, f2, r2)
    elif cat=='com':
        return concat( (rand2(8, f1, r1), rand2(10,f2, r2) , rand2(6, f1, r1) ) , axis = 1 )
    elif cat=='ext':
        return concat( (rand2(8, f1, r1), rand2(14, f2, r2), rand2(2, f1, r1)) , axis = 1 )
    elif cat=='man':
        return concat( ( rand2(8, f1, r1), rand2(4, f2, r2), rand2(12, f1, r1)) , axis = 1  )
    elif cat=='emp':
        return rand2(24, f1, r1)  
    elif cat=='not':
        return concat( ( rand2(4, f2, r2), rand2(4, f1,r1), rand2(16, f2, r2))  , axis = 1 )
    else:
        print('Please specify a category: h24, com, ext, man, emp')

def days(f, cat,  ndays=1):
    d = np.array([[]])
    for i in range(ndays):
        d = concat( ( d, f(cat)), axis = 1)
    return d

def create_week(cat, f):
     #8:00 a 18:00hs de Segunda a Sexta
    if cat==0: 
        return concat( ( [[0]], days(f,'com',5), days(f,'emp',2)), axis =1 )
    #8:00 a 18:00hs de Segunda a Sexta, Sábado pela manhã
    elif cat==1: 
        return concat( ( [[1]], days(f,'com',5), days(f,'man'), days(f,'emp')), axis =1)
    #8:00 às 22:00hs de Segunda a Sábado
    elif cat==2: 
        return concat( ( [[2]], days(f,'ext',6), days(f, 'emp') ), axis =1 )
    #8:00 às 22:00hs de Segunda a Domingo
    elif cat==3: 
        return concat( ( [[3]], days(f,'ext',7)), axis = 1)
    #8:00 às 4:00am todos os dias
    elif cat==4:
        return concat( ( [[4]], days(f,'not', 7)), axis = 1)
    #consumo 24hs por dia
    elif cat==5:
        return concat( ( [[5]], days(f,'h24',5), days(f,'emp',2)), axis = 1)
    #consumo 24hs por dia incluindo Sábado
    elif cat==6: 
        return concat( ( [[6]], days(f,'h24',6), days(f,'emp')), axis =1)
    #consumo 24hs por dia incluindo Sábado e Domingo
    elif cat==7: 
         return concat ( ([[7]], days(f,'h24',7)), axis = 1)
    else:
        print('Please specify a category between 0 and 7')

def create_examples(ne, cat, f):
    """
    Cria uma matriz com um set de exemplos (linhas), cada um com 168 colunas (horas da semana)
    Os valores podem ser (0 ou 1) ou (random(r1), random(r2,r2+f2)) de acordo com o parâmetro (função) f dado com input.
    input:
        ne (integer): número de exemplos (linhas)
        cat (integer): perfil dos exemplos - deve ser um número de 0 a 7
        f: função de preenchimento de valores - pode ser create_rand_day ou create_bin_day
    """
    examples = create_week(cat,f)
    for i in range(1,ne):
        examples = np.concatenate( (examples, create_week(cat,f)) )
    return examples


def save_model(model, model_name):
    """
    Saves a Keras model.
    inputs:
        model
        model_name: name of the file to be saved (with path)
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    model.save(model_name)
    print("Saved model to disk")

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects '''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

def npind(arr):
    """
    input:
        a 1D numpy array
    output:
        a numpy array with the sorted indices of the input
    Ex:
        npa = [134,5,42,66]
        npind(npa)
        **output -> [0,1,2,3]
    """
    return np.arange(arr.shape[0])

def createDataset(examples_per_label):
    dset2 = create_examples(examples_per_label, 0, create_rand_day)
    for label in range(1,8):
        dset2 = concat( (dset2, create_examples(examples_per_label, label, create_rand_day)) )    
    return(dset2.astype(int))

def plotar2(ds1, w = 10, h = 10):
    for i in range(ds1.shape[0]):
        plt.scatter(np.arange(ds1.shape[1]),ds1[i,:])
    fig = plt.gcf()
    fig.set_size_inches(w,h)

def create_day(i1,f1):
    a = np.random.rand(24) * 10
    if f1 >= i1:
        for i in range(i1,f1):
            a[i] = int(np.random.rand() * 70 + 30)
    else:
        for i in range(f1):
            a[i] = int(np.random.rand() * 70 + 30)
        for i in range(i1,24):
            a[i] = int(np.random.rand() * 70 + 30)
    return a



def create_week_2(label, w):
    week = [label]
    for i in range(5):
        week.extend(create_day(w[0], w[1]))
    week.extend(create_day(w[2], w[3]))
    week.extend(create_day(w[4], w[5]))
    return np.asarray(week)


def create_examples_2(ne,*args):
    examples = []
    for i in range(ne):
        examples.append(create_week_2(*args))
    return np.asarray(examples)



def create_dataset_2(dfr, expcat = 1000):
    """
    inputs:
        expcat: Exemplos por categoria
        x: a Pandas dataframe.
    output:
        A numpy int array with expcat examples of each category
    """
    ds = []
    for i in range(dfr.shape[0]):
        ds.extend(create_examples_2( max(1,int(expcat/dfr.iloc[i,0])) , dfr.iloc[i,1], dfr.iloc[i, 2:8]))
    return np.asarray(ds)




