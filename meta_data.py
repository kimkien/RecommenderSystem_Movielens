# created by ducva
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def user_metadata():
    cols = ["id","gender","age","occupation","zip_code"]
    occupations = []
    user = pd.read_csv("Data/ml-1m/users.dat", sep='::', names=cols, header=None,  encoding='latin-1')
    min_age = user.age.min()
    max_age = user.age.max()
    print(user.head(5))
    # print(max_age)
    gender = {'M':1, 'F':0}
    user['age'] = user['age'].apply(lambda x: (x - min_age) / (max_age - min_age))
    user['gender'] = user['gender'].apply(lambda x: gender[x])
    user = pd.get_dummies(user, columns=['occupation'])
    user_md = []
    for i in range(1, user.shape[0]+1):
        # print(i)
        # print(user.loc[user.id == i, ['gender','age', 'occupation_1', 'occupation_2',
        #                                        'occupation_3', 'occupation_4', 'occupation_5',
        #                                        'occupation_6', 'occupation_7',
        #                                        'occupation_8', 'occupation_9', 'occupation_10',
        #                                        'occupation_11', 'occupation_12', 'occupation_13',
        #                                        'occupation_14', 'occupation_15', 'occupation_16',
        #                                        'occupation_17', 'occupation_18', 'occupation_19',
        #                                        'occupation_20']].values.flatten().tolist())
        user_md.append(str(user.loc[user.id == i, ['age', 'gender', 'occupation_1', 'occupation_2',
                                               'occupation_3', 'occupation_4', 'occupation_5',
                                               'occupation_6', 'occupation_7',
                                               'occupation_8', 'occupation_9', 'occupation_10',
                                               'occupation_11', 'occupation_12', 'occupation_13',
                                               'occupation_14', 'occupation_15', 'occupation_16',
                                               'occupation_17', 'occupation_18', 'occupation_19',
                                               'occupation_20']].values.flatten().tolist()))

    # print(len(user_md)) # 23 categories
    arr= np.array(user_md)
    print(arr)
    u_vector = pd.DataFrame(user_md)
    u_vector.index= user.id
    print(u_vector)
    # u_vector = pd.DataFrame(user_md, columns=['vector'])
    u_vector.to_csv("Data/u.vector")

def occupation_embedding():
    occupation = """administrator
artist
doctor
educator
engineer
entertainment
executive
healthcare
homemaker
lawyer
librarian
marketing
none
other
programmer
retired
salesman
scientist
student
technician
writer"""
    col_name = """occupation_administrator
occupation_artist
occupation_doctor
occupation_educator
occupation_engineer
occupation_entertainment
occupation_executive
occupation_healthcare
occupation_homemaker
occupation_lawyer
occupation_librarian
occupation_marketing
occupation_none
occupation_other
occupation_programmer
occupation_retired
occupation_salesman
occupation_scientist
occupation_student
occupation_technician
occupation_writer"""
    occupations = col_name.lower().split('\n')
    return occupations

def movie_metadata():
    item = pd.read_csv("Data/ml-1m/1M_extended_enough.csv", index_col=[0])
    print(item.head(5))
    min_year = item.year.min()
    max_year = item.year.max()
    min_runtimes = item.runtimes.min()
    max_runtimes = item.runtimes.max()
    # print(user.head(5))
    # # print(max_age)
    item_df=pd.DataFrame()
    # item_df["movieId"]= item["movieId"]
    item_df['year'] = item['year'].apply(lambda x: (x - min_year) / (max_year - min_year))
    item_df['runtimes'] = item['runtimes'].apply(lambda x: (x - min_runtimes) / (max_runtimes - min_runtimes))

    # ls_direc = encode_director(item)
    # ls_cast = encode_cast(item)
    ls_genres = encode_genres(item)

    ls_concat=[]
    for i in range(item.shape[0]):
        # ls_genres[i].extend(ls_direc[i])
        # ls_genres[i].extend(ls_cast[i])
        ls_genres[i].extend(item_df.iloc[i,:].tolist())
        ls_concat.append(str(ls_genres[i]))

    print(np.shape(ls_concat))
    vector_df= pd.DataFrame(ls_concat, columns=['vector'], index=item["movieId"])
    vector_df.to_csv("Data/i.vector", index=True, header= False)

def encode_genres(df_genres):
    dict_genres= {"Action":0, "Adventure":1,
	"Animation":2,
	"Children\'s":3,
	"Comedy": 4,
	"Crime":5,
	"Documentary":6,
	"Drama":7,
	"Fantasy":8,
	"Film-Noir":9,
	"Horror":10,
	"Musical":11,
	"Mystery":12,
	"Romance":13,
	"Sci-Fi":14,
	"Thriller":15,
	"War":16,
	"Western":17}
    ls_encoded= list()
    for group_genr in df_genres["genres"].tolist():
        vector= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        group_genr= group_genr.split("|")
        for temp in group_genr:
            vector[dict_genres[temp]]=1
        ls_encoded.append(vector)
    return ls_encoded

def encode_cast(df_cast):
    # get name of cast
    ls_cast = set()
    dict_cast= dict()
    for rel in df_cast["cast_of_most"].str.split("|").values:
        ls_cast = ls_cast.union(set(rel))
    ls_cast = sorted(ls_cast)
    ls_encoded_direc = encoder(ls_cast)
    i=0
    for direc in ls_cast:
        dict_cast[direc] = ls_encoded_direc[i]
        i += 1
    ls_code=list()
    for direc in df_cast["cast_of_most"].tolist():
        ls_code.append(dict_cast[direc])
    return ls_code

def encode_director(df_direc):
    # get name of cast
    ls_direc = set()
    dict_direc= dict()
    for rel in df_direc["director_of_most"].str.split("|").values:
        ls_direc = ls_direc.union(set(rel))
    ls_direc = sorted(ls_direc)
    ls_encoded_direc = encoder(ls_direc)
    i=0
    for direc in ls_direc:
        dict_direc[direc] = ls_encoded_direc[i]
        i += 1
    ls_code= list()
    for direc in df_direc["director_of_most"].tolist():
        # print(direc)
        # print(dict_direc[direc])
        ls_code.append(dict_direc[direc])
    return  ls_code

def encoder(ls_obj):
	np_obj= np.array(ls_obj)
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(np_obj)
	# print(integer_encoded)
	encoded = to_categorical(integer_encoded)
	return encoded

def read_meta():
    PATH_META_USER = "Data/u.vector"
    PATH_META_ITEM = "Data/meta_item.csv"
    user_meta= pd.read_csv(PATH_META_USER, index_col=[0])
    # item_meta= pd.read_csv(PATH_META_ITEM)
    print(user_meta.iloc[6037,])

# def create_negative_train(num_negative=4):


if __name__=="__main__":
    # user_metadata()
    movie_metadata()
    # read_meta()
    #print(occupation_embedding())

