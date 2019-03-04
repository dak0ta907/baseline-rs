# import libraries
import pandas as pd
import numpy as np
import math
import pickle
from parsers import *

# debug mode: if it is set True, only use partial dataset for the purpose of debug or demonstration
debug_mode = False
# load_existing_w_matrix: it it is set True, the previous built similarity matrix will be loaded instead of building one
load_existing_w_matrix = False


# Set the file path where the similarity matrix will be persisted
if debug_mode == True:
    DEFAULT_PARTICLE_PATH = 'w_matrix_debug.pkl'
else:
    DEFAULT_PARTICLE_PATH = 'w_matrix.pkl'

ratings = pd.read_csv("datasets/ratings.csv", encoding='"ISO-8859-1"')
movies = pd.read_csv("datasets/movies.csv", encoding='"ISO-8859-1"')
tags = pd.read_csv("datasets/tags.csv", encoding='"ISO-8859-1"')


# use partial dataset for debug mode
if debug_mode == True:
    ratings = ratings[(ratings['movieId'] < 100) & (ratings['userId'] < 100)]
    movies = movies[movies['movieId'] < 100]

#this might not need to be here, and changing to just use the raw ratings makes the answer consistent

# split the ratings into training and test
# ratings = ratings.sample(frac=0.7)
# ratings_test = ratings.drop(ratings.index)


# calculate adjusted ratings based on training data
rating_mean= ratings.groupby(['movieId'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})[['movieId','rating_mean']]
adjusted_ratings = pd.merge(ratings,rating_mean,on = 'movieId', how = 'left', sort = False)
adjusted_ratings['rating_adjusted']=adjusted_ratings['rating']-adjusted_ratings['rating_mean']
# replace 0 adjusted rating values to 1*e-8 in order to avoid 0 denominator
# adjusted_ratings.loc[adjusted_ratings['rating_adjusted'] == 0, 'rating_adjusted'] = 1e-8


#adding my code to try and get ahead of this, this properly weights the matrix
Movielist = parse_mid()
movieDictionary = genMovieDictionary(Movielist)#create a dictionary of all of our movies, key = MID, val = Movie Class
userRatingsDict = genUserRatingList()#create a dictionary of all of our movie ratings
movieDict = assembleMovieMatricies(movieDictionary, userRatingsDict)

for key in movieDict.keys():
#    print(movieDict[key][0])
   centerMatrix(movieDict[key][1])
#    print(movieDict[key][1]



# function of building the item-to-item weight matrix
def build_w_matrix(adjusted_ratings, load_existing_w_matrix):
    # print("building weight matrix")
    i = 0
    # define weight matrix
    w_matrix_columns = ['movie_1', 'movie_2', 'weight']
    w_matrix=pd.DataFrame(columns=w_matrix_columns)

    # load weight matrix from pickle file
    if load_existing_w_matrix:
        with open(DEFAULT_PARTICLE_PATH, 'rb') as input:
            w_matrix = pickle.load(input)
        input.close()

    # calculate the similarity values
    else:
        comblist = itertools.combinations(movieDict.keys(), 2)
        for comb in comblist:
            sim_value = cos_sim(movieDict[comb[0]][1], movieDict[comb[1]][1])
            w_matrix = w_matrix.append(pd.Series([str(comb[0]), str(comb[1]), sim_value], index=w_matrix_columns), ignore_index=True)

            i = i + 1
            if debug_mode == True:
                if i == 100:
                    break

        # output weight matrix to pickle file
        with open(DEFAULT_PARTICLE_PATH, 'wb') as output:
            pickle.dump(w_matrix, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    # print(w_matrix)
    return w_matrix

# run the function to build similarity matrix
w_matrix = build_w_matrix(adjusted_ratings, load_existing_w_matrix)

# calculate the predicted ratings
def predict(userId, movieId, w_matrix, adjusted_ratings, rating_mean):
    # fix missing mean rating which was caused by no ratings for the given movie
    # mean_rating exists for movieId
    if rating_mean[rating_mean['movieId'] == movieId].shape[0] > 0:
        mean_rating = rating_mean[rating_mean['movieId'] == movieId]['rating_mean'].iloc[0]
    # mean_rating does not exist for movieId(which may be caused by no ratings for the movie)
    else:
        mean_rating = 2.5

    # calculate the rating of the given movie by the given user
    user_other_ratings = adjusted_ratings[adjusted_ratings['userId'] == userId]
    user_distinct_movies = np.unique(user_other_ratings['movieId'])
    sum_weighted_other_ratings = 0
    sum_weghts = 0
    for movie_j in user_distinct_movies:
        if rating_mean[rating_mean['movieId'] == movie_j].shape[0] > 0:
            rating_mean_j = rating_mean[rating_mean['movieId'] == movie_j]['rating_mean'].iloc[0]
        else:
            rating_mean_j = 2.5
        # only calculate the weighted values when the weight between movie_1 and movie_2 exists in weight matrix
        w_movie_1_2 = w_matrix[(w_matrix['movie_1'] == movieId) & (w_matrix['movie_2'] == movie_j)]
        if w_movie_1_2.shape[0] > 0:
            user_rating_j = user_other_ratings[user_other_ratings['movieId']==movie_j]
            sum_weighted_other_ratings += (user_rating_j['rating'].iloc[0] - rating_mean_j) * w_movie_1_2['weight'].iloc[0]
            sum_weghts += np.abs(w_movie_1_2['weight'].iloc[0])

    # if sum_weights is 0 (which may be because of no ratings from new users), use the mean ratings
    if sum_weghts == 0:
        predicted_rating = mean_rating
    # sum_weights is bigger than 0
    else:
        predicted_rating = mean_rating + sum_weighted_other_ratings/sum_weghts

    return predicted_rating

# make recommendations
def recommend(userID, w_matrix, adjusted_ratings, rating_mean, amount=5):
    distinct_movies = np.unique(adjusted_ratings['movieId'])
    user_ratings_all_movies = pd.DataFrame(columns=['movieId', 'rating'])
    user_rating = adjusted_ratings[adjusted_ratings['userId']==userID]

    # calculate the ratings for all movies that the user hasn't rated
    i = 0
    for movie in distinct_movies:
        user_rating = user_rating[user_rating['movieId']==movie]
        if user_rating.shape[0] > 0:
            rating_value = user_ratings_all_movies.loc[i, 'rating'] = user_rating.loc[0, movie]
        else:
            rating_value = user_ratings_all_movies.loc[i, 'rating'] = predict(userID, movie, w_matrix, adjusted_ratings, rating_mean)
        user_ratings_all_movies.loc[i] = [movie, rating_value]

        i = i + 1

    # select top 10 movies rated by the user
    recommendations = user_ratings_all_movies.sort_values(by=['rating'], ascending=False).head(amount)
    return recommendations


# get a recommendation list for each user
i = 0
for user in np.unique(ratings['userId']):
    i+= 1
    if(i == 10 and debug_mode == True):
        break
    try:
        recommended_movies = recommend(user, w_matrix, adjusted_ratings, rating_mean)
        print("User-id",user," ",recommended_movies["movieId"].to_list())
        
    except:
        print("User-id",user," unable to recommend")
