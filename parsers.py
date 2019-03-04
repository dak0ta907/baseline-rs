import re # for regex parsing  to strip out lots of BS
import numpy as np
import itertools

def parse_mid(): #strips movie ids
    processed_list = []
    with open('datasets/movies.csv','r', encoding="latin1") as f2:
        lines = f2.readlines()
        for i in range (1,len(lines)):
            line = lines[i].rstrip()
            line = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), line)
            chunks = line.split(',')

            chunks[0] = int(chunks[0]) 
            chunks[1] = chunks[1]
            processed_list.append(chunks)

            # print(chunks)
        f2.close()
        return processed_list


#gives us UID, MID, rating, timestamp
def parse_ratings():
    processed_list = []
    with open('datasets/ratings.csv','r') as f:
        lines = f.readlines()
        for i in range (1,len(lines)):
            line = lines[i].rstrip()
            line = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), line)
            #replace all random separators with commas for easier use
            chunks = line.split(',')
            #casting to usable datatypes premptively
            chunks[0] = int(chunks[0])
            chunks[1] = int(chunks[1])
            chunks[2] = float(chunks[2])
            chunks[3] = int(chunks[3])
            processed_list.append(chunks)
            # print(chunks)
        f.close()
        return processed_list

def genMovieMatrix(MID, ratingsList):
    userRatings = ratingsList
    movieMatrix = []
    for key in userRatings.keys():#for each user get their movie ratings dictionary
        movieMatrix.append(userRatings[key].get(MID,0)) #append to our movie matrix, the rating of the movie from the user dictionary, rate 0 if not found
    #print(movieMatrix) #prints our movie matrix list which we can calculate stuff on 
    #print(len(movieMatrix)) #prints 671 for our 671 users :)
    # print(movieMatrix)
    return movieMatrix

    #generate a dictionary with tuple values (name, rating list)
def genMovieDictionary(movielist):
    completeDictionary = {}
    for movie in movielist:
        completeDictionary[movie[0]] = (movie[1], [])
    return completeDictionary

def genUserRatingList():
    ratingDictionary = {}
    ratings = parse_ratings()
    for rate in ratings:
        #we make a dictionary of dictionaries, so we can access quickly by [UID][MID] to get users rating of a movie
        #this also lets us quickly do ratingDictionary[UID].get(MID,0) to try and get their movie rating by ID, and default to 0 otherwise
        ratingDictionary.setdefault(rate[0],{}).update({rate[1]:rate[2]})
        #rate[0] is UID, rate[1] is MID, rate[2] is Rating
        #ratingDictionary adds a user with an  empty dictionary as default if necessary, then updates it by adding a MID key with a rating value
    #for key, val in ratingDictionary.items():
    #    print(key,val)
    #print(ratingDictionary[1].get(31,0))
    #print(len(ratingDictionary.keys()))
    return ratingDictionary

#put the user review matricies into the movie classes
def assembleMovieMatricies(movieDictionary, ratingDictionary):
    for key in movieDictionary.keys():
        ratingList = genMovieMatrix(key, ratingDictionary)
        for rating in ratingList:
            movieDictionary[key][1].append(rating)
    return movieDictionary

def centerMatrix(matrix):
    i = 0
    total = 0
    for rating in matrix:
        if rating != 0:
            i += 1
            total += rating
    if i == 0:
        return
    total = total/i
    j = 0
    for rating in matrix:
        if rating != 0:
            matrix[j] -= total
        j += 1
    #print(matrix)

def cos_sim(a, b):
    dot_product = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a != 0 and norm_b != 0:
        return dot_product / (norm_a * norm_b)
    else:
        return 0