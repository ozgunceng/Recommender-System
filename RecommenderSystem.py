# Import Pandas 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
# Import linear_kernel 
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.metrics.pairwise import cosine_similarity 
corpus=[ 'Julie loves me more than Linda loves me', 
        'Jane likes me more than Julie loves me',  
        'harry likes kiwi fruit'] 
    
# Creating coun vectorizer object. Note: we have removed the stop words 
vectorizer = CountVectorizer(stop_words='english') 
vectors = vectorizer.fit_transform(corpus) 
print('\n ---Corpus converted to term-frequency vector:--\n', vectors.toarray()) 
#Array mapping from feature integer indices to feature name or we can say the unique te
rms present in the corpus 
print('\n ---Unique terms in the corpus:--- \n', vectorizer.get_feature_names()) 
#Compute similarity score 
cosine_sim = cosine_similarity(vectors, vectors) 
cosine_sim2 = linear_kernel(vectors, vectors) 
print('\n--Cosine similarity using cosine similarity function:--\n',cosine_sim) 
print('\n--Cosine similarity using Linear Kernel function:--\n',cosine_sim2) 
# You can write your own function as well for computing cosine similarity like followin
g: 
""" 
doc1=vectors.toarray()[0, :] 
doc2=vectors.toarray()[1,:] 
doc3=vectors.toarray()[2,:] 
cos_sim_doc1_doc2 = dot(doc1, doc2)/(norm(doc1)*norm(doc2)) 
cos_sim_doc1_doc3 = dot(doc1, doc3)/(norm(doc1)*norm(doc3)) 
cos_sim_doc2_doc3 = dot(doc2, doc3)/(norm(doc2)*norm(doc3)) 
print(cos_sim_doc1_doc2) 
print(cos_sim_doc1_doc3) 
print(cos_sim_doc2_doc3) 
"""



# Import Pandas 
import pandas as pd 
#Import TfIdfVectorizer from scikit-learn 
from sklearn.feature_extraction.text import TfidfVectorizer 
# Import linear_kernel 
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.metrics.pairwise import cosine_similarity 
corpus=[ 'Julie loves me more than Linda loves me', 
        'Jane likes me more than Julie loves me',  
        'harry likes kiwi fruit'] 
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a' 
tfidf = TfidfVectorizer(stop_words='english') 
#tfidf = TfidfVectorizer() 
#Construct the required TF-IDF matrix by fitting and transforming the data 
tfidf_matrix = tfidf.fit_transform(corpus) 
#Output the shape of tfidf_matrix 
print(tfidf_matrix.shape) 
print(tfidf_matrix.toarray()) 
#Array mapping from feature integer indices to feature name. 
#print(tfidf.get_feature_names()) 
# Compute the cosine similarity matrix 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 
cosine_sim2 = cosine_similarity(tfidf_matrix, tfidf_matrix) 
print('\n--Cosine similarity using cosine similarity function:--\n',cosine_sim) 
print('\n--Cosine similarity using Linear Kernel function:--\n',cosine_sim2)


# Import Pandas 
import pandas as pd 
#Import TfIdfVectorizer from scikit-learn 
from sklearn.feature_extraction.text import TfidfVectorizer 
# Import linear_kernel 
from sklearn.metrics.pairwise import linear_kernel

# Load Movies Metadata 
metadata = pd.read_csv('dataset\movies_metadata.csv', low_memory=False)
# Print the first three rows 
print(metadata.head(3)) 
#Print plot overviews of the first 5 movies. 
print(metadata['overview'].head())
#Replace NaN with an empty string 
metadata['overview'] = metadata['overview'].fillna('')

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a' 
tfidf = TfidfVectorizer(stop_words='english') 
#Construct the required TF-IDF matrix by fitting and transforming the data 
tfidf_matrix = tfidf.fit_transform(metadata['overview']) 
#Output the shape of tfidf_matrix 
print(tfidf_matrix.shape) 
#Array mapping from feature integer indices to feature name. 
print(tfidf.get_feature_names()[5000:5010])

# Compute the cosine similarity matrix 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 
print('cosine similiarity matrix shape:', cosine_sim.shape) 
#Construct a reverse map of indices and movie titles 
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates() 
print(indices[:10]) 

# Function that takes in movie title as input and outputs most similar movies 
def get_recommendations(title, cosine_sim=cosine_sim): 
   # Get the index of the movie that matches the title 
   idx = indices[title] 
   # Get the pairwsie similarity scores of all movies with that movie 
   sim_scores = list(enumerate(cosine_sim[idx])) 
   # Sort the movies based on the similarity scores 
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
   # Get the scores of the 10 most similar movies 
   sim_scores = sim_scores[1:11] 
   # Get the movie indices 
   movie_indices = [i[0] for i in sim_scores] 
   # Return the top 10 most similar movies 
   return metadata['title'].iloc[movie_indices] 
get_recommendations('Father of the Bride Part II') 


