#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
#%%
# ?counting words
text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
print(count_matrix.toarray())

#%%
# ?finding similarity wrt words
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)




# %%
