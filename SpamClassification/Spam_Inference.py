from NLP_Preprocessing import *
from sklearn.externals import joblib

fd = open('InferenceEmails/spamSample1.txt','r')
email = fd.read()
email = clean(email)
fd.close()
email = email.split(' ')
print email

VocabList = readVocabList('vocab.txt')
word_indices = word_to_vocab_indices(email,VocabList)
features = get_feature_vector(word_indices,len(VocabList))


model1 = joblib.load('spamNaivemodel.pkl')
model2 = joblib.load('spamSVMmodel.pkl')

print features.shape
spam = model1.predict(np.transpose(features))
print spam
