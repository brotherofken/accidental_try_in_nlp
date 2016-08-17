## Random code that I accidentally wrote to process some text dataset

Source files aren't supposed to use as normal Python scripts. They was used in Spyder IDE and imitate IPython Notebook structure.

### tokenization.py

Use this file in order to split sentences into tokens and filter data.
Text filters implemented in task_utils.py

1. Delete digits
2. Transliterate from <br>
   'qwertyuiopasdfghjklzxcvbnm' to <br>
   'квертиуиопасдфгхжклзхсвбнм' <br>
   this helps when some Russian words contain Latin characters.
3. Delete short Latin words of length 2 ('aa', 'bb', 'bb' и т.д.)
4. Delete short Russian words of length 2 ('аа', 'бб', 'вв' и т.д.)
5. Delete hyperlinks, e.g. words beginning with http
6. Coalesce sequences of duplicate characters to length of two, e.g. 'aaaa' -> 'aa'
7. remove non-Russian sentences
8. Remove stop words from the list:
  ```
['abex','act','amp','amp','app','club','http','php','post',
 'questions','vkontakte','clickunder','pop','card','uid','esquire',
 'oss','soso','video','audio','graffiti','imm']
 ```
9. Remove sentences shorter than 10 tokens

### lemmatization.py
Lemmatisation of Russian words.
```
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
lemmatised = morph.parse(word)[0].normal_form
```

### stemming.py
Stemming of Russian words.
```
from nltk.stem.snowball import RussianStemmer
lmtzr = RussianStemmer()
stemmed = lmtzr.stem(word)
```

### vect_cluster.py

Naive attempt to cluster sentences by content.

1. Generate word vector using GloVe
2. Word vectors clustering using MiniBatchKMeans into 500~1000 clusters.
3. Sentence feature vector is a bag of clusters (like BoW or term-document matrix?), or simply histogram of clusters that occured in sentence. <br>
Maximum value in this matrix is limited to 5
4. Apply Truncated SVD
5. Cluster sentence feature vectors using MiniBatchKMeans.

### find_patterns.py

Attempt to find similar sentences manually.

Distance between two sentences is distance between mean of their word vectors.
First sentence is a query which consists of keywords, e.g.
```
tokens_filter = 'поздравляю тебя вас день отмечать отмечает днем днём поздравления праздником праздничком прошедшим желаю чтоб чтобы'.split(' ')
```
Is a query for congratulation message.
Then we just filter messages that passes distance threshold.
