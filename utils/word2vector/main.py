from gensim.models import Word2Vec
import time


# 读取数据集
def read_txt():
    with open("/Users/chenyn/chenyn's/研究生/DataSet/My dect/word2vector/corpus/corpus.txt") as f:
        raw_text = f.read()
        sentences = [sentence.replace('\n', '') for sentence in raw_text.split('.') if
                     (sentence.replace('\n', '')) != ' ' and
                     sentence.replace('\n', '') != '']

        res = []
        for sentence in sentences:
            res.append([word for word in sentence.split()])
        return res


sentences = read_txt()

print('start')

time_start = time.time()
model = Word2Vec(sentences=sentences, vector_size=300, window=3, min_count=0, sg=1, negative=5)
time_end = time.time()
print('3_end:', time_end - time_start)

time_start = time.time()
model = Word2Vec(sentences=sentences, vector_size=300, window=4, min_count=0, sg=1, negative=5)
time_end = time.time()
print('4_end:', time_end - time_start)

time_start = time.time()
model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=0, sg=1, negative=5)
time_end = time.time()
print('5_end:', time_end - time_start)

time_start = time.time()
model = Word2Vec(sentences=sentences, vector_size=300, window=7, min_count=0, sg=1, negative=5)
time_end = time.time()
print('7_end:', time_end - time_start)

time_start = time.time()
model = Word2Vec(sentences=sentences, vector_size=300, window=9, min_count=0, sg=1, negative=5)
time_end = time.time()
print('9_end:', time_end - time_start)

# model.save("word_vector.model")
print('end')
# test
# model = Word2Vec.load("word_vector.model")
