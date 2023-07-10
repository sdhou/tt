import faiss
from transformers import AutoTokenizer
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("/opt/code/chatglm", trust_remote_code=True)
source = ['今天你身体好吗', '第一个向量为 0', '你呀傻逼']
for row in range(len(source)):
    line = tokenizer.encode(source[row])
    line = np.pad(line, (0, 64-len(line)), 'constant', constant_values=(-1, -1))
    source[row] = line
source = np.array(source, dtype=int)
index = faiss.IndexFlatL2(64)
index.add(source)


def search(q):
    b2 = tokenizer.encode(q)
    c2 = np.pad(b2, (0, 64-len(b2)), 'constant', constant_values=(-1, -1))
    ddd1 = np.array([c2])
    D, I = index.search(ddd1, 1)
    ret = [x for x in source[I[0][0]] if x != -1]
    return D[0][0], tokenizer.decode(ret)


print(search('你呀'))
