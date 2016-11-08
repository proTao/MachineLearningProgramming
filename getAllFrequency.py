temp_dic = {}  # 用于读入频数
# 载入频数
for i in [1, 2, 3, 4, 5, 6]:
    with open("result/age_" + str(i) + ".pkl", "rb") as f:
        temp_dic[i] = pickle.load(f)
# 统计所有出现过的单词
word_set = set()
for i in [1, 2, 3, 4, 5, 6]:
    word_set = word_set.union(temp_dic[i])
set_length = len(word_set)
print(set_length)
# 按照多项式模型计算单词出现概率
for i in [1, 2, 3, 4, 5, 6]:
    for word in temp_dic[i]:
        age_word_prob[i][word] = (
            1 + temp_dic[i][word]) / (age_count[i] + set_length)