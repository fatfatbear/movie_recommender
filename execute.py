
from data_util import *
from model import *
import matplotlib.pyplot as plt


def rating_movie(mv_net, user_id_val, movie_id_val):
    categories = np.zeros([1, 18])
    categories[0] = movies.values[movieid2idx[movie_id_val]][2]

    titles = np.zeros([1, sentences_size])
    titles[0] = movies.values[movieid2idx[movie_id_val]][1]

    inference_val = mv_net.model([np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                                  np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
                                  categories,
                                  titles])

    return (inference_val.numpy())


def recommend_same_type_movie(movie_id_val, top_k=20):
    norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
    normalized_movie_matrics = movie_matrics / norm_movie_matrics

    # 推荐同类型的电影
    probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results

# 推荐您喜欢的电影
# 思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，取评分最高的top_k个，同样加了些随机选择部分。
def recommend_your_favorite_movie(user_id_val, top_k=10):
    # 推荐您喜欢的电影
    probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])

    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results

# 看过这个电影的人还看了（喜欢）哪些电影
#- 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
#- 然后计算这几个人对所有电影的评分
#- 选择每个人评分最高的电影作为推荐
#- 同样加入了随机选择
import random


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[0][-top_k:]
    #     print(normalized_users_matrics.numpy().shape)
    #     print(probs_user_favorite_similarity.numpy()[0][favorite_user_id])
    #     print(favorite_user_id.shape)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

    print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
    probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
    probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
    p = np.argmax(sim, 1)
    print("喜欢看这个电影的人还喜欢看：")

    if len(set(p)) < 5:
        results = set(p)
    else:
        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results




if __name__ == "__main__":
    mv_net = mv_network()
    mv_net.training(features, targets_values, epochs=5)

    plt.plot(mv_net.losses['train'], label='Training loss')
    plt.legend()
    _ = plt.ylim()

    plt.plot(mv_net.losses['test'], label='Test loss')
    plt.legend()
    _ = plt.ylim()

    rating_movie(mv_net, 234, 1401)

    movie_layer_model = keras.models.Model(inputs=[mv_net.model.input[4], mv_net.model.input[5], mv_net.model.input[6]],
                                           outputs=mv_net.model.get_layer("movie_combine_layer_flat").output)
    movie_matrics = []

    for item in movies.values:
        categories = np.zeros([1, 18])
        categories[0] = item.take(2)

        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)

        movie_combine_layer_flat_val = movie_layer_model([np.reshape(item.take(0), [1, 1]), categories, titles])
        movie_matrics.append(movie_combine_layer_flat_val)

    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))
    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))

    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))

    ## 生成User特征矩阵
    #将训练好的用户特征组合成用户特征矩阵并保存到本地
    user_layer_model = keras.models.Model(
        inputs=[mv_net.model.input[0], mv_net.model.input[1], mv_net.model.input[2], mv_net.model.input[3]],
        outputs=mv_net.model.get_layer("user_combine_layer_flat").output)
    users_matrics = []

    for item in users.values:
        user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0), [1, 1]),
                                                        np.reshape(item.take(1), [1, 1]),
                                                        np.reshape(item.take(2), [1, 1]),
                                                        np.reshape(item.take(3), [1, 1])])
        users_matrics.append(user_combine_layer_flat_val)

    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

    #使用生产的用户特征矩阵和电影特征矩阵做电影推
    recommend_same_type_movie(1401, 20)

    recommend_your_favorite_movie(234, 10)

    recommend_other_favorite_movie(1401, 20)

