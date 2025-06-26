
import numpy as np
import pickle

from utils.evt_fitting import query_weibull
from utils.evt_fitting import weibull_tailfitting
from utils.openmax_utils import *

IMG_DIM = 28
NCLASSES = 10
ALPHA_RANK = 1
WEIBULL_TAIL_SIZE = 10
MODEL_PATH = 'models/weibull_model.pkl'

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# 修改配置参数
IMG_DIM = 28
NCLASSES = 4  # 明确设置为4分类
ALPHA_RANK = 1
WEIBULL_TAIL_SIZE = 10
MODEL_PATH = 'models/weibull_model.pkl'

def create_model(model, data):
    """修正后的模型创建函数"""
    # 获取数据并验证
    X_train, y_train_onehot = data.get_all()
    y_train_onehot = y_train_onehot.astype(np.int32)  # 强制类型转换
    
    # 转换为类别索引并获取标签
    y_train = np.argmax(y_train_onehot, axis=1)
    print(y_train)
    labels = [int(x) for x in np.unique(y_train)]  # 确保标签为整数
    
    # 获取模型激活值
    logits_output, softmax_output = get_activations(X_train, model)
    
    # 筛选正确分类样本
    correct_index = get_correct_classified(softmax_output, y_train_onehot)
    logits_correct = logits_output[correct_index]
    y_correct = np.argmax(y_train_onehot[correct_index], axis=1)
    
    # 按类别组织数据（使用整数标签）
    av_map = {label: logits_correct[y_correct == label] for label in labels}
    
    # 计算统计量
    feature_mean = []
    feature_distance = []
    for label in labels:  # 现在label是整数
        mean = compute_mean_vector(av_map[label])
        distance = compute_distance_dict(mean, av_map[label])
        feature_mean.append(mean)
        feature_distance.append(distance)
    
    # 构建Weibull模型
    build_weibull(feature_mean, feature_distance, labels, WEIBULL_TAIL_SIZE)

def build_weibull(mean_list, distance_list, class_labels, tail):
    """修正后的Weibull构建函数"""
    weibull_model = {}
    
    # 使用enumerate确保整数索引
    for idx, label in enumerate(class_labels):
        try:
            weibull = weibull_tailfitting(
                mean_vector=mean_list[idx],
                distance=distance_list[idx]['eucos'],  # 使用eucos距离
                tailsize=tail
            )
            weibull_model[int(label)] = weibull  # 键强制为整数
        except Exception as e:
            print(f"Class {label} fitting failed: {str(e)}")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(weibull_model, f)

def recalibrate_scores(weibull_model, labels, activation_vector, alpharank=ALPHA_RANK, distance_type='eucos'):
    ranked_list = activation_vector.argsort().ravel()[::-1]
    alpha_weights = [
        ((alpharank+1) - i) / float(alpharank) for i in range(1, alpharank+1)
    ]
    ranked_alpha = np.zeros(NCLASSES)

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # print(ranked_alpha)

    openmax_scores = []
    openmax_scores_u = []

    for label in labels:
        weibull = query_weibull(label, weibull_model, distance_type)
        av_distance = compute_distance(
            weibull[1], activation_vector.ravel())
        wscore = weibull[2][0].w_score(av_distance)
        # print(f'wscore_{label}: {wscore}')
        modified_score = activation_vector[0][label] * \
            (1 - wscore*ranked_alpha[label])
        openmax_scores += [modified_score]
        openmax_scores_u += [activation_vector[0][label] - modified_score]

    openmax_scores = np.array(openmax_scores)
    openmax_scores_u = np.array(openmax_scores_u)

    # print(f'activation_vector: {activation_vector}')
    # print(f'openmax_scores: {openmax_scores}')
    # print(f'openmax_scores_u: {np.sum(openmax_scores_u)}')

    openmax_probab, prob_u = compute_openmax_probability(
        openmax_scores, openmax_scores_u)
    return openmax_probab, prob_u


def compute_openmax_probability(openmax_scores, openmax_scores_u):
    e_k = np.exp(openmax_scores)
    e_u = np.exp(np.sum(openmax_scores_u))
    openmax_arr = np.concatenate((e_k, e_u), axis=None)
    total_denominator = np.sum(openmax_arr)
    prob_k = e_k / total_denominator
    prob_u = e_u / total_denominator
    res = np.concatenate((prob_k, prob_u), axis=None)
    return res, prob_u


def compute_openmax(activation_vector):
    with open(MODEL_PATH, 'rb') as file:
        weibull_model = pickle.load(file)
    openmax, prob_u = recalibrate_scores(
        weibull_model, labels, activation_vector)
    return openmax, prob_u
