#include "sampler_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

namespace llaisys::ops::cpu {

// 定义一个结构体，把原始的索引(编号)和计算出的概率绑定在一起
// 这样在排序的时候，我们就不会“弄丢”这个概率属于哪个词了
struct TokenProb {
    int32_t index;
    float prob;
};

int32_t sample(const float *logits, size_t vocab_size, float temperature, float top_p, size_t top_k) {
    // ==========================================
    // 步骤 0：退化保护 (Argmax)
    // 如果温度极其接近 0，或者 top_k 被设为了 1，直接退化为贪心搜索，省去所有计算
    // ==========================================
    if (temperature <= 1e-5f || top_k == 1) {
        int32_t best_idx = 0;
        float max_val = logits[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best_idx = static_cast<int32_t>(i);
            }
        }
        return best_idx;
    }

    // ==========================================
    // 步骤 1 & 2：应用 Temperature 和“数值稳定的 Softmax”
    // ==========================================
    // 先找出最大的 logit
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    std::vector<TokenProb> probs(vocab_size);
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i].index = static_cast<int32_t>(i);
        // 【核心考点】：减去 max_logit 防止 exp 算爆溢出，同时除以温度
        probs[i].prob = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i].prob;
    }

    // 归一化，变成真正的概率分布 (总和为 1)
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i].prob /= sum_exp;
    }

    // ==========================================
    // 步骤 3：降序排序
    // ==========================================
    // 传入一个 Lambda 表达式，告诉 C++ 按照 prob 的大小从大到小排
    std::sort(probs.begin(), probs.end(), [](const TokenProb &a, const TokenProb &b) {
        return a.prob > b.prob;
    });

    // ==========================================
    // 步骤 4：Top-K 与 Top-P 截断
    // ==========================================
    size_t valid_count = 0;
    float cum_prob = 0.0f;
    
    // 如果没有设置 top_k (比如传了 0 或极大值)，将其设为词表大小
    if (top_k == 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }

    // 寻找截断点
    for (size_t i = 0; i < vocab_size; i++) {
        valid_count++;
        cum_prob += probs[i].prob;
        // 一旦概率累加超过了 Top-P 阈值，或者保留的词数达到了 Top-K，立刻停止
        if (cum_prob >= top_p || valid_count >= top_k) {
            break;
        }
    }

    // ==========================================
    // 步骤 5：掷骰子 (轮盘赌抽样)
    // ==========================================
    // 截断之后，剩下的这些候选词的概率加起来不到 1 了，我们算一下它们新的总和
    float truncated_sum = 0.0f;
    for (size_t i = 0; i < valid_count; i++) {
        truncated_sum += probs[i].prob;
    }

    // 初始化 C++11 的高质量随机数生成器 (梅森旋转算法 mt19937)
    std::random_device rd;
    std::mt19937 gen(rd());
    // 在 0 到 truncated_sum 之间摇一个随机浮点数 r
    std::uniform_real_distribution<float> dis(0.0f, truncated_sum);
    float r = dis(gen);

    // 轮盘赌开始：用 r 挨个减去每个词的概率，r 变成负数的那一刻，落到了哪个词就是哪个词
    float current_sum = 0.0f;
    for (size_t i = 0; i < valid_count; i++) {
        current_sum += probs[i].prob;
        if (r <= current_sum) {
            return probs[i].index;
        }
    }

    // 保底返回（极小概率由于浮点数精度误差走到这里），返回候选池里最小的那个
    return probs[valid_count - 1].index;
}

} // namespace llaisys::ops::cpu