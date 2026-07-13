#include "sample_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <stdexcept>

namespace llaisys::ops::cpu {

struct TokenProb {
    float prob;
    int index;
};

void sample_f32(int64_t* next_token_id, const float* logits, size_t vocab_size, 
                float temperature, int top_k, float top_p) {
    
    // 1. 如果温度极低 (贪心策略)，直接退化为 Argmax，速度最快
    if (temperature < 1e-5f) {
        float max_val = logits[0];
        int max_idx = 0;
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        *next_token_id = max_idx;
        return;
    }

    // 2. 找到最大值，用于安全的 Softmax (防止指数爆炸)
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 3. 应用 Temperature 并计算 Softmax 的分母
    std::vector<TokenProb> probs(vocab_size);
    float sum_prob = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        // Logits 除以温度后再求指数
        float p = std::exp((logits[i] - max_logit) / temperature);
        probs[i] = {p, (int)i};
        sum_prob += p;
    }

    // 4. 归一化为标准概率分布 (总和为 1.0)
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i].prob /= sum_prob;
    }

    // 5. 按照概率从大到小排序
    std::sort(probs.begin(), probs.end(), [](const TokenProb& a, const TokenProb& b) {
        return a.prob > b.prob;
    });

    // 6. Top-K 截断
    size_t active_size = vocab_size;
    if (top_k > 0 && (size_t)top_k < active_size) {
        active_size = top_k;
    }

    // 7. Top-P (核采样) 截断
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative_prob = 0.0f;
        size_t p_size = 0;
        for (size_t i = 0; i < active_size; ++i) {
            cumulative_prob += probs[i].prob;
            p_size++;
            if (cumulative_prob >= top_p) {
                break;
            }
        }
        active_size = p_size;
    }

    // 8. 对截断后的候选集重新归一化
    float active_sum = 0.0f;
    for (size_t i = 0; i < active_size; ++i) {
        active_sum += probs[i].prob;
    }
    
    // 9. 掷骰子：生成 0~1 的随机数，执行多项式采样 (Multinomial Sampling)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen) * active_sum; // 直接映射到未完全归一化的总和上

    float accum = 0.0f;
    for (size_t i = 0; i < active_size; ++i) {
        accum += probs[i].prob;
        if (accum >= r) {
            *next_token_id = probs[i].index;
            return;
        }
    }

    // 保底机制
    *next_token_id = probs[active_size - 1].index;
}

} // namespace llaisys::ops::cpu