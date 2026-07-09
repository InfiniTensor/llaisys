import os
from transformers import AutoTokenizer
from python.llaisys.models.qwen2 import Qwen2
from python.llaisys.libllaisys import DeviceType

MODEL_PATH = "/home/koishiyo/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" 

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Qwen2(MODEL_PATH, device=DeviceType.CPU)

    print("\n🚀 LLAISYS 有状态流式推理引擎已启动！")
    is_first_turn = True

    while True:
        user_input = input("🧑 你: ")
        if user_input.strip().lower() in ['quit', 'exit']: break
        if not user_input.strip(): continue

        if is_first_turn:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            is_first_turn = False
        else:
            # 强行补上上一轮丢失的 <|im_end|>，维持完美的 KV Cache！
            prompt_str = f"<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        # encode 时必须加 add_special_tokens=False，防止污染 C++ 里的 Cache
        new_input_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        print("🤖 AI: ", end="", flush=True)

        all_generated_tokens = []
        printed_text_len = 0

        # 👑 终极替换：调用我们刚才写的流式生成器！
        for next_token in model.stream_generate(
            new_input_tokens,
            max_new_tokens=400,   
            temperature=0.7,
            top_p=0.9
        ):
            all_generated_tokens.append(next_token)
            
            # 完整解码当前生成的所有 token
            current_text = tokenizer.decode(all_generated_tokens, skip_special_tokens=True)
            
            # 算出这次新增了哪些字符（完美解决中文多 Token 乱码问题）
            new_text = current_text[printed_text_len:]
            
            if new_text:
                print(new_text, end="", flush=True)
                printed_text_len = len(current_text)

        print("\n") # 这轮对话结束，换行

if __name__ == "__main__":
    main()