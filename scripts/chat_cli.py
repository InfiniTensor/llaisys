import argparse
import sys
import llaisys
from llaisys import DeviceType
from transformers import AutoTokenizer

def chat_cli():
    parser = argparse.ArgumentParser(description="LLAISYS Chat CLI")
    parser.add_argument("--model", type=str, default="/home/wsl/model/DeepSeek-R1-Distill-Qwen-1.5B", help="Path to the model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--use_session", action="store_true", default=True, help="Use session for KV cache reuse (faster multi-turn)")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading model from {args.model}...")
    try:
        model = llaisys.models.Qwen2(args.model, DeviceType.CPU)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "=" * 50)
    print("LLAISYS Chat CLI (Type 'exit' or 'quit' to stop)")
    print("=" * 50 + "\n")

    # Keep chat history
    history = []
    
    # Create a session if requested
    session = None
    if args.use_session:
        print("Creating session for KV cache reuse...")
        session = model.create_session()

    while True:
        try:
            user_input = input("\033[1;32mUser: \033[0m").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        # Format prompt with history
        prompt_text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt_text)

        print("\033[1;34mAssistant: \033[0m", end="", flush=True)

        # Stream generation
        stream_gen = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            stream=True,
            session=session
        )

        full_response = ""
        try:
            for token_id in stream_gen:
                # Simple decoding (may have issues with multi-byte chars at boundaries)
                # In production, use a stateful decoder
                word = tokenizer.decode([token_id], skip_special_tokens=True)
                print(word, end="", flush=True)
                full_response += word
        except KeyboardInterrupt:
            print("\n[Generation stopped by user]")
        
        print("\n")
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    chat_cli()
