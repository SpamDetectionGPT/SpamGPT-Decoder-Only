import torch
from spamGPT import SpamGPT
from config import SpamGPTConfig
from inference import run_inference, enc, get_spam_ham_probabilities, special_tokens

def load_model(checkpoint_path):
    # Initialize model
    model = SpamGPT(SpamGPTConfig())
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def generate_email(model, prompt, temperature=1, device="cpu"):
    allowed_special = {"<SOE>", "<EOE>", "<SOP>", "<EOP>", "<SPAM>", "<HAM>"}
    ids = enc.encode(prompt, allowed_special=allowed_special)
    input_tokens = torch.tensor([ids], dtype=torch.long, device=device)  # <-- fixed

    spam_prob, ham_prob, top_token = get_spam_ham_probabilities(
        input_tokens, model, temp=temperature
    )
    return spam_prob, ham_prob, top_token


def generate_complete_email(model, prompt, max_length=200, temperature=0.8, device="cpu"):
    allowed_special = {"<SOE>", "<EOE>", "<SOP>", "<EOP>", "<SPAM>", "<HAM>"}
    ids = enc.encode("<SOE>HUGE SALE!!! \n\n URGENT \n\n CLICK HERE TO GET YOUR DISCOUNT \n\n <EOE><SOP><SPAM><SOP> [CLS] ", allowed_special=allowed_special)
    input_tokens = torch.tensor([ids], dtype=torch.long, device=device)  # <-- fixed

    generated_tokens = run_inference(
        input_tokens, max_length, model, temperature, enc,
        endtoken=special_tokens["<EOP>"]
    )
    return enc.decode(generated_tokens[0].tolist())

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model("checkpoints/best_model.pt")
    model = model.to(device)
    model.eval()
    
    # Example prompts
    prompts = [
    # ---------- SPAM ----------
    """<SOE>Hello I am from Microsoft. I am calling to tell you that you have won a $1000 gift card. Please click the link below to claim your prize. <EOE><SOP>""",

]
    
    # Generate emails for each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        # Get spam/ham probabilities
        spam_prob, ham_prob, highest_prob_token = generate_email(model, prompt, device=device)
        print(f"Spam Probability: {spam_prob:.4f}")
        print(f"Ham Probability: {ham_prob:.4f}")
        print(f"Highest Probability Token: {highest_prob_token}")
        # Generate complete email
        generated_email = generate_complete_email(model, prompt, device=device)
        print("\nGenerated Email:")
        print(generated_email)
        print("-" * 50)