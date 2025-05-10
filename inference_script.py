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

def generate_email(model, prompt, max_length=200, temperature=1, device="cpu"):
    # Encode the prompt with allowed special tokens
    input_tokens = torch.tensor([[enc.encode(prompt, allowed_special={"<EOE>", "<SOP>", "<EOP>", "<SOE>", "<SPAM>", "<HAM>"})[0]]], device=device)
    
    # Get spam and ham probabilities
    spam_prob, ham_prob, highest_prob_token = get_spam_ham_probabilities(input_tokens, model, temp=temperature)
    
    return spam_prob, ham_prob, highest_prob_token

def generate_complete_email(model, prompt, max_length=200, temperature=1, device="cpu"):
    # Encode the prompt with allowed special tokens
    input_tokens = torch.tensor([[enc.encode(prompt, allowed_special={"<EOE>", "<SOP>", "<EOP>", "<SOE>", "<SPAM>", "<HAM>"})[0]]], device=device)
    
    # Generate the complete email
    generated_tokens = run_inference(input_tokens, max_length, model, temperature, enc, endtoken=special_tokens["<EOP>"])
    
    # Decode the generated tokens
    generated_text = enc.decode(generated_tokens[0].tolist())
    
    return generated_text

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model("checkpoints/best_model.pt")
    model = model.to(device)
    model.eval()
    
    # Example prompts
    prompts = [
        "<SOE>[CLS] email_from: \"Google\" <no-reply@google.com> [SEP] email_to: user@example.com [SEP] subject: Unusual Sign-In Activity [SEP] message: We detected a new sign-in to your Google account from a new device. If this was you, you can ignore this message. If not, please secure your account here: http://google-secure-account.com. [SEP]<EOE><SOP>"
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