import json

with open(f"combined_ham.json", "rb") as h:
    ham = json.load(h)
    ham = ham["dataset"]
    ham_len = len(ham)

with open(f"combined_spam.json", "rb") as h:
    spam = json.load(h)
    spam = spam["dataset"]
    spam_len = len(spam)

print(f"Spam dataset length: {spam_len}")
print(f"Ham dataset length: {ham_len}")
print(f"Total dataset length: {spam_len + ham_len}")