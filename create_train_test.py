import random
import json
random.seed(42)

with open("./combined_spam.json", "r") as f:
    data = json.load(f)

data = data["dataset"]
random.shuffle(data)

train_data_spam = data[:int(len(data) * 0.8)]
print(f"Train spam: {len(train_data_spam)}")
test_data_spam = data[int(len(data) * 0.8):]
print(f"Test spam: {len(test_data_spam)}")

with open("./train_data_spam.json", "w") as f:
    json.dump({"dataset": train_data_spam}, f)

with open("./test_data_spam.json", "w") as f:
    json.dump({"dataset": test_data_spam}, f)

with open("combined_ham.json", "r") as f:
    data = json.load(f)

data = data["dataset"]
random.shuffle(data)

train_data_ham = data[:int(len(data) * 0.8)]
print(f"Train ham: {len(train_data_ham)}")
test_data_ham = data[int(len(data) * 0.8):]
print(f"Test ham: {len(test_data_ham)}")

with open("./train_data_ham.json", "w") as f:
    json.dump({"dataset": train_data_ham}, f)

with open("./test_data_ham.json", "w") as f:
    json.dump({"dataset": test_data_ham}, f)

print("Done")