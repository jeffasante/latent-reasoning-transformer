
import torch
import random
import matplotlib.pyplot as plt

from model import RecurrentGPT, Config, stoi, itos


# GENERATOR (2 Digits ONLY)
def generate_math_batch(batch_size=32):
    inputs, targets = [], []
    for _ in range(batch_size):
        # 2 DIGITS (e.g. 50+50=100)
        # This is easy enough to learn in 5 minutes
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        
        problem = f"{a}+{b}={a+b}"
        problem = problem.ljust(Config.block_size, ' ') 
        
        encoded = [stoi[p] for p in problem]
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        inputs.append(x)
        targets.append(y)
        
    return torch.stack(inputs).to(Config.device), torch.stack(targets).to(Config.device)

# TRAINING LOOP
model = RecurrentGPT().to(Config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Standard LR

print("Starting Training (2-3 Digit Addition)...")
print("We need Loss < 0.1 for this to work.\n")

for step in range(3001):
    xb, yb = generate_math_batch(64)
    train_depth = random.randint(1, 8) 
    
    logits, loss = model(xb, targets=yb, recur_depth=train_depth)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} (Depth {train_depth})")


# EVALUATION 
print("\nCHECKING PREDICTIONS ")
xb, yb = generate_math_batch(1)
with torch.no_grad():
    logits, _ = model(xb, recur_depth=8)
    preds = torch.argmax(logits, dim=2)

input_str = "".join([itos[i.item()] for i in xb[0]])
pred_str = "".join([itos[i.item()] for i in preds[0]])
# Note: Prediction is shifted by 1, so we align visually
print(f"Input: {input_str}")
print(f"Pred:  {pred_str}")


print('\nLets see accuracy at different depths:')
# EVALUATION (SMART VERSION) 
def evaluate_smart(depth):
    # Test on 200 samples
    total = 200
    correct = 0
    xb, yb = generate_math_batch(total)
    
    with torch.no_grad():
        logits, _ = model(xb, recur_depth=depth)
    
    preds = torch.argmax(logits, dim=2)
    
    for i in range(total):
        # Convert to strings
        target_str = "".join([itos[x.item()] for x in yb[i]])
        pred_str = "".join([itos[x.item()] for x in preds[i]])
        
        # Split at '=' to ignore the prompt
        if '=' in target_str:
            # We only check the part AFTER the equals sign
            target_ans = target_str.split('=')[1].strip()
            
            if '=' in pred_str:
                pred_ans = pred_str.split('=')[1].strip()
                
                # Check if answers match
                if target_ans == pred_ans:
                    correct += 1
            
    return correct / total

print("Calculating Accuracy on Answer Only (Ignoring Prompt Errors)...")
usable_accuracies = []
for d in [1, 2, 4, 8]:
    acc = evaluate_smart(d)
    usable_accuracies.append(acc*100)
    print(f"Depth {d:2d}: Accuracy = {acc*100:.1f}%")
    
# PLOTTING RESULTS
depths = ['Depth 1', 'Depth 2', 'Depth 4', 'Depth 8']
accuracy = usable_accuracies
colors = ['#ff9999', '#66b3ff', '#99ff99', '#66b3ff'] # Red for fail, Blue/Green for success

plt.figure(figsize=(8, 5))
bars = plt.bar(depths, accuracy, color=colors, edgecolor='black')

plt.title('Impact of Recurrent Depth on Arithmetic Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 110)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.savefig('demo_results.png', dpi=300)
print("Chart saved as demo_results.png!")
plt.show()