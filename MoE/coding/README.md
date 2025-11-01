### 1. Start new training:

```
python train_coding.py --epochs 1000 --batch_size 16 --model_arch deepseek
```

### 2. Resume from latest checkpoint:

```
python train_coding.py --resume_from latest --epochs 1500
```

### 3. Resume from specific checkpoint:

```
python train_coding.py --resume_from checkpoints/checkpoint_epoch_500.pt --epochs 2000
```
### 4. Load saved model for inference only:

```
# In a separate script
model = load_model(MoEGPT, "saved_models/model_final.pt")
tokenizer = load_tokenizer("saved_models")
output = inference_from_saved("saved_models/model_final.pt", "saved_models/tokenizer.json", 
                             "def binary_search(arr, target):\n")
```