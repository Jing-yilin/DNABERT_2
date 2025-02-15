import torch
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer

def predict_promoter(sequence, model_path, tokenizer_path):
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    
    # 将序列转换为模型输入
    inputs = tokenizer(sequence, return_tensors='pt')
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)
    
    return {
        'prediction': prediction.item(),
        'probability': probabilities[0][prediction.item()].item()
    }

if __name__ == "__main__":
    # 测试序列
    test_sequences = [
        # 已知的启动子序列
        "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCTACTAGAGAAAGAGGAGAAATACTAGATGCGTAAAGGAGAAGAACTTTTCACTGGAGTTGTC",
        # 随机 DNA 序列
        "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC",
    ]
    
    model_path = "output/promoter/checkpoint-30"
    tokenizer_path = "zhihan1996/DNABERT-2-117M"
    
    print("\nPredicting promoter sequences...")
    for sequence in test_sequences:
        result = predict_promoter(sequence, model_path, tokenizer_path)
        print(f"\nSequence: {sequence[:50]}...")
        print(f"Prediction: {'Promoter' if result['prediction'] == 1 else 'Non-promoter'}")
        print(f"Confidence: {result['probability']:.4f}")