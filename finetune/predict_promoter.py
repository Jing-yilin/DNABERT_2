import torch
import argparse
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer

def analyze_sequence(sequence):
    """分析 DNA 序列中的特征"""
    features = {}
    
    # 检查 -35 区域（TTGACA）
    minus35_region = sequence[0:6]
    features['-35_region'] = minus35_region
    features['-35_match'] = minus35_region == 'TTGACA'
    
    # 检查 -10 区域（TATAAT），通常在 -35 区域后 15-21 bp
    for i in range(15, 22):
        if i + 6 <= len(sequence):
            region = sequence[i:i+6]
            if 'TATAAT' in region:
                features['-10_region'] = region
                features['-10_position'] = i
                features['-10_match'] = True
                break
    else:
        features['-10_region'] = 'Not found'
        features['-10_position'] = -1
        features['-10_match'] = False
    
    # 计算 GC 含量
    gc_count = sequence.count('G') + sequence.count('C')
    features['gc_content'] = gc_count / len(sequence)
    
    return features

def predict_promoter(sequence, model_path, tokenizer_path, threshold=0.5):
    """预测序列是否为启动子"""
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    model.eval()
    
    # 将序列转换为模型输入
    inputs = tokenizer(sequence, return_tensors='pt')
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = (probabilities[0][1] > threshold).int()
    
    # 分析序列特征
    features = analyze_sequence(sequence)
    
    return {
        'sequence': sequence,
        'prediction': prediction.item(),
        'probability': probabilities[0][1].item(),
        'features': features
    }

def format_result(result):
    """格式化预测结果"""
    output = []
    output.append(f"\nSequence: {result['sequence']}")
    output.append(f"Prediction: {'Promoter' if result['prediction'] == 1 else 'Non-promoter'}")
    output.append(f"Confidence: {result['probability']:.4f}")
    output.append("\nSequence Features:")
    output.append(f"  -35 region: {result['features']['-35_region']} ({'Match' if result['features']['-35_match'] else 'No match'})")
    output.append(f"  -10 region: {result['features']['-10_region']} at position {result['features']['-10_position']} ({'Match' if result['features']['-10_match'] else 'No match'})")
    output.append(f"  GC content: {result['features']['gc_content']:.2%}")
    return "\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict promoter sequences using DNABERT-2')
    parser.add_argument('--sequence', type=str, help='DNA sequence to analyze')
    parser.add_argument('--model_path', type=str, default='output/ecoli_promoter/checkpoint-100',
                        help='Path to the model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='zhihan1996/DNABERT-2-117M',
                        help='Path to the tokenizer')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for promoter prediction')
    
    args = parser.parse_args()
    
    # 如果没有提供序列，使用示例序列
    if args.sequence is None:
        test_sequences = [
            # 已知的启动子序列
            "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCTACTAGAGAAAGAGGAGAAATACTAGATGCGTAAAGGAGAAGAACTTTTCACTGGAGTTGTC",
            # 随机 DNA 序列
            "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC",
            # 弱启动子序列
            "TTGACAGCTAGCTCAGTCCTAGGTATTATGCTAGCTACTAGAGAAAGAGGAGAAATACTAGATGCGTAAAGGAGAAGAACTTTTCACTGGAGTTGTC",
        ]
        
        print("\nAnalyzing example sequences...")
        for sequence in test_sequences:
            result = predict_promoter(sequence, args.model_path, args.tokenizer_path, args.threshold)
            print(format_result(result))
    else:
        result = predict_promoter(args.sequence, args.model_path, args.tokenizer_path, args.threshold)
        print(format_result(result))