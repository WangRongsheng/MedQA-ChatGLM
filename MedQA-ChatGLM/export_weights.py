from utils import load_pretrained, ModelArguments
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('-finetuning_weights_path', '--finetuning_weights_path', dest='fwp', type=str, required=True)
    parser.add_argument('-save_weights_path', '--save_weights_path', dest='swp', type=str, required=True)
    # 解析参数
    args = parser.parse_args()
    
    model_args = ModelArguments(checkpoint_dir=args.fwp)
    model, tokenizer = load_pretrained(model_args)
    # 保存合并权重
    model.base_model.model.save_pretrained(args.swp)
    # 保存 Tokenizer
    tokenizer.save_pretrained(args.swp)
    
    print('合并模型完成，保存在：', args.swp)