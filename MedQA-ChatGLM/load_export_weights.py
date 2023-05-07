from transformers import AutoConfig, AutoModel, AutoTokenizer
import argparse

def get_model(load_path):
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(load_path, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(load_path, config=config, trust_remote_code=True).half().cuda()
    model = model.eval()
    
    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('-save_weights_path', '--save_weights_path', dest='swp', type=str, required=True)
    # 解析参数
    args = parser.parse_args()
    
    tokenizer, model = get_model(args.swp)
    print(model)
    print('加载完成')

