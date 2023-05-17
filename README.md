> 超大规模医学问答微调权重即将开源！！

# MedQA-ChatGLM

![](./images/model.png)

# 资源

|项目|数据集|底座模型|
|:-|:-|:-|
|[ChatMed](https://github.com/michael-wzhu/ChatMed)|[Consult](https://huggingface.co/michaelwzhu/ChatMed-Consult) 包含50w+在线问诊+ChatGPT回复，TCM中医药诊疗数据集未公开|LLaMA-7B|
|[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)|[HealthCareMagic-100k](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing) 包含100k+真实患者与医生对话数据集，[icliniq-10k](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view?usp=sharing) 包含10k+患者与医生对话数据集，[GenMedGPT-5k](https://drive.google.com/file/d/1nDTKZ3wZbZWTkFMBkxlamrzbNz0frugg/view?usp=sharing) 包含5k+由GPT生成的医患对话数据集|LLaMA-7B|
|[Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)|数据集未公开|ChatGLM-6B|
|[Huatuo-Llama-Med-Chinese](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)|数据集未公开|LLaMA-7B|
|[DoctorGLM](https://github.com/xionghonglin/DoctorGLM)|[CMD.](https://huggingface.co/datasets/wangrongsheng/CMD-merged) 、[MedDialog](https://huggingface.co/datasets/wangrongsheng/MedDialog-1.1M) 、ChatDoctor项目数据集|ChatGLM-6B|
|[MedicalGPT-zh](https://github.com/MediaBrain-SJTU/MedicalGPT-zh)|数据未开源|ChatGLM-6B|
|[Medical_NLP](https://github.com/FreedomIntelligence/Medical_NLP) <sup>1</sup>|-|-|

<sup>1</sup> 为相关医学的大模型资源

# 使用

## 1. 安装环境
```python
pip install -r requirements.txt
```
## 2. 微调

### 2.1 LoRA
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/finetune.py \
                              --do_train \
                              --dataset merged-cMedQA \
                              --finetuning_type lora \
                              --output_dir ./med-lora \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 256 \
                              --lr_scheduler_type cosine \
                              --logging_steps 500 \
                              --save_steps 1000 \
                              --learning_rate 5e-5 \
                              --num_train_epochs 10.0 \
                              --fp16
```
### 2.2 Freeze微调
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/finetune.py \
                              --do_train \
                              --dataset merged-cMedQA \
                              --finetuning_type freeze \
                              --output_dir ./med-freeze \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 256 \
                              --lr_scheduler_type cosine \
                              --logging_steps 500 \
                              --save_steps 1000 \
                              --learning_rate 5e-5 \
                              --num_train_epochs 10.0 \
                              --fp16
```

### 2.3 P-Turning V2

```python
CUDA_VISIBLE_DEVICES=1 python MedQA-ChatGLM/finetune.py \
                              --do_train --dataset merged-cMedQA \
                              --finetuning_type p_tuning \
                              --output_dir ./med-p_tuning \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 256 \
                              --lr_scheduler_type cosine \
                              --logging_steps 500 \
                              --save_steps 1000 \
                              --learning_rate 5e-5 \
                              --num_train_epochs 10.0 \
                              --fp16
```

更多参数信息，可以查看[docs/参数详解.md](https://github.com/WangRongsheng/MedQA-ChatGLM/blob/main/docs/%E5%8F%82%E6%95%B0%E8%AF%A6%E8%A7%A3.md) .

## 3. 推理

### 3.1 可视化
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/web_demo.py \
                              --checkpoint_dir med-lora/
                                              (med-freez/)
                                              (med-p_tuning/)
```

### 3.2 命令行
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/infer.py \
                              --checkpoint_dir med-lora/
                                              (med-freez/)
                                              (med-p_tuning/)
```

## 4. 合并（可选）

合并模型：
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/export_weights.py \
                              --finetuning_weights_path ./med-lora \
                              --save_weights_path ./save_lora
```

加载合并模型：
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/load_export_weights.py \
                              --save_weights_path ./save_lora
```

# 结果

|微调方式|模型权重|训练时长|训练轮次|
|:-|:-|:-|:-|
|LoRA|[MedQA-ChatGLM-LoRA](https://huggingface.co/wangrongsheng/MedQA-ChatGLM-LoRA)|28h|10|
|P-Tuning V2|[MedQA-ChatGLM-PTuningV2](https://huggingface.co/wangrongsheng/MedQA-ChatGLM-PTuningV2)|27h|10|
|Freeze|[MedQA-ChatGLM-Freeze](https://huggingface.co/wangrongsheng/MedQA-ChatGLM-Freeze)|28h|10|

<details>
  <summary>训练设置</summary>
  <p>* 实验是在Linux系统，A100 (1X, 80GB)上进行的</p>
</details>

# 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。本项目数据集绝大部分由模型生成，即使符合某些医学事实，也不能被用作实际医学诊断的依据。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

# 参考

1. https://github.com/zhangsheng93/cMedQA2
2. https://github.com/zhangsheng93/cMedQA
3. https://github.com/hiyouga/ChatGLM-Efficient-Tuning
4. https://github.com/THUDM/ChatGLM-6B
