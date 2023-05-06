# MedQA-ChatGLM

# 使用

1. 安装环境
```python
pip install -r requirements.txt
```
2. LoRA微调
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
3. 推理
```python
CUDA_VISIBLE_DEVICES=0 python MedQA-ChatGLM/web_demo.py \
                              --checkpoint_dir med-lora/
```

# 结果

|微调方式|模型权重|
|:-|:-|
|LoRA|[MedQA-ChatGLM-LoRA](https://huggingface.co/wangrongsheng/MedQA-ChatGLM-LoRA)|
|P-Tuning V2|很快公布|
|Freeze|很快公布|

# 参考

1. https://github.com/zhangsheng93/cMedQA2
2. https://github.com/zhangsheng93/cMedQA
3. https://github.com/hiyouga/ChatGLM-Efficient-Tuning
4. https://github.com/THUDM/ChatGLM-6B
