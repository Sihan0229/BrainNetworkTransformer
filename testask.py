from omegaconf import OmegaConf
import yaml
# 加载配置
cfg = OmegaConf.load("/root/autodl-tmp/BrainNetworkTransformer/source/conf/config.yaml")

# 打印 optimizer 配置，验证其是否加载正确
print(cfg.optimizer)  # 应该输出来自 optimizer/adam.yaml 的字典配置

optimizer_dict = yaml.safe_load(cfg)

# 输出结果
print(optimizer_dict)
print(type(optimizer_dict))  # <class 'dict'>