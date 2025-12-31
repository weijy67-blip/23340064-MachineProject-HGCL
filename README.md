HGCL: Heterogeneous Graph Contrastive Learning for Recommendation
基于超图对比学习的推荐系统框架

作者
中山大学人工智能学院23340064 韦江洋

项目简介
HGCL (Heterogeneous Graph Contrastive Learning) 是一个基于超图对比学习的推荐系统框架，该模型创新性地将超图神经网络与对比学习相结合，旨在解决传统推荐模型在数据稀疏性和复杂关系建模方面的局限性。本项目基于SIGIR 2023论文《Heterogeneous Graph Contrastive Learning for Recommendation》实现，并在ReChorus框架基础上进行了扩展和优化。

环境要求
torch==1.12.1
cudatoolkit==10.2.89
numpy==1.22.3
ipython==8.10.0
jupyter==1.0.0
tqdm==4.66.1
pandas==1.4.4
scikit-learn==1.1.3
scipy==1.7.3
pickle
yaml

安装依赖
pip install -r requirements.txt

项目结构
MachinProject
----ReChorus/
    ├── src/                    # 源代码目录
    │   ├── models/            # 模型实现
    │   │   ├── HGCL.py        # HGCL主模型
    │   │   ├── BaseModel.py   # 基础模型类
    │   │   └── ...
    │   ├── helpers/           # 数据加载和处理
    │   ├── runners/           # 训练和评估流程
    │   └── main.py           # 主入口文件
    ├── data/                  # 数据集目录
    │   ├── MovieLens_1M/     # MovieLens-1M数据集
    │   └── Grocery/          # Grocery数据集
    ├── scripts/              # 实用脚本
    │   ├── download_data.py  # 数据下载脚本
    │   ├── analysis.py       # 结果分析脚本
    │   └── visualization.py  # 可视化脚本
    ├── logs/                 # 训练日志
    ├── results/             # 实验结果
    ├── requirements.txt     # 依赖列表
    └── README.md           # 项目说明

使用方法
使用自定义数据集
python scripts/preprocess_data.py --dataset your_dataset --data_path path/to/your/data
模型训练
基本训练命令：
首先cd ReChorus跳转到ReChorus目录
python src/main.py --model_name HGCL --dataset Grocery_and_Gourmet_Food --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6  
python src/main.py --model_name HGCL --dataset MovieLens_1M --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6         
训练对比模型
python src/main.py --model_name BPRMF --dataset Grocery_and_Gourmet_Food --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6  
python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6  
python src/main.py --model_name BPRMF --dataset MovieLens_1M --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6         
python src/main.py --model_name LightGCN --dataset MovieLens_1M --epoch 5 --verbose 1 --emb_size 64 --lr 1e-3 --l2 1e-6 
完整参数配置：
python src/main.py \
    --model_name HGCL \
    --dataset MovieLens_1M \
    --epoch 100 \
    --emb_size 128 \
    --batch_size 1024 \
    --lr 0.001 \
    --l2 1e-6 \
    --tau 0.2 \
    --lam 0.1 \
    --hyper_k 3 \
    --n_layers 2 \
    --early_stop 10 \
    --gpu 0

引用
如果本项目对您的研究有帮助，请引用原始论文：
@inproceedings{chen2023heterogeneous,
  title={Heterogeneous Graph Contrastive Learning for Recommendation},
  author={Chen, Mengru and Huang, Chao and Xia, Lianghao and others},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  year={2023},
  pages={1--9}
}

许可证
本项目采用 MIT 许可证 - 查看 LICENSE文件了解详情。

致谢
感谢ReChorus框架提供的基础设施
感谢论文作者提供的思想启发和基线实现参考
感谢印鉴老师以及助教的幸苦付出

联系方式
如有问题或建议，请通过以下方式联系：
Email: weijy67@mail2.sysu.edu.cn
Issues: GitHub Issues
Bug Report: 请提供详细的重现步骤和环境信息
<div align="center">
如果觉得这个项目有帮助，请给个⭐️星标支持！
</div>
