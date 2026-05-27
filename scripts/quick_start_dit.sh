#!/bin/bash

# DiT 快速启动脚本
# 用法: bash scripts/quick_start_dit.sh [train|eval|compare]

set -e

MODE=${1:-train}

echo "========================================="
echo "DiT 快速启动脚本"
echo "========================================="
echo "模式: $MODE"
echo ""

case $MODE in
    train)
        echo "[1/2] 快速训练 DiT (10 步扩散)"
        python run/run_dit.py \
            --save_path saved_model/DiTtest \
            --epochs 500 \
            --batch_size 1000 \
            --n_timesteps 10 \
            --model_choice DiT1d \
            --attn_block causal \
            --predict_epsilon \
            --rtg_preference score
        
        echo ""
        echo "[2/2] 完整训练 DiT (100 步扩散)"
        echo "提示: 这将需要 2-3 小时"
        read -p "是否继续? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python run/run_dit.py \
                --save_path saved_model/DiTtest \
                --epochs 1000 \
                --batch_size 1000 \
                --n_timesteps 100 \
                --model_choice DiT1d \
                --attn_block causal \
                --predict_epsilon \
                --rtg_preference score \
                --save_every 100
        fi
        ;;
    
    eval)
        echo "评估 DiT 模型"
        
        # 修改策略为 DiT
        cat > bidding_train_env/strategy/__init__.py << 'EOF'
# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
EOF
        
        echo "已切换到 DiT 策略"
        python run/run_evaluate.py
        ;;
    
    compare)
        echo "三模型对比评估: DiT vs DT vs DD"
        python run/compare_dit_dt_dd.py
        ;;
    
    *)
        echo "未知模式: $MODE"
        echo "用法: bash scripts/quick_start_dit.sh [train|eval|compare]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "完成!"
echo "========================================="
