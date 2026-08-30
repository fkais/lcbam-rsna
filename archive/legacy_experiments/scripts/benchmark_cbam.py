import torch
import time
from ultralytics import YOLO

# ================= 配置区域 =================
# 这里填入你【原版 CBAM 模型】的权重路径
# 根据你的表格推断，路径应该是这个：
model_path = r"E:\meddet\runs\rsna_yolov8n_cbam_orig_s0\weights\last.pt"

# 图片大小 (保持和训练一致)
img_size = 512
# ===========================================

def benchmark():
    print(f"🚀 正在加载原版 CBAM 模型: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查路径是否正确？")
        return

    # 1. 获取参数量和 FLOPs
    print("\n📊 --- 模型复杂度 (Model Summary) ---")
    # 这行代码会自动打印 Params 和 FLOPs
    model.info(detailed=True)
    
    # 2. FPS 测速
    print(f"\n⏱️ --- 开始测速 (预热中...) ---")
    input_tensor = torch.zeros((1, 3, img_size, img_size)).to(device)
    
    # 预热
    for _ in range(10):
        model.predict(source=input_tensor, verbose=False)
        
    # 正式测试
    num_runs = 100
    print(f"🔥 正在运行 {num_runs} 次推理...")
    
    t_start = time.time()
    for _ in range(num_runs):
        model.predict(source=input_tensor, verbose=False)
    t_end = time.time()
    
    avg_time = (t_end - t_start) / num_runs * 1000 # 毫秒
    fps = 1000 / avg_time
    
    print("\n" + "="*40)
    print(f"   原版 CBAM 测速结果")
    print("="*40)
    print(f"   平均时间: {avg_time:.2f} ms")
    print(f"   FPS      : {fps:.2f}")
    print("="*40)
    print("👉 请把上面打印出的 [Params], [FLOPs], [FPS] 记下来告诉我！")

if __name__ == "__main__":
    benchmark()