import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ================= 配置区域 =================
# 1. 两个模型的路径
path_baseline = r"E:\meddet\runs\rsna_yolov8n_s0\weights\best.pt"
path_ours = r"E:\meddet\runs\rsna_yolov8n_lcbam_s0\weights\best.pt"

# 2. 验证集图片文件夹 (全量搜索！)
image_dir = r"E:\meddet\datasets\rsna\images\val"

# 3. 筛选严苛程度
# conf_thres: 置信度阈值。设为 0.25 是比较标准的。
# 如果你发现还是找不到差异，可以稍微调高到 0.3 或 0.35
conf_thres = 0.25 
# ===========================================

def find_differences():
    print(f"📂 正在扫描图片文件夹: {image_dir}")
    if not os.path.exists(image_dir):
        print("❌ 错误：找不到图片文件夹！")
        return

    # 获取所有图片
    all_imgs = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    # 为了快点找到，我们随机打乱一下，或者只看前500张
    # import random
    # random.shuffle(all_imgs)
    
    print(f"🔍 准备在 {len(all_imgs)} 张图片中寻找“差异样本”...")
    print("⏳ 正在加载模型...")
    
    try:
        model_base = YOLO(path_baseline)
        model_ours = YOLO(path_ours)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    count = 0
    max_save = 5 # 只需要找 5 张完美的图就够了

    for i, img_path in enumerate(all_imgs):
        try:
            # 预测
            res_base = model_base.predict(img_path, conf=conf_thres, verbose=False)[0]
            res_ours = model_ours.predict(img_path, conf=conf_thres, verbose=False)[0]

            # === 核心逻辑：找不同 ===
            # 情况 A: Baseline 一个都没检测到 (0)，但我们检测到了 (>=1) -> 完美素材！
            # 情况 B: Baseline 检测到的数量 比 我们少 (漏检了) -> 完美素材！
            
            len_base = len(res_base.boxes)
            len_ours = len(res_ours.boxes)

            is_golden_sample = False
            reason = ""

            if len_base == 0 and len_ours > 0:
                is_golden_sample = True
                reason = "Baseline漏检"
            elif len_base < len_ours:
                is_golden_sample = True
                reason = "Ours检出更多"
            
            # 如果是黄金样本，生成对比图
            if is_golden_sample:
                print(f"✨ 发现第 {count+1} 张神图！({os.path.basename(img_path)}) -> 原因: {reason}")
                
                # 绘图
                plot_base = res_base.plot(line_width=2, font_size=1)
                plot_ours = res_ours.plot(line_width=2, font_size=1)

                # 统一大小
                h, w = plot_base.shape[:2]
                plot_ours = cv2.resize(plot_ours, (w, h))

                # 加标题
                def add_header(img, text, color):
                    header = np.ones((50, w, 3), dtype=np.uint8) * 255 
                    cv2.putText(header, text, (int(w*0.05), 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    return np.vstack([header, img])

                final_base = add_header(plot_base, f"Baseline: {len_base} objects", (100, 100, 100))
                final_ours = add_header(plot_ours, f"Ours: {len_ours} objects", (0, 0, 255))

                # 拼接
                combined = np.hstack([final_base, final_ours])
                
                # 保存
                save_name = f"DIFF_{count+1}_{reason}.png"
                cv2.imwrite(save_name, combined)
                
                count += 1
                if count >= max_save:
                    print("\n🎉 已找到 5 张完美对比图！停止搜索。")
                    print("👉 请在当前目录查看 DIFF_x.png 系列图片。")
                    break
            
            # 每处理 100 张显示一下进度
            if (i+1) % 100 == 0:
                print(f"   已扫描 {i+1} 张，找到 {count} 张差异图...")

        except Exception as e:
            continue

    if count == 0:
        print("😓 扫描结束，没找到明显的漏检差异。")
        print("建议：尝试在代码第 18 行，把 conf_thres 稍微调高一点 (比如 0.35)，差异会更明显。")

if __name__ == "__main__":
    find_differences()