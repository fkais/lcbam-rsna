import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import shutil

# ================= 配置区域 =================
# 1. 三个模型的路径 (确认无误)
path_base = r"E:\meddet\runs\rsna_yolov8n_s0\weights\best.pt"
path_cbam = r"E:\meddet\runs\rsna_yolov8n_cbam_orig_s0\weights\best.pt"
path_ours = r"E:\meddet\runs\rsna_yolov8n_lcbam_s0\weights\best.pt"

# 2. 验证集图片文件夹 (全量扫描！)
#    之前确认过你的验证集图片在这里:
val_images_dir = r"E:\meddet\datasets\rsna\images\val"

# 3. 筛选严苛程度
#    conf=0.25 是默认值。
#    如果你跑出来图片太多，可以提高到 0.30；如果太少，降低到 0.20
conf_thres = 0.25 

# 4. 结果保存位置
save_dir = r"E:\meddet\golden_samples"
# ===========================================

def add_header(img, text, color):
    h, w = img.shape[:2]
    header = np.ones((50, w, 3), dtype=np.uint8) * 255
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - tw) // 2
    cv2.putText(header, text, (tx, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return np.vstack([header, img])

def find_golden_samples():
    print(f"🚀 启动全自动找茬脚本...")
    print(f"📂 扫描目录: {val_images_dir}")
    
    if not os.path.exists(val_images_dir):
        print("❌ 错误：找不到图片文件夹！")
        return

    # 清空并重建保存目录
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"📂 结果将保存在: {save_dir}")

    # 获取所有图片列表
    all_imgs = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.png'))]
    total_imgs = len(all_imgs)
    print(f"📊 共发现 {total_imgs} 张图片，准备加载模型...")

    # 加载模型
    try:
        model_base = YOLO(path_base)
        model_cbam = YOLO(path_cbam)
        model_ours = YOLO(path_ours)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("🔥 开始扫描 (可能需要几分钟，请耐心等待)...")
    
    found_count = 0
    max_save = 20 # 限制保存数量，找够20张神图就收工，免得硬盘塞满

    for i, img_name in enumerate(all_imgs):
        img_path = os.path.join(val_images_dir, img_name)
        
        try:
            # 预测
            res_base = model_base.predict(img_path, conf=conf_thres, verbose=False)[0]
            res_cbam = model_cbam.predict(img_path, conf=conf_thres, verbose=False)[0]
            res_ours = model_ours.predict(img_path, conf=conf_thres, verbose=False)[0]

            # 获取检测框数量
            n_base = len(res_base.boxes)
            n_cbam = len(res_cbam.boxes)
            n_ours = len(res_ours.boxes)

            # === 核心筛选逻辑：必须满足“别人不行我行” ===
            is_golden = False
            tag = ""

            # 情况1：完美绝杀 (Base=0, CBAM=0, Ours>0)
            if n_base == 0 and n_cbam == 0 and n_ours > 0:
                is_golden = True
                tag = "Perfect_Kill"
            
            # 情况2：各种漏检 (Base漏了 或 CBAM漏了，且 Ours 没漏)
            # 这种图也可以用，用来攻击特定对手
            elif (n_base == 0 or n_cbam == 0) and n_ours > 0:
                # 只有当 Ours 检测到的比 Base 多时才算赢
                if n_ours > n_base: 
                    is_golden = True
                    tag = "Partial_Win"

            # 打印进度 (每100张显示一次)
            if (i+1) % 100 == 0:
                print(f"   进度: {i+1}/{total_imgs} | 已找到神图: {found_count}")

            # 如果是神图，生成对比图并保存
            if is_golden:
                print(f"   ✨ 发现 [{tag}]! 文件: {img_name}")
                
                # 绘图
                p_base = res_base.plot(line_width=2, font_size=1)
                p_cbam = res_cbam.plot(line_width=2, font_size=1)
                p_ours = res_ours.plot(line_width=2, font_size=1)

                # 拼接
                h, w = p_base.shape[:2]
                p_cbam = cv2.resize(p_cbam, (w, h))
                p_ours = cv2.resize(p_ours, (w, h))

                f_base = add_header(p_base, f"Baseline ({n_base})", (100,100,100))
                f_cbam = add_header(p_cbam, f"CBAM ({n_cbam})", (0,165,255))
                f_ours = add_header(p_ours, f"Ours ({n_ours})", (0,0,255))

                combined = np.hstack([f_base, f_cbam, f_ours])
                
                # 保存
                save_name = os.path.join(save_dir, f"{found_count+1}_{tag}_{img_name}")
                cv2.imwrite(save_name, combined)
                
                found_count += 1
                
                if found_count >= max_save:
                    print("\n🛑 已找够 20 张神图，提前收工！")
                    break

        except Exception as e:
            continue

    print("\n" + "="*40)
    print(f"🎉 扫描结束！共找到 {found_count} 张符合条件的对比图。")
    print(f"📂 请立刻打开文件夹查看: {save_dir}")
    print("👉 挑选一张最让你满意的，重命名为 Figure4.png！")
    print("="*40)

if __name__ == "__main__":
    find_golden_samples()