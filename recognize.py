import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, Scale, Checkbutton
import cv2
import os

# 确保中文显示正常
import matplotlib

matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 加载预训练模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# 图像预处理类
class ImagePreprocessor:
    @staticmethod
    def preprocess(img, target_size=(28, 28)):
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')

        # 调整大小到模型输入尺寸
        img = img.resize(target_size, Image.LANCZOS)

        # 反色处理（MNIST是黑底白字，绘制的是白底黑字）
        img = Image.eval(img, lambda x: 255 - x)

        # 归一化处理
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 增加通道维度和批量维度
        img_tensor = torch.tensor(img_array).view(1, 1, *target_size)

        return img_tensor, img


# 手写数字识别应用
class HandwritingRecognitionApp:
    def __init__(self, root, model_path='cnn2.pkl'):
        self.root = root
        self.root.title("鼠标手写数字识别 - 画布优化版")
        self.root.geometry("700x800")
        self.root.resizable(True, True)

        # 加载模型
        self.cnn = CNN()
        if os.path.exists(model_path):
            self.cnn.load_state_dict(torch.load(model_path))
            self.cnn.eval()
            print("模型加载成功")
        else:
            print(f"模型文件 {model_path} 不存在，请先训练模型")
            self.cnn = None

        # 画布参数
        self.initial_canvas_size = 300
        self.model_input_size = 28
        self.canvas_size = self.initial_canvas_size
        self.line_width = 15
        self.show_grid = False
        self.grid_color = "gray"
        self.grid_spacing = 28  # 对应MNIST的28x28尺寸

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        # 顶部控制框架
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill="x", pady=5)

        # 画布大小调节
        ttk.Label(self.control_frame, text="画布大小:").pack(side="left", padx=5)
        self.canvas_size_scale = Scale(self.control_frame, from_=100, to=500, orient="horizontal",
                                       length=200, command=self.update_canvas_size)
        self.canvas_size_scale.set(self.initial_canvas_size)
        self.canvas_size_scale.pack(side="left", padx=5)
        ttk.Label(self.control_frame, text=f"{self.initial_canvas_size}px").pack(side="left", padx=2)

        # 线条宽度调节
        ttk.Label(self.control_frame, text="线条宽度:").pack(side="left", padx=15)
        self.line_width_scale = Scale(self.control_frame, from_=5, to=30, orient="horizontal",
                                      length=100, command=self.update_line_width)
        self.line_width_scale.set(self.line_width)
        self.line_width_scale.pack(side="left", padx=5)
        ttk.Label(self.control_frame, text=f"{self.line_width}px").pack(side="left", padx=2)

        # 网格辅助线
        self.grid_var = tk.BooleanVar(value=self.show_grid)
        self.grid_checkbox = Checkbutton(self.control_frame, text="显示网格", variable=self.grid_var,
                                         command=self.toggle_grid)
        self.grid_checkbox.pack(side="right", padx=10)

        # 画布框架
        self.canvas_frame = ttk.LabelFrame(self.main_frame, text="请用鼠标绘制数字", padding="5")
        self.canvas_frame.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg="white", highlightthickness=2, highlightbackground="gray")
        self.canvas.pack(fill="both", expand=True)

        # 创建PIL图像用于存储绘制内容
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.pil_image)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Configure>", self.on_canvas_configure)  # 处理窗口调整大小

        # 按钮框架
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(btn_frame, text="清除", command=self.clear_canvas).pack(side="left", padx=5, fill="y")
        ttk.Button(btn_frame, text="识别", command=self.recognize).pack(side="left", padx=5, fill="y")
        ttk.Button(btn_frame, text="重置画布", command=self.reset_canvas).pack(side="left", padx=5, fill="y")

        # 识别结果框架
        result_frame = ttk.LabelFrame(self.main_frame, text="识别结果", padding="10")
        result_frame.pack(fill="both", expand=True, pady=10)

        ttk.Label(result_frame, text="预测数字:").pack(anchor="w")
        self.result_label = ttk.Label(result_frame, text="--", font=("Arial", 60), foreground="blue")
        self.result_label.pack(pady=10)

        ttk.Label(result_frame, text="置信度分布:").pack(anchor="w")
        self.confidence_text = tk.Text(result_frame, height=3, width=50, wrap=tk.WORD)
        self.confidence_text.pack(fill="both", pady=5)
        self.confidence_text.config(state="disabled")

        # 处理后图像框架
        process_frame = ttk.LabelFrame(self.main_frame, text="处理后的图像", padding="10")
        process_frame.pack(fill="both", expand=True, pady=10)

        self.processed_img_label = ttk.Label(process_frame)
        self.processed_img_label.pack(fill="both", expand=True)

        # 绘制状态
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.preprocessor = ImagePreprocessor()
        self.tk_image = None
        self.canvas_dirty = False  # 标记画布是否有修改

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        self.canvas_dirty = True

    def draw_line(self, event):
        if not self.drawing:
            return

        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在PIL图像上绘制
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=self.line_width)
            # 更新画布显示
            self.update_canvas()
        self.last_x, self.last_y = x, y
        self.canvas_dirty = True

    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def on_canvas_configure(self, event):
        # 处理窗口调整大小时的画布重绘
        if self.canvas_dirty:
            self.update_canvas()

    def update_canvas_size(self, size):
        # 更新画布大小
        self.canvas_size = int(size)
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)
        # 调整PIL图像大小
        new_pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        if self.pil_image.size != (self.canvas_size, self.canvas_size):
            # 如果尺寸变化，缩放原有图像
            if self.pil_image.size[0] > 0 and self.pil_image.size[1] > 0:
                scaled_image = self.pil_image.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
                new_pil_image.paste(scaled_image)
        self.pil_image = new_pil_image
        self.draw = ImageDraw.Draw(self.pil_image)
        self.update_canvas()
        self.canvas_size_scale.config(label=f"{self.canvas_size}px")
        self.canvas_dirty = True

    def update_line_width(self, width):
        # 更新线条宽度
        self.line_width = int(width)
        self.line_width_scale.config(label=f"{self.line_width}px")

    def toggle_grid(self):
        # 切换网格显示
        self.show_grid = self.grid_var.get()
        self.update_canvas()

    def update_canvas(self):
        # 绘制网格
        if self.show_grid:
            grid_pil = self.pil_image.copy()
            draw_grid = ImageDraw.Draw(grid_pil)
            # 绘制垂直网格线
            for x in range(0, self.canvas_size, self.grid_spacing):
                draw_grid.line([x, 0, x, self.canvas_size], fill=self.grid_color, width=1)
            # 绘制水平网格线
            for y in range(0, self.canvas_size, self.grid_spacing):
                draw_grid.line([0, y, self.canvas_size, y], fill=self.grid_color, width=1)
            self.tk_image = ImageTk.PhotoImage(grid_pil)
        else:
            self.tk_image = ImageTk.PhotoImage(self.pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def reset_canvas(self):
        # 重置画布到初始状态
        self.canvas_size_scale.set(self.initial_canvas_size)
        self.update_canvas_size(self.initial_canvas_size)
        self.line_width_scale.set(self.line_width)
        self.update_line_width(self.line_width)
        self.grid_var.set(self.show_grid)
        self.toggle_grid()
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.update_canvas()
        self.result_label.config(text="--")
        self.confidence_text.config(state="normal")
        self.confidence_text.delete(1.0, tk.END)
        self.confidence_text.config(state="disabled")
        self.processed_img_label.config(image="")
        self.canvas_dirty = False

    def recognize(self):
        if self.cnn is None:
            self.result_label.config(text="模型未加载")
            return

        try:
            # 预处理图像
            img_tensor, processed_img = self.preprocessor.preprocess(self.pil_image)

            # 显示处理后的图像
            processed_photo = ImageTk.PhotoImage(processed_img)
            self.processed_img_label.config(image=processed_photo)
            self.processed_img_label.photo = processed_photo  # 保持引用

            # 模型预测
            with torch.no_grad():
                output = self.cnn(img_tensor)
                probabilities = torch.softmax(output, dim=1).numpy()[0]
                predicted = np.argmax(probabilities)

                # 显示结果
                self.result_label.config(text=str(predicted))

                # 显示置信度分布（按从高到低排序）
                self.confidence_text.config(state="normal")
                self.confidence_text.delete(1.0, tk.END)
                for i, prob in sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True):
                    self.confidence_text.insert(tk.END, f"数字 {i}: {prob * 100:.2f}%\n")
                self.confidence_text.config(state="disabled")

        except Exception as e:
            print(f"识别错误: {e}")
            self.result_label.config(text="识别错误")


# 主函数
def main():
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()