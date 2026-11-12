from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk

from predict import load_prediction_bundle, predict_image
from utils import DEFAULT_MODEL_PATH


APP_WIDTH = 1120
APP_HEIGHT = 620

CATEGORY_LABELS = {
    "fruit": "水果",
    "vegetable": "蔬菜",
    "unknown": "未知",
}

CLASS_LABELS = {
    "apple": "苹果",
    "banana": "香蕉",
    "beetroot": "甜菜根",
    "bell pepper": "甜椒",
    "cabbage": "卷心菜",
    "capsicum": "辣椒",
    "carrot": "胡萝卜",
    "cauliflower": "花椰菜",
    "chilli pepper": "辣椒",
    "corn": "玉米",
    "cucumber": "黄瓜",
    "eggplant": "茄子",
    "garlic": "大蒜",
    "ginger": "姜",
    "grapes": "葡萄",
    "jalepeno": "墨西哥辣椒",
    "kiwi": "猕猴桃",
    "lemon": "柠檬",
    "lettuce": "生菜",
    "mango": "芒果",
    "onion": "洋葱",
    "orange": "橙子",
    "paprika": "辣椒粉",
    "pear": "梨",
    "peach": "桃子",
    "peas": "豌豆",
    "pineapple": "菠萝",
    "pomegranate": "石榴",
    "potato": "马铃薯",
    "raddish": "萝卜",
    "soy beans": "大豆",
    "spinach": "菠菜",
    "sweetcorn": "甜玉米",
    "sweetpotato": "甘薯",
    "tomato": "番茄",
    "turnip": "芜菁",
    "watermelon": "西瓜",
}

COLORS = {
    "green": "#1f8f3c",
    "green_dark": "#0f6f2b",
    "green_soft": "#eaf7ec",
    "orange": "#ff7a21",
    "orange_dark": "#e95f11",
    "red": "#e52f36",
    "red_soft": "#fde7e8",
    "text": "#333333",
    "muted": "#666666",
    "line": "#dfe8de",
    "line_green": "#9bd3aa",
    "bg": "#fbfdf9",
    "panel": "#ffffff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a GUI for fruit and vegetable recognition."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a saved .pth checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override checkpoint confidence threshold.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to show.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device.",
    )
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value:.2%}"


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


def class_label(class_name: str) -> str:
    chinese = CLASS_LABELS.get(class_name)
    if chinese:
        return f"{chinese} ({class_name})"
    return class_name


def contain_size(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))


def build_result_text(result: dict) -> tuple[str, str, str]:
    category = category_label(result["broad_category"])
    display_class = class_label(result["predicted_class"])
    confidence = percent(result["confidence"])
    summary = f"大类：{category}    具体类别：{display_class}    置信度：{confidence}"

    if result["is_uncertain"]:
        status = "低置信度：图片可能不属于已知水果/蔬菜类别"
    else:
        status = "识别完成，结果可信"

    lines = [
        f"图片路径：{result['image']}",
        f"判断阈值：{result['threshold']:.2f}    状态提示：{status}",
        "",
        "前三预测：",
    ]
    for index, item in enumerate(result["top_k"], start=1):
        lines.append(f"{index}. {class_label(item['class_name'])}：{percent(item['confidence'])}")
    return summary, status, "\n".join(lines)


def create_rounded_photo(image_path: Path, size: tuple[int, int], radius: int = 24) -> ImageTk.PhotoImage:
    width, height = size
    image = Image.open(image_path).convert("RGB")
    new_width, new_height = contain_size(image.size, size)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    rounded = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    x = (width - new_width) // 2
    y = (height - new_height) // 2
    rounded.paste(image, (x, y))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    rounded.putalpha(mask)
    return ImageTk.PhotoImage(rounded)


class FruitVegetableGUI:
    def __init__(self, root, model_path, threshold, top_k, device):
        if not hasattr(tk.Canvas, "create_round_rectangle"):
            tk.Canvas.create_round_rectangle = create_round_rectangle

        self.root = root
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.top_k = top_k
        self.device = device
        self.bundle = None
        self.selected_image: Path | None = None
        self.preview_photo = None
        self.asset_images = {}

        self.result = None
        self.status_text = "低置信度提示：若最高置信度低于阈值，可能不属于已知类别"
        self.file_text = "暂无选择图片"
        self.category_text = "--"
        self.class_text = "--"
        self.confidence_value = 0.0
        self.top_predictions = []
        self.is_busy = False

        self.choose_button_box = (70, 493, 293, 549)
        self.predict_button_box = (322, 493, 545, 549)
        self.close_box = (1060, 24, 1090, 54)

        self.root.title("水果蔬菜识别系统")
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.minsize(APP_WIDTH, APP_HEIGHT)
        self.root.resizable(False, False)
        self.load_ui_assets()

        self.canvas = tk.Canvas(
            self.root,
            width=APP_WIDTH,
            height=APP_HEIGHT,
            bg=COLORS["bg"],
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        self.redraw()

    def load_ui_assets(self):
        asset_dir = Path("assets")
        for key, filename in {
            "title_fruits": "title_fruits.png",
            "result_bowl": "result_bowl.png",
            "corner_tomato": "corner_tomato.png",
        }.items():
            path = asset_dir / filename
            if path.exists():
                self.asset_images[key] = ImageTk.PhotoImage(Image.open(path).convert("RGBA"))

    def rounded_rect(self, x1, y1, x2, y2, radius, fill, outline="", width=1, tags=None):
        return self.canvas.create_round_rectangle(
            x1,
            y1,
            x2,
            y2,
            radius=radius,
            fill=fill,
            outline=outline,
            width=width,
            tags=tags,
        )

    def redraw(self):
        self.canvas.delete("all")
        self.draw_background()
        self.draw_header()
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_bottom_bar()

    def draw_background(self):
        self.canvas.create_rectangle(0, 0, APP_WIDTH, APP_HEIGHT, fill="#fdfefd", outline="")
        self.canvas.create_line(0, 0, APP_WIDTH, 0, fill="#e5e5e5")

    def draw_header(self):

        if "title_fruits" in self.asset_images:
            self.canvas.create_image(340, 19, image=self.asset_images["title_fruits"], anchor="nw")
        else:
            self.draw_fruit_cluster(345, 48, scale=1.0)
        self.canvas.create_text(
            420,
            55,
            text="水果蔬菜识别系统",
            fill=COLORS["green_dark"],
            font=("Microsoft YaHei UI", 30, "bold"),
            anchor="w",
        )

        self.canvas.create_text(965, 33, text="−", fill="#333333", font=("Arial", 23))
        self.canvas.create_rectangle(1016, 24, 1029, 37, outline="#333333", width=2)
        self.canvas.create_text(1078, 33, text="×", fill="#333333", font=("Arial", 23))

    def draw_left_panel(self):
        self.rounded_rect(64, 100, 550, 407, 24, "#fbfffb", COLORS["line_green"], width=1)
        self.canvas.create_rectangle(84, 116, 530, 389, outline="#b7d9bd", dash=(2, 2))
        self.rounded_rect(88, 122, 526, 385, 22, "#f7fbf6", "", width=0)

        if self.preview_photo:
            self.canvas.create_image(88, 122, image=self.preview_photo, anchor="nw")
        else:
            self.canvas.create_text(
                307,
                252,
                text="请选择一张水果或蔬菜图片",
                fill="#7c897c",
                font=("Microsoft YaHei UI", 16, "bold"),
            )
            self.canvas.create_text(
                307,
                285,
                text="支持 JPG / PNG / BMP / WEBP",
                fill="#8f9b8f",
                font=("Microsoft YaHei UI", 11),
            )

        self.rounded_rect(70, 424, 545, 477, 8, "#ffffff", "#c9c9c9", width=1)
        self.canvas.create_rectangle(242, 440, 267, 464, outline="#a8a8a8", width=2)
        self.canvas.create_line(246, 459, 253, 452, 259, 457, 265, 449, fill="#a8a8a8", width=2)
        self.canvas.create_text(
            284,
            451,
            text=self.file_text,
            fill="#777777",
            font=("Microsoft YaHei UI", 14, "bold"),
            anchor="w",
        )

        self.draw_button(
            self.choose_button_box,
            "选择图片",
            COLORS["green"],
            "#1f7e36",
            icon="folder",
        )
        self.draw_button(
            self.predict_button_box,
            "开始识别",
            COLORS["orange"],
            COLORS["orange_dark"],
            icon="play",
        )

    def draw_right_panel(self):
        x1, y1, x2, y2 = 618, 105, 1085, 282
        self.rounded_rect(x1, y1, x2, y2, 14, "#ffffff", COLORS["line"], width=1)
        self.rounded_rect(620, 89, 740, 126, 10, COLORS["green"], COLORS["green"], width=1)
        self.canvas.create_text(
            680,
            108,
            text="识别结果",
            fill="white",
            font=("Microsoft YaHei UI", 17, "bold"),
        )
        self.canvas.create_line(637, 140, 1065, 140, fill="#eeeeee")
        self.canvas.create_text(640, 168, text="大类", fill=COLORS["text"], font=("Microsoft YaHei UI", 14, "bold"), anchor="w")
        self.canvas.create_text(
            742,
            178,
            text=self.category_text,
            fill=COLORS["green_dark"],
            font=("Microsoft YaHei UI", 38, "bold"),
            anchor="w",
        )
        if "result_bowl" in self.asset_images:
            self.canvas.create_image(968, 144, image=self.asset_images["result_bowl"], anchor="nw")
        else:
            self.draw_fruit_bowl(975, 188)
        self.canvas.create_line(637, 218, 1065, 218, fill="#eeeeee")
        self.canvas.create_text(640, 249, text="具体类别", fill=COLORS["text"], font=("Microsoft YaHei UI", 14, "bold"), anchor="w")
        self.canvas.create_text(
            750,
            252,
            text=self.class_text,
            fill=COLORS["red"],
            font=("Microsoft YaHei UI", 25, "bold"),
            anchor="w",
        )

        self.rounded_rect(618, 294, 1085, 386, 12, "#ffffff", COLORS["line"], width=1)
        self.canvas.create_text(640, 322, text="置信度", fill=COLORS["text"], font=("Microsoft YaHei UI", 14, "bold"), anchor="w")
        self.canvas.create_text(
            750,
            333,
            text=percent(self.confidence_value) if self.confidence_value else "--",
            fill=COLORS["red"],
            font=("Microsoft YaHei UI", 29, "bold"),
            anchor="w",
        )
        self.rounded_rect(748, 356, 1064, 372, 8, "#f7dada", "", width=0)
        fill_width = int(316 * max(0, min(self.confidence_value, 1)))
        if fill_width:
            self.rounded_rect(748, 356, 748 + fill_width, 372, 8, COLORS["red"], "", width=0)

        self.rounded_rect(618, 397, 1085, 552, 12, "#ffffff", COLORS["line"], width=1)
        self.canvas.create_text(640, 422, text="前三预测", fill=COLORS["text"], font=("Microsoft YaHei UI", 14, "bold"), anchor="w")
        self.draw_top_predictions()

    def draw_top_predictions(self):
        if not self.top_predictions:
            rows = [
                ("1", "等待识别", "--", COLORS["red"]),
                ("2", "等待识别", "--", COLORS["orange"]),
                ("3", "等待识别", "--", "#ffb32c"),
            ]
        else:
            rows = []
            rank_colors = [COLORS["red"], COLORS["orange"], "#ffb32c"]
            for idx, item in enumerate(self.top_predictions[:3], start=1):
                rows.append(
                    (
                        str(idx),
                        class_label(item["class_name"]),
                        percent(item["confidence"]),
                        rank_colors[idx - 1],
                    )
                )

        y_positions = [459, 496, 533]
        for y, (rank, name, score, color) in zip(y_positions, rows):
            self.canvas.create_oval(640, y - 11, 664, y + 13, fill=color, outline="")
            self.canvas.create_text(652, y + 1, text=rank, fill="white", font=("Arial", 12, "bold"))
            self.canvas.create_text(
                688,
                y,
                text=name,
                fill=color if rank == "1" else "#8a5a4b",
                font=("Microsoft YaHei UI", 13, "bold"),
                anchor="w",
            )
            self.canvas.create_text(
                1058,
                y,
                text=score,
                fill=color if rank == "1" else "#8a5a4b",
                font=("Microsoft YaHei UI", 12, "bold"),
                anchor="e",
            )

    def draw_bottom_bar(self):
        self.canvas.create_rectangle(0, 566, APP_WIDTH, APP_HEIGHT, fill="#eef8ee", outline="#d4ead7")
        if "corner_tomato" in self.asset_images:
            self.canvas.create_image(-23, 526, image=self.asset_images["corner_tomato"], anchor="nw")
        else:
            self.draw_corner_tomato()
        self.canvas.create_oval(174, 582, 190, 598, fill=COLORS["green"], outline="")
        self.canvas.create_polygon(182, 589, 187, 586, 185, 593, fill="#dff5df", outline="")
        self.canvas.create_text(
            205,
            592,
            text=self.status_text,
            fill=COLORS["green_dark"],
            font=("Microsoft YaHei UI", 13, "bold"),
            anchor="w",
        )
        threshold = self.threshold if self.threshold is not None else 0.65
        self.canvas.create_text(
            930,
            592,
            text=f"阈值：{threshold:.2f}",
            fill=COLORS["green_dark"],
            font=("Microsoft YaHei UI", 13, "bold"),
            anchor="w",
        )
        self.draw_gear(1057, 592)

    def draw_button(self, box, text, fill, outline, icon):
        x1, y1, x2, y2 = box
        self.rounded_rect(x1 + 2, y1 + 4, x2 + 2, y2 + 4, 8, "#d3d3d3", "", width=0)
        self.rounded_rect(x1, y1, x2, y2, 8, fill, outline, width=1)
        icon_x = x1 + 50
        center_y = (y1 + y2) // 2
        if icon == "folder":
            self.canvas.create_rectangle(icon_x - 14, center_y - 6, icon_x + 16, center_y + 13, fill="white", outline="")
            self.canvas.create_rectangle(icon_x - 13, center_y - 12, icon_x + 2, center_y - 5, fill="white", outline="")
        else:
            self.canvas.create_oval(icon_x - 14, center_y - 14, icon_x + 14, center_y + 14, fill="white", outline="")
            self.canvas.create_polygon(icon_x - 4, center_y - 8, icon_x - 4, center_y + 8, icon_x + 8, center_y, fill=fill, outline="")
        self.canvas.create_text(
            x1 + 80,
            center_y,
            text=text,
            fill="white",
            font=("Microsoft YaHei UI", 22, "bold"),
            anchor="w",
        )

    def draw_fruit_cluster(self, x, y, scale=1.0):
        s = scale
        self.canvas.create_oval(x, y - 17 * s, x + 32 * s, y + 15 * s, fill="#e63832", outline="")
        self.canvas.create_oval(x + 22 * s, y - 4 * s, x + 54 * s, y + 26 * s, fill="#ff8423", outline="")
        self.canvas.create_oval(x + 47 * s, y - 14 * s, x + 76 * s, y + 17 * s, fill="#f0b12d", outline="")
        self.canvas.create_oval(x + 17 * s, y - 33 * s, x + 37 * s, y - 15 * s, fill="#2faa46", outline="")
        self.canvas.create_oval(x + 52 * s, y - 30 * s, x + 70 * s, y - 13 * s, fill="#2faa46", outline="")
        self.canvas.create_line(x + 34 * s, y - 18 * s, x + 37 * s, y - 34 * s, fill="#8b5a2b", width=2)
        self.canvas.create_line(x + 61 * s, y - 13 * s, x + 63 * s, y - 27 * s, fill="#8b5a2b", width=2)

    def draw_fruit_bowl(self, x, y):
        self.canvas.create_oval(x - 20, y - 30, x + 18, y + 5, fill="#e43a31", outline="")
        self.canvas.create_oval(x + 5, y - 33, x + 42, y + 3, fill="#ff8625", outline="")
        self.canvas.create_oval(x + 32, y - 27, x + 66, y + 6, fill="#f1ae2b", outline="")
        self.canvas.create_arc(x - 18, y - 2, x + 76, y + 58, start=180, extent=180, fill="#c98337", outline="#c98337")
        self.canvas.create_arc(x - 11, y + 7, x + 69, y + 51, start=180, extent=180, fill="#f0b15f", outline="")
        self.canvas.create_oval(x + 12, y - 52, x + 29, y - 36, fill="#2e9b3e", outline="")
        self.canvas.create_oval(x + 43, y - 49, x + 60, y - 34, fill="#2e9b3e", outline="")

    def draw_corner_tomato(self):
        self.canvas.create_oval(-19, 558, 60, 637, fill="#ef3a28", outline="")
        self.canvas.create_oval(-7, 571, 47, 625, outline="#ff7b58", width=4)
        self.canvas.create_line(19, 584, 33, 612, fill="#ff8b6d", width=3)
        self.canvas.create_line(36, 585, 11, 611, fill="#ff8b6d", width=3)
        self.canvas.create_oval(16, 580, 25, 589, fill="#2e9b3e", outline="")
        self.canvas.create_polygon(22, 548, 38, 517, 42, 557, fill="#75bd43", outline="")
        self.canvas.create_polygon(0, 564, 9, 525, 25, 562, fill="#7ec64d", outline="")

    def draw_gear(self, x, y):
        self.canvas.create_oval(x - 11, y - 11, x + 11, y + 11, fill=COLORS["green"], outline="")
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#eef8ee", outline="")
        for dx, dy in [(0, -15), (0, 15), (-15, 0), (15, 0)]:
            self.canvas.create_rectangle(x + dx - 2, y + dy - 2, x + dx + 2, y + dy + 2, fill=COLORS["green"], outline="")

    def on_click(self, event):
        x, y = event.x, event.y
        if self.inside(self.choose_button_box, x, y):
            self.choose_image()
        elif self.inside(self.predict_button_box, x, y):
            self.start_prediction()
        elif self.inside(self.close_box, x, y):
            self.root.destroy()

    def on_motion(self, event):
        x, y = event.x, event.y
        if self.inside(self.choose_button_box, x, y) or self.inside(self.predict_button_box, x, y):
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="")

    @staticmethod
    def inside(box, x, y):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return

        self.selected_image = Path(file_path)
        self.file_text = self.selected_image.name
        self.preview_photo = create_rounded_photo(self.selected_image, (438, 263))
        self.status_text = "图片已载入，请点击“开始识别”"
        self.category_text = "--"
        self.class_text = "--"
        self.confidence_value = 0.0
        self.top_predictions = []
        self.redraw()

    def start_prediction(self):
        if self.selected_image is None:
            messagebox.showwarning("提示", "请先选择一张图片。")
            return
        if self.is_busy:
            return

        self.is_busy = True
        self.status_text = "正在加载模型并识别图片，请稍等..."
        self.redraw()
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()

    def run_prediction(self):
        try:
            if self.bundle is None:
                self.bundle = load_prediction_bundle(self.model_path, device_name=self.device)
            result = predict_image(
                self.selected_image,
                self.bundle,
                top_k=self.top_k,
                threshold=self.threshold,
            )
        except Exception as exc:
            self.root.after(0, self.show_error, exc)
            return
        self.root.after(0, self.show_result, result)

    def show_result(self, result):
        _summary, status, _detail = build_result_text(result)
        self.result = result
        self.category_text = category_label(result["broad_category"])
        self.class_text = class_label(result["predicted_class"])
        self.confidence_value = result["confidence"]
        self.top_predictions = result["top_k"]
        self.status_text = status
        self.is_busy = False
        self.redraw()

    def show_error(self, exc):
        self.is_busy = False
        self.status_text = "识别失败，请检查模型文件或图片格式"
        self.redraw()
        messagebox.showerror("识别失败", str(exc))


def main():
    if not hasattr(tk.Canvas, "create_round_rectangle"):
        tk.Canvas.create_round_rectangle = create_round_rectangle

    args = parse_args()
    root = tk.Tk()
    FruitVegetableGUI(
        root,
        model_path=args.model_path,
        threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
    )
    root.mainloop()


def create_round_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


if __name__ == "__main__":
    main()
