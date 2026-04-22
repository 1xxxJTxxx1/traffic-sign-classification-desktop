from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .config import (
    BEST_WEIGHTS_PATH,
    DATASET_META_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REFERENCE_ICONS_DIR,
    TRAIN_REPORT_PATH,
)
from .inference import PredictionResult, TrafficSignPredictor

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp"}


class TrafficSignApp(tk.Tk):
    def __init__(self, predictor: TrafficSignPredictor) -> None:
        super().__init__()
        self.predictor = predictor

        self.title("Классификация дорожных знаков")
        self.geometry("1450x900")
        self.minsize(1220, 760)

        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.last_open_dir: Path = self._resolve_initial_open_dir()

        self._canvas_image_photo: ImageTk.PhotoImage | None = None
        self._model_preview_photo: ImageTk.PhotoImage | None = None
        self._icon_photo: ImageTk.PhotoImage | None = None

        self.image_view = {
            "scale": 1.0,
            "offset_x": 0,
            "offset_y": 0,
            "disp_w": 1,
            "disp_h": 1,
        }

        self.history_rows: list[dict] = []
        self.batch_rows: list[dict] = []

        self._build_ui()
        self._load_model_info()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_single = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_info = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_single, text="Одиночное изображение")
        self.notebook.add(self.tab_batch, text="Пакетная обработка")
        self.notebook.add(self.tab_info, text="О модели")

        self._build_single_tab()
        self._build_batch_tab()
        self._build_info_tab()

    def _build_single_tab(self) -> None:
        controls = ttk.Frame(self.tab_single, padding=(8, 8))
        controls.pack(fill="x")

        self.btn_open = ttk.Button(controls, text="Открыть изображение", command=self._open_image)
        self.btn_open.pack(side="left", padx=(0, 8))

        self.btn_classify = ttk.Button(controls, text="Классифицировать", command=self._start_single_inference)
        self.btn_classify.pack(side="left", padx=(0, 8))

        self.single_status_var = tk.StringVar(value="Готово")
        ttk.Label(controls, textvariable=self.single_status_var).pack(side="right")

        content = ttk.Panedwindow(self.tab_single, orient="horizontal")
        content.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(content)
        right = ttk.Frame(content, width=360)
        content.add(left, weight=3)
        content.add(right, weight=1)

        self.canvas = tk.Canvas(left, bg="#202632", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_canvas())

        previews = ttk.Frame(left, padding=(0, 8, 0, 0))
        previews.pack(fill="x")

        model_view_frame = ttk.LabelFrame(previews, text="Что подается в сеть", padding=6)
        model_view_frame.pack(fill="both", expand=True)
        self.model_preview_label = ttk.Label(model_view_frame, text="Нет данных")
        self.model_preview_label.pack(fill="both", expand=True)

        result_frame = ttk.LabelFrame(right, text="Результат", padding=10)
        result_frame.pack(fill="x", padx=4, pady=(2, 8))

        self.result_sign_var = tk.StringVar(value="-")
        self.result_conf_var = tk.StringVar(value="-")

        ttk.Label(result_frame, text="Предсказанный знак:").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.result_sign_var, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        ttk.Label(result_frame, text="Уверенность:").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.result_conf_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(result_frame, text="Статус:").pack(anchor="w")
        self.status_badge = tk.Label(result_frame, text="-", bg="#BDBDBD", fg="black", width=16)
        self.status_badge.pack(anchor="w", pady=(0, 8))

        icon_frame = ttk.LabelFrame(right, text="Эталонная иконка", padding=8)
        icon_frame.pack(fill="x", padx=4, pady=(0, 8))
        self.icon_label = ttk.Label(icon_frame, text="Нет")
        self.icon_label.pack(fill="x")

        top3_frame = ttk.LabelFrame(right, text="Top-3 класса", padding=8)
        top3_frame.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        self.top3_tree = ttk.Treeview(top3_frame, columns=("class", "prob"), show="headings", height=4)
        self.top3_tree.heading("class", text="Класс")
        self.top3_tree.heading("prob", text="Вероятность")
        self.top3_tree.column("class", width=220, anchor="w")
        self.top3_tree.column("prob", width=90, anchor="center")
        self.top3_tree.pack(fill="both", expand=True)

        history_frame = ttk.LabelFrame(right, text="История последних результатов", padding=8)
        history_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self.history_tree = ttk.Treeview(history_frame, columns=("time", "file", "pred", "conf"), show="headings", height=7)
        for col, title, width in (
            ("time", "Время", 70),
            ("file", "Файл", 140),
            ("pred", "Класс", 160),
            ("conf", "Увер.", 70),
        ):
            self.history_tree.heading(col, text=title)
            self.history_tree.column(col, width=width, anchor="w")
        self.history_tree.pack(fill="both", expand=True)

    def _build_batch_tab(self) -> None:
        top = ttk.Frame(self.tab_batch, padding=8)
        top.pack(fill="x")

        self.batch_folder_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.batch_folder_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(top, text="Выбрать папку", command=self._choose_batch_folder).pack(side="left", padx=(0, 8))
        self.btn_batch_run = ttk.Button(top, text="Запустить пакетную обработку", command=self._start_batch_processing)
        self.btn_batch_run.pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Экспорт CSV", command=self._export_batch_csv).pack(side="left", padx=(0, 8))

        self.batch_only_uncertain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="Показывать только сомнительные",
            variable=self.batch_only_uncertain_var,
            command=self._refresh_batch_tree,
        ).pack(side="left", padx=(0, 8))

        self.batch_threshold_var = tk.DoubleVar(value=0.60)
        ttk.Label(top, text="Порог:").pack(side="left")
        threshold_spin = ttk.Spinbox(
            top,
            from_=0.30,
            to=0.95,
            increment=0.01,
            textvariable=self.batch_threshold_var,
            width=6,
            command=self._refresh_batch_tree,
        )
        threshold_spin.pack(side="left")

        table_frame = ttk.Frame(self.tab_batch, padding=(8, 0, 8, 8))
        table_frame.pack(fill="both", expand=True)

        self.batch_tree = ttk.Treeview(
            table_frame,
            columns=("file", "pred", "conf", "status"),
            show="headings",
        )
        self.batch_tree.heading("file", text="Файл")
        self.batch_tree.heading("pred", text="Предсказание")
        self.batch_tree.heading("conf", text="Уверенность")
        self.batch_tree.heading("status", text="Статус")

        self.batch_tree.column("file", width=340, anchor="w")
        self.batch_tree.column("pred", width=420, anchor="w")
        self.batch_tree.column("conf", width=120, anchor="center")
        self.batch_tree.column("status", width=120, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=vsb.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.batch_status_var = tk.StringVar(value="Ожидание")
        ttk.Label(self.tab_batch, textvariable=self.batch_status_var, padding=(10, 0, 10, 8)).pack(anchor="w")

    def _build_info_tab(self) -> None:
        frame = ttk.Frame(self.tab_info, padding=10)
        frame.pack(fill="both", expand=True)

        self.info_text = tk.Text(frame, wrap="word")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.configure(state="disabled")

    @staticmethod
    def _resolve_initial_open_dir() -> Path:
        demo_dir = PROCESSED_DATA_DIR / "demo"
        if demo_dir.exists():
            return demo_dir
        return PROJECT_ROOT

    def _open_image(self) -> None:
        initial_dir = self.last_open_dir
        if self.current_image_path is not None and self.current_image_path.parent.exists():
            initial_dir = self.current_image_path.parent
        elif not initial_dir.exists():
            initial_dir = PROJECT_ROOT

        path = filedialog.askopenfilename(
            title="Выберите изображение",
            initialdir=str(initial_dir),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.ppm *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{exc}")
            return

        self.current_image = image
        self.current_image_path = Path(path)
        if self.current_image_path.parent.exists():
            self.last_open_dir = self.current_image_path.parent
        self._render_canvas()
        self._update_model_preview(None)
        self.single_status_var.set(f"Загружено: {self.current_image_path.name}")

    def _render_canvas(self) -> None:
        self.canvas.delete("all")

        if self.current_image is None:
            self.canvas.create_text(30, 30, anchor="nw", fill="white", text="Откройте изображение для классификации")
            return

        canvas_w = max(200, self.canvas.winfo_width())
        canvas_h = max(200, self.canvas.winfo_height())

        img_w, img_h = self.current_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2

        image_resized = self.current_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._canvas_image_photo = ImageTk.PhotoImage(image_resized)
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self._canvas_image_photo)

        self.image_view = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "disp_w": disp_w,
            "disp_h": disp_h,
        }

    def _update_model_preview(self, image: Image.Image | None) -> None:
        if image is None:
            self.model_preview_label.configure(text="Нет данных", image="")
            self._model_preview_photo = None
            return

        thumb = image.copy()
        thumb.thumbnail((260, 180), Image.Resampling.NEAREST)
        self._model_preview_photo = ImageTk.PhotoImage(thumb)
        self.model_preview_label.configure(image=self._model_preview_photo, text="")

    def _start_single_inference(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Внимание", "Сначала откройте изображение.")
            return

        self.btn_classify.configure(state="disabled")
        self.single_status_var.set("Идет обработка...")

        thread = threading.Thread(target=self._single_inference_worker, daemon=True)
        thread.start()

    def _single_inference_worker(self) -> None:
        try:
            result = self.predictor.predict(self.current_image, roi=None)
            self.after(0, lambda: self._apply_prediction_result(result))
        except Exception as exc:
            self.after(0, lambda: self._single_inference_failed(exc))

    def _single_inference_failed(self, exc: Exception) -> None:
        self.btn_classify.configure(state="normal")
        self.single_status_var.set("Ошибка")
        messagebox.showerror("Ошибка инференса", str(exc))

    def _apply_prediction_result(self, result: PredictionResult) -> None:
        self.btn_classify.configure(state="normal")
        self.single_status_var.set("Готово")

        self.result_sign_var.set(result.label_name)
        self.result_conf_var.set(f"{result.confidence * 100:.2f}%")

        status_colors = {
            "уверенно": "#86EFAC",
            "средне": "#FDE68A",
            "сомнительно": "#FCA5A5",
        }
        self.status_badge.configure(text=result.status, bg=status_colors.get(result.status, "#D4D4D8"))

        for row_id in self.top3_tree.get_children():
            self.top3_tree.delete(row_id)
        for _, class_name, prob in result.top3:
            self.top3_tree.insert("", "end", values=(class_name, f"{prob * 100:.2f}%"))

        icon_path = REFERENCE_ICONS_DIR / f"{result.label_id:02d}.png"
        if icon_path.exists():
            icon_img = Image.open(icon_path).convert("RGB")
            icon_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
            self._icon_photo = ImageTk.PhotoImage(icon_img)
            self.icon_label.configure(image=self._icon_photo, text="")
        else:
            self.icon_label.configure(text="Иконка не найдена", image="")
            self._icon_photo = None

        self._update_model_preview(result.input_preview)
        self._append_history(result)

    def _append_history(self, result: PredictionResult) -> None:
        row = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "file": self.current_image_path.name if self.current_image_path else "-",
            "pred": result.label_name,
            "conf": f"{result.confidence * 100:.1f}%",
        }
        self.history_rows.append(row)
        self.history_rows = self.history_rows[-20:]

        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for item in reversed(self.history_rows):
            self.history_tree.insert("", "end", values=(item["time"], item["file"], item["pred"], item["conf"]))

    def _choose_batch_folder(self) -> None:
        initial_dir = self.last_open_dir if self.last_open_dir.exists() else PROJECT_ROOT
        folder = filedialog.askdirectory(title="Выберите папку с изображениями", initialdir=str(initial_dir))
        if folder:
            chosen = Path(folder)
            if chosen.exists():
                self.last_open_dir = chosen
            self.batch_folder_var.set(folder)

    def _start_batch_processing(self) -> None:
        folder = self.batch_folder_var.get().strip()
        if not folder:
            messagebox.showwarning("Внимание", "Выберите папку с изображениями.")
            return

        path = Path(folder)
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Ошибка", "Указанная папка не существует.")
            return

        self.btn_batch_run.configure(state="disabled")
        self.batch_status_var.set("Идет пакетная обработка...")

        thread = threading.Thread(target=self._batch_worker, args=(path,), daemon=True)
        thread.start()

    def _batch_worker(self, folder: Path) -> None:
        image_files = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXT and p.is_file()]
        image_files.sort()

        if not image_files:
            self.after(0, lambda: self._batch_finished([], "В выбранной папке нет изображений."))
            return

        rows: list[dict] = []
        for idx, img_path in enumerate(image_files, start=1):
            try:
                image = Image.open(img_path).convert("RGB")
                res = self.predictor.predict(image, roi=None)
                rows.append(
                    {
                        "file": str(img_path),
                        "pred": res.label_name,
                        "confidence": float(res.confidence),
                        "status": res.status,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": str(img_path),
                        "pred": f"Ошибка: {exc}",
                        "confidence": 0.0,
                        "status": "ошибка",
                    }
                )

            if idx % 10 == 0 or idx == len(image_files):
                self.after(0, lambda i=idx, n=len(image_files): self.batch_status_var.set(f"Обработано {i}/{n}"))

        self.after(0, lambda: self._batch_finished(rows, f"Готово. Обработано: {len(rows)}"))

    def _batch_finished(self, rows: list[dict], status_text: str) -> None:
        self.batch_rows = rows
        self.btn_batch_run.configure(state="normal")
        self.batch_status_var.set(status_text)
        self._refresh_batch_tree()

    def _refresh_batch_tree(self) -> None:
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        threshold = float(self.batch_threshold_var.get())
        only_uncertain = bool(self.batch_only_uncertain_var.get())

        for row in self.batch_rows:
            is_uncertain = row["confidence"] < threshold
            if only_uncertain and not is_uncertain:
                continue
            self.batch_tree.insert(
                "",
                "end",
                values=(
                    Path(row["file"]).name,
                    row["pred"],
                    f"{row['confidence'] * 100:.2f}%",
                    row["status"],
                ),
            )

    def _export_batch_csv(self) -> None:
        if not self.batch_rows:
            messagebox.showwarning("Внимание", "Нет результатов для экспорта.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not save_path:
            return

        import csv

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "pred", "confidence", "status"])
            writer.writeheader()
            for row in self.batch_rows:
                writer.writerow(row)

        self.batch_status_var.set(f"CSV сохранен: {save_path}")

    def _load_model_info(self) -> None:
        lines: list[str] = []
        lines.append("Информация о модели\n")
        lines.append(f"Путь к лучшим весам: {BEST_WEIGHTS_PATH}\n")

        if BEST_WEIGHTS_PATH.exists():
            try:
                import torch

                ckpt = torch.load(BEST_WEIGHTS_PATH, map_location="cpu")
                lines.append(f"Архитектура: {ckpt.get('model_name', 'TrafficSignCNN')}\n")
                lines.append(f"Размер входа: {ckpt.get('input_size', 64)}x{ckpt.get('input_size', 64)}\n")
                lines.append(f"Число классов: {ckpt.get('num_classes', 43)}\n")
                lines.append(f"Дата обучения: {ckpt.get('trained_at', '-') }\n")
                lines.append(f"Порог сомнительного результата: {ckpt.get('uncertain_threshold', 0.6)}\n")
            except Exception as exc:
                lines.append(f"Не удалось прочитать checkpoint: {exc}\n")
        else:
            lines.append("Веса пока не найдены. Запустите обучение (mode=train).\n")

        if TRAIN_REPORT_PATH.exists():
            try:
                with TRAIN_REPORT_PATH.open("r", encoding="utf-8") as f:
                    report = json.load(f)
                lines.append("\nРезультаты обучения:\n")
                lines.append(f"Best epoch: {report.get('best_epoch', '-') }\n")
                lines.append(f"Test accuracy: {report.get('test_accuracy', 0.0) * 100:.2f}%\n")
                lines.append(f"Test F1 macro: {report.get('test_f1_macro', 0.0):.4f}\n")
            except Exception as exc:
                lines.append(f"Не удалось прочитать training_report.json: {exc}\n")

        if DATASET_META_PATH.exists():
            try:
                with DATASET_META_PATH.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                lines.append("\nДатасет:\n")
                lines.append(f"Источник: {meta.get('dataset', '-') }\n")
                lines.append(f"Создан: {meta.get('created_at', '-') }\n")
                lines.append(f"Split counts: {meta.get('split_counts', {})}\n")
            except Exception as exc:
                lines.append(f"Не удалось прочитать dataset_meta.json: {exc}\n")

        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "".join(lines))
        self.info_text.configure(state="disabled")


def launch_app() -> None:
    predictor = TrafficSignPredictor()
    app = TrafficSignApp(predictor)
    app.mainloop()
