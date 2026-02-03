import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint, QPointF, QSizeF
from PyQt6.QtGui import QImage, QPainter, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QGroupBox,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
    QToolButton,
    QLabel,
)

from webtoonmtl.ui.widgets import ImageContainer
from webtoonmtl.core.mtlcore import MtlCore
from webtoonmtl.ui.colors import COLORS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("WebtoonMTL")
        self.setMinimumSize(900, 700)

        self.original_image = None
        self._text_boxes_data = []
        self._mtl_core = None
        self._text_inputs = []

        self._setup_ui()
        self._apply_theme()

    def _get_mtl_core(self):
        """Lazy initialization of MtlCore."""
        if self._mtl_core is None:
            self._mtl_core = MtlCore()
        return self._mtl_core

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setSpacing(10)

        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self.import_image)
        controls_layout.addWidget(self.import_btn)

        self.ocr_btn = QPushButton("Translate")
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.ocr_btn.setEnabled(False)
        self.ocr_btn.setToolTip("Extract and translate Korean text from image")
        controls_layout.addWidget(self.ocr_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.save_btn)

        self.clear_text_btn = QPushButton("Clear Text")
        self.clear_text_btn.clicked.connect(self.clear_text)
        self.clear_text_btn.setEnabled(False)
        controls_layout.addWidget(self.clear_text_btn)

        controls_layout.addStretch()
        main_layout.addWidget(controls_group)

        text_group = QGroupBox("Translation Text")
        text_layout = QVBoxLayout(text_group)
        text_layout.setSpacing(10)

        self.add_textbox_btn = QPushButton("+ Add Text Box")
        self.add_textbox_btn.clicked.connect(self.add_text_box)
        self.add_textbox_btn.setEnabled(False)
        text_layout.addWidget(self.add_textbox_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet(f"background-color: {COLORS['surface0']};")

        self.text_boxes_container = QWidget()
        self.text_boxes_layout = QVBoxLayout(self.text_boxes_container)
        self.text_boxes_layout.setSpacing(8)
        self.text_boxes_layout.setContentsMargins(8, 8, 8, 8)
        self.text_boxes_layout.addStretch()

        scroll.setWidget(self.text_boxes_container)
        text_layout.addWidget(scroll)

        main_layout.addWidget(text_group)

        self.image_container = ImageContainer()
        self.image_container.setMinimumSize(600, 400)
        main_layout.addWidget(self.image_container, stretch=1)

    def _apply_theme(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {COLORS["base"]};
            }}
            QWidget {{
                background-color: {COLORS["base"]};
                color: {COLORS["text"]};
                font-family: 'Segoe UI', sans-serif;
            }}
            QGroupBox {{
                border: 2px solid {COLORS["surface1"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {COLORS["mauve"]};
            }}
            QPushButton {{
                background-color: {COLORS["surface1"]};
                border: 2px solid {COLORS["surface2"]};
                border-radius: 6px;
                padding: 8px 16px;
                color: {COLORS["text"]};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS["surface2"]};
                border-color: {COLORS["blue"]};
            }}
            QPushButton:pressed {{
                background-color: {COLORS["blue"]};
                color: {COLORS["base"]};
            }}
            QPushButton:disabled {{
                background-color: {COLORS["surface0"]};
                border-color: {COLORS["overlay"]};
                color: {COLORS["overlay"]};
            }}
            QLineEdit {{
                background-color: {COLORS["surface0"]};
                border: 2px solid {COLORS["surface1"]};
                border-radius: 6px;
                padding: 8px;
                color: {COLORS["text"]};
            }}
            QLineEdit:focus {{
                border-color: {COLORS["blue"]};
            }}
            QLabel {{
                color: {COLORS["text"]};
            }}
        """
        )

    def on_text_changed(self, index, text):
        self.image_container.updateText(index, text)

    def import_image(self):
        """Import an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)",
        )

        if file_path:
            self.original_image = QImage(file_path)
            if not self.original_image.isNull():
                self._text_boxes_data = []
                self.display_image()
                self.save_btn.setEnabled(True)
                self.ocr_btn.setEnabled(True)
                self.add_textbox_btn.setEnabled(True)
                self.image_container.removeTextOverlay()
                self.clear_text_btn.setEnabled(False)
                self._clear_text_inputs()
            else:
                self.image_container.image_label.setText("Failed to load image")

    def run_ocr(self):
        if self.original_image is None:
            return

        # Create progress dialog
        progress = QProgressDialog("Loading OCR model...", None, 0, 0, self)
        progress.setWindowTitle("OCR Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        try:
            mtl_core = self._get_mtl_core()

            progress.setLabelText("Extracting text...")
            QApplication.processEvents()

            temp_path = Path.home() / ".webtoonmtl_temp.png"
            self.original_image.save(str(temp_path))

            results = mtl_core.image_to_translation(str(temp_path))

            temp_path.unlink(missing_ok=True)

            progress.close()

            if results:
                self._clear_text_inputs()
                self.image_container.removeTextOverlay()
                self._text_boxes_data = []

                img_width = self.original_image.width()
                img_height = self.original_image.height()

                for i, result in enumerate(results):
                    text = result["text"]
                    bbox = result["bbox"]

                    self._add_text_input(text)

                    # Calculate position from bounding box
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]

                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)

                    center_x = (min_x + max_x) / 2 / img_width
                    center_y = (min_y + max_y) / 2 / img_height
                    width = (max_x - min_x) / img_width
                    height = (max_y - min_y) / img_height

                    rel_pos = QPointF(center_x, center_y)
                    rel_size = QSizeF(max(width, 0.1), max(height, 0.05))

                    self._text_boxes_data.append(
                        {"rel_pos": rel_pos, "rel_size": rel_size}
                    )
                    self.add_translation(i)

                self.clear_text_btn.setEnabled(True)

                QMessageBox.information(
                    self,
                    "OCR Complete",
                    f"Successfully extracted and translated {len(results)} text segment(s).",
                )
            else:
                QMessageBox.warning(
                    self, "No Text Found", "No Korean text was detected in the image."
                )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, "OCR Error", f"Failed to process image:\n{str(e)}"
            )

    def display_image(self):
        """Display the current image scaled to fit the container."""
        if self.original_image is None:
            return

        pixmap = QPixmap.fromImage(self.original_image)

        container_size = self.image_container.size()
        scaled_size = pixmap.size()
        scaled_size.scale(container_size, Qt.AspectRatioMode.KeepAspectRatio)

        offset_x = (container_size.width() - scaled_size.width()) // 2
        offset_y = (container_size.height() - scaled_size.height()) // 2

        scaled_pixmap = pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        scale = scaled_size.width() / pixmap.width()

        displayed_rect = scaled_pixmap.rect().translated(offset_x, offset_y)

        self.image_container.setPixmap(
            scaled_pixmap,
            offset=QPoint(offset_x, offset_y),
            scale=scale,
            image_rect=displayed_rect,
        )

    def add_translation(self, index=None):
        """Add a draggable and resizable text overlay."""
        if index is None:
            index = len(self._text_inputs)

        if index < len(self._text_inputs):
            text = self._text_inputs[index].text() or "Translated Text"
        else:
            text = "Translated Text"

        rel_pos = None
        rel_size = None
        if index < len(self._text_boxes_data):
            rel_pos = self._text_boxes_data[index].get("rel_pos")
            rel_size = self._text_boxes_data[index].get("rel_size")

        self.image_container.addTextOverlay(text, rel_pos=rel_pos, rel_size=rel_size)
        self.clear_text_btn.setEnabled(True)

    def clear_text(self):
        self._text_boxes_data = []
        self.image_container.removeTextOverlay()
        self.clear_text_btn.setEnabled(False)
        self._clear_text_inputs()

    def save_image(self):
        if self.original_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(Path.home() / "translated.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )

        if file_path:
            final_pixmap = QPixmap.fromImage(self.original_image)

            text_infos = self.image_container.getTextInfo()
            if text_infos:
                painter = QPainter(final_pixmap)

                displayed_scale = self.image_container.getImageScale()
                orig_width = self.original_image.width()
                orig_height = self.original_image.height()

                for text_info in text_infos:
                    original_font_size = int(text_info["font_size"] / displayed_scale)

                    font = QFont(text_info["font_family"], original_font_size)
                    font.setBold(text_info["font_bold"])
                    painter.setFont(font)

                    orig_w = int(text_info["rel_w"] * orig_width)
                    orig_h = int(text_info["rel_h"] * orig_height)

                    center_x = int(text_info["rel_x"] * orig_width)
                    center_y = int(text_info["rel_y"] * orig_height)

                    x = center_x - orig_w // 2
                    y = center_y - orig_h // 2

                    bg_rect = painter.boundingRect(
                        x,
                        y,
                        orig_w,
                        orig_h,
                        Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                        text_info["text"],
                    )

                    padding_x = 8
                    padding_y = 4
                    bg_rect.adjust(-padding_x, -padding_y, padding_x, padding_y)

                    painter.fillRect(bg_rect, QColor("white"))

                    painter.setPen(QColor("black"))
                    painter.drawText(
                        bg_rect,
                        Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                        text_info["text"],
                    )

                painter.end()

            final_pixmap.save(file_path)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if self.original_image is not None:
            text_count = self.image_container.getTextBoxCount()
            self._text_boxes_data = []
            for i in range(text_count):
                info = self.image_container.getTextInfo(i)
                if info:
                    self._text_boxes_data.append(
                        {
                            "rel_pos": QPointF(info["rel_x"], info["rel_y"]),
                            "rel_size": QSizeF(info["rel_w"], info["rel_h"]),
                        }
                    )

            self.display_image()

            if self._text_boxes_data and self.image_container.getTextBoxCount() == 0:
                for i, data in enumerate(self._text_boxes_data):
                    text = (
                        self._text_inputs[i].text()
                        if i < len(self._text_inputs)
                        else "Translated Text"
                    )
                    self.image_container.addTextOverlay(
                        text, rel_pos=data["rel_pos"], rel_size=data["rel_size"]
                    )

    def add_text_box(self):
        index = len(self._text_inputs)
        self._add_text_input("Translated Text")
        self._text_boxes_data.append({"rel_pos": None, "rel_size": None})
        self.add_translation(index)
        self.clear_text_btn.setEnabled(True)

    def _add_text_input(self, text=""):
        index = len(self._text_inputs)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        label = QLabel(f"{index + 1}:")
        label.setStyleSheet(f"color: {COLORS['text']};")
        row_layout.addWidget(label)

        text_input = QLineEdit()
        text_input.setPlaceholderText(f"Enter text for box {index + 1}...")
        text_input.setText(text)
        text_input.textChanged.connect(
            lambda t, idx=index: self.on_text_changed(idx, t)
        )
        text_input.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {COLORS["surface0"]};
                border: 2px solid {COLORS["surface1"]};
                border-radius: 6px;
                padding: 8px;
                color: {COLORS["text"]};
            }}
            QLineEdit:focus {{
                border-color: {COLORS["blue"]};
            }}
        """
        )
        row_layout.addWidget(text_input, stretch=1)

        remove_btn = QToolButton()
        remove_btn.setText("×")
        remove_btn.setToolTip("Remove this text box")
        remove_btn.clicked.connect(lambda: self._remove_text_input(index))
        remove_btn.setStyleSheet(
            f"""
            QToolButton {{
                background-color: {COLORS["surface1"]};
                border: 1px solid {COLORS["surface2"]};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS["red"]};
                font-weight: bold;
            }}
            QToolButton:hover {{
                background-color: {COLORS["red"]};
                color: {COLORS["base"]};
            }}
        """
        )
        row_layout.addWidget(remove_btn)

        self.text_boxes_layout.insertWidget(
            self.text_boxes_layout.count() - 1, row_widget
        )
        self._text_inputs.append(text_input)

    def _remove_text_input(self, index):
        if index < len(self._text_inputs):
            self.image_container.removeTextOverlay(index)

            item = self.text_boxes_layout.itemAt(index)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            self._text_inputs.pop(index)
            if index < len(self._text_boxes_data):
                self._text_boxes_data.pop(index)

            for i in range(self.text_boxes_layout.count() - 1):
                item = self.text_boxes_layout.itemAt(i)
                if item:
                    row_widget = item.widget()
                    if row_widget:
                        for child in row_widget.findChildren(QLabel):
                            child.setText(f"{i + 1}:")
                            break

                        for child in row_widget.findChildren(QToolButton):
                            child.clicked.disconnect()
                            child.clicked.connect(
                                lambda checked, idx=i: self._remove_text_input(idx)
                            )
                            break

            if len(self._text_inputs) == 0:
                self.clear_text_btn.setEnabled(False)

    def _clear_text_inputs(self):
        while self.text_boxes_layout.count() > 1:
            item = self.text_boxes_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        self._text_inputs.clear()


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
