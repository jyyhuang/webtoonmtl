from typing import Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from webtoonmtl.ui.colors import COLORS


class SetupWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cancelled = False

    def run(self):
        from webtoonmtl.core.setup import (
            download_ocr_model,
            download_translation_model,
            mark_setup_complete,
        )

        try:
            self.progress.emit("ocr", 0)
            if self._cancelled:
                self.finished.emit(False)
                return

            def ocr_progress(percent: int, message: str):
                if self._cancelled:
                    raise InterruptedError("Setup cancelled")
                self.progress.emit("ocr", percent)

            if not download_ocr_model(ocr_progress):
                self.error.emit("Failed to download OCR model")
                self.finished.emit(False)
                return

            if self._cancelled:
                self.finished.emit(False)
                return

            self.progress.emit("translation", 0)

            def translation_progress(percent: int, message: str):
                if self._cancelled:
                    raise InterruptedError("Setup cancelled")
                self.progress.emit("translation", percent)

            if not download_translation_model(translation_progress):
                self.error.emit("Failed to download translation model")
                self.finished.emit(False)
                return

            mark_setup_complete()
            self.finished.emit(True)

        except InterruptedError:
            self.finished.emit(False)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

    def cancel(self):
        self._cancelled = True


class ModelProgressWidget(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        header.addWidget(self.name_label)
        header.addStretch()

        self.status_label = QLabel("Waiting...")
        self.status_label.setStyleSheet(f"color: {COLORS['overlay']};")
        header.addWidget(self.status_label)

        layout.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: {COLORS["surface0"]};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS["blue"]};
                border-radius: 4px;
            }}
            """
        )
        layout.addWidget(self.progress_bar)

    def set_progress(self, percent: int):
        self.progress_bar.setValue(min(100, max(0, percent)))
        if percent < 100:
            self.status_label.setText("Downloading...")
            self.status_label.setStyleSheet(f"color: {COLORS['base']};")
        else:
            self.status_label.setText("Done")
            self.status_label.setStyleSheet(f"color: {COLORS['base']};")

    def set_error(self, message: str):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet(f"color: {COLORS['red']};")

    def set_waiting(self):
        self.status_label.setText("Waiting...")
        self.status_label.setStyleSheet(f"color: {COLORS['overlay']};")


class SetupWizard(QDialog):
    setup_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("WebtoonMTL Setup")
        self.setMinimumSize(450, 280)
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)

        self._worker = None
        self._setup_complete = False

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Downloading Required Models")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {COLORS['mauve']};"
        )
        layout.addWidget(title)

        desc = QLabel(
            "WebtoonMTL needs to download OCR and translation models.\n"
            "This may take a few minutes depending on your connection."
        )
        desc.setStyleSheet(f"color: {COLORS['text']};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        models_widget = QWidget()
        models_layout = QVBoxLayout(models_widget)
        models_layout.setContentsMargins(0, 8, 0, 8)
        models_layout.setSpacing(8)

        self.ocr_progress = ModelProgressWidget("EasyOCR (Korean Text Recognition)")
        models_layout.addWidget(self.ocr_progress)

        self.translation_progress = ModelProgressWidget(
            "Helsinki-NLP Translation Model"
        )
        models_layout.addWidget(self.translation_progress)

        layout.addWidget(models_widget)

        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {COLORS['red']};")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _apply_theme(self):
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {COLORS["base"]};
            }}
            QLabel {{
                color: {COLORS["text"]};
            }}
            QPushButton {{
                background-color: {COLORS["surface1"]};
                border: 2px solid {COLORS["surface2"]};
                border-radius: 6px;
                padding: 8px 24px;
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
            """
        )

    def start_setup(self):
        self._worker = SetupWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, model: str, percent: int):
        if model == "ocr":
            self.ocr_progress.set_progress(percent)
        elif model == "translation":
            self.translation_progress.set_progress(percent)

    def _on_error(self, message: str):
        self.error_label.setText(f"Error: {message}")
        self.error_label.show()

    def _on_finished(self, success: bool):
        if success:
            self._setup_complete = True
            self.setup_complete.emit()
            self.accept()
        else:
            self.cancel_btn.setText("Close")

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self.reject()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        event.accept()
