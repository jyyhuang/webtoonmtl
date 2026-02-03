from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QFont, QPainter, QColor, QPen
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout
from webtoonmtl.ui.colors import COLORS


class ResizableTextLabel(QLabel):

    HANDLE_SIZE = 10
    MIN_SIZE = QSize(50, 30)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"""
            QLabel {{
                background-color: white;
                padding: 8px;
                color: black;
            }}
        """
        )

        # Font scaling setup
        self._base_font_size = 10
        self._base_size = QSize(200, 60)
        self._font_family = "Segoe UI"

        self.setFont(QFont(self._font_family, self._base_font_size))
        self.setText("Translated Text")
        self.adjustSize()

        self.setMinimumSize(self.MIN_SIZE)

        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.drag_start_position = QPoint()
        self.resize_start_geometry = None
        self.resize_start_pos = QPoint()

        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _get_resize_corner(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        handle = self.HANDLE_SIZE

        # Check corners first
        if x < handle and y < handle:
            return "top-left"
        elif x > w - handle and y < handle:
            return "top-right"
        elif x < handle and y > h - handle:
            return "bottom-left"
        elif x > w - handle and y > h - handle:
            return "bottom-right"

        return None

    def _get_cursor_for_corner(self, corner):
        """Get appropriate cursor for resize corner."""
        cursors = {
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
        }
        return cursors.get(corner, Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            local_pos = ev.pos()
            corner = self._get_resize_corner(local_pos)

            if corner:
                # Start resizing
                self.resizing = True
                self.resize_corner = corner
                self.resize_start_geometry = self.geometry()
                self.resize_start_pos = ev.globalPosition().toPoint()
            else:
                # Start dragging
                self.dragging = True
                self.drag_start_position = (
                    ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

            ev.accept()

    def _update_font_size(self):
        current_area = self.width() * self.height()
        base_area = self._base_size.width() * self._base_size.height()
        scale_factor = (current_area / base_area) ** 0.5

        new_size = max(8, int(self._base_font_size * scale_factor * 0.6))

        font = QFont(self._font_family, new_size)
        self.setFont(font)

    def mouseMoveEvent(self, ev):
        global_pos = ev.globalPosition().toPoint()

        if self.resizing:
            delta = global_pos - self.resize_start_pos
            rect = self.resize_start_geometry

            if self.resize_corner == "bottom-right":
                new_w = max(self.MIN_SIZE.width(), rect.width() + delta.x())
                new_h = max(self.MIN_SIZE.height(), rect.height() + delta.y())
                self.resize(new_w, new_h)
            elif self.resize_corner == "bottom-left":
                new_w = max(self.MIN_SIZE.width(), rect.width() - delta.x())
                new_h = max(self.MIN_SIZE.height(), rect.height() + delta.y())
                new_x = rect.left() + (rect.width() - new_w)
                self.setGeometry(new_x, rect.top(), new_w, new_h)
            elif self.resize_corner == "top-right":
                new_w = max(self.MIN_SIZE.width(), rect.width() + delta.x())
                new_h = max(self.MIN_SIZE.height(), rect.height() - delta.y())
                new_y = rect.top() + (rect.height() - new_h)
                self.setGeometry(rect.left(), new_y, new_w, new_h)
            elif self.resize_corner == "top-left":
                new_w = max(self.MIN_SIZE.width(), rect.width() - delta.x())
                new_h = max(self.MIN_SIZE.height(), rect.height() - delta.y())
                new_x = rect.left() + (rect.width() - new_w)
                new_y = rect.top() + (rect.height() - new_h)
                self.setGeometry(new_x, new_y, new_w, new_h)

            self._update_font_size()

            ev.accept()

        elif self.dragging:
            new_pos = global_pos - self.drag_start_position
            parent = self.parentWidget()
            if parent:
                max_x = parent.width() - self.width()
                max_y = parent.height() - self.height()
                new_pos.setX(max(0, min(new_pos.x(), max_x)))
                new_pos.setY(max(0, min(new_pos.y(), max_y)))
            self.move(new_pos)
            ev.accept()

        else:
            local_pos = ev.pos()
            corner = self._get_resize_corner(local_pos)
            self.setCursor(self._get_cursor_for_corner(corner))

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                self.dragging = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self.resizing:
                self.resizing = False
                self.resize_corner = None
                local_pos = ev.pos()
                corner = self._get_resize_corner(local_pos)
                self.setCursor(self._get_cursor_for_corner(corner))
            ev.accept()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        # Draw resize handles
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        handle_color = QColor(COLORS["mauve"])
        painter.setBrush(handle_color)
        painter.setPen(QPen(handle_color, 1))

        handle = self.HANDLE_SIZE
        w, h = self.width(), self.height()

        corners = [
            (0, 0, handle, handle),  # top-left
            (w - handle, 0, w, handle),  # top-right
            (0, h - handle, handle, h),  # bottom-left
            (w - handle, h - handle, w, h),  # bottom-right
        ]

        for x1, y1, x2, y2 in corners:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def getSize(self):
        return self.size()


class ImageContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['surface0']};")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)

        self.text_labels = []
        self._image_offset = QPoint(0, 0)
        self._image_scale = 1.0
        self._displayed_image_rect = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.image_label.setGeometry(self.rect())

    def setPixmap(self, pixmap, offset=None, scale=None, image_rect=None):
        """Set the pixmap and store the image transformation info."""
        self.image_label.setPixmap(pixmap)
        if offset is not None:
            self._image_offset = offset
        if scale is not None:
            self._image_scale = scale
        if image_rect is not None:
            self._displayed_image_rect = image_rect

    def pixmap(self):
        return self.image_label.pixmap()

    def getImageOffset(self):
        return self._image_offset

    def getImageScale(self):
        return self._image_scale

    def getDisplayedImageRect(self):
        if self._displayed_image_rect:
            return self._displayed_image_rect
        pixmap = self.pixmap()
        if pixmap:
            return pixmap.rect().translated(self._image_offset)
        return None

    def addTextOverlay(self, text="Translated Text", rel_pos=None, rel_size=None):
        """
        Add a resizable text overlay.

        Args:
            text: The text to display
            rel_pos: QPointF with relative position (0.0-1.0) within the image, or None for center
            rel_size: QSize with relative size (0.0-1.0) of image dimensions, or None for default

        Returns:
            The created ResizableTextLabel instance
        """
        text_label = ResizableTextLabel(self)
        text_label.setText(text)

        if rel_size:
            img_rect = self.getDisplayedImageRect()
            if img_rect:
                w = int(rel_size.width() * img_rect.width())
                h = int(rel_size.height() * img_rect.height())
                text_label.resize(max(w, 50), max(h, 30))
        else:
            text_label.adjustSize()

        img_rect = self.getDisplayedImageRect()
        if img_rect is not None:
            if rel_pos is None:
                # Center the text
                x = img_rect.x() + (img_rect.width() - text_label.width()) // 2
                y = img_rect.y() + (img_rect.height() - text_label.height()) // 2
            else:
                # Position at relative coordinates
                x = (
                    img_rect.x()
                    + int(rel_pos.x() * img_rect.width())
                    - text_label.width() // 2
                )
                y = (
                    img_rect.y()
                    + int(rel_pos.y() * img_rect.height())
                    - text_label.height() // 2
                )

            x = max(0, min(x, self.width() - text_label.width()))
            y = max(0, min(y, self.height() - text_label.height()))

            text_label.move(x, y)

        text_label.show()
        text_label.raise_()
        self.text_labels.append(text_label)
        return text_label

    def updateText(self, index, text):
        if 0 <= index < len(self.text_labels):
            self.text_labels[index].setText(text)

    def getTextInfo(self, index=None) -> dict | list:
        """
        Get text information for saving.

        Args:
            index: If specified, returns info for that text box only. Otherwise returns list of all.

        Returns:
            dict with text, relative position (0.0-1.0), relative size, and font info,
            or list of such dicts if index is None
        """
        img_rect = self.getDisplayedImageRect()
        if img_rect is None:
            return None if index is not None else []

        def _get_single_info(text_label):
            text_rect = text_label.geometry()
            text_center = text_rect.center()

            rel_x = (text_center.x() - img_rect.x()) / img_rect.width()
            rel_y = (text_center.y() - img_rect.y()) / img_rect.height()

            rel_w = text_rect.width() / img_rect.width()
            rel_h = text_rect.height() / img_rect.height()

            return {
                "text": text_label.text(),
                "rel_x": rel_x,
                "rel_y": rel_y,
                "rel_w": rel_w,
                "rel_h": rel_h,
                "font_family": text_label.font().family(),
                "font_size": text_label.font().pointSize(),
                "font_bold": text_label.font().bold(),
            }

        if index is not None:
            if 0 <= index < len(self.text_labels):
                return _get_single_info(self.text_labels[index])
            return None
        else:
            return [_get_single_info(label) for label in self.text_labels]

    def removeTextOverlay(self, index=None):
        """
        Remove text overlay(s).

        Args:
            index: If specified, removes only that text box. Otherwise removes all.
        """
        if index is not None:
            if 0 <= index < len(self.text_labels):
                self.text_labels[index].deleteLater()
                self.text_labels.pop(index)
        else:
            for label in self.text_labels:
                label.deleteLater()
            self.text_labels.clear()

    def getTextBoxCount(self):
        return len(self.text_labels)

    def getTextBoxText(self, index):
        if 0 <= index < len(self.text_labels):
            return self.text_labels[index].text()
        return ""
