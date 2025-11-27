import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt6 - Thermo King Camera Test")
        self.setGeometry(500, 500, 1500, 800)

        # Set grey background
        self.setStyleSheet("background-color: #bfbfbf;")

        # Main layout (vertical)
        main_layout = QVBoxLayout()

        # ---- TITLE ----
        title = QLabel("Thermo King Camera Test")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        main_layout.addWidget(title)

        # ---- BUTTONS ----
        btn1 = QPushButton("Button 1")
        btn2 = QPushButton("Button 2")
        btn3 = QPushButton("Button 3")

        # Make buttons bigger
        for btn in (btn1, btn2, btn3):
            btn.setFixedHeight(80)
            btn.setFont(QFont("Arial", 20))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    border: 2px solid #555;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)

        # Connect button actions
        btn1.clicked.connect(self.button1_clicked)
        btn2.clicked.connect(self.button2_clicked)
        btn3.clicked.connect(self.button3_clicked)

        # Add buttons to layout
        main_layout.addWidget(btn1)
        main_layout.addWidget(btn2)
        main_layout.addWidget(btn3)

        # ---- IMAGES AT BOTTOM-LEFT ----
        image_layout = QHBoxLayout()

        # First image (tk.png)
        image_label1 = QLabel()
        pixmap1 = QPixmap("tk.png")
        image_label1.setPixmap(pixmap1)
        image_label1.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

        # Second image (TraneTechnologies.png)
        image_label2 = QLabel()
        pixmap2 = QPixmap("TraneTechnologies.png")
        image_label2.setPixmap(pixmap2)
        image_label2.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

        # Add both images side-by-side
        image_layout.addWidget(image_label1)
        image_layout.addSpacing(20)  # space between images
        image_layout.addWidget(image_label2)
        image_layout.addStretch()  # keeps them pushed to the left

        # Add image layout to bottom of main layout
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

    # ---- Button functions ----
    def button1_clicked(self):
        QMessageBox.information(self, "Button 1", "You clicked Button 1!")

    def button2_clicked(self):
        QMessageBox.warning(self, "Button 2", "Button 2 was pressed!")

    def button3_clicked(self):
        QMessageBox.critical(self, "Button 3", "Button 3 triggered a critical message!")


# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
