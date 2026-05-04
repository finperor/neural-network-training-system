# ┌────────────────────────────────────────────────────────────┐
# │  程序入口模块                                               │
# │  启动 PyQt6 应用程序并显示主窗口                            │
# └────────────────────────────────────────────────────────────┘

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """应用程序主入口
    创建 QApplication 实例，设置 Fusion 样式以提高跨平台一致性，
    然后显示主窗口并进入事件循环。
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")          # Fusion 样式：现代扁平化外观
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
