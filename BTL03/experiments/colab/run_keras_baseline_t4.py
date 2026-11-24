"""Notebook helper để Member_03 chạy Keras legacy trên Google Colab T4.

Script này không tự nhập notebook gốc vì notebook có đường dẫn/dataset và
thứ tự cell riêng. Nó chỉ xác nhận runtime, liệt kê quy trình chạy an toàn,
và tạo thư mục output tách biệt để người chạy lưu bản sao kết quả.
"""

from pathlib import Path
import platform
import sys


class KerasBaselineColabGuide:
    """Hướng dẫn kiểm tra runtime và lưu notebook chạy lại tách biệt."""

    NOTEBOOKS = (
        "legacy_keras/01_mlp/Func_apxm_MLP.ipynb",
        "legacy_keras/01_mlp/MNIST_MLP.ipynb",
        "legacy_keras/02_cnn_mlp/1_MNIST_CNN.ipynb",
        "legacy_keras/02_cnn_mlp/2_Human_Gender_CNN.ipynb",
        "legacy_keras/02_cnn_mlp/3_Emotion_Reg_CNN.ipynb",
        "legacy_keras/03_cnn_lstm/MNIST_CNN_LSTM.ipynb",
        "legacy_keras/04_training_utilities/2_Human_Gender_CNN.ipynb",
        "legacy_keras/04_training_utilities/3_Emotion_Reg_CNN.ipynb",
        "legacy_keras/05_others/1_segnet.ipynb",
        "legacy_keras/05_others/2_obj_detection.ipynb",
    )

    def __init__(self, project_root: Path) -> None:
        """Lưu gốc dự án để kiểm tra file mà không sửa notebook legacy."""
        self.project_root = project_root.resolve()

    def run(self) -> None:
        """In thông tin runtime và hướng dẫn chạy baseline trong Colab."""
        print(f"Python: {sys.version.split()[0]} | OS: {platform.platform()}")
        try:
            import tensorflow as tf

            print(f"TensorFlow: {tf.__version__}")
            devices = tf.config.list_physical_devices("GPU")
            print(f"GPU devices: {devices or 'NONE'}")
            if not devices:
                print("CẢNH BÁO: chưa thấy GPU; không khởi chạy các notebook nặng.")
        except ImportError:
            print("TensorFlow/Keras chưa cài trong interpreter này; không chạy notebook.")
        print("\nNotebook legacy (chỉ đọc; chạy bản sao trên Colab, không ghi đè file gốc):")
        for relative in self.NOTEBOOKS:
            source = self.project_root / "BTL03" / relative
            print(f"{'OK' if source.is_file() else 'MISSING'} {relative}")
        output = self.project_root / "BTL03" / "experiments" / "keras_reruns"
        output.mkdir(parents=True, exist_ok=True)
        print(f"\nLưu bản sao/output run vào: {output}")
        print("Ghi runtime GPU model, TF/Keras version, seed, dataset/hash, CSV logs và execution time.")


def main() -> None:
    """Điểm vào chạy kiểm tra runtime Colab."""
    root = Path(__file__).resolve().parents[3]
    KerasBaselineColabGuide(root).run()


if __name__ == "__main__":
    main()