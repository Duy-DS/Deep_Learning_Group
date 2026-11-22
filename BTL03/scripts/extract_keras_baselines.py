"""Trích xuất metadata và execution outputs từ 10 notebook Keras legacy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NOT_RECORDED = "NOT_RECORDED"


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata cố định của một thí nghiệm theo checklist legacy."""

    exp_id: str
    group: str
    relative_path: str
    model_name: str
    input_shape: str
    output_shape: str
    optimizer: str
    learning_rate: str
    loss: str
    batch_size: str
    configured_epochs: str
    primary_metric: str
    metric_kind: str
    notes: str
    analytic_parameters: str = NOT_RECORDED
    parameter_provenance: str = "model.summary() output"
    parameter_formula: str = NOT_RECORDED


SPECS = [
    BaselineSpec("EXP-01", "01_mlp", "01_mlp/Func_apxm_MLP.ipynb", "Func_apxm_MLP", "(2,)", "(None, 1)", "SGD", "0.01", "mean_squared_error", "128", "C (loop sweep)", "MSE", "regression", "Sweep A/B/C; no saved numeric evaluation output or timing.", parameter_provenance="No saved summary; B varies in sweeps.", parameter_formula="4*B+1; 65 only when B=16"),
    BaselineSpec("EXP-02", "01_mlp", "01_mlp/MNIST_MLP.ipynb", "MNIST_MLP", "(28, 28)", "(None, 10)", "SGD", "0.01", "categorical_crossentropy", "128", "30", "accuracy", "classification", "Final cell explicitly prints Test loss/accuracy."),
    BaselineSpec("EXP-03", "02_cnn_mlp", "02_cnn_mlp/1_MNIST_CNN.ipynb", "1_MNIST_CNN", "(28, 28, 1)", "(None, 10)", "SGD", "0.01", "categorical_crossentropy", "128", "30", "accuracy", "classification", "Final cell explicitly prints Test loss/accuracy."),
    BaselineSpec("EXP-04", "02_cnn_mlp", "02_cnn_mlp/2_Human_Gender_CNN.ipynb", "2_Human_Gender_CNN", "(64, 64, 1)", "(None, 2)", "Adam", "0.0001", "categorical_crossentropy", "128", "30", "accuracy", "classification", "Basic variant; np.rot90(k=3, axes=(1,2)); no callbacks."),
    BaselineSpec("EXP-05", "02_cnn_mlp", "02_cnn_mlp/3_Emotion_Reg_CNN.ipynb", "3_Emotion_Reg_CNN", "(48, 48, 1)", "(None, 7)", "Adam", "0.0001", "categorical_crossentropy", "128", "15", "accuracy", "classification", "Basic variant; notebook names seven FER classes."),
    BaselineSpec("EXP-06", "03_cnn_lstm", "03_cnn_lstm/MNIST_CNN_LSTM.ipynb", "MNIST_CNN_LSTM", "(8, 28, 28, 1)", "(None, 1)", "SGD", "0.01", "mean_absolute_error", "128", "30", "MAE", "regression", "TimeDistributed CNN + two LSTM layers; logs val_loss, no test evaluate output."),
    BaselineSpec("EXP-07", "04_training_utilities", "04_training_utilities/2_Human_Gender_CNN.ipynb", "2_Human_Gender_CNN_callbacks", "(64, 64, 1)", "(None, 2)", "Adam", "0.001", "categorical_crossentropy", "128", "100000 (EarlyStopping)", "accuracy", "classification", "ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard; execution stopped at epoch 32."),
    BaselineSpec("EXP-08", "04_training_utilities", "04_training_utilities/3_Emotion_Reg_CNN.ipynb", "3_Emotion_Reg_CNN_callbacks", "(48, 48, 1)", "(None, 7)", "Adam", "0.0001", "categorical_crossentropy", "128", "100 (EarlyStopping)", "accuracy", "classification", "ImageDataGenerator + EarlyStopping; execution stopped at epoch 57."),
    BaselineSpec("EXP-09", "05_others", "05_others/1_segnet.ipynb", "SegNet", "(96, 96, 1)", "(None, 9216, 7)", "Adam", "0.0001", "categorical_crossentropy", "8", "15", "accuracy", "segmentation", "7-class pixel labels; UpSampling2D decoder; Keras reports accuracy; pixel-level interpretation follows segmentation labels."),
    BaselineSpec("EXP-10", "05_others", "05_others/2_obj_detection.ipynb", "FCN_Object_Detection", "(64, 64, 3)", "(None, 1)", "adam", NOT_RECORDED, "mse", "128", "5", "accuracy", "legacy_regression_metric", "Vehicle score/regression; reported accuracy is not IoU/mAP; no training time printed."),
]

MODEL_NOTES = {
    "EXP-01": "Single hidden Dense sigmoid width B plus sigmoid output; B sweeps 8,16,...,512. Params=4B+1 (65 only at B=16). Training data size A and epochs C vary by loop; output is regression MSE, no saved result.",
    "EXP-02": "Flatten → Dense(512,ReLU) → Dropout(0.2) → Dense(512,ReLU) → Dropout(0.2) → Dense(10,softmax); pixels divided by 255.",
    "EXP-03": "Conv2D(64)→pool→Conv2D(128)→pool→Conv2D(256)→pool→Flatten→Dense(512)×2 with Dropout→10-way softmax.",
    "EXP-04": "Five CNN/pooling blocks, Flatten, two Dense(1024) layers; np.rot90(k=3, axes=(1,2)) before model; grayscale input per model.summary.",
    "EXP-05": "FER2013 48×48 grayscale, 2304 pixel columns + seven one-hot labels; CNN uses valid/same convolutions and Dense(1024), 7-way softmax.",
    "EXP-06": "8-frame MNIST sequence; TimeDistributed Conv/Pool/Dropout/Flatten → LSTM(128, return_sequences=True) → LSTM(128) → Dense(1); MAE.",
    "EXP-07": "Same Gender CNN architecture as EXP-04; ModelCheckpoint, EarlyStopping(patience=5), ReduceLROnPlateau(patience=5), TensorBoard; observed 32/100000 epochs.",
    "EXP-08": "Same Emotion CNN architecture as EXP-05; ImageDataGenerator and EarlyStopping(patience=5); observed 57/100 epochs.",
    "EXP-09": "96×96 single-channel input; seven-class pixel segmentation; VGG-style encoder-decoder, UpSampling2D and softmax; 29,458,211 total parameters.",
    "EXP-10": "64×64 RGB vehicle score model; Lambda normalization x/127.5−1, Conv2D head with tanh output, MSE; printed accuracy is not detection IoU/mAP.",
}

CSV_FIELDS = [
    "exp_id", "group", "model_name", "notebook_path", "input_shape_nhwc",
    "output_shape", "total_params", "trainable_params", "non_trainable_params",
    "params_provenance", "parameter_formula", "optimizer", "learning_rate", "loss", "batch_size",
    "configured_epochs", "epochs_observed", "train_loss_last_epoch",
    "val_loss_last_epoch", "train_metric_name", "train_metric_last_epoch",
    "val_metric_name", "val_metric_last_epoch",
    "notebook_printed_eval_loss", "notebook_printed_eval_metric_name", "notebook_printed_eval_metric",
    "notebook_eval_label", "evaluation_data_split", "training_time_seconds",
    "time_provenance", "evidence_source", "metric_kind", "notes",
]

EVALUATION_SPLITS = {
    "EXP-01": "x_test/y_test in notebook; exact sample count NOT_RECORDED",
    "EXP-02": "MNIST test split from keras.datasets.mnist.load_data; same split passed as validation_data and later evaluate",
    "EXP-03": "MNIST test split from keras.datasets.mnist.load_data; same split passed as validation_data and later evaluate",
    "EXP-04": "Notebook variable x_test loaded from filenames containing val_x/val_y (validation file)",
    "EXP-05": "fer2013_publictest_onehot.csv; notebook variable x_testing",
    "EXP-06": "No evaluate/test cell output saved; fit log contains validation loss",
    "EXP-07": "Notebook x_test loaded from WIKI validation file; calls it Test in print label",
    "EXP-08": "fer2013_publictest_onehot.csv; notebook variable x_testing",
    "EXP-09": "data/segnet/test/depth and paired labels; notebook test arrays",
    "EXP-10": "Random train_test_split holdout (test_size=0.30); split seed not fixed in notebook",
}

REPORTED_OUTPUT_RULES = {
    "EXP-04": "training log x_test loads filenames named val_x/val_y; final evaluate uses that same split",
    "EXP-05": "training validation_data and final evaluate both use fer2013_publictest_onehot.csv",
    "EXP-07": "training validation_data and final evaluate both use WIKI validation arrays",
    "EXP-08": "training validation_data and final evaluate both use FER public-test CSV",
    "EXP-09": "fit validation_data and final evaluate use notebook x_test/y_test",
    "EXP-10": "fit validation_data and final evaluate use the same X_test/Y_test holdout",
}


def _output_text(output: dict[str, Any]) -> str:
    """Nối text output của một notebook cell mà không thực thi cell."""
    chunks: list[str] = []
    stream = output.get("text", "")
    chunks.append("".join(stream) if isinstance(stream, list) else str(stream))
    for value in output.get("data", {}).values():
        if isinstance(value, list):
            chunks.append("".join(str(part) for part in value))
        elif isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks)


def _summary_value(text: str, label: str) -> str:
    """Lấy chính xác giá trị một dòng summary, tránh khớp nhầm substring."""
    match = re.search(rf"(?m)^\s*{re.escape(label)}:\s*([\d,]+)", text)
    return match.group(1).replace(",", "") if match else NOT_RECORDED


def _last_epoch_metrics(text: str) -> tuple[str, dict[str, str]]:
    """Trích số liệu progress bar ở epoch cuối được notebook lưu."""
    matches = list(re.finditer(r"Epoch\s+(\d+)/(\d+)\s*\r?\n([^\r\n]+)", text))
    if not matches:
        return NOT_RECORDED, {}
    match = matches[-1]
    metrics: dict[str, str] = {}
    for key, value in re.findall(r"\b(loss|val_loss|accuracy|val_accuracy|acc|val_acc):\s*([0-9.eE+-]+)", match.group(3)):
        normalized = {"acc": "accuracy", "val_acc": "val_accuracy"}.get(key, key)
        metrics[normalized] = value
    return match.group(1), metrics


def _collect_cells(notebook: dict[str, Any]) -> tuple[str, str, list[tuple[int, str]]]:
    """Thu thập mã nguồn, outputs và provenance theo chỉ số cell 0-based."""
    source_parts: list[str] = []
    output_parts: list[str] = []
    evidence: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source_parts.append(f"# CELL {index}\n" + "".join(cell.get("source", [])))
        for output in cell.get("outputs", []):
            text = _output_text(output)
            if text.strip():
                output_parts.append(text)
                evidence.append((index, text))
    return "\n".join(source_parts), "\n".join(output_parts), evidence


def _first_source_cell(evidence: list[tuple[int, str]], token: str) -> str:
    """Trả số cell đầu tiên có chứa token nguồn, nếu có."""
    for index, text in evidence:
        if token in text:
            return str(index)
    return NOT_RECORDED


def extract_row(project_root: Path, spec: BaselineSpec) -> dict[str, str]:
    """Trích metadata và metrics đã lưu từ một notebook baseline."""
    path = project_root / "legacy_keras" / spec.relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source, output, evidence = _collect_cells(notebook)
    epoch, train_metrics = _last_epoch_metrics(output)

    total = _summary_value(output, "Total params")
    trainable = _summary_value(output, "Trainable params")
    non_trainable = _summary_value(output, "Non-trainable params")
    params_provenance = "model.summary output" if total != NOT_RECORDED else "NOT_RECORDED; no saved model.summary output"
    if spec.analytic_parameters != NOT_RECORDED:
        total = spec.analytic_parameters.split()[0]
        trainable = total
        non_trainable = "0"
        params_provenance = spec.parameter_provenance

    test_loss_match = re.findall(r"Test (?:score|loss):\s*([0-9.eE+-]+)", output)
    test_accuracy_match = re.findall(r"Test accuracy:\s*([0-9.eE+-]+)", output, re.I)
    time_match = re.findall(
        r">\s*training time(?:\s+is)?\s*:?\s*([0-9.eE+-]+)\s*(minutes|s)\b",
        output,
        re.I,
    )
    time_seconds = NOT_RECORDED
    if time_match:
        value, unit = time_match[-1]
        time_seconds = f"{float(value) * (60 if unit.lower().startswith('min') else 1):.6f}"

    fit_cells = [
        index for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
        and re.search(r"(?:model|autoencoder)\.fit(?:_generator)?\s*\(", "".join(cell.get("source", [])))
    ]
    compile_cells = [
        index for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
        and re.search(r"(?:model|autoencoder)\.compile\s*\(", "".join(cell.get("source", [])))
    ]
    summary_cells = [index for index, text in evidence if "Total params:" in text]
    epoch_output_cells = [
        index for index, text in evidence if re.search(r"Epoch\s+\d+/\d+", text)
    ]
    test_output_cells = [
        index for index, text in evidence
        if re.search(r"Test (?:score|loss|accuracy):", text, re.I)
    ]
    source_evidence = f"legacy_keras/{spec.relative_path}"
    source_evidence += f"; summary_cells={summary_cells or NOT_RECORDED}"
    source_evidence += f"; epoch_output_cells={epoch_output_cells or NOT_RECORDED}"
    source_evidence += f"; test_output_cells={test_output_cells or NOT_RECORDED}"
    source_evidence += f"; fit_cell={fit_cells[-1] if fit_cells else NOT_RECORDED}"
    source_evidence += f"; compile_cell={compile_cells[-1] if compile_cells else NOT_RECORDED}"

    notebook_eval_metric = test_accuracy_match[-1] if test_accuracy_match else NOT_RECORDED
    notebook_eval_metric_name = spec.primary_metric if notebook_eval_metric != NOT_RECORDED else NOT_RECORDED

    evaluation_data_split = EVALUATION_SPLITS[spec.exp_id]
    if spec.exp_id in {"EXP-04", "EXP-05", "EXP-07", "EXP-08"}:
        evaluation_data_split += "; printed Test metric reuses validation/public-test data, not an independent final holdout"
    if spec.exp_id in REPORTED_OUTPUT_RULES:
        evaluation_data_split += "; " + REPORTED_OUTPUT_RULES[spec.exp_id]

    return {
        "exp_id": spec.exp_id,
        "group": spec.group,
        "model_name": spec.model_name,
        "notebook_path": f"legacy_keras/{spec.relative_path}",
        "input_shape_nhwc": spec.input_shape,
        "output_shape": spec.output_shape,
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "params_provenance": params_provenance,
        "parameter_formula": spec.parameter_formula,
        "optimizer": spec.optimizer,
        "learning_rate": spec.learning_rate,
        "loss": spec.loss,
        "batch_size": spec.batch_size,
        "configured_epochs": spec.configured_epochs,
        "epochs_observed": epoch,
        "train_loss_last_epoch": train_metrics.get("loss", NOT_RECORDED),
        "val_loss_last_epoch": train_metrics.get("val_loss", NOT_RECORDED),
        "train_metric_name": "accuracy" if "accuracy" in train_metrics else NOT_RECORDED,
        "train_metric_last_epoch": train_metrics.get("accuracy", NOT_RECORDED),
        "val_metric_name": "accuracy" if "val_accuracy" in train_metrics else NOT_RECORDED,
        "val_metric_last_epoch": train_metrics.get("val_accuracy", NOT_RECORDED),
        "notebook_printed_eval_loss": test_loss_match[-1] if test_loss_match else NOT_RECORDED,
        "notebook_printed_eval_metric_name": notebook_eval_metric_name,
        "notebook_printed_eval_metric": notebook_eval_metric,
        "notebook_eval_label": "Test" if test_loss_match or test_accuracy_match else NOT_RECORDED,
        "evaluation_data_split": evaluation_data_split,
        "training_time_seconds": time_seconds,
        "time_provenance": "Notebook printed runtime; hardware not identified" if time_seconds != NOT_RECORDED else NOT_RECORDED,
        "evidence_source": source_evidence,
        "metric_kind": spec.metric_kind,
        "notes": spec.notes + " " + MODEL_NOTES[spec.exp_id],
    }


def write_csv(project_root: Path, destination: Path) -> int:
    """Ghi đúng 10 dòng CSV; không chạy notebook hay tạo metric mới."""
    rows = [extract_row(project_root, spec) for spec in SPECS]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    """Tìm project root từ vị trí script và xuất baseline CSV."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/csv_logs/keras_baseline_10_models.csv"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    destination = args.output if args.output.is_absolute() else project_root / args.output
    count = write_csv(project_root, destination)
    print(f"Đã trích xuất {count} notebook; không thực thi Keras. CSV: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())