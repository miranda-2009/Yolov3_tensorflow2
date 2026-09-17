# YOLOv3 Object Detection with TensorFlow 2

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

使用 **TensorFlow 2 / Keras** 實作 YOLOv3 物件偵測。本專案以教學用 Colab Notebook 為基礎，加入詳細繁體中文註解，說明從 Darknet 權重轉換、模型架構、預測框解碼、非極大值抑制（NMS）到圖片推論的完整流程。

> This project implements YOLOv3 object detection with TensorFlow 2 and Keras. The notebook contains detailed Traditional Chinese explanations for learning and review.

## 專案內容｜Project Overview

Notebook 涵蓋以下主題：

- 在 Google Colab 準備 TensorFlow 2 執行環境
- 下載 YOLOv3 官方預訓練權重
- 將 Darknet 權重格式轉換為 Keras 可使用的格式
- 建立 Darknet-53 骨幹網路與殘差區塊
- 建立 YOLOv3 三尺度偵測頭
- 說明 Anchor boxes 與 Anchor masks
- 將模型輸出解碼為邊界框、物件分數及類別機率
- 使用 Non-Maximum Suppression 移除重複預測框
- 說明 IoU 與 YOLOv3 Loss 的計算概念
- 上傳圖片並顯示物件偵測結果

## YOLOv3 架構簡介｜Architecture

YOLOv3 使用 **Darknet-53** 作為特徵擷取網路，並在三個不同尺度進行物件偵測：

| 輸出尺度 | 常用網格大小（輸入 416×416） | 主要偵測目標 |
|---|---:|---|
| Large-scale feature map | 52×52 | 小型物件 |
| Medium-scale feature map | 26×26 | 中型物件 |
| Small-scale feature map | 13×13 | 大型物件 |

每個網格使用 3 組 Anchor boxes。每組 Anchor 會預測：

```text
x, y, width, height, objectness, class probabilities
```

本專案使用在 **COCO dataset** 上訓練的 YOLOv3 權重，可辨識 80 種常見物件，例如 person、car、dog、cat、bicycle 等。

## 專案結構｜Repository Structure

```text
YOLOv3-TensorFlow2/
├── README.md
└── YOLOv3_TensorFlow2.ipynb
```

## 使用方式｜How to Run

1. 下載本專案或將 Notebook 儲存到自己的 Google Drive。
2. 使用 [Google Colab](https://colab.research.google.com/) 開啟 Notebook。
3. 建議在選單選擇：`執行階段 → 變更執行階段類型 → T4 GPU`。
4. 依照 Notebook 順序執行所有儲存格。
5. 出現檔案上傳按鈕時，選擇一張 JPG 或 PNG 圖片。
6. 模型會輸出畫有偵測框、類別名稱及信心分數的圖片。


## 主要套件｜Requirements

本專案主要使用：

- Python 3
- TensorFlow 2 / Keras
- NumPy
- OpenCV
- absl-py
- Google Colab

在 Colab 中，大部分套件已預先安裝。YOLOv3 權重則由 Notebook 使用以下指令下載：

```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

## 推論流程｜Inference Pipeline

1. 將輸入圖片縮放為 416×416。
2. 將像素值除以 255，正規化到 0～1。
3. 經過 Darknet-53 擷取圖片特徵。
4. 在 13×13、26×26、52×52 三種尺度產生候選框。
5. 將原始輸出解碼為正規化邊界框座標。
6. 計算 `objectness × class probability` 作為類別分數。
7. 使用 NMS 移除重複且高度重疊的預測框。
8. 將保留的預測框、類別名稱及分數畫回原圖。

## 重要參數｜Important Parameters

```python
yolo_iou_threshold = 0.6
yolo_score_threshold = 0.6
size = 416
num_classes = 80
```

| 參數 | 說明 |
|---|---|
| `yolo_iou_threshold` | NMS 判斷預測框是否過度重疊的 IoU 門檻 |
| `yolo_score_threshold` | 保留預測框所需的最低信心分數 |
| `size` | 模型輸入圖片的寬與高 |
| `num_classes` | COCO 預訓練模型的物件類別數 |

若偵測框太少，可以稍微降低 `yolo_score_threshold`；但門檻太低也會增加誤判。

## 輸出結果｜Output

推論完成後，Notebook 會：

- 在 Colab 中顯示偵測結果
- 在圖片上標示 Bounding Box
- 顯示 COCO 類別名稱
- 顯示模型信心分數
- 將結果儲存為 `output.jpg`





### YOLOv3 權重下載失敗

官方網站有時可能暫時無法連線。此時可稍後重試，或先自行下載 `yolov3.weights`，再上傳到 Colab 工作目錄。請只使用可信任的來源。

## 學習重點｜What I Learned

透過此專案可以理解：

- One-stage object detector 的基本流程
- CNN、殘差連接與多尺度特徵融合
- YOLOv3 如何使用 Anchor boxes 產生預測框
- IoU 與 Non-Maximum Suppression 的用途
- Darknet 與 TensorFlow/Keras 權重格式的差異
- 如何在 Colab 載入模型並完成圖片推論

## 限制｜Limitations

- 此 Notebook 主要用於教學與圖片推論示範。
- 預訓練模型僅能辨識 COCO 的 80 個類別。
- 執行結果可能因 Colab 的 TensorFlow、Keras 或 NumPy 版本而異。

## References

- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [YOLOv3 TensorFlow 2 Implementation](https://github.com/zzh8829/yolov3-tf2)
- [COCO Dataset](https://cocodataset.org/)
- [DataHacker](https://datahacker.rs/)

## Acknowledgements

本專案以課堂教材為基礎，程式架構參考公開的 YOLOv3 TensorFlow 2 實作；本人針對程式流程加入繁體中文註解與學習整理。原始模型與相關實作之著作權歸各自作者所有。

---

如果這份專案對你有幫助，歡迎在 GitHub 上給予 ⭐。
