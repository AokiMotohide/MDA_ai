# MDA DA3 実行コマンドリファレンス

MDA DA3 版のカスタム再構成スクリプトの起動方法と出力仕様。
VGGT 用 `run_vggt.py` と同じアプリ連携・JSON 読み込みに寄せていますが、
使用モデルと点群生成ロジックは MDA DA3 です。

## スクリプト

| ファイル | 使用モデル | ライセンス | 用途 |
|---|---|---|---|
| `run_mda_da3.py` | `mda_mog_sky_l2` | Apache-2.0 | MDA DA3 推論、カメラ JSON、GLB 出力 |

モデル checkpoint は以下を使います。

```text
checkpoints/MDA/DA3_MOG_Sky_LogL2.ckpt
```

未構築の場合は MDA repo root で先に実行します。

```powershell
.\scripts\setup_windows_mda_da3.ps1
```

## クイックスタート

PowerShell で MDA repo root から実行します。

```powershell
conda run -n mda python python_mda_customScript\run_mda_da3.py -i my_images
```

`my_images/` 直下の `.png` / `.jpg` / `.jpeg` をソート順で読み込みます。

## オプション

| オプション | 既定値 | 内容 |
|---|---:|---|
| `--image-folder`, `-i` | `my_images` | 入力画像フォルダ |
| `--conf-thres`, `-c` | `60.0` | 下位 X% の低信頼度点を GLB から除外 |
| `--mask-sky` / `--no-mask-sky` | ON | MDA の sky mask で空領域を除外 |
| `--mask-black-bg` | OFF | 黒背景ピクセルを除外 |
| `--mask-white-bg` | OFF | 白背景ピクセルを除外 |
| `--no-show-cam` | OFF | GLB 内のカメラ可視化を無効化 |
| `--size` | `512` | MDA 推論の長辺サイズ |
| `--max-chunk` | `1` | 1 forward に入れる最大フレーム数 |
| `--model-name` | `mda_mog_sky_l2` | MDA model registry 名 |
| `--env-name` | `mda` | 想定 conda 環境名の表示用 |
| `--num-max-points` | `1000000` | GLB 点群の最大点数 |

8GB VRAM 環境では `--max-chunk 1` が安定です。複数画像の相対カメラ推定を重視する場合は、
VRAM に余裕がある範囲で `--max-chunk` を増やしてください。

## 出力

スクリプトは `--image-folder` で指定したフォルダ直下に以下を作ります。

| パス | 内容 |
|---|---|
| `<image-folder>/all_cameras_parameters.json` | VGGT 互換カメラ JSON |
| `<image-folder>/reconstructed_scene.glb` | MDA depth/conf/sky から生成した点群 GLB |

## JSON 形式

`all_cameras_parameters.json` は VGGT 版と同じ schema です。

```json
{
    "image001.jpg": {
        "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "extrinsics": [[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz]],
        "width": 504,
        "height": 280,
        "original_width": 504,
        "original_height": 284
    }
}
```

- `intrinsics` は pinhole camera の 3x3 K 行列です。
- `extrinsics` は OpenCV 規約の World-to-Camera `[R|t]` 3x4 行列です。
- `width` / `height` は MDA が実際に処理した解像度です。
- `original_width` / `original_height` は入力画像の元解像度です。
- 歪み係数は出力しません。MDA DA3 のこの推論経路は VGGT と同じく pinhole camera のみで、カメラ歪みまでは推定しません。

## VGGT 版との主な違い

- VGGT の `prediction-mode` はありません。MDA は depth/conf/camera から GLB 点群を作ります。
- VGGT の ONNX SkyMask ではなく、MDA の MoG sky component / `sky_mask` を使います。
- MDA は `depth_conf` と MoG の選択深度を使います。
- GLB のファイル名は同じですが、点群密度や見た目は VGGT と一致しません。
- JSON schema は同じですが、モデルが異なるため数値は VGGT と一致しません。

## 進捗マーカー

標準出力には VGGT 版と同じ形式の構造化マーカーを出します。

```text
▶ [PHASE:<phase_name>] (<progress>) <message>
```

使用する phase は以下です。

```text
startup, env_check, model_load, preprocess, infer, infer_done,
camera_calc, world_unproj, save_json, save_glb, save_glb_done, done
```

既存の `VGGTPhaseParser` と同じ正規表現で解析できます。

## トラブルシュート

### `DA3_MOG_Sky_LogL2.ckpt` が見つからない

MDA repo root で以下を実行してください。

```powershell
.\scripts\setup_windows_mda_da3.ps1
```

### `CUDA out of memory`

- `--max-chunk 1` を使う
- 入力画像枚数を減らす
- `--size 384` などに下げる
- 他の GPU プロセスを終了する

### 相対カメラ姿勢が弱い

`--max-chunk 1` は省 VRAM ですが、フレーム間 attention を使いにくくなります。
VRAM に余裕がある場合は `--max-chunk 2` 以上を試してください。

### `MDA root was not found`

通常は `python_mda_customScript` を MDA repo root 直下に置けば自動検出されます。
別の場所から実行する場合は `MDA_ROOT` を指定してください。

```powershell
$env:MDA_ROOT = "C:\aokiDev\MDA_ai"
conda run -n mda python C:\path\to\run_mda_da3.py -i C:\path\to\images
```
