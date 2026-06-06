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
| `--conf-thres`, `-c` | `80.0` | 上位 X% の高信頼度点を GLB に使用。`80` は下位 20% を除外 |
| `--mask-sky` / `--no-mask-sky` | OFF | MDA の sky mask で空領域を除外 |
| `--mask-black-bg` | OFF | 黒背景ピクセルを除外 |
| `--mask-white-bg` | OFF | 白背景ピクセルを除外 |
| `--no-show-cam` | OFF | GLB 内のカメラ可視化を無効化 |
| `--size` | `518` | MDA 推論の長辺サイズ。VGGT カスタムスクリプトの既定 target size と同じ |
| `--max-chunk` | `0` | 1 forward に入れる最大フレーム数。`0` は全画像を同一 forward |
| `--max-images` | `0` | 使用する最大画像枚数。`0` は全画像、`>0` は均等間引き |
| `--oom-action` | `exit` | CUDA OOM 時の動作。`exit` または `lower-size` |
| `--retry-size` | `384` | `--oom-action lower-size` の再実行サイズ |
| `--model-name` | `mda_mog_sky_l2` | MDA model registry 名 |
| `--env-name` | `mda` | 想定 conda 環境名の表示用 |
| `--num-max-points` | `1000000` | GLB 点群の最大点数 |

VGGT のような一体点群を狙う場合は、既定の `--max-chunk 0` のまま実行します。
これは公式 `run_inference_video.py` と同じく、同一シーンの画像を single forward の multi-view mode に入れる使い方です。

VGGT カスタムスクリプトは `load_and_preprocess_images()` の既定 `target_size=518` を使います。
MDA 版も既定 `--size 518` に合わせていますが、MDA 側は入力画像比率と patch size の倍数に合わせて実際の処理解像度を丸めます。

`--max-chunk 1` は省 VRAM 用ですが、各画像をほぼ独立推論するため cross-frame attention が切れます。
相対カメラ推定が弱くなり、カメラ位置が同一点付近に潰れた点群になりやすいため、完成された空間を得たい場合は非推奨です。

## 信頼度フィルタ

`--conf-thres` は「使用する上位パーセンテージ」です。
既定の `80` は、信頼度が高い上位 80% の点を使い、低信頼度の下位 20% を GLB から除外します。

| 指定値 | 動作 |
|---:|---|
| `100` | 信頼度では除外しない |
| `90` | 上位 90% を使用、下位 10% を除外 |
| `80` | 上位 80% を使用、下位 20% を除外 |
| `70` | 上位 70% を使用、下位 30% を除外 |
| `40` | 上位 40% を使用、下位 60% を除外 |
| `0` | 信頼度フィルタで全点を除外 |

## 出力

スクリプトは `--image-folder` で指定したフォルダ直下に以下を作ります。

| パス | 内容 |
|---|---|
| `<image-folder>/all_cameras_parameters.json` | VGGT 互換カメラ JSON |
| `<image-folder>/reconstructed_scene.glb` | MDA depth/conf から生成した点群 GLB。`--mask-sky` 指定時のみ sky mask を適用 |

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

## DA3/MDA の3D構成

DA3/MDA は古典的な特徴点マッチング、ICP、bundle adjustment で複数点群を後処理位置合わせする仕組みではありません。
同一 forward の multi-view 推論で depth、confidence、intrinsics、extrinsics を推定し、そのカメラで各 depth map を world 座標へ unproject して結合します。

GLB export の alignment は first camera 基準の glTF 座標変換と点群中心合わせです。
これは表示用の座標整理であり、画像間の幾何対応を最適化する処理ではありません。
DA3 標準 API には COLMAP 入力や pose-conditioned depth の経路もありますが、このカスタムスクリプトは VGGT JSON 互換を優先し、MDA DA3 の推定 camera/depth をそのまま使います。

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

- 既定は `--oom-action exit` なので、OOM が出たら再実行せず終了します。
- 1 回だけ解像度を下げて自動再実行する場合は `--oom-action lower-size --retry-size 384` を指定します。
- それでも不足する場合は `--retry-size 336` など、さらに小さい値で手動再実行します。
- 他の GPU プロセスを終了する

`--max-chunk` を分割すると実行は通りやすくなりますが、相対カメラ推定が弱くなります。
VGGT 的な一体点群を優先する場合は、chunk 分割より先に解像度を下げます。

### `No such operator xformers::swiglu_packedw`

conda 環境外のユーザー site-packages にある古い `xformers` を Python が拾うと発生します。
`run_mda_da3.py` は起動時にユーザー site-packages を `sys.path` から外します。
`demo.py` など別入口を直接実行する場合は、以下のようにユーザー site-packages を無効化してください。

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n mda python demo.py assets\examples\mono\painting\painting.jpeg --model_name mda_mog_sky_l2 --size 512 --max_chunk 1 --no-viewer
```

### 相対カメラ姿勢が弱い

`--max-chunk` が使用画像枚数より小さいと、フレーム間 attention が chunk ごとに分断されます。
まず `--max-chunk 0` で全画像を同一 forward に入れてください。
OOM する場合は `--oom-action lower-size --retry-size 384` のように解像度を下げます。

### `MDA root was not found`

通常は `python_mda_customScript` を MDA repo root 直下に置けば自動検出されます。
別の場所から実行する場合は `MDA_ROOT` を指定してください。

```powershell
$env:MDA_ROOT = "C:\aokiDev\MDA_ai"
conda run -n mda python C:\path\to\run_mda_da3.py -i C:\path\to\images
```
