# MDA ワークスペースの VGGT 実行

`mda` conda 環境に VGGT を追加し、MDA DA3 版と同じ出力契約で実行する。

## 初回セットアップ

```powershell
.\scripts\setup_windows_vggt.ps1
```

この処理は `facebookresearch/vggt` の Python パッケージと SkyMask 用の
`onnxruntime` を `mda` 環境へ導入する。VGGT 重みは初回実行時に Hugging
Face キャッシュへ保存する。

## 実行

```powershell
conda run -n mda python python_mda_customScript\run_vggt.py -i my_images
```

商用モデルを使う場合は、先に Hugging Face で
[`facebook/VGGT-1B-Commercial`](https://huggingface.co/facebook/VGGT-1B-Commercial)
の利用規約へ同意し、Read 権限のトークンで `huggingface-cli login` を行う。

```powershell
conda run -n mda python python_mda_customScript\run_vggt_commercial.py -i my_images
```

重みだけを先に取得する場合は次を実行する。

```powershell
.\scripts\setup_windows_vggt.ps1 -InstallCommercialModel
```

## オプションと出力

指定先の VGGT カスタムスクリプトと同じオプションを受け付ける。

| オプション | 既定値 | 内容 |
|---|---:|---|
| `--image-folder`, `-i` | `my_images` | 入力画像フォルダ |
| `--conf-thres`, `-c` | `60` | 下位 X% の信頼度点を除外 |
| `--mask-sky` / `--no-mask-sky` | ON | SkyMask の有効化切替 |
| `--mask-black-bg` | OFF | 黒背景を除外 |
| `--mask-white-bg` | OFF | 白背景を除外 |
| `--no-show-cam` | OFF | GLB のカメラ可視化を無効化 |
| `--prediction-mode` | `Depthmap and Camera Branch` | 深度由来または Pointmap 由来の点群を選択 |

出力は入力フォルダ直下の `all_cameras_parameters.json` と
`reconstructed_scene.glb`。JSON は `intrinsics`、`extrinsics`、`width`、
`height`、`original_width`、`original_height` を後方互換で維持した Camera Parameters v2である。画像前処理変換、歪みモデル、GLB基準のcamera-to-scene、推定元情報も保存する。
