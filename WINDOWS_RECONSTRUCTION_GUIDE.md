# Windows 実行ガイド

MDA DA3 と VGGT の実行方法、入力条件、パラメータ、出力ファイルを説明します。商用利用では `run_vggt_commercial.py` を使います。会社 PC では、公式 MDA と社内配布のカスタムスクリプト ZIP を使って環境を構築します。詳細は [会社 PC 向け商用 VGGT 環境構築ガイド](WINDOWS_BEGINNER_SETUP.md) を参照します。

## 1. 入力画像を用意する

1 シーンごとに 1 フォルダを作成し、`.png`、`.jpg`、`.jpeg` の画像を直接入れます。サブフォルダ内の画像は読み込みません。

```text
C:\work\scene01\
  0001.jpg
  0002.jpg
  0003.jpg
```

画像は同じ場所を異なる位置から撮影したものを使います。被写体がほとんど重ならない画像、強いブレ、撮影途中で大きく変形した被写体では、カメラ姿勢と点群が崩れます。

以降のコマンドでは、Miniforge が導入した Conda を明示的に使います。PowerShell を開き直して `conda` コマンドが認識されない場合でも動作します。

```powershell
$Conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
```

## 2. 最初の MDA DA3 実行

まず 2〜5 枚の画像で動作確認します。

```powershell
Set-Location C:\work\MDA_ai
& $Conda run -n mda python python_mda_customScript\run_mda_da3.py -i C:\work\scene01 --oom-action lower-size --retry-size 384
```

`--max-chunk` は指定しません。既定の `0` は全画像を同じ forward に入れるため、画像間の相対カメラ推定を維持します。CUDA out of memory が起きた場合だけ、解像度を 384 に下げて 1 回再試行します。

## 3. MDA DA3 のパラメータ

| パラメータ | 既定値 | 指定例 | 効果 |
|---|---:|---|---|
| `--image-folder`, `-i` | `my_images` | `-i C:\work\scene01` | 入力フォルダ |
| `--size` | `518` | `--size 384` | 推論解像度。小さくすると VRAM 使用量は下がり、細部は減る |
| `--max-chunk` | `0` | `--max-chunk 0` | 0 は全画像を一括推論。1 以上は分割するため、相対姿勢が弱くなる |
| `--max-images` | `0` | `--max-images 30` | 0 は全画像。30 は始点と終点を含めて 30 枚へ均等間引き |
| `--oom-action` | `exit` | `--oom-action lower-size` | OOM 発生時に自動で 1 回だけ再試行する |
| `--retry-size` | `384` | `--retry-size 336` | OOM 再試行時の解像度 |
| `--conf-thres`, `-c` | `80` | `-c 70` | 上位 X% の信頼度点を使用する。70 は下位 30% を除外 |
| `--mask-sky` | OFF | `--mask-sky` | MDA の空マスクで空領域を除外 |
| `--mask-black-bg` | OFF | `--mask-black-bg` | 黒背景を除外 |
| `--mask-white-bg` | OFF | `--mask-white-bg` | 白背景を除外 |
| `--no-show-cam` | OFF | `--no-show-cam` | GLB 内のカメラ可視化を無効化 |
| `--num-max-points` | `1000000` | `--num-max-points 300000` | GLB に残す最大点数。小さくするとファイルは軽くなる |

`--conf-thres` は絶対値ではなく割合です。MDA DA3 の `80` は上位 80% を残し、下位 20% を除外します。

## 4. 商用 VGGT を実行する

商用モデルの規約同意と認証が完了している場合は、次を使います。

```powershell
Set-Location C:\work\MDA_ai
& $Conda run -n mda python python_mda_customScript\run_vggt_commercial.py -i C:\work\scene01
```

初回はモデル重みの取得に時間がかかります。商用版は認証が完了していないと `Access denied` で終了します。商用モデルは Meta の VGGT を直接実行します。MDA の MoG 深度モデルを重ねて実行する経路ではありません。

## 5. VGGT のパラメータ

| パラメータ | 既定値 | 指定例 | 効果 |
|---|---:|---|---|
| `--image-folder`, `-i` | `my_images` | `-i C:\work\scene01` | 入力フォルダ |
| `--conf-thres`, `-c` | `60` | `-c 50` | 下位 X% の信頼度点を除外する。60 は上位 40% を残す |
| `--mask-sky` / `--no-mask-sky` | ON | `--no-mask-sky` | SkyMask の有効・無効 |
| `--mask-black-bg` | OFF | `--mask-black-bg` | 黒背景を除外 |
| `--mask-white-bg` | OFF | `--mask-white-bg` | 白背景を除外 |
| `--no-show-cam` | OFF | `--no-show-cam` | GLB 内のカメラ可視化を無効化 |
| `--prediction-mode` | `Depthmap and Camera Branch` | `--prediction-mode "Pointmap Branch"` | 点群を depth または pointmap から生成 |

MDA DA3 と VGGT では `--conf-thres` の意味が逆です。MDA DA3 の `80` は上位 80% を残します。VGGT の `60` は下位 60% を除外し、上位 40% を残します。

## 6. 出力を確認する

両スクリプトとも、入力フォルダに次の 2 ファイルを作成します。

| ファイル | 内容 |
|---|---|
| `all_cameras_parameters.json` | Camera Parameters v2。内部行列、World-to-Camera外部行列、解像度に加え、画像前処理変換、歪みモデル、GLB基準のcamera-to-scene、推定元情報を保存 |
| `reconstructed_scene.glb` | 点群と任意のカメラ可視化を含む GLB |

カメラJSONは従来の6項目を維持したCamera Parameters v2です。旧読込側は追加項目を無視でき、新読込側は不足項目へ既定値を設定します。

## 7. 実行結果が崩れる場合の判断

| 症状 | 先に確認する項目 | 対処 |
|---|---|---|
| カメラ位置が同じ場所に集まる | `--max-chunk` | `--max-chunk 0` で再実行する。画像数を減らす場合は `--max-images` を使う |
| 点が多すぎて見づらい | `--conf-thres` | MDA は値を下げる。VGGT は値を上げる |
| GLB が重い | `--num-max-points` | MDA DA3 は 300000 などに下げる。VGGT は信頼度フィルタを強める |
| 空や背景が残る | 背景マスク | MDA は `--mask-sky`。VGGT は既定の SkyMask を使う。白背景・黒背景では対応する背景マスクを追加する |
| OOM が発生する | 解像度と入力画像数 | `--oom-action lower-size --retry-size 384` を付ける。改善しなければ `--retry-size 336` を使う |
