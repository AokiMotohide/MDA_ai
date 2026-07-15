# Windows 初心者向け環境構築ガイド

この手順は `C:\aokiDev\MDA_ai` で MDA DA3 と VGGT を動かすための Windows 用手順です。想定外の GPU ドライバー不具合、ネットワーク遮断、Hugging Face の権限拒否までは自動復旧できません。各手順の確認結果を見れば、失敗箇所を特定できます。

## 1. 事前条件

| 項目 | 必要な状態 | 確認コマンド |
|---|---|---|
| OS | Windows 10 または Windows 11 | `winver` |
| GPU | NVIDIA GPU。CPU 実行は非常に遅い | `nvidia-smi` |
| 空き容量 | MDA、VGGT、出力用に十分な空き容量。商用 VGGT 重みだけで約 5 GB | エクスプローラー |
| ネットワーク | GitHub、PyTorch、Hugging Face に接続可能 | ブラウザー |
| 実行場所 | MDA リポジトリのルート | `Get-Location` |

`nvidia-smi` が見つからない場合は NVIDIA ドライバーをインストールして再起動します。GPU がない環境では構築自体はできますが、実用的な推論時間になりません。

## 2. PowerShell を開く

PowerShell を開き、MDA リポジトリへ移動します。

```powershell
Set-Location C:\aokiDev\MDA_ai
Get-Location
```

最後の出力が `C:\aokiDev\MDA_ai` なら正しい場所です。

PowerShell のスクリプト実行が禁止されている場合だけ、現在の PowerShell で次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

この設定は開いている PowerShell を閉じると元に戻ります。

## 3. MDA DA3 を構築する

次を 1 回実行します。

```powershell
.\scripts\setup_windows_mda_da3.ps1
```

スクリプトは不足している Miniforge3 と ffmpeg を `winget` で導入し、`mda` という conda 環境を作成します。続いて PyTorch、xformers、MDA 本体、DA3 MDA 重みを導入します。

完了後、次を実行します。

```powershell
C:\Users\aokim\miniforge3\Scripts\conda.exe run -n mda python -c "import torch, xformers; print(torch.__version__); print(torch.cuda.is_available()); print(xformers.__version__)"
Test-Path .\checkpoints\MDA\DA3_MOG_Sky_LogL2.ckpt
```

最後の行が `True` なら DA3 MDA 重みは配置済みです。`torch.cuda.is_available()` が `True` なら GPU 推論を使えます。

`conda.exe` の場所が異なる場合は、次で conda の実体を確認します。

```powershell
Get-Command conda
```

以降の `C:\Users\aokim\miniforge3\Scripts\conda.exe` は、その結果のパスに置き換えます。

## 4. VGGT を構築する

DA3 の構築が完了してから次を実行します。

```powershell
.\scripts\setup_windows_vggt.ps1
```

この処理は既存の `mda` 環境へ VGGT と SkyMask 用の `onnxruntime` を追加します。非商用 VGGT 重みは初回推論時に Hugging Face キャッシュへ保存されます。

導入確認は次です。

```powershell
C:\Users\aokim\miniforge3\Scripts\conda.exe run -n mda python -c "from vggt.models.vggt import VGGT; import onnxruntime; print('VGGT: OK'); print('SkyMask: OK')"
```

`VGGT: OK` と `SkyMask: OK` が出れば導入は完了です。

## 5. 商用 VGGT 重みを使う場合

商用版 `facebook/VGGT-1B-Commercial` は Hugging Face のゲート付きモデルです。規約への同意と Read 権限トークンによる認証がない状態ではダウンロードできません。

1. ブラウザーで `https://huggingface.co/facebook/VGGT-1B-Commercial` を開く
2. Hugging Face にログインし、利用規約へ同意する
3. `https://huggingface.co/settings/tokens` で Read 権限のトークンを作成する
4. PowerShell で認証する

```powershell
C:\Users\aokim\miniforge3\Scripts\conda.exe run -n mda huggingface-cli login
```

トークンを入力後、重みを取得します。

```powershell
.\scripts\setup_windows_vggt.ps1 -InstallCommercialModel
```

## 6. よくある失敗

| 表示または症状 | 原因 | 対処 |
|---|---|---|
| `conda was not found` | Miniforge3 が未導入、または新しい PowerShell を開いていない | `setup_windows_mda_da3.ps1` を実行し直す。完了後に PowerShell を開き直す |
| `DA3_MOG_Sky_LogL2.ckpt was not found` | 重みのダウンロードが未完了 | `setup_windows_mda_da3.ps1` を再実行する |
| `No such operator xformers...` | ユーザー領域の古い xformers を読み込んでいる | カスタムスクリプトから実行する。独自コマンドでは `$env:PYTHONNOUSERSITE='1'` を設定する |
| `CUDA out of memory` | 入力画像数または解像度に対して VRAM が不足 | 実行ガイドの `--oom-action lower-size` を使う。`--max-chunk` を先に小さくしない |
| 商用モデルで `Access denied` | Hugging Face の規約同意または認証が未完了 | 前節の 1〜4 を完了する |
| `torch.cuda.is_available()` が `False` | NVIDIA ドライバー、GPU、CUDA 対応 PyTorch のいずれかが不一致 | `nvidia-smi` を確認し、ドライバーを更新後に環境構築をやり直す |
