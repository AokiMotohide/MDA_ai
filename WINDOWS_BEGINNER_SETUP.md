# 会社 PC 向け Windows 環境構築ガイド

この手順は、Python、Conda、Git、MDA、VGGT が何も入っていない Windows PC で、GitHub から MDA_ai を取得し、商用利用向けの `VGGT-1B-Commercial` を使うまでの手順です。実行ファイル、Python 環境、モデル重みを別の PC からコピーしません。

このリポジトリは MDA のソースコードとカスタム実行スクリプトを含みます。商用再構成では `run_vggt_commercial.py` が Meta の `VGGT-1B-Commercial` を直接実行します。MDA の `VGGT_MOG_LogL2.ckpt` を商用 VGGT 重みへ置き換える処理は実装していません。

## 0. 完了条件

以下がすべて終われば環境構築は完了です。

1. GitHub から `AokiMotohide/MDA_ai` を clone している
2. `mda` conda 環境が作成されている
3. `vggt` と `onnxruntime` が `mda` 環境に入っている
4. Hugging Face で商用 VGGT の利用規約に同意し、ローカル認証が終わっている
5. `VGGT-1B-Commercial` の重み取得が完了している
6. 商用 VGGT スクリプトの `--help` が表示される

## 1. 事前条件

| 項目 | 必要な状態 | 確認方法 |
|---|---|---|
| OS | Windows 10 または Windows 11 | `winver` |
| GPU | NVIDIA GPU。CPU 実行は実用的な時間にならない | `nvidia-smi` |
| 管理権限 | 不要。`winget` はユーザー領域へ導入する | PowerShell を通常起動する |
| 空き容量 | ソース、Python 環境、PyTorch、商用 VGGT 重み、出力用に 25 GB 以上 | エクスプローラー |
| ネットワーク | GitHub、PyTorch、Hugging Face へ HTTPS 接続できる | ブラウザー |
| Hugging Face アカウント | 商用モデルの規約同意と認証に使う | [Hugging Face](https://huggingface.co/join) |

Python、Conda、Git は事前に導入しません。Git は次節で導入します。Python と Conda は第 6 節のセットアップが Miniforge として導入します。Miniforge は Python と Conda を同時に提供するため、Python の公式インストーラーを別途導入する必要はありません。

`nvidia-smi` が見つからない場合は NVIDIA ドライバーを導入して再起動します。会社ネットワークで GitHub または Hugging Face が遮断されている場合は、先に IT 管理者へ HTTPS 接続許可を依頼します。モデル重みや Python 環境を USB 経由で持ち込む運用にはしません。

## 2. Git を導入する

PowerShell を通常起動し、Git の有無を確認します。

```powershell
git --version
```

バージョンが表示されれば次節へ進みます。`git` が見つからない場合は次を実行します。

```powershell
winget install --id Git.Git --exact --source winget --accept-package-agreements --accept-source-agreements
```

`winget` も見つからない場合は、Microsoft Store の **アプリ インストーラー**を導入して PowerShell を開き直します。会社のポリシーで Microsoft Store が使えない場合は、IT 管理者へ Git の導入を依頼します。Git の公式配布元は [git-scm.com](https://git-scm.com/download/win) です。

完了後は PowerShell を閉じて開き直し、もう一度 `git --version` を実行します。

## 3. GitHub からソースを取得する

作業フォルダを作り、GitHub リポジトリを clone します。`C:\work` は任意のローカル作業フォルダに置き換えられます。OneDrive やネットワークドライブは使いません。

```powershell
New-Item -ItemType Directory -Force C:\work | Out-Null
Set-Location C:\work
git clone https://github.com/AokiMotohide/MDA_ai.git
Set-Location C:\work\MDA_ai
git submodule update --init --recursive
git status
```

`On branch main` と表示され、`nothing to commit, working tree clean` が出れば取得は完了です。

以後は必ずこのフォルダでコマンドを実行します。

```powershell
Set-Location C:\work\MDA_ai
```

更新版を取得する場合は次を実行します。推論結果や入力画像をリポジトリ内へ置かない限り、このコマンドで失われるデータはありません。

```powershell
git pull --ff-only origin main
git submodule update --init --recursive
```

## 4. 商用 VGGT の利用規約へ同意する

商用モデルは [facebook/VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) です。公式リポジトリは、商用利用にはこの新しい商用チェックポイントを使うよう案内しています。[VGGT 公式リポジトリ](https://github.com/facebookresearch/vggt)

以下をブラウザーで完了します。

1. Hugging Face にログインする
2. [商用 VGGT モデルページ](https://huggingface.co/facebook/VGGT-1B-Commercial) を開く
3. 利用規約を確認して同意する
4. [Access Tokens](https://huggingface.co/settings/tokens) で Read 権限のトークンを作成する

この手順を飛ばすと、後の重み取得は `Access denied` で必ず停止します。トークンを Git、Markdown、チャット、共有フォルダへ保存しません。

## 5. PowerShell のスクリプト実行をこの画面だけ許可する

スクリプト実行が許可されていない場合だけ、現在の PowerShell で次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

この設定は PowerShell を閉じると消えます。会社全体の実行ポリシーは変更しません。

## 6. Miniforge、MDA、商用 VGGT を導入する

リポジトリのルートで次を実行します。

```powershell
Set-Location C:\work\MDA_ai
.\scripts\setup_windows_commercial_vggt.ps1
```

途中で Hugging Face のトークン入力が求められます。第 4 節で作った Read 権限トークンを貼り付けて Enter を押します。画面に文字が表示されない動作でも入力は受け付けています。

このスクリプトは次の順で処理します。

1. 不足している Miniforge3 と ffmpeg を `winget` で導入する。Miniforge3 には Python と Conda が含まれる
2. Python 3.10 の `mda` conda 環境を作成する
3. CUDA 12.8 用 PyTorch、xformers、MDA の必要パッケージを導入する
4. MDA DA3 の重みを取得せずに基本環境だけを構築する
5. VGGT と SkyMask 用 `onnxruntime` を導入する
6. Hugging Face 認証後に `VGGT-1B-Commercial` の重みを取得する

商用 VGGT 重みは約 5 GB あります。ダウンロード中に PowerShell を閉じないでください。会社のプロキシ環境では `HTTPS_PROXY` の設定が必要になる場合があります。この設定値は IT 管理者から取得します。

## 7. 構築結果を確認する

セットアップが完了したら次を順に実行します。

```powershell
Set-Location C:\work\MDA_ai
$Conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
& $Conda run -n mda python -c "import torch, xformers; from vggt.models.vggt import VGGT; import onnxruntime; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('xformers=', xformers.__version__); print('VGGT=OK'); print('SkyMask=OK')"
& $Conda run -n mda python python_mda_customScript\run_vggt_commercial.py --help
```

最初のコマンドで `VGGT=OK` と `SkyMask=OK` が出て、`cuda=True` なら GPU 実行環境です。二つ目のコマンドで `run_vggt_commercial.py` のオプション一覧が出れば、スクリプトの起動準備も完了しています。

`conda.exe` が見つからない場合は、次を実行して表示された `conda.exe` のパスを `$Conda` に設定します。

```powershell
Get-ChildItem -Path $env:USERPROFILE, $env:LOCALAPPDATA -Filter conda.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
```

## 8. 商用重みだけを再取得する場合

環境構築後に商用重みの取得が失敗した場合は、規約同意とログイン状態を確認してから次を実行します。

```powershell
Set-Location C:\work\MDA_ai
$Conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
& $Conda run -n mda huggingface-cli login
.\scripts\setup_windows_vggt.ps1 -InstallCommercialModel
```

## 9. 失敗時の切り分け

| 表示または症状 | 原因 | 対処 |
|---|---|---|
| `winget` が見つからない | Windows App Installer が未導入、または会社ポリシーで無効 | IT 管理者に App Installer または Miniforge3 の導入を依頼する |
| `git` が見つからない | Git 導入後の PowerShell を開き直していない | PowerShell を閉じて開き直す |
| `conda was not found` | Miniforge3 導入が未完了 | `setup_windows_commercial_vggt.ps1` を再実行する |
| `Access denied`、`401`、`403` | 商用 VGGT の規約同意または Hugging Face 認証が未完了 | 第 4 節を完了し、`huggingface-cli login` をやり直す |
| `CUDA out of memory` | 入力枚数または解像度に対して VRAM が不足 | [実行ガイド](WINDOWS_RECONSTRUCTION_GUIDE.md) の OOM 対処を使う |
| `torch.cuda.is_available()` が `False` | NVIDIA ドライバー、GPU、CUDA 対応 PyTorch の不整合 | `nvidia-smi` を確認し、ドライバー更新後に第 6 節をやり直す |
| `No such operator xformers...` | ユーザー領域の古い xformers を読み込んでいる | `$env:PYTHONNOUSERSITE='1'` を設定してから再実行する |
