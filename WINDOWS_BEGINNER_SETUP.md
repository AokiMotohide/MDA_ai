# 会社 PC 向け 商用 VGGT 環境構築ガイド

この手順は、個人 GitHub の `AokiMotohide/MDA_ai` にアクセスできない会社 PC を対象にします。Python、Conda、Git、MDA、VGGT が何も入っていない状態から開始します。

会社 PC は公式 MDA リポジトリ `biansy000/MDA` だけを GitHub から取得します。商用 VGGT 対応のカスタムスクリプトは、事前に社内承認済みの経路で配布された**ソースコード ZIP**を追加します。実行ファイル、Python 仮想環境、モデル重みを別 PC からコピーしません。

## 0. 構成と完了条件

| 区分 | 取得元 | 会社 PC での配置先 |
|---|---|---|
| MDA 本体 | [公式 MDA](https://github.com/biansy000/MDA) | `C:\work\MDA` |
| 商用 VGGT カスタムスクリプト | 社内承認済みのソース ZIP | `C:\work\MDA\python_mda_customScript` と `C:\work\MDA\scripts` |
| 商用 VGGT 重み | [Hugging Face の公式モデルページ](https://huggingface.co/facebook/VGGT-1B-Commercial) | Hugging Face キャッシュ |

以下が終われば環境構築は完了です。

1. `C:\work\MDA` が公式 MDA の clone になっている
2. カスタムスクリプト ZIP を展開し、必要な 6 ファイルがある
3. `mda` conda 環境が作成されている
4. Hugging Face で商用モデルの規約同意とローカル認証が終わっている
5. `VGGT-1B-Commercial` の重み取得が完了している
6. `run_vggt_commercial.py --help` が表示される

## 1. 事前条件

| 項目 | 必要な状態 | 確認方法 |
|---|---|---|
| OS | Windows 10 または Windows 11 | `winver` |
| GPU | NVIDIA GPU。CPU 実行は実用的な時間にならない | `nvidia-smi` |
| 空き容量 | Python 環境、PyTorch、商用 VGGT 重み、出力用に 25 GB 以上 | エクスプローラー |
| ネットワーク | 公式 MDA GitHub、PyTorch、Hugging Face へ HTTPS 接続できる | ブラウザー |
| Hugging Face アカウント | 商用モデルの規約同意と認証に使う | [Hugging Face](https://huggingface.co/join) |
| カスタム ZIP | `MDA_CommercialVGGT_CustomScripts.zip` | 社内共有フォルダ、Teams、社内成果物管理など |

Python、Conda、Git は事前に導入しません。Git は第 2 節で導入します。Python と Conda は第 6 節で Miniforge として導入されるため、Python の公式インストーラーを別途導入する必要はありません。

`nvidia-smi` が見つからない場合は NVIDIA ドライバーを導入して再起動します。公式 MDA GitHub まで遮断されている場合は、この手順を進められません。IT 管理者へ `github.com/biansy000/MDA`、PyTorch、Hugging Face への HTTPS 接続許可を依頼します。

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

Git の導入後は PowerShell を閉じて開き直し、もう一度 `git --version` を実行します。

## 3. 公式 MDA を GitHub から取得する

個人リポジトリは clone しません。公式 MDA だけを取得します。

```powershell
New-Item -ItemType Directory -Force C:\work | Out-Null
Set-Location C:\work
git clone https://github.com/biansy000/MDA.git
Set-Location C:\work\MDA
git submodule update --init --recursive
git remote -v
```

`origin` が `https://github.com/biansy000/MDA.git` を示せば正しい取得先です。以後の作業場所は `C:\work\MDA` です。

公式 MDA を更新する場合は次を実行します。カスタムスクリプトを別フォルダへ置く必要はありません。

```powershell
Set-Location C:\work\MDA
git pull --ff-only origin main
git submodule update --init --recursive
```

## 4. カスタムスクリプトを追加する

商用 VGGT 用のカスタムスクリプトは公式 MDA に含まれません。個人 GitHub に接続できない会社 PC では、`MDA_CommercialVGGT_CustomScripts.zip` を社内承認済みの経路で受け取ります。この ZIP は Python と PowerShell のソースコードだけを含み、実行ファイル、Python 環境、モデル重みは含みません。

ZIP を `C:\work` に置いた場合は、次を実行します。

```powershell
Set-Location C:\work\MDA
Expand-Archive -LiteralPath C:\work\MDA_CommercialVGGT_CustomScripts.zip -DestinationPath C:\work\MDA -Force
```

展開後、次の 6 ファイルがすべて存在することを確認します。

```powershell
$RequiredFiles = @(
    '.\python_mda_customScript\run_vggt.py',
    '.\python_mda_customScript\run_vggt_commercial.py',
    '.\python_mda_customScript\run_vggt_common.py',
    '.\scripts\setup_windows_mda_da3.ps1',
    '.\scripts\setup_windows_vggt.ps1',
    '.\scripts\setup_windows_commercial_vggt.ps1'
)
$RequiredFiles | ForEach-Object { "{0} : {1}" -f $_, (Test-Path -LiteralPath $_) }
```

6 行すべてが `True` なら正しく展開されています。`False` がある場合は ZIP の版が不足しています。`CUSTOM_SCRIPT_PACKAGE_MANIFEST.md` を含む最新版の ZIP を受け取ります。

## 5. 商用 VGGT の利用規約へ同意する

商用モデルは [facebook/VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) です。公式 VGGT リポジトリは、商用利用にはこの商用チェックポイントを使うよう案内しています。[VGGT 公式リポジトリ](https://github.com/facebookresearch/vggt)

以下をブラウザーで完了します。

1. Hugging Face にログインする
2. [商用 VGGT モデルページ](https://huggingface.co/facebook/VGGT-1B-Commercial) を開く
3. 利用規約を確認して同意する
4. [Access Tokens](https://huggingface.co/settings/tokens) で Read 権限のトークンを作成する

この手順を飛ばすと、重み取得は `Access denied`、`401`、`403` で停止します。トークンを ZIP、Git、Markdown、チャット、共有フォルダへ保存しません。

## 6. PowerShell のスクリプト実行をこの画面だけ許可する

スクリプト実行が許可されていない場合だけ、現在の PowerShell で次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

この設定は PowerShell を閉じると消えます。会社全体の実行ポリシーは変更しません。

## 7. Miniforge、MDA、商用 VGGT を導入する

公式 MDA に追加したカスタムスクリプトから、次を実行します。

```powershell
Set-Location C:\work\MDA
.\scripts\setup_windows_commercial_vggt.ps1
```

途中で Hugging Face のトークン入力が求められます。第 5 節で作った Read 権限トークンを貼り付けて Enter を押します。画面に文字が表示されない動作でも入力は受け付けています。

このスクリプトは次の順で処理します。

1. 不足している Miniforge3 と ffmpeg を `winget` で導入する。Miniforge3 には Python と Conda が含まれる
2. Python 3.10 の `mda` conda 環境を作成する
3. CUDA 12.8 用 PyTorch、xformers、公式 MDA の必要パッケージを導入する
4. 商用 VGGT に不要な MDA DA3 重みは取得しない
5. VGGT と SkyMask 用 `onnxruntime` を導入する
6. Hugging Face 認証後に `VGGT-1B-Commercial` の重みを取得する

商用 VGGT 重みは約 5 GB あります。ダウンロード中に PowerShell を閉じないでください。会社のプロキシ環境では `HTTPS_PROXY` の設定が必要になる場合があります。この設定値は IT 管理者から取得します。

## 8. 構築結果を確認する

セットアップが完了したら次を順に実行します。

```powershell
Set-Location C:\work\MDA
$Conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
& $Conda run -n mda python -c "import torch, xformers; from vggt.models.vggt import VGGT; import onnxruntime; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('xformers=', xformers.__version__); print('VGGT=OK'); print('SkyMask=OK')"
& $Conda run -n mda python python_mda_customScript\run_vggt_commercial.py --help
```

最初のコマンドで `VGGT=OK` と `SkyMask=OK` が出て、`cuda=True` なら GPU 実行環境です。二つ目のコマンドで `run_vggt_commercial.py` のオプション一覧が出れば、スクリプトの起動準備も完了しています。

`conda.exe` が見つからない場合は、次を実行して表示された `conda.exe` のパスを `$Conda` に設定します。

```powershell
Get-ChildItem -Path $env:USERPROFILE, $env:LOCALAPPDATA -Filter conda.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
```

## 9. 失敗時の切り分け

| 表示または症状 | 原因 | 対処 |
|---|---|---|
| `git clone` が失敗する | 公式 MDA GitHub への接続が遮断されている | IT 管理者へ `github.com/biansy000/MDA` の HTTPS 接続許可を依頼する |
| 個人 GitHub にアクセスできない | 会社のアクセス制限 | 正常な前提。個人 GitHub を使わず、第 4 節の社内配布 ZIP を使う |
| 6 ファイルの確認で `False` が出る | カスタム ZIP の内容が不足している | `CUSTOM_SCRIPT_PACKAGE_MANIFEST.md` を含む最新版を受け取る |
| `winget` が見つからない | Windows App Installer が未導入、または会社ポリシーで無効 | IT 管理者に App Installer または Miniforge3 の導入を依頼する |
| `conda was not found` | Miniforge3 導入が未完了 | 第 7 節のセットアップを再実行する |
| `Access denied`、`401`、`403` | 商用 VGGT の規約同意または Hugging Face 認証が未完了 | 第 5 節を完了し、`huggingface-cli login` をやり直す |
| `CUDA out of memory` | 入力枚数または解像度に対して VRAM が不足 | [実行ガイド](WINDOWS_RECONSTRUCTION_GUIDE.md) の OOM 対処を使う |
| `torch.cuda.is_available()` が `False` | NVIDIA ドライバー、GPU、CUDA 対応 PyTorch の不整合 | `nvidia-smi` を確認し、ドライバー更新後に第 7 節をやり直す |
