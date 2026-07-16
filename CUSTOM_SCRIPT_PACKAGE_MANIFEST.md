# 商用 VGGT カスタムスクリプト配布物

`MDA_CommercialVGGT_CustomScripts.zip` は、公式 MDA に追加するソースコード配布物です。Python 実行環境、モデル重み、実行ファイルを含めません。

展開先は公式 MDA リポジトリのルートです。

```text
C:\work\MDA\
```

必須ファイルは以下です。

```text
python_mda_customScript\run_vggt.py
python_mda_customScript\run_vggt_commercial.py
python_mda_customScript\run_vggt_common.py
scripts\setup_windows_mda_da3.ps1
scripts\setup_windows_vggt.ps1
scripts\setup_windows_commercial_vggt.ps1
```

この配布物は公式 MDA の `src\`、`configs\`、`demo.py`、`pyproject.toml` を上書きしません。

作成元の管理 PC では、次を実行して ZIP を作成します。

```powershell
.\scripts\export_commercial_vggt_custom_scripts.ps1 -OutputDirectory C:\work\delivery
```

出力された ZIP を会社の承認済みファイル共有、Teams、社内成果物管理へ登録します。個人 GitHub から会社 PC へ取得する手順は使いません。
