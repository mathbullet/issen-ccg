# Issen

Issenは、CCGBankから独立に学習した英語のCCGパーサです。
文字CNNと2層の双方向LSTMで各語のCCGカテゴリ・係り先・品詞を推定し、C++で実装したA*探索で構文木を求めます。
推論はCPU上のONNX Runtimeで行い、モデルは量子化していないfloat32で約16MBです。
構文木はAUTO、JSON、Jigg XMLで出力でき、Jigg XMLはccg2lambdaの意味解析にそのまま渡せます。

## インストール

Python 3.11以上と、C++17に対応したコンパイラが必要です。
インストール時に探索器をビルドします。

```sh
uv add "issen-ccg[infer,morphology] @ git+https://github.com/mathbullet/issen-ccg" --tag v0.1.0
```

`infer` はONNX Runtime、`morphology` はJigg出力のlemma生成に使うMorphoDiTaを導入します。
AUTOとJSONだけを使う場合、`morphology` は不要です。

## モデル

モデルはパッケージに含まれていません。
次の2つを入手し、実行時にパスを指定してください。

- Issenのモデル：[Releases](https://github.com/mathbullet/issen-ccg/releases) の `issen-ccg-model-v0.1.0.tar.gz` を展開したディレクトリです。`--model` に指定します。
- MorphoDiTaの英語モデル：Jigg出力でのみ必要です。[配布ZIP](https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/46fe97c1-1ea9-4121-a46c-f1ed27c57a50/content) に含まれる `english-morphium-wsj-140407-no_negation.tagger` を `--morphodita-model` に指定します。このモデルはCC BY-NC-SA（非商用）で配布されています。詳細は [MorphoDiTaのマニュアル](https://ufal.mff.cuni.cz/morphodita/users-manual) を参照してください。

## CLI

入力は1行1文で、トークンを空白で区切ります。

```sh
echo 'John likes Mary .' | uv run issen-ccg --model issenccg
echo 'The women are sketching houses .' | uv run issen-ccg \
  --model issenccg --format jigg \
  --morphodita-model english-morphium-wsj-140407-no_negation.tagger
```

- `--format` は `auto`（既定）、`json`、`jigg` から選べます。`jigg` では `--morphodita-model` が必須です。
- Jigg出力の各トークンには、入力トークンを `surf`、小文字のlemmaを `base`、Issenが予測した品詞を `pos` として書きます。
- 1文は1〜512トークンです。513トークン以上の文は解析せず、`too_long` の失敗として結果に残します。空行は読み飛ばします。
- 探索の上限は `--max-nodes`（既定300,000）と `--timeout-ms`（既定10,000）で変更できます。上限に達した文は、構文木を作らず `node_limit` または `timeout` の失敗として結果に残します。
- 終了コードは、すべての文を解析できた場合に0、解析に失敗した文がある場合に3、引数の誤りやモデルを読み込めない場合に2です。失敗した文があっても、成功した文の結果は出力します。

## Python

```python
from issen_ccg import load_parser
from issen_ccg.adapters.morphodita import MorphoditaLemmatizer
from issen_ccg.adapters.tree_output import to_jigg

sentences = [("The", "women", "are", "sketching", "houses", ".")]
parser = load_parser("issenccg")
results = parser.parse(sentences)
if results[0].success:
    print(results[0].tree.lexical_categories)

lemmatizer = MorphoditaLemmatizer("english-morphium-wsj-140407-no_negation.tagger")
lemmas = [lemmatizer.lemmatize(tokens) for tokens in sentences]
xml = to_jigg(sentences, results, lemmas_by_sentence=lemmas)
```

空の文を渡すと `ValueError` になります。

## ライセンス

コードはMITライセンスです。

Issenのモデルは [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) で配布し、非商用の利用に限ります。
モデルはCCGbank 1.1（LDC2005T13）で学習しており、単語ベクトルの初期値にGloVe 6Bを使っています。
商用で利用する場合は、Linguistic Data ConsortiumとのCCGbankの契約が別途必要です。
