# Tunny Hands on

## Tunny について

Tunny は、オープンソースのハイパーパラメータ自動最適化フレームワークである Optuna を使用した、Grasshopper の最適化コンポーネントです。

以下、公式サイトより転載

> オープンソースの自動ハイパーパラメーター最適化フレームワークである Optuna™ は、ハイパーパラメーターを最適化するための試行錯誤プロセスを自動化します。最適化ターゲットに基づいて最適なハイパーパラメータ値を自動的に見つけます。Optuna はフレームワークに依存せず、Chainer、Scikit-learn、Pytorch などを含むほとんどの Python フレームワークで使用できます。
>
> Optuna は PFN プロジェクトで使用され、良好な結果が得られています。その一例が、 Google AI Open Images 2018 – Object Detection Track コンペティションでの 2 位受賞です。

Optuna 公式サイト

https://optuna.org/

### インストール

通常の Grasshopper コンポーネントと同様にインストールることができますが、注意点として Windows でしか動作しないので注意してください。

### ライセンス

Tunny は MIT ライセンスの下でライセンスされています。
Copyright© 2022, hrntsm

Tunny は Python ランタイムといくつかの Python パッケージを使用します。
これらは、独自のライセンスに依存します。
ライセンスの詳細については、PYTHON_PACKAGE_LICENSE を参照してください。

### 特徴

- Python の最適化ツールである Optuna をラップした Grasshopper の最適化コンポーネントであり、複数の最新のテクノロジーが反映されている。
  - Optuna は現在でも開発されており、今後も最新のテクノロジーが反映されていくと予想される
- 複数の実行アルゴリズムを実装
  - ベイズ最適化に Grasshopper で唯一の対応
    - 1 トライアルに時間がかかる場合に非常に有効
  - 単純な最適化だけでなく、ランダムやグリッドなど、解空間を確認する機能がある
- 最適化が経過時間指定で実行できる
- 最適化内容がマイステップ保存され、一旦最適化を止めた後に追加で最適化を実行することができる
- 結果分析用にダッシュボードに対応

### 使い方

https://github.com/hrntsm/Tunny#dolphin-usage

いくつかサンプルを使って使い方の説明

## 理論的背景

遺伝的アルゴリズムは皆さんよく知っていると思うので、他になく、Tunny が実装している TPE の理論的背景として ベイズ最適化 について紹介します。

### 回帰の話

線形回帰

![image](https://user-images.githubusercontent.com/23289252/182602369-01358e28-1b8d-421b-a979-061d46db1915.png)![]()

ベイズ線形回帰

![image](https://user-images.githubusercontent.com/23289252/182602474-36459c35-9433-4b4f-842d-4e3957c64f1c.png)
![image](https://user-images.githubusercontent.com/23289252/182602484-fd9b548f-3343-47d2-b2b6-c8045657e67b.png)

ガウス過程回帰

![image](https://user-images.githubusercontent.com/23289252/182602675-2d0da71c-d8bd-4d3a-abf5-555489ce5185.png)
