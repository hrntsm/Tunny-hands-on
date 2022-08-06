# Tunny Hands on

## 必要なもの

- [Wallacei](https://www.food4rhino.com/en/app/wallacei)
- [Tunny](https://www.food4rhino.com/en/app/tunny)
- Google Colaboratory

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
これらは、それら独自のライセンスに依存します。
ライセンスの詳細については、PYTHON_PACKAGE_LICENSE を参照してください。

### 特徴

- Python の最適化ツールである Optuna をラップした Grasshopper の最適化コンポーネントであり、複数の最新のテクノロジーが反映されている。
  - Optuna は現在でも開発されており、今後も最新のテクノロジーが反映されていくと予想される
- 複数の実行アルゴリズムを実装
  - ベイズ最適化に Grasshopper で唯一の対応
    - 1 トライアルに時間がかかる場合に非常に有効
  - 単純な最適化だけでなく、ランダムやグリッドなど、解空間を確認する機能がある
- 最適化が経過時間指定で実行できる
- 最適化内容が毎ステップ保存され、一旦最適化を止めた後に追加で最適化を実行することができる
- 結果分析用にダッシュボードに対応

### 使い方

https://github.com/hrntsm/Tunny#dolphin-usage

いくつかサンプルを使って使い方の説明

### 結局どのアルゴリズムを使えばよいの？

- Tunny のサポートしているアルゴリズムで場合分け
  - 少ない回数しか評価できない場合
    - BO（TPE）
    - BO（GP）
  - 多くの個体を評価したい場合
    - GA（NSGAII）
  - 多峰性がない単一目的最適化
    - CMA-ES
  - 解空間を確認したい
    - Random
    - Grid
- 目的関数の数で場合分け
  - 多目的
    - TPE、GP、NSGAII
  - 単一目的
    - TPE、GP、NSGAII、CMA-ES、Random、Grid
- ベイズ最適化の中での場合分け
  - 探索空間が低次元の問題では GP（GaussianProcess）
  - 高次元、条件パラメータを含む問題では TPE
- 具体的な評価回数が大小について
  - 推奨される評価回数
    - TPE：100~1000
    - GP：10~100
    - NSGAII：100~10000
    - CMA-ES：100~10000
    - Grid：いくらでも
    - Random：いくらでも
  - 結果の計算コスト
    - TPE：$O(dn \log{n})$
    - GP：$O(n^3)$
    - NSGAII：$O(mnp)$
    - CMA-ES：$O(d^3)$
    - Grid：$O(d)$
    - Random：$O(dn)$
    - $d$ は変数の数、$n$ はこれまでの trial 回数、$m$ は目的関数の数、$p$ は世代に含まれる個体数

詳細（ただし Tunny が全てをサポートしているわけではない）

- https://github.com/optuna/optuna/releases/tag/v3.0.0-b1

何回トライアルしたいか、何回トライアルすることが可能かから選択するのがよい。

例

1. Boolean 演算や構造解析、環境解析があって非常に重い、流れるのに 1 分かかる Grasshopper ファイルで最適化をしたい！
2. 多目的最適化でパレート解を 50 個体くらい取得したい
3. 収束していてほしいから少なくとも合計 500 個体は流した方が良い？
   1. GA で考えると 50 個体 × 10 世代 or 25 個体 × 20 世代 など
4. 500 trial × 1 分 = 500 分 → 約 8 時間！！！
5. 許容できる？
   1. Yes → GA でやる
   2. No → ベイズ最適化もあり
      1. 問題にもよるが、ベイズ最適化の場合は 100 個体でもある程度の結果が出る可能性がある
      2. 100 trial × 1 分 = 100 分 → 約 1.5 時間！！！
6. 単純にたくさん trial したほうが良い値が見つける可能性が上がることはその通りなので、大きく見たときの優劣はない
   1. [ノーフリーランチ定理](https://ja.wikipedia.org/wiki/%E3%83%8E%E3%83%BC%E3%83%95%E3%83%AA%E3%83%BC%E3%83%A9%E3%83%B3%E3%83%81%E5%AE%9A%E7%90%86)

## 理論的背景

遺伝的アルゴリズムは皆さんよく知っていると思うので、他になく、Tunny が実装している「ベイズ最適化」の基本的な部分であるガウス過程回帰について紹介します。

### 回帰

回帰（Regression）とは？

> デジタル大辞泉「回帰」の解説
>
> かい‐き〔クワイ‐〕【回帰】
>
> ［名］(スル)ひとまわりして、もとの所に帰ること。「伝統への回帰」

x が与えられたとき y を予測するのが回帰モデル

わかりやすいものは最小二乗近似

https://ja.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%97%E6%B3%95

線形単回帰

![image](https://user-images.githubusercontent.com/23289252/182602369-01358e28-1b8d-421b-a979-061d46db1915.png)![]()

ベイズ線形単回帰

![image](https://user-images.githubusercontent.com/23289252/182602474-36459c35-9433-4b4f-842d-4e3957c64f1c.png)
![image](https://user-images.githubusercontent.com/23289252/182602484-fd9b548f-3343-47d2-b2b6-c8045657e67b.png)

ガウス過程回帰

![image](https://user-images.githubusercontent.com/23289252/182602675-2d0da71c-d8bd-4d3a-abf5-555489ce5185.png)
