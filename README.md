# pg-diginet

**數字透網**：手寫數字經小型 MLP 前向傳播——訊號粒子沿連線流動、隱藏節點呼吸發光、softmax 機率即時跳動；可點隱藏節點看像素權重熱圖。純前端，無建置步驟。

僅供作為 [Playgrounds（遊樂場）](https://play.samkuo.me/) 範本小品。教學可視化——**不是**正式 OCR／生產辨識服務。

## 一鍵開 SAM 小

**[一鍵開 SAM 小](https://play.samkuo.me/?open=sampot%2Fpg-diginet&name=%E6%95%B8%E5%AD%97%E9%80%8F%E7%B6%B2)**

```
https://play.samkuo.me/?open=sampot/pg-diginet&name=數字透網
```

## 試玩（本機）

```bash
npx --yes serve .
```

## 重訓權重（可選）

```bash
python3 scripts/train_mlp.py
```

會下載 MNIST、訓練 `16×16 → 24 → 10`，覆寫 `weights.json`。

## License

MIT
