# 結論

## 1. 哪一種方法收斂較快
在本次設定（Cliff Walking, epsilon=0.1, alpha=0.1, gamma=0.9, episodes=500）下，  
Q-learning 收斂較快。

- Q-learning convergence episode: 145
- SARSA convergence episode: 382

因此若只看「前期到中期學習速度」，Q-learning 在本實驗中較有優勢。

## 2. 哪一種方法較穩定
在本次實驗中，SARSA 較穩定。

- Q-learning 最後 100 回合 reward 標準差: 75.94
- SARSA 最後 100 回合 reward 標準差: 19.63

此外在 epsilon=0.1 的探索評估中：
- Q-learning 掉崖次數: 68 / 200
- SARSA 掉崖次數: 8 / 200

可見 SARSA 不只波動較小，也更能避免高風險失誤。

## 3. 何種情境下選擇 Q-learning 或 SARSA
- 選擇 Q-learning：
  - 你希望更快逼近理論上高價值策略。
  - 環境對失誤容忍度高，或可接受訓練/執行過程的較高風險。
  - 你更重視最終最短路徑或理論最優性。

- 選擇 SARSA：
  - 你重視訓練與執行時的穩定性與安全性。
  - 環境存在明顯高懲罰區域（如 cliff），探索失誤成本高。
  - 你希望策略能反映實際探索行為（on-policy）而非只追理論最優。

## 4. 本專案最終總結
本專案結果與理論一致：
- Q-learning：收斂快，但風險與波動較高。
- SARSA：收斂較慢，但更保守、更穩定，且在高風險環境下表現更安全。
