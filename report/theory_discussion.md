# 理論比較與討論

## 1. Q-learning 為 Off-policy
Q-learning 的更新目標是：

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

重點是 `max_a' Q(s',a')`。  
它用的是「下一狀態理論上最好的動作價值」，不管當下實際採取的是哪個動作，因此屬於 Off-policy。

在本專案實作中，對應到 [q_learning.py](c:\Users\User\Desktop\repository\G1-DRL-HW2-Q-learning-And-SARSA\algorithms\q_learning.py) 的 `q_learning_target(...)`：
- `done=False` 時：`reward + gamma * next_max_q`
- `done=True` 時：`reward`

## 2. SARSA 為 On-policy
SARSA 的更新目標是：

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]`

這裡的 `a'` 是在 `s'` 由當前策略（本實驗為 epsilon-greedy）實際選出的動作，所以會把探索行為的風險直接納入更新，屬於 On-policy。

在本專案實作中，對應到 [sarsa.py](c:\Users\User\Desktop\repository\G1-DRL-HW2-Q-learning-And-SARSA\algorithms\sarsa.py) 的 `sarsa_target(...)`：
- `done=False` 時：`reward + gamma * next_q`
- `done=True` 時：`reward`

## 3. 風險、穩定性、最優性差異
在 Cliff Walking 這類高風險環境：

- Q-learning：
  - 由於目標值使用 `max`，容易偏向「貼近懸崖但理論步數更短」的路徑。
  - 好處是通常更快學到高估值方向（收斂速度常較快）。
  - 代價是探索期波動較大，且實際執行時更容易踩到高風險區域。

- SARSA：
  - 因為更新使用實際採取行為，會「感受到」epsilon 探索可能帶來的掉崖風險。
  - 因此傾向學到較保守、離懸崖更遠的策略。
  - 代價是學習速度可能較慢，但通常更穩定。

## 4. 與本次實驗現象對照
根據結果分析輸出 [comparison_metrics.json](c:\Users\User\Desktop\repository\G1-DRL-HW2-Q-learning-And-SARSA\report\comparison_metrics.json)：

- 收斂速度：Q-learning 較快（145 vs 382）
- 穩定性：SARSA 較穩（tail std 19.63 < 75.94）
- 路徑風格：SARSA 更保守（risk score 2.625 > 1.0）
- 探索風險：SARSA 掉崖更少（8 vs 68）

這與理論預期一致：  
Q-learning 更偏向理論最優、速度快但風險高；SARSA 更反映實際探索、速度慢一些但更安全穩定。
