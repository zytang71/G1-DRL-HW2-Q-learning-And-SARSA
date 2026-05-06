# Q-learning 與 SARSA：Cliff Walking 整合報告

## 作業目的
本作業在相同環境與相同參數下，實作並比較 Q-learning 與 SARSA，觀察兩者的學習速度、穩定性、策略風格與探索風險差異。

## 實驗環境與問題設定
- 環境：Cliff Walking（4x12 Gridworld）
- 起點：左下角
- 終點：右下角
- 懸崖：底部起終點之間區域
- 動作：上、下、左、右
- 獎勵：
  - 每步 `-1`
  - 掉入懸崖 `-100` 並回到起點
  - 到達終點回合結束

## 演算法與設定
- 策略：epsilon-greedy（`epsilon=0.1`）
- 學習率：`alpha=0.1`
- 折扣因子：`gamma=0.9`
- 訓練回合數：`500`
- 每回合步數上限：`1000`
- 公平比較方式：兩個演算法使用同一套環境規格、超參數、回合數與評估流程

### Q-learning（Off-policy）
更新式：
`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

特性：
- 使用下一狀態的最佳可能動作價值
- 傾向追求理論最優策略
- 在高風險區域可能更激進

### SARSA（On-policy）
更新式：
`Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]`

特性：
- 使用實際在 `s'` 採取的動作價值
- 直接反映探索策略風險
- 通常更保守、更穩定

## 實驗結果

### 收斂速度
- Q-learning 收斂回合：`145`
- SARSA 收斂回合：`382`

解讀：本實驗下 Q-learning 收斂較快。

### 訓練穩定性（最後 100 回合）
- Q-learning reward 標準差：`75.94`
- SARSA reward 標準差：`19.63`

解讀：SARSA 波動明顯較小，穩定性較佳。

### 最終策略風格
- Q-learning greedy path 長度：`13`
- SARSA greedy path 長度：`17`
- Q-learning 風險分數（越高越保守）：`1.00`
- SARSA 風險分數：`2.625`

解讀：Q-learning 偏短路徑、較貼近懸崖；SARSA 路徑較長、較保守。

### 探索影響（epsilon=0.1 評估 200 回合）
- Q-learning 掉崖次數：`68`
- SARSA 掉崖次數：`8`
- Q-learning 平均每回合 reward：`-52.865`
- SARSA 平均每回合 reward：`-22.74`

解讀：在探索存在時，SARSA 在此環境有更好的安全性與平均表現。

## 理論與實驗對照
本次觀察與理論一致：
- Q-learning：收斂較快，但風險與波動較高。
- SARSA：收斂較慢，但更穩定、更安全。

在 Cliff Walking 這種「探索失誤成本很高」的環境，SARSA 的 on-policy 特性會更直接地將風險納入更新，因此更容易學到保守策略。

## 最終結論與選用建議
- 若重視「較快逼近理論最優」且可接受較高風險，選 Q-learning。
- 若重視「執行穩定性與安全性」，尤其在高懲罰環境，選 SARSA。

## 可重現流程
在 `conda env DRL` 下執行：
1. `python train.py --algorithm both`
2. `python analyze_results.py`

以上流程可重建訓練輸出、比較指標與圖表。
