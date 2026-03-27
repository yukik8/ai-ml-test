# 1

Epoch [1/5]                                                                                                                    
  Train Loss: 0.5024 | Train Acc: 0.7602
  Val   Loss: 0.4439 | Val   Acc: 0.8852
              precision    recall  f1-score   support

         bad       0.93      0.58      0.72        67
        good       0.88      0.99      0.93       203

    accuracy                           0.89       270
   macro avg       0.90      0.78      0.82       270
weighted avg       0.89      0.89      0.88       270

Best model saved.
Epoch [2/5]                                                                                                                    
  Train Loss: 0.3500 | Train Acc: 0.8454
  Val   Loss: 0.4973 | Val   Acc: 0.9037
              precision    recall  f1-score   support

         bad       0.96      0.64      0.77        67
        good       0.89      0.99      0.94       203

    accuracy                           0.90       270
   macro avg       0.92      0.82      0.85       270
weighted avg       0.91      0.90      0.90       270

Best model saved.
Epoch [3/5]                                                                                                                    
  Train Loss: 0.3229 | Train Acc: 0.8676
  Val   Loss: 0.3611 | Val   Acc: 0.7889
Epoch [4/5]                                                                                                                    
  Train Loss: 0.2765 | Train Acc: 0.8759
  Val   Loss: 0.2410 | Val   Acc: 0.9444
              precision    recall  f1-score   support

         bad       0.93      0.84      0.88        67
        good       0.95      0.98      0.96       203

    accuracy                           0.94       270
   macro avg       0.94      0.91      0.92       270
weighted avg       0.94      0.94      0.94       270

Best model saved.
Epoch [5/5]                                                                                                                    
  Train Loss: 0.2896 | Train Acc: 0.8880
  Val   Loss: 0.2715 | Val   Acc: 0.9000
Training finished. Best Val Acc: 0.9444

# 2

- increased epochs (5 → 20)
- introduced LR schedular
- increased resolution (224 → 320)
- increased bad class weight
    - bad precision ↓, bad recall ↑
    - goodをbadとしてしまうよりも、badをgoodとしてしまう方が問題

Epoch [1/20]                                                                                                                   
  Train Loss: 0.4622 | Train Acc: 0.7704
  Val   Loss: 1.1522 | Val   Acc: 0.7926
              precision    recall  f1-score   support

         bad       1.00      0.16      0.28        67
        good       0.78      1.00      0.88       203

    accuracy                           0.79       270
   macro avg       0.89      0.58      0.58       270
weighted avg       0.84      0.79      0.73       270

Best model saved.
Epoch [2/20]                                                                                                                   
  Train Loss: 0.3567 | Train Acc: 0.8556
  Val   Loss: 0.4131 | Val   Acc: 0.9037
              precision    recall  f1-score   support

         bad       1.00      0.61      0.76        67
        good       0.89      1.00      0.94       203

    accuracy                           0.90       270
   macro avg       0.94      0.81      0.85       270
weighted avg       0.91      0.90      0.90       270

Best model saved.
Epoch [3/20]                                                                                                                   
  Train Loss: 0.2635 | Train Acc: 0.8833
  Val   Loss: 0.2223 | Val   Acc: 0.9519
              precision    recall  f1-score   support

         bad       0.89      0.93      0.91        67
        good       0.97      0.96      0.97       203

    accuracy                           0.95       270
   macro avg       0.93      0.94      0.94       270
weighted avg       0.95      0.95      0.95       270

Best model saved.
Epoch [4/20]                                                                                                                   
  Train Loss: 0.2448 | Train Acc: 0.9093
  Val   Loss: 0.1637 | Val   Acc: 0.9630
              precision    recall  f1-score   support

         bad       0.91      0.94      0.93        67
        good       0.98      0.97      0.98       203

    accuracy                           0.96       270
   macro avg       0.95      0.96      0.95       270
weighted avg       0.96      0.96      0.96       270

Best model saved.
Epoch [5/20]                                                                                                                   
  Train Loss: 0.2133 | Train Acc: 0.9000
  Val   Loss: 0.1746 | Val   Acc: 0.9630
Epoch [6/20]                                                                                                                   
  Train Loss: 0.1998 | Train Acc: 0.9204
  Val   Loss: 0.1027 | Val   Acc: 0.9704
              precision    recall  f1-score   support

         bad       0.94      0.94      0.94        67
        good       0.98      0.98      0.98       203

    accuracy                           0.97       270
   macro avg       0.96      0.96      0.96       270
weighted avg       0.97      0.97      0.97       270

Best model saved.
Epoch [7/20]                                                                                                                   
  Train Loss: 0.2138 | Train Acc: 0.9130
  Val   Loss: 0.3618 | Val   Acc: 0.9296
Epoch [8/20]                                                                                                                   
  Train Loss: 0.1517 | Train Acc: 0.9306
  Val   Loss: 1.1088 | Val   Acc: 0.8481
Epoch [9/20]                                                                                                                   
  Train Loss: 0.2001 | Train Acc: 0.9269
  Val   Loss: 0.1510 | Val   Acc: 0.9148
Epoch [10/20]                                                                                                                  
  Train Loss: 0.1648 | Train Acc: 0.9370
  Val   Loss: 0.1170 | Val   Acc: 0.9407
Epoch [11/20]                                                                                                                                                       
  Train Loss: 0.1660 | Train Acc: 0.9426
  Val   Loss: 0.0723 | Val   Acc: 0.9815
              precision    recall  f1-score   support

         bad       0.97      0.96      0.96        67
        good       0.99      0.99      0.99       203

    accuracy                           0.98       270
   macro avg       0.98      0.97      0.98       270
weighted avg       0.98      0.98      0.98       270

Best model saved.
Epoch [12/20]                                                                                                                                                       
  Train Loss: 0.1264 | Train Acc: 0.9565
  Val   Loss: 0.0724 | Val   Acc: 0.9667
Epoch [13/20]                                                                                                                                                       
  Train Loss: 0.1298 | Train Acc: 0.9500
  Val   Loss: 0.0973 | Val   Acc: 0.9815
Epoch [14/20]                                                                                                                                                       
  Train Loss: 0.0962 | Train Acc: 0.9694
  Val   Loss: 0.0841 | Val   Acc: 0.9556
Epoch [15/20]                                                                                                                                                       
  Train Loss: 0.1231 | Train Acc: 0.9574
  Val   Loss: 0.0800 | Val   Acc: 0.9630
Epoch [16/20]                                                                                                                                                       
  Train Loss: 0.0901 | Train Acc: 0.9630
  Val   Loss: 0.0584 | Val   Acc: 0.9852
              precision    recall  f1-score   support

         bad       0.96      0.99      0.97        67
        good       1.00      0.99      0.99       203

    accuracy                           0.99       270
   macro avg       0.98      0.99      0.98       270
weighted avg       0.99      0.99      0.99       270

Best model saved.
Epoch [17/20]                                                                                                                                                       
  Train Loss: 0.0971 | Train Acc: 0.9676
  Val   Loss: 0.0762 | Val   Acc: 0.9815
Epoch [18/20]                                                                                                                                                       
  Train Loss: 0.0839 | Train Acc: 0.9648
  Val   Loss: 0.0663 | Val   Acc: 0.9815
Epoch [19/20]                                                                                                                                                       
  Train Loss: 0.1019 | Train Acc: 0.9639
  Val   Loss: 0.0752 | Val   Acc: 0.9815
Epoch [20/20]                                                                                                                                                       
  Train Loss: 0.0914 | Train Acc: 0.9741
  Val   Loss: 0.0503 | Val   Acc: 0.9852
Training finished. Best Val Acc: 0.9852

