# PhysioNet

This repo contains MIT-BIH data preprocessing and a sample deep model.

The original data can be found at https://physionet.org/content/mitdb/1.0.0/

# Task

(1) Classify ECG heart beats into 5 classes: N, S, V, F, Q

(2) Classify ECG heart beats into 2 classes: V, non-V (need manually modify test_mitdb.label2index)

# Usage
```
# (1) unzip raw data
unzip data/mit-bih-arrhythmia-database-1.0.0.zip

# (2) preprocess to get npy
python preprocess.py

# (3) train
python test_mitdb.py
```

# Label Description

## (1) Beat Level
| Code | Group | Description                                                                       | 中文                                                                                                       |
|------|-------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| N    | N     | Normal beat (displayed as"·" by the PhysioBank ATM, LightWAVE, pschart, and psfd) | 正常                                                                                                       |
| L    | N     | Left bundle branch block beat                                                     | 左束支传导阻滞心搏                                                                                         |
| R    | N     | Right bundle branch block beat                                                    | 右束支传导阻滞心搏                                                                                         |
| B    | N     | Bundle branch block beat (unspecified)                                            | 束支传导阻滞心搏                                                                                           |
| A    | S     | Atrial premature beat                                                             | 房性早搏，房性期前收缩(心室搏动还未结束新房就开始搏动)                                                     |
| a    | S     | Aberrated atrial premature beat                                                   | 异常房性早搏                                                                                               |
| J    | S     | Nodal (junctional) premature beat                                                 | 交界性早搏                                                                                                 |
| S    | S     | Supraventricular premature or ectopic beat (atrial or nodal)                      | 室上性早搏(发生于心房或者房室结的统称为室上性，室上性早搏，是说早搏介于房性和室性之间，属于轻微的心率失常) |
| V    | V     | Premature ventricular contraction                                                 | 室性收缩                                                                                                   |
| r    | V     | R-on-T premature ventricular contraction                                          | R落在T上的室性早搏                                                                                         |
| F    | F     | Fusion of ventricular and normal beat                                             | 心室融合心跳                                                                                               |
| e    | S     | Atrial escape beat                                                                | 房性逸搏(被动性异位心律)                                                                                   |
| j    | S     | Nodal (junctional) escape beat                                                    | 交界性逸搏                                                                                                 |
| n    | S     | Supraventricular escape beat (atrial or nodal)                                    | 室上性逸搏                                                                                                 |
| E    | V     | Ventricular escape beat                                                           | 室性逸搏                                                                                                   |
| /    | Q     | Paced beat                                                                        | 起搏心搏                                                                                                   |
| f    | Q     | Fusion of paced and normal beat                                                   | 起搏融合心跳                                                                                               |
| Q    | Q     | Unclassifiable beat                                                               | 未分类心跳                                                                                                 |
| ?    | Q     | Beat not classified during learning                                               | 其他                                                                                                       |

## (2) Rhythm Level

| Code  | Group | Description                      | 中文                       |
|-------|-------|----------------------------------|----------------------------|
| (AB   |       | Atrial bigeminy                  | 房性早搏(二联律)           |
| (AFIB |       | Atrial fibrillation              | 心房颤动                   |
| (AFL  |       | Atrial flutter                   | 心房扑动                   |
| (B    |       | Ventricular bigeminy             | 室性早搏二联律             |
| (BII  |       | 2° heart block                   | 2°心脏传导阻滞             |
| (IVR  |       | Idioventricular rhythm           | 室性自主节律               |
| (N    |       | Normal sinus rhythm              | 正常窦性节律               |
| (NOD  |       | Nodal (A-V junctional) rhythm    | 结性心律                   |
| (P    |       | Paced rhythm                     | 起搏心律                   |
| (PREX |       | Pre-excitation (WPW)             | 预激综合征                 |
| (SBR  |       | Sinus bradycardia                | 窦性心动过缓               |
| (SVTA |       | Supraventricular tachyarrhythmia | 室上性心律失常             |
| (T    |       | Ventricular trigeminy            | 室性早搏三联律（心律失常） |
| (VFL  |       | Ventricular flutter              | 心室扑动                   |
| (VT   |       | Ventricular tachycardia          | 室性心动过速               |




