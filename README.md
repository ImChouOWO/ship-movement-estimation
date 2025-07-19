# ship-movement-estimation

### 流程簡要
1. 偵測船隻並賦予追蹤編號（ID）

2. 使用 UNet 模型進行海水區域分割，萃取地平線位置

3. 根據地平線與相機內參推估船隻的實際距離（單目視覺深度估測）

4. 記錄同一物體在不同幀的空間位置與時間，估算其移動速度

5. 將 ID、速度（km/h）與深度（m）標示於畫面上並輸出影片

---

### 距離計算

1. 相機參數（內參矩陣 K）

    ```
    K = | fx  0  cx |
        | 0  fy  cy |
        | 0   0   1 |
    ```
- `fx`, `fy` 為焦距（以像素為單位）

- `cx`, `cy` 為主點位置（通常為影像中心）
>[!NOTE]
>矩陣可將像素點從影像座標系轉換為相機座標系
---
2. 地平線幾何：像素點 → 深度距離（Z）

    


    ```
    z = (camera_height * fy) / Δv
    ```

- camera_height: 相機離地高度（假設已知）

- fy: 相機在 y 軸的焦距

- Δv: 像素點到當前地平線的垂直距離（越遠代表越近）

`針對每個目標的中心點 (u, v)，根據其與地平線的垂直距離 Δv = v - horizon_y[u]，可推估該物體的前後距離 z`

>[!NOTE]
>基於地面平面假設與相似三角形原理來建立距離模型 `需調整`
---
3. 像素 → 相機座標（透過內參逆矩陣）
```
P_cam = z * inv(K) @ [u, v, 1]
```

> [!NOTE]
> 得到 z 之後，即可回推出該像素在相機座標系的 3D 位置
---
4. 相機 → 世界座標（透過外參 R、T）
```
R = scipy.spatial.transform.Rotation.from_euler('xyz', imu_angles).as_matrix()
```
>[!NOTE]
>如果相機有 IMU 裝置（或固定在船上），可取得其姿態角度 (roll, pitch, yaw)，轉換成旋轉矩陣 R

```
P_world = R @ P_cam + T
```
- `R`: 相機在世界中的旋轉（外參）

- `T`: 相機的位移（此專案假設為 `[0, 0, 0]`）
>[!NOTE]
>接著把點從相機座標轉換為世界座標

參考：Zhang, Q., & Singh, S. (2017). Visual-Inertial Monocular Horizon-Based Range Estimation. IEEE International Conference on Robotics and Automation (ICRA).
