# Racecar Gym

* 競賽簡報連結：[https://docs.google.com/presentation/d/1J6RE2CaqmXGGYuwT6-_0erj9Yp3HIR30mk6lciRfBpY/edit?usp=sharing](https://docs.google.com/presentation/d/1J6RE2CaqmXGGYuwT6-_0erj9Yp3HIR30mk6lciRfBpY/edit?usp=sharing)
> **注意：** 中文版內容尚未驗證，請參考英文版以獲取重要資訊。

## 競賽規則
* 請使用 Pytorch 完成比賽。
* 期中競賽僅使用本文件提供的環境，請勿在提交的代碼中包含其他第三方庫。
* 期中競賽分為兩輪，第一輪的前 10 名將進入第二輪。最後僅有一位參賽者能獲得冠軍。
* 我們將使用 scenarios/validation.yml 中描述的傳感器和設置進行競賽。

## 如何提交你的代碼和強化學習 (RL) 智能體？
* 將所有模型的權重和代碼放入 agent 資料夾中。
* 將 agent 資料夾壓縮成 zip 文件，並上傳到 Moodle。
  > 提交位置將在未來提供。

## 如何驗證你的代碼和智能體是否可以正常運行？
* 重新下載項目
    > 以下簡稱為`新項目`
* 按照下方指令重新創建新的虛擬環境
    > 此步驟確保你的代碼能在本項目提供的標準環境中執行
    ``` shell
    # uninstall env
    conda deactivate
    conda env remove -n racing
    # re-install env
    conda env create -f environment.yml
    conda activate racing
    pip install -e .
   ```
* 將你的 agent 資料夾放入`新項目`中
* 執行 `python check_env_script.py`
* 如果能正常執行，則表示你的提交文件無誤。

## 如何控制賽車並訓練 RL 智能體？
> **重要注意事項！**
> *validation.yml* 和 *validation2.yml* 場景僅用於算法驗證，請勿用於訓練。使用這些場景進行訓練可能會導致未知錯誤。  
> 這兩個場景僅用來驗證算法。
* 修改 `agent/Agent.py` 腳本中的 `Agent.get_action` 函數。
* 在 `Agent.get_action` 函數內實現你的 RL 算法來控制賽車。

## 如果遇到環境安裝錯誤應該怎麼辦？
* 請參閱下方的**FAQ**部分。

---

## 安裝

> 以下步驟僅在 Windows 10 上驗證過
1. 安裝 [Anaconda](https://www.anaconda.com/download/success)
2. 在項目資料夾中執行以下命令：
    ```shell
    # For windows
    conda env create -f environment.yml
    conda activate racing
    pip install -e .
    ```
3. 為了測試環境，請執行以下命令
    ```shell
    conda activate racing
    python check_env_script.py
    ```

---

## 安裝過程遇到問題!!!
* 參考[英文版README.md](README.md)

---

## Project structure
![img.png](docs/ProjectStructure01.png)
![img_1.png](docs/ProjectStructure02.png)

---

## FAQ
### error: Microsoft Visual C++ 14.0 or greater is required.
1. Download and install Visual Studio 2022 Installer ([here](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/))
2. Install c++ package, As shown below
![img.png](docs/VisualStudioInstall.png)

3. Delete the racing environment and re-install again
   ```
    # uninstall env
    conda deactivate
    conda env remove -n racing
    # re-install env
    conda env create -f environment.yml
    conda activate racing
    pip install -e .
   ```

## Acknowledgments
* This project is modified from [axelbr/racecar_gym](https://github.com/axelbr/racecar_gym.git)