# 发布部署指南

让其他人**不经过你本地电脑**也能访问本软件，常见有三种方式，按难度从低到高排列。

---

## 方式一：Streamlit Community Cloud（推荐，免费）

适合：软著演示、课题组内分享、快速获得公网 HTTPS 链接。

### 前提

- 代码已推送到 **GitHub**（公开仓库，或私有仓库 + Streamlit 账号授权）
- 有 GitHub 账号

### 步骤

1. 打开 [https://share.streamlit.io](https://share.streamlit.io) ，用 GitHub 登录  
2. 点击 **New app**  
3. 填写部署信息：

   | 配置项 | 填写内容 |
   |--------|----------|
   | Repository | 你的仓库，例如 `username/vertehurtdateprediction` |
   | Branch | `main` 或 `master` |
   | Main file path | **`verte_hurt_software_demo/app.py`** |

4. **Advanced settings** → **Python version** 选 **3.11**（或与仓库根目录 `.python-version` 一致）  
5. 点击 **Deploy**，等待 3～10 分钟构建完成  
6. 获得形如 `https://xxx.streamlit.app` 的地址，发给他人即可访问  

### 说明

- 仓库已包含 `.python-version`（3.11）并使用 `opencv-python-headless`，避免云端默认 Python 3.14 导致 `import cv2` 失败  
- 若仍报 OpenCV / cv2 错误：在 Cloud 控制台 **Manage app → Settings** 确认 Python 为 3.11，然后 **Reboot app** 或 **Clear cache and redeploy**

- `requirements.txt`、`packages.txt`（中文字体）已放在 `verte_hurt_software_demo/` 目录，云端会自动安装  
- 免费版实例休眠后首次打开可能需等待十几秒  
- 云端**不持久保存**你训练的检查点，他人打开后需自行「生成演示数据 → 训练」；属正常现象  
- 不要将含真实患者信息的 DICOM/影像上传到公网演示环境  

### 仅部署 demo 子目录（可选）

若希望仓库根目录更干净，可单独建一个 GitHub 仓库，只包含 `verte_hurt_software_demo/` 内全部文件，此时 Main file path 填 **`app.py`** 即可。

---

## 方式二：云服务器 + Docker（适合长期、可控）

适合：医院/实验室自有服务器、需要固定域名、内网穿透到公网。

### 前提

- 一台有公网 IP 的云主机（阿里云、腾讯云、华为云等）
- 已安装 Docker 与 Docker Compose

### 步骤

1. 将 `verte_hurt_software_demo` 目录上传到服务器，例如 `/opt/verte_hurt_software_demo`  

2. 在服务器上执行：

```bash
cd /opt/verte_hurt_software_demo
docker compose up -d --build
```

3. 在云厂商**安全组**中放行 **8501** 端口（或改用 80/443 反向代理，见下）  
4. 浏览器访问：`http://<服务器公网IP>:8501`  

### 绑定域名与 HTTPS（建议）

在服务器安装 Nginx，将域名反代到 `127.0.0.1:8501`，并用 Let’s Encrypt 申请证书。Nginx 配置示例：

```nginx
server {
    listen 80;
    server_name demo.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 常用运维命令

```bash
docker compose logs -f      # 查看日志
docker compose restart      # 重启
docker compose down         # 停止
```

---

## 方式三：云服务器直接运行（无 Docker）

```bash
cd /opt/verte_hurt_software_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Linux 安装中文字体（图表中文）
sudo apt-get update && sudo apt-get install -y fonts-noto-cjk

# 后台运行（示例）
nohup python3 -m streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  > streamlit.log 2>&1 &
```

放行 8501 端口后，通过 `http://<公网IP>:8501` 访问。

生产环境建议用 **systemd** 或 **supervisor** 托管进程，避免 SSH 断开后服务退出。

---

## 临时分享（仅演示几分钟）

若只想临时给同事看一眼，可用内网穿透（**不适合正式发布**）：

```bash
# 本地先启动
python3 -m streamlit run app.py

# 另开终端（需安装 ngrok 并登录）
ngrok http 8501
```

ngrok 会生成一个临时公网 URL，关闭终端后失效。

---

## 发布前检查清单

- [ ] 代码已推送到 GitHub（若用 Streamlit Cloud）  
- [ ] 确认 `verte_hurt_software_demo/requirements.txt` 完整  
- [ ] 公网环境**不要**放入真实患者数据  
- [ ] 在云端打开一次，走通：生成演示数据 → 训练 → 预测 → 报告  
- [ ] 图表中文在 Linux 下正常（已配置 `packages.txt` / Docker 字体）  

---

## 常见问题

**Q：别人打开很慢？**  
A：Streamlit 免费版冷启动会慢；可换云服务器 Docker 常驻部署。

**Q：训练后的模型别人看不到？**  
A：云端实例重启后检查点会丢失；每人需自行训练，或你将 `checkpoints/*.pt` 一并提交到仓库（体积较小可接受）。

**Q：能否只允许单位内访问？**  
A：用云服务器部署在内网，或通过 Nginx 加 IP 白名单 / 基础认证；Streamlit 本身无账号体系，敏感场景需额外网关。

---

## 推荐选择

| 场景 | 推荐方式 |
|------|----------|
| 软著材料、快速分享链接 | Streamlit Cloud |
| 单位长期对外服务 | 云服务器 + Docker + Nginx HTTPS |
| 临时演示 5 分钟 | ngrok |

如需我帮你在当前仓库写好 GitHub Actions 自动部署，可说明使用的平台（Streamlit Cloud / 自有服务器）。
