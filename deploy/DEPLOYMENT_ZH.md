# 魔搭免费后台与中英文展示端部署说明

本次执行路径为：GitHub 管理代码和自动更新；Cloudflare Pages 独立托管中文、英文展示站；魔搭 ModelScope Docker 创空间承载 Python 上传预测、人工校准及密码下载。Oracle 不再作为当前部署前提。

## 当前进度

两个展示站已生产发布：中文 `https://bicyclus-wing-atlas-zh.pages.dev/`、英文 `https://bicyclus-wing-atlas-en.pages.dev/`，27 项公网展示检查通过。代码已发布 GitHub，两个发布包也已在 GitHub 运行器重建并通过完整哈希校验。魔搭镜像已构建运行，Secrets 已配置，`ATLAS_SETUP_ONLY=false`，首次初始化正在下载验证完整数据，真实预测尚待验收。全量私有数据远端核验通过：含清单 4,140 文件、19,966,259,403 字节，大小与 SHA-256 全部匹配，匿名读清单返回 404。Cloudflare 自动发布仍需专用 Pages Edit Token；不需要重复插件授权或 GitHub 登录。完整体系以工作目录 `FINAL_REVIEW_ZH.md` 的逐项结果为准。

此账号当前资源 API 确认 `platform/2v-cpu-8g-mem` 为免费 Docker 档位，已实际完成构建。运行容器返回虚拟文件系统容量，不能由磁盘数字推断真实存储额度；仍须完成全量私有数据下载、哈希校验及运行测试。免费休眠和额度限制仍然适用。

## 一、账号授权及资源核验

1. 魔搭注册和阿里云绑定已由用户报告完成，无需重复注册。
2. 在个人账号设置中完成阿里云账号绑定，并在魔搭完成实名认证。Docker 创空间要求这两项条件，见[官方 Docker 文档](https://modelscope.cn/docs/studios/docker)。实名认证资料只在官方页面提交，不要发送到聊天中。
3. 进入创空间的创建页面，选择 Docker，查看是否可选免费 CPU 资源及其额度。保持免费配置；本项目没有授权付费开通或自动切换付费档位。
4. 用户名及本机加密令牌已核验，无需重新配置。授权助手位于发布工作目录的 `account_connection`。密码、API Token、身份证和银行卡信息均不应出现在聊天里。

当前私有数据包解包后约 18.6 GiB，4,139 个文件。建议可用持久磁盘至少 30 GiB，为预览缓存、上传样本和运行结果预留空间；同时核对私有数据集的免费存储额度。若控制台容量不足，应先明确缺口，不能通过公开上传原图或删减最终科学数据来绕过。

## 二、发布代码与私有数据

`repository` 是可交给 GitHub 的代码目录。它与相邻的 `protected_data` 必须保持分离。不要把整个 `platform_full_cloud_20261020` 工作目录上传到公共仓库。

代码已发布到 `https://github.com/huangmengxuan351-cpu/bicyclus-wing-atlas`。GitHub 官方 CLI 已确认登录；无需重新授权。公共仓库只保存代码、界面、小型图谱元数据和减小分辨率后的示意图。286 个展示预览包由 CI 从已发布 Pages 站点按 SHA-256 获取，不进入 Git 历史。

已创建正式私有数据集 `EVA030806/bicyclus-wing-atlas-private-data`，完整上传、私有可见性、授权读回和匿名访问拦截已通过。此包包含最终图谱、原始数组、最终模型及运行依赖，不包含被移除的 hth 数据；数据集容量通过不等于容器运行盘验收通过。

`deploy/upload_private_data.py` 只向已有且 API 确认私有的数据集上传，先核验每个文件。它从本地环境变量 `MODELSCOPE_API_KEY` 读取写入凭据；不要把凭据写入命令行参数、代码或公共环境变量。调用格式为：

```text
python deploy/upload_private_data.py --data-dir <protected_data目录> --repo <用户名/私有数据集名> --manifest-sha256 <data_manifest.json的SHA-256>
```

上传、断点续传和完整远端清单核验已在真实账号通过；4,140 个文件的大小及 SHA-256 均一致，清单授权读回成功，匿名读取被拒绝。无需先复制成另一个 20 GB 压缩包。`protected_data` 本身就是完整、可核验的数据文件夹。

## 三、Docker 创空间与后续构建

已创建后端空间 `EVA030806/bicyclus-wing-atlas-backend`，使用免费 2 核 8 GB Docker 档位，镜像已构建并运行。Secrets 已获授权并写入，`ATLAS_SETUP_ONLY=false`，正在初始化全量私有数据，尚未通过真实预测验收。代码中不放私有数据；代码同步到创空间默认 `master` 分支。应用监听 `0.0.0.0:7860`，只选择账号中明确标记免费的 Docker 硬件。

配置以下普通变量：

| 名称 | 作用 |
| --- | --- |
| `ATLAS_DATA_REPO` | 私有数据集的 `用户名/仓库名` |
| `ATLAS_DATA_REVISION` | 已上传数据的版本；优先固定提交，初次可用 `master` 并配合固定清单哈希 |
| `ATLAS_DATA_MANIFEST_SHA256` | 本地 `protected_data/data_manifest.json` 的 SHA-256 |
| `ATLAS_DATA_DIR` | `/mnt/workspace/wing-atlas-data` |
| `ATLAS_STATE` | `/mnt/workspace/wing-atlas-state` |
| `ATLAS_DOWNLOAD_ENABLED` | `true`；需要关闭下载时设为 `false` 并重启 |
| `ATLAS_SETUP_ONLY` | 首次构建设为 `true`；完整数据和 Secrets 就绪后再改为 `false` 并部署 |

配置以下 **Secrets**，不要使用普通 Variables：

| Secret | 作用 |
| --- | --- |
| `ATLAS_DATA_TOKEN` | 访问该私有数据集的魔搭凭据，优先单独配置只读令牌 |
| `ATLAS_DOWNLOAD_PASSWORD` | 新生成的独立云端下载密码；用户名固定为 `atlas_download`，不复制本地密码 |

首次启动从私有数据集下载到 `/mnt/workspace` 并核验清单及文件哈希。之后正常重启复用已验证的数据；更换发布清单会重新核验。此目录仍需遵守平台关于转移或重命名创空间时的数据保留规则。当前支持的科学计算、上传预测、人工对齐、结果查询和受保护下载沿用本地平台实现。

魔搭保留请求头 `Authorization`、`X-modelscope-*`、`X-studio-*`。Pages 将用户的下载认证转成 `X-Atlas-Download-Authorization`，云端入口再交给现有服务端核验。访问网站无需密码；正确密码只放行目录允许的原始文件。公开预览本来就可以保存，本方案不限制截图。

## 四、分别发布中文和英文 Pages

本工作目录下 `repository/dist/zh` 与 `repository/dist/en` 是两个完整展示发布文件夹，各自包含自己的首页、界面、Worker、目录和预览包。只上传某一种语言文件夹的**内部文件**到对应项目。

两个项目分别是 `bicyclus-wing-atlas-zh` 与 `bicyclus-wing-atlas-en`。Cloudflare 控制台进入 Workers & Pages → 对应项目 → 创建部署/上传资源，分别上传 `dist/zh`、`dist/en` 内容。必须包含 `_worker.js` 和 `_routes.json`，不能只上传 HTML。第一次发布必须使用完整本地文件夹或已上传资源的插件最终发布请求；CI 只有在预览包网址已公开可用后才能自举。

已实测魔搭用于程序调用的地址为 `https://studio-eva030806-bicyclus-wing-atlas-backend.api-inference.modelscope.net`。中文版 Pages 变量 `ATLAS_ORIGIN` 在该地址后追加 `/zh`，英文版追加 `/en`。普通 `.ms.show` 浏览器地址拒绝程序请求，不能直接作为本项目的 Pages 转发源站。

两种 Pages 的服务端 Secret `MODELSCOPE_STUDIO_TOKEN` 均已获授权并配置。Worker 仅在请求魔搭 API 专用域名时加入该 Bearer 凭据；它不会进入浏览器，也不会替代原图下载密码。不要把 Token 放入普通 Variables、wrangler 配置、HTML 或 GitHub 源码。

不得保留本机隧道或旧 Sites 地址作为回退。

尚未配置预测地址时，图谱展示可以独立运行，预测和原始下载明确返回服务未连接；此时不能把展示上线写成全功能上线。

## 五、启用 GitHub 自动更新

GitHub 仓库 Settings → Secrets and variables → Actions：

| 类型 | 名称 | 内容 |
| --- | --- | --- |
| Variable | `CLOUDFLARE_ACCOUNT_ID` | Cloudflare 账号 ID |
| Variable | `ATLAS_ORIGIN` | 上述魔搭 API 专用 HTTPS 主机地址，**不带 `/zh` 或 `/en`**；构建脚本分别追加 |
| Variable | `PAGES_BOOTSTRAP_READY` | 首次完整 Pages 发布及预览包验收完成后设为 `true` |
| Variable | `MODELSCOPE_STUDIO` | 已创建的 Docker 创空间 `用户名/空间名` |
| Secret | `CLOUDFLARE_API_TOKEN` | 限定到目标账号的 Pages 编辑凭据 |
| Secret | `MODELSCOPE_API_KEY` | 更新本人创空间代码和请求部署的魔搭凭据 |

`.github/workflows/pages.yml` 分别构建和发布两个独立 Pages 项目；静态预览包下载后逐个核验 SHA-256。`.github/workflows/modelscope.yml` 更新已有 Docker 创空间，先检查账号 API 返回的免费硬件状态，不创建或切换付费资源。代码通过普通 Git 推送保留远端历史，不强制推送。

两站已完成首次生产发布和 27 项公网检查。Pages 工作流可先在 GitHub 运行器验证预览包构建；自动发布步骤在 `PAGES_BOOTSTRAP_READY=true` 后启用。现有魔搭 Token 已写入 Actions Secret；Cloudflare 专用 Pages Edit Token 尚待控制台创建与本机隐藏输入。插件 OAuth 接口不支持创建 API Token，无需重复连接插件。完成后台真实验收后才启用后台更新变量。公开源文件与科学数据的使用授权是两件事，软件许可证仍应由项目所有者明确。

## 六、修改后台数据与最终验收

后续数据更新：修改私有数据源 → 重新运行数据打包并生成清单 → 上传私有数据集的新提交 → 更新 Studio 的版本与清单哈希 → 重启并验收。无需更改科学计算程序。展示目录和预览包同时变动时，还需生成新的展示包和 SHA-256 清单并发布两种语言。

最终云端验收须在电脑上的服务全部停止后，从外部网络分别打开两个 Pages 主入口，检查图谱、解剖、多基因叠加、预测上传、人工校准、结果和历史；核验未授权原图请求 401、错误密码 401、正确密码原图哈希一致，且其他访客不能读到个人预测记录。后台 `/healthz` 的 ready 只表示运行状态，不能替代完整功能验收。

当前本地验收结果保存在相邻目录的 `linux_http_validation.json`、`pages_adapter_validation.json` 和 `FINAL_REVIEW_ZH.md`，以报告实际通过状态为准。Docker 镜像构建、魔搭资源部署、GitHub CI 与云端跨网络访问都需要单独记录，不能用本地测试替代。

官方参考：[ModelScope Studio 指引](https://modelscope.cn/skills/modelscope/modelscope-studio)、[Pages CI 直接上传](https://developers.cloudflare.com/pages/how-to/use-direct-upload-with-continuous-integration/)、[Pages 配置](https://developers.cloudflare.com/pages/functions/wrangler-configuration/)。免费额度可能调整，休眠会影响预测可用性；这不能承诺 24 小时无间断服务。

## 七、凭据配置边界

以下存储目的地均已获用户明确授权。旧本地下载密码没有复制；云端使用单独生成的密码：

- 魔搭 `EVA030806/bicyclus-wing-atlas-backend` 私有 Secrets：`ATLAS_DATA_TOKEN`、`ATLAS_DOWNLOAD_PASSWORD`，已配置。
- 两个 Cloudflare Pages 项目的私有 Secret：`MODELSCOPE_STUDIO_TOKEN`，已配置。
- GitHub `huangmengxuan351-cpu/bicyclus-wing-atlas` Actions Secrets：`MODELSCOPE_API_KEY` 已配置，`CLOUDFLARE_API_TOKEN` 等待账号中创建专用令牌。

上述凭据不应在聊天里提供。GitHub CLI 登录已完成；不要重复设备授权。Cloudflare Token 的具体创建与隐藏输入步骤在工作目录 `CLOUDFLARE_CI_TOKEN_ZH.md`。
