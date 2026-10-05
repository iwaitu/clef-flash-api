# Clef Flash API

将 [Clef Flash NVFP4](https://huggingface.co/alpha-x-ai/clef-flash-NVFP4) 封装为常驻 GPU 的 HTTP 服务。
支持文字和 JSON 的结构化判断：选项选择（`choice`）、真假概率（`noul`）和等级评分（`score`）。
模型加载并预热后才开放接口，客户端可重复调用，避免每次请求重新加载模型。

- Docker Hub：[iwaitu/clef-flash-api](https://hub.docker.com/r/iwaitu/clef-flash-api)
- 镜像：`iwaitu/clef-flash-api:latest`，平台 `linux/amd64`
- 默认服务地址：`http://127.0.0.1:18190`
- 交互接口文档：`http://127.0.0.1:18190/docs`

镜像包含 API 和运行环境。**模型权重需单独下载、挂载**，仓库和镜像均不包含模型文件。
本服务提供 Clef 原生 `/v1/systemone` 协议；客户端应按照本文请求格式调用。

## 运行要求

- NVIDIA Blackwell GPU，当前实测 RTX 5090；4096-token 配置单实例约使用 9.5 GiB 模型显存，另需留出运行余量。
- 兼容 CUDA 13 的 NVIDIA 驱动。
- Linux：Docker 与 NVIDIA Container Toolkit。
- Windows：启用 WSL2 和 GPU 支持的 Docker Desktop。
- 完整的 `alpha-x-ai/clef-flash-NVFP4` 模型目录，包括模型自带 Python 代码、配置、权重和 `joint_head.safetensors`。

API 在离线模式下使用已下载的模型，部署机器无需另外安装 Python 或 vLLM。
其他 GPU 型号尚未实测。

## 使用现成镜像

克隆仓库后，复制 `.env.example` 为 `.env`，将 `CLEF_MODEL_DIR` 改为目标机器的模型目录。

Linux 示例：

```bash
git clone https://github.com/iwaitu/clef-flash-api.git
cd clef-flash-api
cp .env.example .env
# 编辑 .env：CLEF_MODEL_DIR=/srv/models/clef-flash-NVFP4
docker compose pull
docker compose up -d --no-build
docker compose ps
docker compose logs -f
```

Windows PowerShell 示例：

```powershell
git clone https://github.com/iwaitu/clef-flash-api.git
Set-Location clef-flash-api
Copy-Item .env.example .env
# 编辑 .env：CLEF_MODEL_DIR=E:/LLMRoot/clef-flash-NVFP4
docker compose pull
docker compose up -d --no-build
docker compose ps
```

容器状态变为 `healthy` 后即可访问 `/docs`。首次启动需要加载权重和编译 GPU 内核，可能需要数分钟；
当前 Windows 模型目录挂载环境的首次加载实测约 644 秒。编译缓存保存在独立 Docker volume，后续启动可复用。

停止服务：

```bash
docker compose down
```

不使用 Compose 的 Linux 示例：

```bash
docker run -d --name clef-flash-api --gpus all --init --restart unless-stopped \
  --shm-size=2g -p 127.0.0.1:18190:8000 \
  -v /srv/models/clef-flash-NVFP4:/models/clef-flash:ro \
  -v clef-flash-api-cache:/root/.cache \
  iwaitu/clef-flash-api:latest
```

默认只向本机开放端口。远程访问可修改 `.env` 中的 `CLEF_BIND_ADDRESS`，由部署环境提供访问控制。
同一 Compose 网络内的其他容器可以使用 `http://clef-flash-api:8000`。

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 预热后的就绪状态、显存和请求数量 |
| GET | `/v1/models` | 模型名称与能力 |
| POST | `/v1/systemone` | 结构化判断 |
| GET | `/docs` | 可直接填写请求并测试的 Swagger 文档 |
| GET | `/openapi.json` | 接口定义 |

完整的三种问题类型示例见 [example-request.json](example-request.json)。
`state` 可以是文字、对象、数组等 JSON 值，不能为 null；一份请求可以同时包含多个问题。

```bash
curl http://127.0.0.1:18190/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @example-request.json
```

请求示例：

```json
{
  "model": "clef-flash",
  "state": "小明有3个苹果，又买了2个苹果。",
  "questions": {
    "answer": {
      "type": "choice",
      "instructions": "现在共有几个苹果？",
      "criteria": {"three": "3个", "five": "5个", "six": "6个"}
    }
  }
}
```

| 类型 | criteria 格式 | 返回内容 |
|---|---|---|
| `choice` | 选项 ID 到描述的对象 | 所选 ID、confidence、全部 probabilities |
| `noul` | 可省略；可用 true/false 描述自定义语义 | `noul`，命题为真的概率 |
| `score` | 从低到高的描述数组 | 从 0 开始的等级期望值、confidence、probabilities |

`instructions` 可省略，届时模型使用问题 ID 作为指令。问题名称和选项描述应明确；
业务上的“证据不足”应明确列为 choice 的选项，业务判断阈值应在实际部署环境中校准。

返回示例，数值仅用于说明格式：

```json
{
  "model": "clef-flash",
  "answers": {
    "answer": {
      "type": "choice",
      "choice": "five",
      "confidence": 0.97,
      "probabilities": {"three": 0.01, "five": 0.97, "six": 0.02}
    }
  },
  "usage": {"input_tokens": 190, "output_tokens": 0},
  "timing": {"queue_ms": 0.1, "inference_ms": 60.0, "total_ms": 60.1}
}
```

`inference_ms` 包含输入编码、模型和判断头执行、结果转换；`queue_ms` 是服务内等待时间，
`total_ms` 不包含客户端传输与请求验证。响应含 `X-Request-ID` 和 `X-Elapsed-Ms`，成功推理另含 `Server-Timing`。

## 配置与限制

`.env.example` 提供 Compose 配置。服务进程支持以下环境变量，覆盖 `config.json`：

| 变量 | 容器默认值 | 用途 |
|---|---|---|
| `CLEF_MODEL_PATH` | `/models/clef-flash` | 容器内模型路径 |
| `CLEF_HOST` | `0.0.0.0` | 容器内监听地址 |
| `CLEF_PORT` | `8000` | 容器内监听端口 |
| `CLEF_MAX_MODEL_LEN` | `4096` | 完整输入的 token 上限 |
| `CLEF_KV_CACHE_GIB` | `0.5` | KV cache 大小 |
| `CLEF_MAX_PENDING_REQUESTS` | `16` | 正在执行及排队请求数量上限 |
| `CLEF_MAX_BODY_BYTES` | `262144` | 请求体字节上限 |

- 仅支持文字和 JSON，未启用图像或视频。
- 输入上限包括 state、问题和选项；超长返回 422 `context_length_exceeded`，不会静默截断。
- 每份请求最多 32 个问题，每个问题最多 64 个选项。
- 参数错误返回 422，超大请求体返回 413，队列满返回 429。
- 单进程、单 GPU 执行线程；模型只加载一次，不应增加 Uvicorn workers 或开启 reload。
- NVFP4 使用 FlashInfer cuDNN 后端；Dockerfile 固定基础镜像摘要及依赖版本。

改变上下文长度会影响显存、编译时间和延迟；当前验证范围为 4096 tokens。

## 从 .NET 调用

建议长期复用 HttpClient。容器内 `127.0.0.1` 指向容器自身，跨容器调用需共享网络并使用服务名。

```csharp
using System.Net.Http.Json;
using System.Text.Json.Nodes;

using var client = new HttpClient {
    BaseAddress = new Uri("http://127.0.0.1:18190"),
    Timeout = TimeSpan.FromSeconds(60)
};
var request = new {
    model = "clef-flash",
    state = "本部门对本方案无原则性修改意见。",
    questions = new {
        opinion = new {
            type = "choice",
            instructions = "这句话是否明确点名了具体专题？仅依据原文判断。",
            criteria = new {
                explicit_topic = "明确点名了具体专题或对象",
                general_only = "只是对整个方案的一般性意见",
                unknown = "原文不足以判断"
            }
        }
    }
};
using var response = await client.PostAsJsonAsync("/v1/systemone", request);
response.EnsureSuccessStatusCode();
var result = await response.Content.ReadFromJsonAsync<JsonNode>();
Console.WriteLine(result?["answers"]?["opinion"]?["choice"]);
```

## 从源码构建及离线导出

Linux / Windows 均可构建：

```bash
docker build --platform linux/amd64 -t iwaitu/clef-flash-api:latest .
```

PowerShell 辅助脚本：

```powershell
.\build-image.ps1
.\export-image.ps1
```

导出文件附带 SHA256 校验文件，可使用 `docker load -i clef-flash-api-0.1.0-linux-amd64.tar` 离线导入。
模型目录仍需单独带到目标机器。构建上下文采用白名单，模型、缓存和日志不会进入镜像。

## 验证与响应时间

2026-10-06 的 RTX 5090 / Windows Docker Desktop 测试：8 组、13 个简单问题，240 次请求全部回答正确。
另验证三种问题类型、错误输入、超长输入拒绝和 8 个并发请求。

| 客户端 / 测试 | HTTP P50 | HTTP P95 |
|---|---|---|
| Python urllib，8 组测试各自 P50 的中位数 | 67.08 ms | — |
| Windows .NET HttpClient 持续连接，中文应用题 30 次 | 64.43 ms | 72.09 ms |

输入为 149–343 tokens 的短文本，这组简单测试不代表长材料速度或真实业务准确率。
与之前原生 WSL 环境的判断结果一致，概率最大绝对差为 0.0036，未达到 0.001 的概率一致性检查。
测试脚本分别记录功能结果与概率比较结果。

服务启动后，可在安装 Python 的客户端运行标准库验收脚本：

```bash
# 主机调用使用 18190；从同网络容器调用时可设置为 http://clef-flash-api:8000
python verify_api.py
```

可通过 `CLEF_API_BASE_URL` 设置测试地址。`test-fixtures.json` 包含全部测试输入和原生概率基线。
脚本生成 `api-test-results.json`，功能失败会返回非零退出码；概率差异单独记录为 `MATCH` 或 `DIFFERENT`。
PowerShell 可执行 `.\test-api.ps1` 发起示例请求，或 `.\benchmark-windows.ps1` 测试持续连接下的响应时间。

模型及其自带代码的使用条件以 [模型仓库](https://huggingface.co/alpha-x-ai/clef-flash-NVFP4) 为准。
