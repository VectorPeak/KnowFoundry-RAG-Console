# KnowForge RAG Platform Versioning

本文件用于维护项目仓库的版本、分支和阶段开发边界，不属于学生讲义内容。

## 当前基线

- 一期稳定基线标签：`v1.0.0-phase1-baseline`
- 基线提交：`e5b3b1c`
- 基线含义：多场景企业级 RAG 一期项目的完整代码、讲义、动画、测试和 Docker 部署版本。

## 长期分支

| 分支 | 用途 | 规则 |
| --- | --- | --- |
| `main` | 稳定主线 | 只合并已经验证过的阶段版本或维护修复 |
| `phase1-maintenance` | 一期维护 | 修复一期代码、讲义、命令、Docker、测试，不加入二期新功能 |
| `phase2-graphrag` | 二期开发 | 在一期稳定基础上开发 GraphRAG 和二期能力 |

## 日常开发流程

### 一期发现问题

在 `phase1-maintenance` 修改：

```powershell
git checkout phase1-maintenance
git pull
```

修改完成后运行必要检查：

```powershell
docker compose --env-file .env.compose config --quiet
python scripts/check_project_guardrails.py
python scripts/check_docs_consistency.py
python scripts/check_no_polyfill_io.py
pytest tests -q
```

通过后提交并推送：

```powershell
git add .
git commit -m "Fix phase1 ..."
git push
```

确认稳定后再合并回 `main`，并同步到二期分支：

```powershell
git checkout main
git merge phase1-maintenance
git push

git checkout phase2-graphrag
git merge main
git push
```

### 二期开发

在 `phase2-graphrag` 修改：

```powershell
git checkout phase2-graphrag
git pull
```

二期功能未验证完成前，不直接合并回 `main`。

## 阶段标签

每个阶段完成后都打一个不可变标签，方便回滚和对照：

```powershell
git tag v2.0.0-phase2-baseline
git push origin v2.0.0-phase2-baseline
```

建议命名：

- `v1.0.0-phase1-baseline`
- `v1.0.1-phase1-maintenance`
- `v1.1.0-codealong-complete`
- `v2.0.0-phase2-baseline`
- `v3.0.0-phase3-baseline`

`v1.1.0-codealong-complete` 表示 `codealong/` 已经完成第 05 章到第 19 章的课堂版跟敲闭环。这个标签不是生产功能版本，而是课程代码阶段节点。

## 跟敲项目代码边界

跟敲型课程代码放在本仓库顶层 `codealong/` 目录：

```text
codealong/
  README.md
  CODEALONG_PLAN.md
  chapters/
    ch05_intent_classification/
    ch06_retrieval_strategy/
    ...
```

`codealong/` 由 Git 管理，但不会进入主项目 Docker 镜像。它和主项目源码保持清晰边界：

- 主项目源码仍放在 `qa_core/`、`app.py`、`scripts/`、`scenarios/` 等目录。
- 学生跟敲代码只放在 `codealong/chapters/`。
- 主项目测试不默认扫描 `codealong/`。
- 跟敲章节需要自己的 README、源码和测试，保证每章结束都有独立闭环。

如果一期主项目发生修复，先判断是否影响课堂跟敲；只同步真正需要学生理解和亲手实现的部分，不把完整生产工程细节直接塞进跟敲目录。
