"""数据域隔离模型与检索过滤工具。

DataScope 是检索安全边界的基础模型，统一携带 tenant_id、dataset_id、visibility
和用户角色。入库时这些字段会写入 Milvus metadata；检索时会转换成 Milvus expr，
确保多租户、多数据集和角色权限不会互相串数据。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
DEFAULT_TENANT_ID = "default"
DEFAULT_DATASET_ID = "default"
DEFAULT_VISIBILITY = "public"
DEFAULT_USER_ROLE = "public"

def _clean_token(value: str | None, default: str) -> str:
    """清洗短标识，去除空白并提供默认值。

    参数：
        value: 原始输入值，可能为空或只有空白字符。
        default: 清洗后为空时使用的默认值。

    返回：
        清洗后的非空字符串。
    """
    cleaned = str(value or "").strip()
    return cleaned or default


def _clean_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """清洗字符串列表，顺序去重。

    参数：
        values: 原始字符串列表、元组或 None。

    返回：
        清洗后且按原顺序去重的列表。
    """
    result: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def escape_expr_value(value: str) -> str:
    """转义 Milvus expr 中的字符串值。

    参数：
        value: 原始字符串值。

    返回：
        可安全放入 Milvus 双引号表达式中的字符串。
    """
    return str(value).replace('"', '\\"')


@dataclass(frozen=True)
class DataScope:
    """入库或检索时使用的数据域，包含租户、数据集、可见级别和角色，用于 Milvus 过滤。（★★★ 核心）

    数据隔离是安全边界：不同租户、数据集之间的数据绝对不能互相可见；可见级别（public/internal/private）
    和角色控制确保用户只能访问授权范围内的知识库内容。

    字段说明：
      - tenant_id：租户标识，默认 default。
      - dataset_id：数据集标识，默认 default。
      - visibility：可见级别，支持 public/internal/private。
      - user_roles：当前用户拥有的角色列表。
    """

    tenant_id: str = DEFAULT_TENANT_ID
    dataset_id: str = DEFAULT_DATASET_ID
    visibility: str = DEFAULT_VISIBILITY
    user_roles: list[str] = field(default_factory=lambda: [DEFAULT_USER_ROLE])

    @classmethod
    def from_request(
        cls,
        *,
        tenant_id: str | None = None,
        dataset_id: str | None = None,
        visibility: str | None = None,
        user_roles: list[str] | tuple[str, ...] | None = None,
        user_role: str | None = None,
    ) -> "DataScope":
        """从 API 请求构建数据域，user_role 和 user_roles 合并去重。（★★ 理解）

        兼容两套角色传参方式：user_role（单值，来自 JWT token）和 user_roles（多值，来自 API 显式参数）；
        二者合并去重确保一个用户有多个角色时仍能正确匹配。

        参数：
            tenant_id: 请求中的租户标识。
            dataset_id: 请求中的数据集标识。
            visibility: 请求中的可见级别。
            user_roles: 请求中的多角色列表。
            user_role: 请求中的单个角色，会合并进 user_roles。

        返回：
            清洗后的不可变 DataScope 实例。
        """
        roles = _clean_list(user_roles)
        single_role = _clean_token(user_role, "")
        if single_role and single_role not in roles:
            roles.append(single_role)
        if not roles:
            roles = [DEFAULT_USER_ROLE]
        return cls(
            tenant_id=_clean_token(tenant_id, DEFAULT_TENANT_ID),
            dataset_id=_clean_token(dataset_id, DEFAULT_DATASET_ID),
            visibility=_clean_token(visibility, DEFAULT_VISIBILITY),
            user_roles=roles,
        )

    def metadata(self, *, allowed_roles: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        """返回写入 Milvus metadata 的数据域字段。

        参数：
            allowed_roles: 可访问该数据的角色列表；为空时使用当前 DataScope 的 user_roles。

        返回：
            包含 tenant_id、dataset_id、visibility、allowed_roles 的 metadata 字典。
        """
        roles = _clean_list(allowed_roles) or list(self.user_roles)
        return {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "visibility": self.visibility,
            "allowed_roles": roles,
        }

    def as_dict(self) -> dict[str, Any]:
        """返回 API 诊断使用的结构化数据域。

        返回：
            包含 tenant_id、dataset_id、visibility、user_roles 的字典。
        """
        return {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "visibility": self.visibility,
            "user_roles": list(self.user_roles),
        }

    def expr_clauses(self) -> list[str]:
        """转换为 Milvus 过滤子句。

        示例：
            ``DataScope(visibility="private", user_roles=["admin"]).expr_clauses()``
            returns::

                [
                    'tenant_id == "default"',
                    'dataset_id == "default"',
                    '(visibility == "public" or visibility == "private")',
                    '(array_contains(allowed_roles, "admin"))',
                ]

        执行流程：
          1. 添加 tenant_id 过滤。
          2. 添加 dataset_id 过滤。
          3. 构建 visibility 过滤：public 始终可见，internal/private 只在请求级别允许时加入。
          4. 构建角色过滤：用户任一角色命中 allowed_roles 即可访问。
          5. 返回所有 Milvus expr 子句。

        返回：
            Milvus 布尔表达式子句列表。
        """
        clauses = [
            f'tenant_id == "{escape_expr_value(self.tenant_id)}"',
            f'dataset_id == "{escape_expr_value(self.dataset_id)}"',
        ]
        # 可见级别策略：public 始终可见（任何角色都能看到公开内容）；
        # 内部/私有级别需要增加对应匹配，确保用户不能越级看到高保密内容
        allowed_visibility = ["public"]
        if self.visibility in {"internal", "private"}:
            allowed_visibility.append(self.visibility)
        visibility_expr = " or ".join(f'visibility == "{escape_expr_value(item)}"' for item in allowed_visibility)
        clauses.append(f"({visibility_expr})")

        # 角色过滤：用户必须至少拥有一个 allowed_roles 中的角色才能访问该文档；
        # 任意角色匹配即可（OR 语义），适用于一个用户有多个业务角色的场景
        role_exprs = [f'array_contains(allowed_roles, "{escape_expr_value(role)}")' for role in self.user_roles]
        if role_exprs:
            clauses.append(f"({' or '.join(role_exprs)})")
        return clauses


def resolve_data_scope(
    *,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_roles: list[str] | tuple[str, ...] | None = None,
    user_role: str | None = None,
) -> DataScope:
    """构建当前请求或入库任务的数据域。

    参数：
        tenant_id: 租户标识。
        dataset_id: 数据集标识。
        visibility: 可见级别。
        user_roles: 多角色列表。
        user_role: 单个角色。

    返回：
        清洗后的 DataScope 实例。
    """
    return DataScope.from_request(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_roles=user_roles,
        user_role=user_role,
    )
