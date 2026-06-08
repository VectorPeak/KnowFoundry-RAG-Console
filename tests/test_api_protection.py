"""API 基础保护能力测试。

当前项目要求管理令牌是必需环境配置，不再提供“本地为空则放行”的降级路径。这里直接
测试 `qa_core.api.dependencies` 的依赖函数，避免启动真实服务。
"""

from __future__ import annotations

import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from qa_core.api import dependencies as api_deps
from qa_core.api.error_handlers import register_api_exception_handlers
from qa_core.api.error_handlers import raise_bad_request, raise_not_found, raise_too_many_requests


class ApiProtectionTests(unittest.TestCase):
    """验证管理令牌和限流行为可控。"""

    def test_admin_token_requires_configured_token(self) -> None:
        original = api_deps.settings.admin_api_token
        api_deps.settings.admin_api_token = ""
        try:
            with self.assertRaises(HTTPException) as ctx:
                api_deps.require_admin_token(None)
            self.assertEqual(ctx.exception.status_code, 500)
        finally:
            api_deps.settings.admin_api_token = original

    def test_admin_token_rejects_wrong_token_when_enabled(self) -> None:
        original = api_deps.settings.admin_api_token
        api_deps.settings.admin_api_token = "secret"
        try:
            with self.assertRaises(HTTPException) as ctx:
                api_deps.require_admin_token("bad")
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIsNone(api_deps.require_admin_token("secret"))
        finally:
            api_deps.settings.admin_api_token = original

    def test_rate_limit_can_block_after_limit(self) -> None:
        original_limit = api_deps.settings.api_rate_limit_per_minute
        api_deps.settings.api_rate_limit_per_minute = 2
        api_deps.RATE_BUCKETS.clear()
        try:
            self.assertTrue(api_deps.check_rate_limit("unit-test"))
            self.assertTrue(api_deps.check_rate_limit("unit-test"))
            self.assertFalse(api_deps.check_rate_limit("unit-test"))
        finally:
            api_deps.settings.api_rate_limit_per_minute = original_limit
            api_deps.RATE_BUCKETS.clear()

    def test_value_error_becomes_http_400(self) -> None:
        app = FastAPI()
        register_api_exception_handlers(app)

        @app.get("/value-error")
        def value_error_route():
            raise ValueError("业务分类不合法")

        response = TestClient(app).get("/value-error")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "业务分类不合法"})

    def test_unexpected_error_becomes_stable_http_500(self) -> None:
        app = FastAPI()
        register_api_exception_handlers(app)

        @app.get("/unexpected-error")
        def unexpected_error_route():
            raise RuntimeError("database password leaked in stack")

        response = TestClient(app, raise_server_exceptions=False).get("/unexpected-error")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "服务内部错误，请查看后端日志"})

    def test_http_error_helpers_keep_status_code_semantics(self) -> None:
        cases = [
            (raise_bad_request, 400),
            (raise_not_found, 404),
            (raise_too_many_requests, 429),
        ]

        for raiser, status_code in cases:
            with self.subTest(status_code=status_code):
                with self.assertRaises(HTTPException) as ctx:
                    raiser("错误信息")
                self.assertEqual(ctx.exception.status_code, status_code)
                self.assertEqual(ctx.exception.detail, "错误信息")


if __name__ == "__main__":
    unittest.main()
