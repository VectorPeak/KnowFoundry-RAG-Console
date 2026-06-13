from qa_core.config.preflight import _is_placeholder


def test_is_placeholder_rejects_empty_and_sample_values() -> None:
    assert _is_placeholder("")
    assert _is_placeholder("replace-with-real-key")
    assert _is_placeholder("请替换为真实可用的模型服务 Key")
    assert _is_placeholder("请替换为随机长令牌")


def test_is_placeholder_accepts_realistic_values() -> None:
    assert not _is_placeholder("sk-prod-abc123456789")
    assert not _is_placeholder("admin-token-abc123456789")
