"""MySQL 控制面存储测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from sqlalchemy import text

from qa_core.governance.kb_versions import KB_ACTIVE_TABLE, KB_VERSIONS_TABLE, KnowledgeBaseVersionStore
from qa_core.indexing.manifest import INDEX_MANIFEST_TABLE, IndexManifest


class MySqlMetadataStoreTests(unittest.TestCase):
    """验证知识库版本和文档 manifest 都落在 MySQL。"""

    scenario_id = "enterprise_knowledge"

    def setUp(self) -> None:
        self.version_store = KnowledgeBaseVersionStore(self.scenario_id)
        self.version_store.ensure_tables()
        self.original_pointer = self.version_store._active_pointer()
        self._cleanup_versions()

        self.manifest = IndexManifest()
        self.manifest.ensure_table()
        self._cleanup_manifest()

    def tearDown(self) -> None:
        self._cleanup_versions()
        with self.version_store.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {KB_ACTIVE_TABLE}
                        (scenario_id, active_kb_version, previous_kb_version)
                    VALUES
                        (:scenario_id, :active_kb_version, :previous_kb_version)
                    ON DUPLICATE KEY UPDATE
                        active_kb_version=VALUES(active_kb_version),
                        previous_kb_version=VALUES(previous_kb_version)
                    """
                ),
                {
                    "scenario_id": self.scenario_id,
                    "active_kb_version": self.original_pointer[0],
                    "previous_kb_version": self.original_pointer[1],
                },
            )
        self._cleanup_manifest()

    def _cleanup_versions(self) -> None:
        with self.version_store.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {KB_VERSIONS_TABLE}
                    WHERE scenario_id=:scenario_id AND kb_version LIKE 'kb_mysql_unit_%'
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def _cleanup_manifest(self) -> None:
        with self.manifest.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {INDEX_MANIFEST_TABLE}
                    WHERE scenario_id=:scenario_id AND kb_version LIKE 'kb_mysql_unit_%'
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def test_kb_versions_use_mysql_control_plane(self) -> None:
        v1 = self.version_store.ensure_version("kb_mysql_unit_v1")
        v2 = self.version_store.ensure_version("kb_mysql_unit_v2")
        self.version_store.activate_version(v1.kb_version)
        self.version_store.activate_version(v2.kb_version)

        payload = self.version_store.as_payload()

        self.assertEqual(payload["metadata_store"], "mysql")
        self.assertEqual(payload["active_version"], "kb_mysql_unit_v2")
        self.assertEqual(self.version_store.get("kb_mysql_unit_v1").status, "STAGED")
        self.assertEqual(self.version_store.get("kb_mysql_unit_v2").status, "ACTIVE")

    def test_index_manifest_uses_mysql_records(self) -> None:
        path = Path("scenarios/enterprise_knowledge/data/hr_data/onboarding.md")
        self.manifest.update(
            "hr",
            path,
            "fingerprint-1",
            ["chunk-a", "chunk-b"],
            scenario_id=self.scenario_id,
            kb_version="kb_mysql_unit_v1",
            embedding_model_version="embed-v1",
            chunk_schema_version="schema-v1",
        )

        record = self.manifest.get("hr", path, "kb_mysql_unit_v1", self.scenario_id)
        records = self.manifest.iter_records(scenario_id=self.scenario_id, kb_version="kb_mysql_unit_v1")

        self.assertIsNotNone(record)
        self.assertEqual(record.chunk_ids, ["chunk-a", "chunk-b"])
        self.assertEqual(len(records), 1)
        removed = self.manifest.remove_by_key(record.key)
        self.assertEqual(removed.fingerprint, "fingerprint-1")
        self.assertIsNone(self.manifest.get("hr", path, "kb_mysql_unit_v1", self.scenario_id))


if __name__ == "__main__":
    unittest.main()
