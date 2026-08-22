from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_campaign_episodes import (  # noqa: E402
    build_campaign_episodes,
    filter_code_patch,
    preimage_context,
    resolve_commit,
)


PATCH = """diff --git a/docs/note.md b/docs/note.md
index 1111111..2222222 100644
--- a/docs/note.md
+++ b/docs/note.md
@@ -1 +1 @@
-old prose
+new prose
diff --git a/src/ops/kernel.cu b/src/ops/kernel.cu
index 3333333..4444444 100644
--- a/src/ops/kernel.cu
+++ b/src/ops/kernel.cu
@@ -10,3 +10,3 @@ void launch() {
   keep();
-  old_path();
+  new_path();
 }
"""


class PatchExtractionTests(unittest.TestCase):
    def test_filter_code_patch_excludes_non_code_files(self) -> None:
        filtered = filter_code_patch(PATCH)
        self.assertNotIn("docs/note.md", filtered)
        self.assertIn("src/ops/kernel.cu", filtered)
        self.assertIn("+  new_path();", filtered)

    def test_preimage_contains_context_and_deleted_lines_only(self) -> None:
        preimage = preimage_context(PATCH)
        self.assertIn("keep();", preimage)
        self.assertIn("-  old_path();", preimage)
        self.assertNotIn("new_path();", preimage)
        self.assertNotIn("docs/note.md", preimage)

    def test_resolve_commit_requires_a_unique_prefix(self) -> None:
        records = {
            "abcdef0123456789abcdef0123456789abcdef01": {},
            "1234567890abcdef1234567890abcdef12345678": {},
        }
        self.assertEqual(
            resolve_commit("abcdef0", records),
            "abcdef0123456789abcdef0123456789abcdef01",
        )
        self.assertIsNone(resolve_commit("deadbee", records))

        ambiguous = {
            "abcdef0123456789abcdef0123456789abcdef01": {},
            "abcdef0999999999999999999999999999999999": {},
        }
        with self.assertRaisesRegex(ValueError, "ambiguous commit prefix"):
            resolve_commit("abcdef0", ambiguous)


class ClosedCampaignIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes, cls.stats = build_campaign_episodes()

    def test_all_training_views_are_materialized(self) -> None:
        self.assertGreater(self.stats["candidate_families"], 0)
        self.assertEqual(
            set(self.stats["by_view"]),
            {"diagnosis", "implementation", "judgment", "orchestration_reporting"},
        )

    def test_training_split_contains_positive_and_negative_candidate_work(self) -> None:
        training = [
            episode
            for episode in self.episodes
            if episode["split_group"]["assigned_split"] == "train"
        ]
        self.assertTrue(any(item["task_view"] == "diagnosis" for item in training))
        self.assertTrue(any(item["task_view"] == "implementation" for item in training))
        self.assertTrue(any(item["outcome"]["disposition"] == "accepted" for item in training))
        self.assertTrue(any(item["outcome"]["disposition"] == "rejected" for item in training))

    def test_sm86_remains_a_complete_cross_architecture_holdout(self) -> None:
        sm86 = [
            episode
            for episode in self.episodes
            if episode["scope"]["architectures"] == ["sm_86"]
        ]
        self.assertGreater(len(sm86), 0)
        self.assertTrue(
            all(
                item["split_group"]["assigned_split"] == "test"
                and item["split_group"]["evaluation_tier"] == "cross_arch"
                for item in sm86
            )
        )

    def test_latest_qwen_consolidation_acceptance_is_preserved_as_a_holdout(self) -> None:
        prefix = "ginfer-auto-sm120a-rtx5090-qwen38-consolidation-20260822-c10-02-"
        accepted = [
            episode for episode in self.episodes if episode["episode_id"].startswith(prefix)
        ]
        self.assertEqual(
            {episode["task_view"] for episode in accepted},
            {"diagnosis", "implementation", "judgment", "orchestration_reporting"},
        )
        self.assertTrue(
            all(
                episode["outcome"]["disposition"] == "accepted"
                and episode["outcome"]["candidate_commit"]
                == "37a7080c2988aa81c9991eb0bd9b8efac2ee273e"
                and episode["split_group"]["assigned_split"] == "test"
                for episode in accepted
            )
        )

    def test_implementation_targets_are_exact_code_diffs_without_result_leakage(self) -> None:
        implementations = [
            episode for episode in self.episodes if episode["task_view"] == "implementation"
        ]
        self.assertGreater(len(implementations), 0)
        for episode in implementations:
            events = {event["event_id"]: event for event in episode["events"]}
            input_events = [events[event_id] for event_id in episode["view"]["input_event_ids"]]
            target_events = [events[event_id] for event_id in episode["view"]["target_event_ids"]]
            self.assertTrue(
                any(
                    event["event_type"] == "patch"
                    and "diff --git" in str(event["payload"])
                    for event in target_events
                ),
                episode["episode_id"],
            )
            self.assertTrue(
                all(
                    event["information_class"]
                    in {"task_context", "pre_candidate_evidence"}
                    for event in input_events
                ),
                episode["episode_id"],
            )

    def test_rejected_committed_candidates_have_exact_restorations(self) -> None:
        for episode in self.episodes:
            outcome = episode["outcome"]
            if outcome["disposition"] == "rejected" and outcome["candidate_commit"]:
                self.assertTrue(outcome["restoration_commit"], episode["episode_id"])

    def test_judgment_uses_the_exact_patch_when_history_is_available(self) -> None:
        judgments = [
            episode for episode in self.episodes if episode["task_view"] == "judgment"
        ]
        exact = 0
        for episode in judgments:
            events = {event["event_id"]: event for event in episode["events"]}
            candidate_events = [
                events[event_id]
                for event_id in episode["view"]["input_event_ids"]
                if events[event_id]["information_class"] == "candidate_artifact"
            ]
            if any("diff --git" in str(event["payload"]) for event in candidate_events):
                exact += 1
        self.assertEqual(exact, self.stats["candidate_families_with_exact_patch"])


if __name__ == "__main__":
    unittest.main()
