import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "faces" / "classify_face_candidates_vlm.py"
SPEC = importlib.util.spec_from_file_location("face_vlm_classifier", MODULE_PATH)
face_vlm_classifier = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(face_vlm_classifier)


class FaceVlmClassifierHelpersTest(unittest.TestCase):
    def test_parse_volume_tokens_normalizes_inputs(self) -> None:
        self.assertEqual(
            face_vlm_classifier.parse_volume_tokens("2, VOL00008,vol9"),
            ["VOL00002", "VOL00008", "VOL00009"],
        )

    def test_parse_processed_image_path_extracts_metadata(self) -> None:
        path = Path("/Volumes/T7/Epstine_data/epstine_images/vol00008/0002/efta00016800-p03-img-01.webp")
        parsed = face_vlm_classifier.parse_processed_image_path(path)
        assert parsed is not None
        self.assertEqual(parsed["volume"], "VOL00008")
        self.assertEqual(parsed["subdir"], "0002")
        self.assertEqual(parsed["efta_id"], "EFTA00016800")
        self.assertEqual(parsed["page"], 3)
        self.assertEqual(parsed["image_index"], 1)

    def test_load_env_file_strips_quotes_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "export OPENROUTER_API_KEY='abc123'",
                        'OPENROUTER_TITLE="Face Ranker"',
                        "OPENROUTER_NO_FALLBACKS=true",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = face_vlm_classifier.load_env_file(env_path)

        self.assertEqual(loaded["OPENROUTER_API_KEY"], "abc123")
        self.assertEqual(loaded["OPENROUTER_TITLE"], "Face Ranker")
        self.assertEqual(loaded["OPENROUTER_NO_FALLBACKS"], "true")

    def test_shortlist_labels_combines_embedding_scores_and_mentions(self) -> None:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        gallery = {
            "bill-clinton": [np.asarray([0.99, 0.01], dtype=np.float32)],
            "jeffrey-epstein": [np.asarray([0.8, 0.2], dtype=np.float32)],
            "ghislaine-maxwell": [np.asarray([0.0, 1.0], dtype=np.float32)],
        }
        shortlist, score_map = face_vlm_classifier.shortlist_labels(
            query_embedding=query,
            gallery=gallery,
            mention_counts={"ghislaine-maxwell": 3},
            shortlist_size=3,
            mention_shortlist_size=1,
            min_similarity=0.5,
        )

        self.assertEqual(shortlist[0], "bill-clinton")
        self.assertIn("ghislaine-maxwell", shortlist)
        self.assertGreater(score_map["bill-clinton"], score_map["ghislaine-maxwell"])

    def test_normalize_model_result_filters_unknown_labels(self) -> None:
        normalized = face_vlm_classifier.normalize_model_result(
            {
                "best_identity": "bill-clinton",
                "confidence": 0.91,
                "match_basis": "Facial features and visible passport text both align.",
                "alternate_identities": ["jeffrey-epstein", "not-allowed"],
                "visible_people_count_estimate": "2",
                "setting": "Indoor event",
                "scene_tags": ["Indoor", "podium", "podium", "!!!"],
                "notes": "Matched hairstyle and face shape.",
            },
            ["bill-clinton", "jeffrey-epstein"],
        )

        self.assertEqual(normalized["best_identity"], "bill-clinton")
        self.assertEqual(normalized["confidence"], "high")
        self.assertEqual(normalized["alternate_identities"], ["jeffrey-epstein"])
        self.assertEqual(normalized["visible_people_count_estimate"], 2)
        self.assertEqual(normalized["setting"], "indoor event")
        self.assertEqual(normalized["scene_tags"], ["indoor", "podium"])


if __name__ == "__main__":
    unittest.main()
