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
    def test_identity_registry_prefers_curated_display_names_and_aliases(self) -> None:
        registry = face_vlm_classifier.build_identity_registry(
            reference_labels=["bill-clinton", "jeffrey-epstein"],
            person_catalog={
                "bill-clinton": {
                    "display_name": "William Jefferson Clinton",
                    "aliases": ["Bill Clinton", "President Clinton"],
                }
            },
        )
        alias_map = face_vlm_classifier.build_identity_alias_map(registry)
        display_name_map = face_vlm_classifier.build_display_name_map(registry)

        canonical_name, canonical_label, known_label = face_vlm_classifier.canonicalize_identity_name(
            "President Clinton",
            alias_map,
            display_name_map,
        )

        self.assertEqual(canonical_name, "William Jefferson Clinton")
        self.assertEqual(canonical_label, "bill-clinton")
        self.assertEqual(known_label, "bill-clinton")

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

    def test_normalize_model_result_preserves_freeform_identity_and_maps_known_labels(self) -> None:
        registry = face_vlm_classifier.build_identity_registry(
            reference_labels=["bill-clinton", "jeffrey-epstein"],
            person_catalog={
                "bill-clinton": {"display_name": "Bill Clinton", "aliases": ["William Jefferson Clinton"]},
                "jeffrey-epstein": {"display_name": "Jeffrey Epstein", "aliases": ["Jeff Epstein"]},
            },
        )
        alias_map = face_vlm_classifier.build_identity_alias_map(registry)
        display_name_map = face_vlm_classifier.build_display_name_map(registry)
        normalized = face_vlm_classifier.normalize_model_result(
            {
                "best_identity": "Bill Clinton",
                "confidence": 0.91,
                "match_basis": "Facial features and visible passport text both align.",
                "alternate_identities": ["Jeffrey Epstein", "Sarah Ferguson"],
                "visible_people_count_estimate": "2",
                "setting": "Indoor event",
                "scene_tags": ["Indoor", "podium", "podium", "!!!"],
                "notes": "Matched hairstyle and face shape.",
            },
            known_alias_map=alias_map,
            display_name_map=display_name_map,
        )

        self.assertEqual(normalized["best_identity"], "Bill Clinton")
        self.assertEqual(normalized["best_identity_label"], "bill-clinton")
        self.assertEqual(normalized["known_reference_label"], "bill-clinton")
        self.assertEqual(normalized["confidence"], "high")
        self.assertEqual(normalized["alternate_identities"], ["Jeffrey Epstein", "Sarah Ferguson"])
        self.assertEqual(
            normalized["alternate_identity_labels"],
            ["jeffrey-epstein", "sarah-ferguson"],
        )
        self.assertEqual(normalized["visible_people_count_estimate"], 2)
        self.assertEqual(normalized["setting"], "indoor event")
        self.assertEqual(normalized["scene_tags"], ["indoor", "podium"])

    def test_select_known_identity_hints_caps_global_list_and_prioritizes_relevant_labels(self) -> None:
        identity_registry = face_vlm_classifier.build_identity_registry(
            reference_labels=["bill-clinton", "jeffrey-epstein", "ghislaine-maxwell", "prince-andrew"],
            person_catalog={},
        )
        hints = face_vlm_classifier.select_known_identity_hints(
            identity_registry=identity_registry,
            shortlist_labels=["prince-andrew"],
            mention_counts={"bill-clinton": 3},
            observed_identity_counts={"jeffrey-epstein": 12},
            limit=2,
        )

        self.assertEqual(hints, ["jeffrey-epstein", "prince-andrew"])

    def test_build_observed_identity_counts_counts_unique_images(self) -> None:
        counts = face_vlm_classifier.build_observed_identity_counts(
            [
                {
                    "faces": [
                        {"known_reference_label": "jeffrey-epstein", "predicted_label": "jeffrey-epstein"},
                        {"known_reference_label": "jeffrey-epstein", "predicted_label": "jeffrey-epstein"},
                        {"known_reference_label": "bill-clinton", "predicted_label": "bill-clinton"},
                    ]
                },
                {
                    "faces": [
                        {"known_reference_label": "jeffrey-epstein", "predicted_label": "jeffrey-epstein"},
                    ]
                },
            ],
            {"jeffrey-epstein", "bill-clinton"},
        )

        self.assertEqual(counts, {"jeffrey-epstein": 2, "bill-clinton": 1})

    def test_format_known_identity_hints_includes_counts(self) -> None:
        registry = face_vlm_classifier.build_identity_registry(
            reference_labels=["bill-clinton", "jeffrey-epstein"],
            person_catalog={},
        )
        display_name_map = face_vlm_classifier.build_display_name_map(registry)
        formatted = face_vlm_classifier.format_known_identity_hints(
            ["jeffrey-epstein", "bill-clinton"],
            display_name_map,
            {"jeffrey-epstein": 9, "bill-clinton": 2},
        )

        self.assertEqual(formatted, "Jeffrey Epstein (9), Bill Clinton (2)")

    def test_build_identity_registry_payload_embeds_counts_and_aliases(self) -> None:
        registry = face_vlm_classifier.build_identity_registry(
            reference_labels=["bill-clinton"],
            person_catalog={
                "bill-clinton": {
                    "display_name": "Bill Clinton",
                    "aliases": ["William Jefferson Clinton", "Bill Clinton"],
                }
            },
        )
        payload = face_vlm_classifier.build_identity_registry_payload(
            identity_registry=registry,
            observed_identity_counts={"bill-clinton": 4},
            person_catalog_path=Path("/tmp/person-catalog.json"),
            known_identity_hint_limit=30,
        )

        self.assertEqual(payload["known_identity_hint_limit"], 30)
        self.assertEqual(payload["ranked_labels_by_observed_image_count"], ["bill-clinton"])
        self.assertEqual(payload["labels"]["bill-clinton"]["observed_image_count"], 4)
        self.assertIn("William Jefferson Clinton", payload["labels"]["bill-clinton"]["aliases"])

    def test_resolve_person_catalog_out_path_defaults_next_to_master_catalog(self) -> None:
        resolved = face_vlm_classifier.resolve_person_catalog_out_path(
            None,
            Path("/tmp/master-image-catalog-vlm.json"),
        )

        self.assertEqual(resolved, Path("/tmp/master-image-catalog-vlm-persons.json"))

    def test_build_prompt_contract_exposes_expected_fields(self) -> None:
        contract = face_vlm_classifier.build_prompt_contract()
        self.assertIn("system_prompt", contract)
        self.assertIn("user_instruction_template", contract)
        self.assertIn("best_identity", contract["user_instruction_template"])
        self.assertIn("investigative journalists", contract["system_prompt"])
        self.assertIn("Known people already cataloged", contract["user_instruction_template"])
        self.assertIn("sorted by prior normalized image counts", contract["user_instruction_template"])
        self.assertNotIn("Allowed labels", contract["user_instruction_template"])
        self.assertEqual(
            contract["output_fields"],
            [
                "best_identity",
                "confidence",
                "match_basis",
                "alternate_identities",
                "visible_people_count_estimate",
                "setting",
                "scene_tags",
                "notes",
            ],
        )


if __name__ == "__main__":
    unittest.main()
