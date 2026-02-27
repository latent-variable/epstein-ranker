import argparse
import csv
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import gpt_ranker


class GptRankerHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.skip_args = argparse.Namespace(
            min_text_chars=60,
            min_text_words=12,
            min_alpha_ratio=0.25,
            min_unique_word_ratio=0.15,
            max_short_token_ratio=0.6,
            min_avg_word_length=3.0,
            min_long_word_count=4,
            max_repeated_char_run=40,
            include_action_items=False,
            justice_files_base_url=gpt_ranker.DEFAULT_JUSTICE_FILES_BASE_URL,
        )

    def test_iter_rows_supports_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            (nested / "first.txt").write_text("hello world", encoding="utf-8")
            (root / "second.txt").write_text("another file", encoding="utf-8")
            (root / "ignore.md").write_text("should not load", encoding="utf-8")

            rows = list(gpt_ranker.iter_rows(root, input_glob="*.txt", include_text=True))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "a/b/first.txt")
        self.assertEqual(rows[1]["filename"], "second.txt")
        self.assertEqual(rows[0]["text"], "hello world")

    def test_iter_rows_supports_csv_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "text"])
                writer.writeheader()
                writer.writerow({"filename": "one.txt", "text": "hello"})
                writer.writerow({"filename": "two.txt", "text": "world"})

            rows = list(gpt_ranker.iter_rows(csv_path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "one.txt")
        self.assertEqual(rows[1]["text"], "world")

    def test_iter_rows_supports_image_mode_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_pdf = root / "doc.pdf"
            image_pdf.write_bytes(b"%PDF-1.4\n")
            rows = list(
                gpt_ranker.iter_rows(
                    root,
                    input_glob="*.pdf",
                    include_text=True,
                    processing_mode="image",
                )
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_kind"], "image")
        self.assertEqual(rows[0]["text"], "")

    def test_iter_rows_splits_pdf_into_parts_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_pdf = root / "doc.pdf"
            image_pdf.write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", return_value=10):
                rows = list(
                    gpt_ranker.iter_rows(
                        root,
                        input_glob="*.pdf",
                        include_text=False,
                        processing_mode="image",
                        pdf_part_pages=4,
                    )
                )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["part_index"], 1)
        self.assertEqual(rows[0]["part_total"], 3)
        self.assertEqual(rows[0]["part_start_page"], 1)
        self.assertEqual(rows[0]["part_end_page"], 4)
        self.assertIn("::part_0001_p00001-00004", rows[0]["source_id"])
        self.assertEqual(rows[1]["part_start_page"], 5)
        self.assertEqual(rows[1]["part_end_page"], 8)
        self.assertEqual(rows[2]["part_start_page"], 9)
        self.assertEqual(rows[2]["part_end_page"], 10)

    def test_iter_rows_resume_fast_path_skips_page_count_for_completed_file_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_pdf = root / "doc.pdf"
            image_pdf.write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(
                gpt_ranker,
                "detect_pdf_page_count",
                side_effect=AssertionError("page count should not be requested"),
            ):
                rows = list(
                    gpt_ranker.iter_rows(
                        root,
                        input_glob="*.pdf",
                        include_text=False,
                        processing_mode="image",
                        pdf_part_pages=4,
                        resume_completed_file_ids={"doc.pdf"},
                    )
                )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_id"], "doc.pdf")
        self.assertEqual(rows[0]["part_total"], 1)

    def test_iter_rows_tracks_pdf_file_index_across_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "b.pdf").write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", side_effect=[5, 3]):
                rows = list(
                    gpt_ranker.iter_rows(
                        root,
                        input_glob="*.pdf",
                        include_text=False,
                        processing_mode="image",
                        pdf_part_pages=2,
                    )
                )

        self.assertEqual([row["source_pdf_index"] for row in rows], [1, 1, 1, 2, 2])

    def test_calculate_workload_pdf_range_uses_global_row_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "b.pdf").write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", side_effect=[5, 3]):
                stats = gpt_ranker.calculate_workload(
                    root,
                    input_glob="*.pdf",
                    processing_mode="image",
                    pdf_part_pages=2,
                    max_rows=None,
                    completed_filenames=set(),
                    start_row=4,
                    end_row=5,
                    start_pdf=2,
                    end_pdf=2,
                )

        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["workload"], 2)

    def test_load_source_id_filter_parses_mixed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"source_id":"IMAGES/0001/A.pdf"}',
                        "IMAGES/0002/B.pdf",
                        '{"metadata":{"source_id":"IMAGES/0003/C.pdf"}}',
                    ]
                ),
                encoding="utf-8",
            )
            loaded = gpt_ranker.load_source_id_filter(path)

        self.assertEqual(
            loaded,
            {"IMAGES/0001/A.pdf", "IMAGES/0002/B.pdf", "IMAGES/0003/C.pdf"},
        )

    def test_classify_failure_reason_detects_content_filter(self) -> None:
        category = gpt_ranker.classify_failure_reason(
            'HTTP 400 ... code":"data_inspection_failed" ... Input data may contain inappropriate content'
        )
        self.assertEqual(category, "provider_content_filter")

    def test_skip_reason_flags_low_quality_rows(self) -> None:
        quality = gpt_ranker.assess_text_quality("x")
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNotNone(reason)
        self.assertIn("too short", reason)

    def test_skip_reason_flags_ocr_noise_tokens(self) -> None:
        text = "~ © B S)Geeeee ee A go 6 4 ls * . ; errr : id ° N + oO oO oO oO oO oO << = Ww Lu."
        quality = gpt_ranker.assess_text_quality(text)
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNotNone(reason)
        self.assertTrue(
            "too many short/noisy tokens" in reason
            or "average token length too low/noisy OCR" in reason
        )

    def test_skip_reason_allows_normal_text(self) -> None:
        text = (
            "This is a normal paragraph with enough words to pass screening and "
            "contains meaningful language for downstream analysis."
        )
        quality = gpt_ranker.assess_text_quality(text)
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNone(reason)

    def test_build_output_records_marks_skipped_status(self) -> None:
        source_row = {"filename": "sample.txt", "text": ""}
        quality = gpt_ranker.assess_text_quality("")
        result = gpt_ranker.build_skipped_model_result("empty text")
        csv_row, json_record = gpt_ranker.build_output_records(
            idx=3,
            source_row=source_row,
            result=result,
            args=self.skip_args,
            config_metadata={"model": "qwen/qwen3-coder-next"},
            quality=quality,
            processing_status="skipped",
            skip_reason="empty text",
        )

        self.assertEqual(csv_row["processing_status"], "skipped")
        self.assertEqual(csv_row["skip_reason"], "empty text")
        self.assertEqual(csv_row["importance_score"], 0)
        self.assertEqual(json_record["metadata"]["processing_status"], "skipped")
        self.assertEqual(json_record["metadata"]["skip_reason"], "empty text")
        self.assertEqual(csv_row["source_pdf_url"], "")

    def test_build_output_records_includes_api_usage_and_cost_fields(self) -> None:
        source_row = {"filename": "sample.txt", "text": ""}
        result = {
            "headline": "h",
            "importance_score": 9,
            "reason": "r",
            "key_insights": [],
            "tags": [],
            "power_mentions": [],
            "agency_involvement": [],
            "lead_types": [],
            "_request_meta": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
                "model_cost": {
                    "source": "estimated",
                    "total_cost_usd": 0.00042,
                },
            },
        }
        csv_row, json_record = gpt_ranker.build_output_records(
            idx=1,
            source_row=source_row,
            result=result,
            args=self.skip_args,
            config_metadata={"model": "qwen/qwen3-vl-30b"},
            quality={},
            processing_status="processed",
            skip_reason="",
        )
        self.assertEqual(csv_row["api_prompt_tokens"], 10)
        self.assertEqual(csv_row["api_completion_tokens"], 3)
        self.assertEqual(csv_row["api_total_tokens"], 13)
        self.assertEqual(csv_row["api_cost_source"], "estimated")
        self.assertEqual(json_record["api_total_tokens"], 13)
        self.assertAlmostEqual(json_record["api_cost_usd"], 0.00042)

    def test_build_output_records_includes_part_metadata(self) -> None:
        source_row = {
            "filename": "VOL00003/IMAGES/0001/EFTA00000001.pdf",
            "source_id": "VOL00003/IMAGES/0001/EFTA00000001.pdf::part_0002_p00025-00048",
            "document_part": "part_0002_of_0005_p00025-00048",
            "part_index": 2,
            "part_total": 5,
            "part_start_page": 25,
            "part_end_page": 48,
            "document_total_pages": 117,
            "text": "",
        }
        result = {
            "headline": "h",
            "importance_score": 42,
            "reason": "r",
            "key_insights": [],
            "tags": [],
            "power_mentions": [],
            "agency_involvement": [],
            "lead_types": [],
        }
        csv_row, json_record = gpt_ranker.build_output_records(
            idx=10,
            source_row=source_row,
            result=result,
            args=self.skip_args,
            config_metadata={"model": "qwen/qwen3-vl-30b"},
            quality={},
            processing_status="processed",
            skip_reason="",
        )

        self.assertEqual(
            csv_row["source_id"],
            "VOL00003/IMAGES/0001/EFTA00000001.pdf::part_0002_p00025-00048",
        )
        self.assertEqual(csv_row["document_part"], "part_0002_of_0005_p00025-00048")
        self.assertEqual(csv_row["part_index"], 2)
        self.assertEqual(csv_row["part_total"], 5)
        self.assertEqual(json_record["document_total_pages"], 117)
        self.assertEqual(json_record["metadata"]["part_start_page"], 25)

    def test_normalize_power_mentions_collapses_initial_aliases(self) -> None:
        mentions = [
            "Jeffrey Epstein",
            "Jeffrey",
            "J.E.",
            "J",
            "J. Epstein",
            "J. E.",
            "J. E. Epstein",
        ]
        normalized = gpt_ranker.normalize_power_mentions(mentions)
        self.assertEqual(normalized, ["Jeffrey Epstein"])

    def test_filter_power_mentions_removes_agencies_and_keeps_person_names(self) -> None:
        mentions = [
            "Jeffrey Epstein",
            "Prince Edward",
            "Donald Trump",
            "Tony Hawk",
            "FBI",
            "NTOC",
            "New York Field Office",
        ]
        filtered = gpt_ranker.filter_power_mentions(
            mentions,
            tags=["Jeffrey Epstein", "Prince Edward", "Tony Hawk", "FBI"],
            reason="Complainant states Prince Edward and Tony Hawk were present on Epstein's island.",
            key_insights=["FBI memo alleges trafficking tied to Jeffrey Epstein."],
            agency_involvement=["FBI", "NTOC", "New York Field Office"],
            source_support_text="jeffrey epstein prince edward tony hawk island",
        )
        self.assertEqual(
            filtered,
            ["Jeffrey Epstein", "Prince Edward", "Donald Trump", "Tony Hawk"],
        )

    def test_filter_power_mentions_keeps_name_supported_by_tags(self) -> None:
        filtered = gpt_ranker.filter_power_mentions(
            ["Tony Hawk"],
            tags=["Tony Hawk", "human trafficking"],
            reason="",
            key_insights=[],
            agency_involvement=[],
        )
        self.assertEqual(filtered, ["Tony Hawk"])

    def test_filter_power_mentions_uses_source_support_text(self) -> None:
        filtered = gpt_ranker.filter_power_mentions(
            ["Tony Hawk", "FBI"],
            tags=["human trafficking"],
            reason="Complainant says Prince Edward was present.",
            key_insights=[],
            agency_involvement=[],
            source_support_text="when tony hawk got married on the island",
        )
        self.assertEqual(filtered, ["Tony Hawk"])

    def test_filter_power_mentions_keeps_unverified_person_when_no_source_text(self) -> None:
        filtered = gpt_ranker.filter_power_mentions(
            ["Tony Hawk"],
            tags=["human trafficking"],
            reason="Complainant alleges trafficking on the island.",
            key_insights=["Prince Edward is mentioned as present."],
            agency_involvement=[],
            source_support_text="",
        )
        self.assertEqual(filtered, ["Tony Hawk"])

    def test_derive_justice_pdf_url_from_dataset_path(self) -> None:
        filename = "DataSet10/IMAGES/0332/EFTA01970985.txt"
        url = gpt_ranker.derive_justice_pdf_url(filename)
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%2010/EFTA01970985.pdf",
        )

    def test_derive_justice_pdf_url_returns_none_when_unmatched(self) -> None:
        self.assertIsNone(gpt_ranker.derive_justice_pdf_url("notes/no_match.txt"))

    def test_derive_justice_pdf_url_from_volume_path(self) -> None:
        url = gpt_ranker.derive_justice_pdf_url(
            "IMAGES/0001/EFTA00000001.pdf",
            source_path="/tmp/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
        )
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%201/EFTA00000001.pdf",
        )

    def test_derive_justice_pdf_url_from_dataset_tag(self) -> None:
        url = gpt_ranker.derive_justice_pdf_url(
            "IMAGES/0001/EFTA00000001.pdf",
            dataset_tag="standardworks_epstein_files_vol00001",
        )
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%201/EFTA00000001.pdf",
        )

    def test_derive_local_source_url_maps_data_path(self) -> None:
        source_url = gpt_ranker.derive_local_source_url(
            "/tmp/project/source/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
            "IMAGES/0001/EFTA00000001.pdf",
            source_files_base_url=None,
        )
        self.assertEqual(
            source_url,
            "/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
        )

    def test_cli_explicit_default_value_overrides_config(self) -> None:
        original_argv = sys.argv[:]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "ranker_config.toml"
                config_path.write_text("sleep = 0.5\n", encoding="utf-8")
                sys.argv = [
                    "gpt_ranker.py",
                    "--config",
                    str(config_path),
                    "--sleep",
                    "0",
                ]
                args = gpt_ranker.parse_args()
                self.assertEqual(args.sleep, 0.0)
        finally:
            sys.argv = original_argv

    def test_apply_dataset_workspace_defaults_sets_isolated_paths(self) -> None:
        args = argparse.Namespace(
            dataset_workspace_root=Path("data/workspaces"),
            dataset_tag=None,
            input=Path("data/new_data"),
            output=Path("data/epstein_ranked.csv"),
            json_output=Path("data/epstein_ranked.jsonl"),
            checkpoint=Path("data/.epstein_checkpoint"),
            chunk_dir=Path("contrib"),
            chunk_manifest=Path("data/chunks.json"),
            run_metadata_file=None,
            known_json=["old.jsonl"],
        )
        gpt_ranker.apply_dataset_workspace_defaults(args, cli_explicit=set())

        self.assertEqual(args.dataset_tag, "new_data")
        self.assertEqual(
            args.output,
            Path("data/workspaces/new_data/results/epstein_ranked.csv"),
        )
        self.assertEqual(
            args.chunk_manifest,
            Path("data/workspaces/new_data/metadata/chunks.json"),
        )
        self.assertEqual(args.known_json, [])

    def test_apply_dataset_workspace_defaults_keeps_explicit_paths(self) -> None:
        args = argparse.Namespace(
            dataset_workspace_root=Path("data/workspaces"),
            dataset_tag="custom",
            input=Path("data/new_data"),
            output=Path("/tmp/custom.csv"),
            json_output=Path("/tmp/custom.jsonl"),
            checkpoint=Path("/tmp/custom.ckpt"),
            chunk_dir=Path("/tmp/chunks"),
            chunk_manifest=Path("/tmp/chunks.json"),
            run_metadata_file=Path("/tmp/run_meta.json"),
            known_json=["keep.jsonl"],
        )
        gpt_ranker.apply_dataset_workspace_defaults(
            args,
            cli_explicit={
                "output",
                "json_output",
                "checkpoint",
                "chunk_dir",
                "chunk_manifest",
                "run_metadata_file",
                "known_json",
            },
        )

        self.assertEqual(args.dataset_tag, "custom")
        self.assertEqual(args.output, Path("/tmp/custom.csv"))
        self.assertEqual(args.chunk_dir, Path("/tmp/chunks"))
        self.assertEqual(args.known_json, ["keep.jsonl"])

    def test_load_resume_completed_ids_uses_checkpoint_without_output_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / ".checkpoint"
            checkpoint.write_text("IMAGES/0001/EFTA00000001.pdf\n", encoding="utf-8")
            args = argparse.Namespace(
                resume=True,
                checkpoint=checkpoint,
                json_output=root / "results.jsonl",
                known_json=[],
                chunk_size=1000,
                chunk_dir=root / "chunks",
            )
            completed = gpt_ranker.load_resume_completed_ids(args)
        self.assertEqual(completed, {"IMAGES/0001/EFTA00000001.pdf"})

    def test_load_resume_completed_ids_bootstraps_from_chunk_outputs_when_checkpoint_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunk_dir = root / "chunks"
            chunk_dir.mkdir(parents=True)
            chunk_file = chunk_dir / "epstein_ranked_00001_01000.jsonl"
            chunk_file.write_text(
                '{"source_id":"IMAGES/0001/EFTA00000001.pdf","filename":"IMAGES/0001/EFTA00000001.pdf"}\n',
                encoding="utf-8",
            )
            checkpoint = root / ".checkpoint"
            checkpoint.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                resume=True,
                checkpoint=checkpoint,
                json_output=root / "results.jsonl",
                known_json=[],
                chunk_size=1000,
                chunk_dir=chunk_dir,
            )
            completed = gpt_ranker.load_resume_completed_ids(args)
            self.assertEqual(completed, {"IMAGES/0001/EFTA00000001.pdf"})
            self.assertIn(
                "IMAGES/0001/EFTA00000001.pdf",
                checkpoint.read_text(encoding="utf-8"),
            )

    def test_chunk_bounds_for_row_index_applies_chunk_size_and_end_row(self) -> None:
        self.assertEqual(
            gpt_ranker.chunk_bounds_for_row_index(18001, chunk_size=1000),
            (18001, 19000),
        )
        self.assertEqual(
            gpt_ranker.chunk_bounds_for_row_index(18555, chunk_size=1000, end_row=18700),
            (18001, 18700),
        )
        self.assertIsNone(gpt_ranker.chunk_bounds_for_row_index(10, chunk_size=0))

    def test_collect_ready_row_indices_keeps_chunk_order_but_allows_other_chunks(self) -> None:
        emit_order = deque()
        emit_order_by_chunk = {
            (17001, 18000): deque([17001, 17002]),
            (18001, 19000): deque([18001, 18002]),
        }
        pending_results = {
            17002: {"type": "record"},
            18001: {"type": "record"},
            18002: {"type": "record"},
        }

        ready = gpt_ranker.collect_ready_row_indices(
            chunk_mode=True,
            emit_order=emit_order,
            emit_order_by_chunk=emit_order_by_chunk,
            pending_results=pending_results,
        )

        self.assertEqual(ready, [18001, 18002])
        self.assertEqual(list(emit_order_by_chunk[(17001, 18000)]), [17001, 17002])
        self.assertNotIn((18001, 19000), emit_order_by_chunk)

    def test_pdf_page_count_cache_persists_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = root / ".pdf_page_counts.json"
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            cache = gpt_ranker.PdfPageCountCache(cache_path)
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", return_value=12) as mocked_detect:
                first = cache.get_page_count(pdf_path)
                cache.flush()
            self.assertEqual(first, 12)
            self.assertEqual(mocked_detect.call_count, 1)

            cache_reloaded = gpt_ranker.PdfPageCountCache(cache_path)
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", return_value=99) as mocked_detect_reloaded:
                second = cache_reloaded.get_page_count(pdf_path)
            self.assertEqual(second, 12)
            self.assertEqual(mocked_detect_reloaded.call_count, 0)

            time.sleep(0.001)
            pdf_path.write_bytes(b"%PDF-1.4\nupdated")
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", return_value=13) as mocked_detect_changed:
                third = cache_reloaded.get_page_count(pdf_path)
            self.assertEqual(third, 13)
            self.assertEqual(mocked_detect_changed.call_count, 1)

    def test_write_run_metadata_writes_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_file = Path(tmpdir) / "meta" / "run.json"
            args = argparse.Namespace(
                run_metadata_file=run_file,
                dataset_tag="standardworks_epstein_files",
                dataset_source_label="StandardWorks",
                dataset_source_url="https://standardworks.ai/epstein-files",
                input=Path("data/new_data"),
                input_glob="*.txt",
                output=Path("out.csv"),
                json_output=Path("out.jsonl"),
                checkpoint=Path("ckpt"),
                chunk_size=1000,
                chunk_dir=Path("chunks"),
                chunk_manifest=Path("chunks.json"),
            )
            gpt_ranker.write_run_metadata(
                args=args,
                prompt_source="prompts/default_system_prompt.txt",
                config_metadata={"model": "qwen/qwen3-coder-next"},
                workload_stats={"total": 10, "already_done": 0, "workload": 10},
                total_dataset_rows=10,
                dataset_profile={"profile_id": "standardworks_epstein_files"},
            )

            payload = run_file.read_text(encoding="utf-8")
            self.assertIn("standardworks_epstein_files", payload)
            self.assertIn("dataset_profile", payload)

    def test_build_request_targets_adds_localhost_fallback(self) -> None:
        targets = gpt_ranker.build_request_targets(
            "http://localhost:5002/v1",
            "auto",
        )
        self.assertIn(("openai", "http://localhost:5002/v1"), targets)
        self.assertIn(("chat", "http://localhost:5002/v1"), targets)
        self.assertIn(("chat", "http://localhost:5555/api/v1"), targets)

    def test_build_request_targets_normalizes_full_chat_route(self) -> None:
        targets = gpt_ranker.build_request_targets(
            "https://example.com/v1/chat/completions",
            "openai",
        )
        self.assertEqual(targets, [("openai", "https://example.com/v1")])

    def test_call_model_chat_mode_parses_output(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", return_value=response_payload) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertTrue(mocked_post.call_args.kwargs["url"].endswith("/chat"))

    def test_call_model_openai_extracts_usage_and_provider_cost(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"headline":"h","importance_score":1,"reason":"r",'
                            '"key_insights":[],"tags":[],"power_mentions":[],'
                            '"agency_involvement":[],"lead_types":[]}'
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 200},
            },
            "cost": 0.00123,
        }
        with mock.patch.object(gpt_ranker, "post_request", return_value=response_payload):
            result = gpt_ranker.call_model(
                endpoint="https://openrouter.ai/api/v1",
                api_format="openai",
                model="qwen/qwen3-vl-30b-a3b-instruct",
                filename="DataSet3/EFTA00004066.pdf",
                text="text",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=120,
                system_prompt="Return JSON",
                api_key="test-key",
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=256,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )

        meta = result["_request_meta"]
        self.assertEqual(meta["usage"]["prompt_tokens"], 1200)
        self.assertEqual(meta["usage"]["completion_tokens"], 300)
        self.assertEqual(meta["usage"]["total_tokens"], 1500)
        self.assertEqual(meta["usage"]["cache_read_tokens"], 200)
        self.assertAlmostEqual(meta["provider_reported_cost_usd"], 0.00123)

    def test_call_model_openai_accepts_array_content_blocks(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    '{"headline":"h","importance_score":1,"reason":"r",'
                                    '"key_insights":[],"tags":[],"power_mentions":[],'
                                    '"agency_involvement":[],"lead_types":[]}'
                                ),
                            }
                        ]
                    }
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", return_value=response_payload):
            result = gpt_ranker.call_model(
                endpoint="https://openrouter.ai/api/v1",
                api_format="openai",
                model="qwen/qwen3-vl-30b-a3b-instruct",
                filename="DataSet3/EFTA00004066.pdf",
                text="text",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=120,
                system_prompt="Return JSON",
                api_key="test-key",
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=256,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")

    def test_call_model_openai_omits_metadata_for_non_openrouter_endpoints(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"headline":"h","importance_score":1,"reason":"r",'
                            '"key_insights":[],"tags":[],"power_mentions":[],'
                            '"agency_involvement":[],"lead_types":[]}'
                        )
                    }
                }
            ]
        }

        captured_payload = {}

        def side_effect(*, url, payload, api_key, extra_headers, timeout):
            captured_payload.update(payload)
            return response_payload

        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effect):
            result = gpt_ranker.call_model(
                endpoint="https://api.example.com/v1",
                api_format="openai",
                model="qwen/qwen3-vl-30b",
                filename="DataSet3/EFTA00004066.pdf",
                text="text",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=120,
                system_prompt="Return JSON",
                api_key="test-key",
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=256,
                reasoning_effort=None,
                image_detail="low",
                config_metadata={"source": "test"},
            )

        self.assertEqual(result["headline"], "h")
        self.assertNotIn("metadata", captured_payload)

    def test_call_model_openai_includes_metadata_for_openrouter(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"headline":"h","importance_score":1,"reason":"r",'
                            '"key_insights":[],"tags":[],"power_mentions":[],'
                            '"agency_involvement":[],"lead_types":[]}'
                        )
                    }
                }
            ]
        }

        captured_payload = {}

        def side_effect(*, url, payload, api_key, extra_headers, timeout):
            captured_payload.update(payload)
            return response_payload

        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effect):
            result = gpt_ranker.call_model(
                endpoint="https://openrouter.ai/api/v1",
                api_format="openai",
                model="qwen/qwen3-vl-30b-a3b-instruct",
                filename="DataSet3/EFTA00004066.pdf",
                text="text",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=120,
                system_prompt="Return JSON",
                api_key="test-key",
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=256,
                reasoning_effort=None,
                image_detail="low",
                config_metadata={"source": "test"},
            )

        self.assertEqual(result["headline"], "h")
        self.assertEqual(captured_payload.get("metadata"), {"source": "test"})

    def test_attach_request_usage_and_cost_estimates_from_prices(self) -> None:
        result = {
            "_request_meta": {
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                }
            }
        }
        args = argparse.Namespace(
            input_price_per_1m=0.13,
            output_price_per_1m=0.52,
            cache_read_price_per_1m=None,
            cache_write_price_per_1m=None,
        )
        gpt_ranker.attach_request_usage_and_cost(result, args)
        model_cost = result["_request_meta"]["model_cost"]
        self.assertEqual(model_cost["source"], "estimated")
        self.assertAlmostEqual(model_cost["input_cost_usd"], 0.13)
        self.assertAlmostEqual(model_cost["output_cost_usd"], 0.52)
        self.assertAlmostEqual(model_cost["total_cost_usd"], 0.65)

    def test_call_model_auto_falls_back_to_chat_endpoint(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }

        def side_effect(*, url, payload, api_key, extra_headers, timeout):
            if url.endswith("/chat/completions"):
                raise gpt_ranker.UnsupportedEndpointError("no completions route")
            return response_payload

        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effect) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:1234/v1",
                api_format="auto",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["importance_score"], 1)
        called_urls = [call.kwargs["url"] for call in mocked_post.call_args_list]
        self.assertIn("http://localhost:1234/v1/chat/completions", called_urls)
        self.assertIn("http://localhost:1234/v1/chat", called_urls)

    def test_call_model_retries_transient_errors(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        side_effects = [
            gpt_ranker.ModelRequestError("temporary outage", retriable=True),
            response_payload,
        ]
        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effects) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=2,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 2)

    def test_call_model_uses_rate_limit_backoff_hint(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        rate_limit_error = gpt_ranker.ModelRequestError(
            "rate limited",
            retriable=True,
            status_code=429,
            retry_after_seconds=7.0,
        )
        with (
            mock.patch.object(gpt_ranker, "post_request", side_effect=[rate_limit_error, response_payload]) as mocked_post,
            mock.patch.object(gpt_ranker._model_client.time, "sleep") as mocked_sleep,
            mock.patch.object(gpt_ranker._model_client.random, "uniform", return_value=0.0),
        ):
            result = gpt_ranker.call_model(
                endpoint="https://example.com/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=2,
                retry_backoff=0.1,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 2)
        self.assertGreaterEqual(mocked_sleep.call_args.args[0], 7.0)

    def test_call_model_retries_malformed_json_output(self) -> None:
        malformed = {
            "output": [
                {
                    "type": "message",
                    "content": '{"headline":"h","importance_score":1,"reason":"r"',
                }
            ]
        }
        valid = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", side_effect=[malformed, valid]) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=2,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 2)

    def test_call_model_retries_empty_messages_within_attempt(self) -> None:
        empty = {
            "output": [
                {
                    "type": "message",
                    "content": "   ",
                }
            ]
        }
        valid = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", side_effect=[empty, empty, valid]) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 3)
        self.assertEqual(result["_request_meta"]["request_attempt"], 3)

    def test_call_model_image_mode_rejects_chat_api(self) -> None:
        with self.assertRaises(RuntimeError):
            gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-vl-30b",
                filename="IMAGES/0001/EFTA00000001.pdf",
                text="",
                input_kind="image",
                image_path=Path("/tmp/nope.pdf"),
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )


if __name__ == "__main__":
    unittest.main()
