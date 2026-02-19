import os
import re
import ast
import glob
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


class LoadMultiConer:
    # Tag vocab according to MultiCoNER
    NER_TAGS: List[str] = [
        "O",  # 0
        "B-Facility","I-Facility",                       # 1-2
        "B-OtherLOC","I-OtherLOC",                       # 3-4
        "B-HumanSettlement","I-HumanSettlement",         # 5-6
        "B-Station","I-Station",                         # 7-8
        "B-VisualWork","I-VisualWork",                   # 9-10
        "B-MusicalWork","I-MusicalWork",                 # 11-12
        "B-WrittenWork","I-WrittenWork",                 # 13-14
        "B-ArtWork","I-ArtWork",                         # 15-16
        "B-Software","I-Software",                       # 17-18
        "B-OtherCW","I-OtherCW",                         # 19-20
        "B-MusicalGRP","I-MusicalGRP",                   # 21-22
        "B-PublicCorp","I-PublicCorp",                   # 23-24
        "B-PrivateCorp","I-PrivateCorp",                 # 25-26
        "B-OtherCorp","I-OtherCorp",                     # 27-28
        "B-AerospaceManufacturer","I-AerospaceManufacturer", # 29-30
        "B-SportsGRP","I-SportsGRP",                     # 31-32
        "B-CarManufacturer","I-CarManufacturer",          # 33-34
        "B-TechCORP","I-TechCORP",                       # 35-36
        "B-ORG","I-ORG",                                 # 37-38
        "B-Scientist","I-Scientist",                     # 39-40
        "B-Artist","I-Artist",                           # 41-42
        "B-Athlete","I-Athlete",                         # 43-44
        "B-Politician","I-Politician",                   # 45-46
        "B-Cleric","I-Cleric",                           # 47-48
        "B-SportsManager","I-SportsManager",             # 49-50
        "B-OtherPER","I-OtherPER",                       # 51-52
        "B-Clothing","I-Clothing",                       # 53-54
        "B-Vehicle","I-Vehicle",                         # 55-56
        "B-Food","I-Food",                               # 57-58
        "B-Drink","I-Drink",                             # 59-60
        "B-OtherPROD","I-OtherPROD",                     # 61-62
        "B-Medication/Vaccine","I-Medication/Vaccine",   # 63-64
        "B-MedicalProcedure","I-MedicalProcedure",       # 65-66
        "B-AnatomicalStructure","I-AnatomicalStructure", # 67-68
        "B-Symptom","I-Symptom",                         # 69-70
        "B-Disease","I-Disease"                          # 71-72
    ]
    NER_TAG2ID: Dict[str, int] = {t: i for i, t in enumerate(NER_TAGS)}

    # Metadata parsing for noisy test subset
    _meta_id_re      = re.compile(r"\bid\s+([^\s]+)")
    _meta_domain_re  = re.compile(r"\bdomain=([^\s]+)")
    _meta_corrupt_re = re.compile(r"\bcorrupt=([^\s]+)")
    _meta_ctoks_re   = re.compile(r"\bchanged_tokens=\[([^\]]*)\]")
    _meta_cidx_re    = re.compile(r"\bchanged_indexes=(\[[^\]]*\])")

    @staticmethod
    def _resolve_dest_dir(dest_dir: str) -> str:
        if os.path.isabs(dest_dir):
            return dest_dir

        cwd_candidate = os.path.abspath(dest_dir)
        if os.path.isdir(cwd_candidate):
            return cwd_candidate

        module_candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", dest_dir)
        )
        if os.path.isdir(module_candidate):
            return module_candidate

        return cwd_candidate

    def __init__(
        self,
        dest_dir: str = "multiconer2023",
        lang_subdir: str = "EN-English",
        train_filename: str = "en_train.conll",
        dev_filename: str = "en_dev.conll",
        test_filename: str = "en_test.conll",
        col_order: Optional[List[str]] = None,
    ):
        self.dest_dir = self._resolve_dest_dir(dest_dir)
        self.lang_subdir = lang_subdir

        self.en_dir = os.path.join(self.dest_dir, self.lang_subdir)
        self.train_path = self._resolve_split_path(train_filename, split="train")
        self.dev_path = self._resolve_split_path(dev_filename, split="dev")
        self.test_path = self._resolve_split_path(test_filename, split="test")

        self.col_order = col_order or [
            "id", "sample_id", "tokens", "ner_tags", "ner_tags_index",
            "domain", "corrupt", "changed_tokens", "changed_indexes", "raw_meta"
        ]

    def _resolve_split_path(self, filename: str, split: str) -> str:
        direct_path = os.path.join(self.en_dir, filename)
        if os.path.isfile(direct_path):
            return direct_path

        candidates = sorted(glob.glob(os.path.join(self.en_dir, f"*_{split}.conll")))
        if len(candidates) == 1:
            return candidates[0]

        if split in {"dev", "test"} and "_train.conll" in filename:
            inferred = filename.replace("_train.conll", f"_{split}.conll")
            inferred_path = os.path.join(self.en_dir, inferred)
            if os.path.isfile(inferred_path):
                return inferred_path

        return direct_path

    @staticmethod
    def _parse_changed_tokens(inner: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for m in re.finditer(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", inner):
            a = m.group(1).strip().strip('"').strip("'")
            b = m.group(2).strip().strip('"').strip("'")
            pairs.append((a, b))
        return pairs

    def parse_metadata(self, line: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "sample_id": None,
            "domain": None,
            "corrupt": False,
            "changed_tokens": [],
            "changed_indexes": [],
            "raw_meta": line.strip(),
        }

        m = self._meta_id_re.search(line)
        if m:
            meta["sample_id"] = m.group(1)

        m = self._meta_domain_re.search(line)
        if m:
            meta["domain"] = m.group(1)

        m = self._meta_corrupt_re.search(line)
        if m:
            meta["corrupt"] = (m.group(1).strip().lower() == "true")

        m = self._meta_ctoks_re.search(line)
        if m:
            meta["changed_tokens"] = self._parse_changed_tokens(m.group(1))

        m = self._meta_cidx_re.search(line)
        if m:
            try:
                meta["changed_indexes"] = ast.literal_eval(m.group(1))
            except Exception:
                meta["changed_indexes"] = []

        return meta

    def load_conll_hf_like(self, path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CoNLL file not found: {path}. "
                f"Resolved dest_dir='{self.dest_dir}', lang_subdir='{self.lang_subdir}'."
            )

        rows: List[Dict[str, Any]] = []

        cur_meta = {
            "sample_id": None,
            "domain": None,
            "corrupt": False,
            "changed_tokens": [],
            "changed_indexes": [],
            "raw_meta": None,
        }
        tokens: List[str] = []
        tags: List[str] = []

        def flush_sentence():
            nonlocal tokens, tags, cur_meta
            if not tokens:
                return

            try:
                tag_ids = [self.NER_TAG2ID[t] for t in tags]
            except KeyError as e:
                raise ValueError(f"Unknown NER tag {e} in file {path}") from None

            rows.append({
                "tokens": tokens,
                "ner_tags": tags,
                "ner_tags_index": tag_ids,
                **cur_meta,
            })
            tokens, tags = [], []

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")

                if not line.strip():
                    flush_sentence()
                    continue

                if line.startswith("#"):
                    cur_meta = self.parse_metadata(line)
                    continue

                cols = line.split()
                if len(cols) < 2:
                    continue

                tokens.append(cols[0])
                tags.append(cols[-1])

        flush_sentence()

        df = pd.DataFrame(rows).reset_index(drop=True)
        df.insert(0, "id", df.index)

        # enforce column order
        df = df[[c for c in self.col_order if c in df.columns]]
        return df

    def load_splits(self) -> Dict[str, pd.DataFrame]:
        ds_train = self.load_conll_hf_like(self.train_path)
        ds_val   = self.load_conll_hf_like(self.dev_path)
        ds_test  = self.load_conll_hf_like(self.test_path)

        ds_test_noisy = ds_test[ds_test["corrupt"]].reset_index(drop=True)

        # align noisy to same column order
        ds_test_noisy = ds_test_noisy[[c for c in self.col_order if c in ds_test_noisy.columns]]

        return {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
            "test_noisy": ds_test_noisy,
        }
