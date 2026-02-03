"""
Tests for ECFP4 and Atom Pair fingerprint generation and data integrity.

These tests verify that:
1. Fingerprints are reproducible (re-computing yields same results)
2. Fingerprint dimensions are correct (2048 bits)
3. Fingerprint values are binary (0 or 1)
4. All SMILES are valid and parseable by RDKit
5. Fingerprint files maintain consistency with source files
6. Identical SMILES produce identical fingerprints (determinism)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Configuration
# =============================================================================

N_BITS = 2048
N_RANDOM_SAMPLES = 50  # Number of random molecules to verify per split

# ECFP4 parameters
MORGAN_RADIUS = 2

# Atom Pair parameters
AP_MIN_LENGTH = 1
AP_MAX_LENGTH = 30


# =============================================================================
# Helper Functions
# =============================================================================


def compute_ecfp4(smiles: str) -> Optional[np.ndarray]:
    """Compute ECFP4 fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_atompair(smiles: str) -> Optional[np.ndarray]:
    """Compute Atom Pair fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
        mol, nBits=N_BITS, minLength=AP_MIN_LENGTH, maxLength=AP_MAX_LENGTH
    )
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _find_fp_file(base: Path, split: str, fp_type: str) -> Optional[Path]:
    """Find fingerprint file for a given split and fingerprint type."""
    candidates = [
        base / split / f"supcon_{split}_{fp_type}.parquet",
        base / split / f"supcon_{split}_V1_{fp_type}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _get_fp_columns(df: pd.DataFrame) -> List[str]:
    """Get fingerprint column names from dataframe."""
    return [c for c in df.columns if c.startswith("fp_")]


def _extract_fingerprint(row: pd.Series, fp_cols: List[str]) -> np.ndarray:
    """Extract fingerprint array from a dataframe row."""
    return row[fp_cols].values.astype(np.uint8)


def _find_source_file(base: Path, split: str) -> Optional[Path]:
    """Find source SupCon split file."""
    candidates = [
        base / split / f"supcon_{split}.parquet",
        base / split / f"supcon_{split}_V1.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# =============================================================================
# ECFP4 Tests
# =============================================================================


class TestECFP4Reproducibility:
    """Test that ECFP4 fingerprints are reproducible when re-computed."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprints_match_recomputed(self, data_dir: Path):
        """
        Randomly sample molecules from each split and verify that
        re-computing fingerprints yields the same result.
        """
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "ecfp4")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                errors.append(f"{split}: No fingerprint columns found")
                continue

            # Random sample
            n_samples = min(N_RANDOM_SAMPLES, len(df))
            sample_df = df.sample(n=n_samples, random_state=42)

            mismatches = []
            for idx, row in sample_df.iterrows():
                smiles = row["smiles"]
                stored_fp = _extract_fingerprint(row, fp_cols)
                recomputed_fp = compute_ecfp4(smiles)

                if recomputed_fp is None:
                    mismatches.append((idx, smiles, "Failed to parse SMILES"))
                elif not np.array_equal(stored_fp, recomputed_fp):
                    diff_bits = np.sum(stored_fp != recomputed_fp)
                    mismatches.append((idx, smiles, f"{diff_bits} bits differ"))

            if mismatches:
                examples = mismatches[:5]
                errors.append(
                    f"{split}: {len(mismatches)}/{n_samples} fingerprints don't match. "
                    f"Examples: {examples}"
                )

        if not any(_find_fp_file(data_dir, s, "ecfp4") for s in splits):
            pytest.skip("No ECFP4 fingerprint files found")

        assert not errors, "\n".join(errors)


class TestECFP4Dimensions:
    """Test ECFP4 fingerprint array dimensions."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprint_has_correct_bit_count(self, data_dir: Path):
        """Verify fingerprint has exactly 2048 bits."""
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "ecfp4")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) != N_BITS:
                errors.append(
                    f"{split}: Expected {N_BITS} fingerprint columns, "
                    f"found {len(fp_cols)}"
                )

        if not any(_find_fp_file(data_dir, s, "ecfp4") for s in splits):
            pytest.skip("No ECFP4 fingerprint files found")

        assert not errors, "\n".join(errors)


class TestECFP4BinaryValues:
    """Test that ECFP4 fingerprint values are binary (0 or 1)."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprint_values_are_binary(self, data_dir: Path):
        """Verify all fingerprint values are 0 or 1."""
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "ecfp4")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                continue

            fp_data = df[fp_cols].values
            unique_values = np.unique(fp_data)

            invalid_values = [v for v in unique_values if v not in (0, 1)]
            if invalid_values:
                errors.append(
                    f"{split}: Found non-binary values in fingerprints: {invalid_values}"
                )

        if not any(_find_fp_file(data_dir, s, "ecfp4") for s in splits):
            pytest.skip("No ECFP4 fingerprint files found")

        assert not errors, "\n".join(errors)


# =============================================================================
# Atom Pair Tests
# =============================================================================


class TestAtomPairReproducibility:
    """Test that Atom Pair fingerprints are reproducible when re-computed."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprints_match_recomputed(self, data_dir: Path):
        """
        Randomly sample molecules from each split and verify that
        re-computing fingerprints yields the same result.
        """
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "atompair")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                errors.append(f"{split}: No fingerprint columns found")
                continue

            # Random sample
            n_samples = min(N_RANDOM_SAMPLES, len(df))
            sample_df = df.sample(n=n_samples, random_state=42)

            mismatches = []
            for idx, row in sample_df.iterrows():
                smiles = row["smiles"]
                stored_fp = _extract_fingerprint(row, fp_cols)
                recomputed_fp = compute_atompair(smiles)

                if recomputed_fp is None:
                    mismatches.append((idx, smiles, "Failed to parse SMILES"))
                elif not np.array_equal(stored_fp, recomputed_fp):
                    diff_bits = np.sum(stored_fp != recomputed_fp)
                    mismatches.append((idx, smiles, f"{diff_bits} bits differ"))

            if mismatches:
                examples = mismatches[:5]
                errors.append(
                    f"{split}: {len(mismatches)}/{n_samples} fingerprints don't match. "
                    f"Examples: {examples}"
                )

        if not any(_find_fp_file(data_dir, s, "atompair") for s in splits):
            pytest.skip("No Atom Pair fingerprint files found")

        assert not errors, "\n".join(errors)


class TestAtomPairDimensions:
    """Test Atom Pair fingerprint array dimensions."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprint_has_correct_bit_count(self, data_dir: Path):
        """Verify fingerprint has exactly 2048 bits."""
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "atompair")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) != N_BITS:
                errors.append(
                    f"{split}: Expected {N_BITS} fingerprint columns, "
                    f"found {len(fp_cols)}"
                )

        if not any(_find_fp_file(data_dir, s, "atompair") for s in splits):
            pytest.skip("No Atom Pair fingerprint files found")

        assert not errors, "\n".join(errors)


class TestAtomPairBinaryValues:
    """Test that Atom Pair fingerprint values are binary (0 or 1)."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_fingerprint_values_are_binary(self, data_dir: Path):
        """Verify all fingerprint values are 0 or 1."""
        splits = ["train", "val", "test"]
        errors = []

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, "atompair")
            if fp_path is None:
                continue

            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                continue

            fp_data = df[fp_cols].values
            unique_values = np.unique(fp_data)

            invalid_values = [v for v in unique_values if v not in (0, 1)]
            if invalid_values:
                errors.append(
                    f"{split}: Found non-binary values in fingerprints: {invalid_values}"
                )

        if not any(_find_fp_file(data_dir, s, "atompair") for s in splits):
            pytest.skip("No Atom Pair fingerprint files found")

        assert not errors, "\n".join(errors)


# =============================================================================
# Shared Tests (apply to both fingerprint types)
# =============================================================================


class TestSMILESValidity:
    """Test that all SMILES in fingerprint files are valid."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    @pytest.mark.parametrize("fp_type", ["ecfp4", "atompair"])
    def test_all_smiles_are_parseable(self, data_dir: Path, fp_type: str):
        """Verify all SMILES can be parsed by RDKit."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, fp_type)
            if fp_path is None:
                continue

            checked_any = True
            df = pd.read_parquet(fp_path)
            invalid_smiles = []

            for idx, smiles in enumerate(df["smiles"]):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    invalid_smiles.append((idx, smiles))

            if invalid_smiles:
                examples = invalid_smiles[:5]
                errors.append(
                    f"{split}: {len(invalid_smiles)} invalid SMILES. "
                    f"Examples: {examples}"
                )

        if not checked_any:
            pytest.skip(f"No {fp_type} fingerprint files found")

        assert not errors, "\n".join(errors)


class TestFingerprintFileConsistency:
    """Test consistency between fingerprint files and source files."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    @pytest.mark.parametrize("fp_type", ["ecfp4", "atompair"])
    def test_row_count_matches_source(self, data_dir: Path, fp_type: str):
        """Verify fingerprint file has same row count as source."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, fp_type)
            source_path = _find_source_file(data_dir, split)

            if fp_path is None or source_path is None:
                continue

            checked_any = True
            fp_df = pd.read_parquet(fp_path)
            source_df = pd.read_parquet(source_path)

            if len(fp_df) != len(source_df):
                errors.append(
                    f"{split}: Fingerprint file has {len(fp_df)} rows, "
                    f"source has {len(source_df)} rows"
                )

        if not checked_any:
            pytest.skip(f"No matching {fp_type} fingerprint and source files found")

        assert not errors, "\n".join(errors)

    @pytest.mark.parametrize("fp_type", ["ecfp4", "atompair"])
    def test_smiles_match_source(self, data_dir: Path, fp_type: str):
        """Verify SMILES in fingerprint file match source file."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, fp_type)
            source_path = _find_source_file(data_dir, split)

            if fp_path is None or source_path is None:
                continue

            checked_any = True
            fp_df = pd.read_parquet(fp_path)
            source_df = pd.read_parquet(source_path)

            fp_smiles = set(fp_df["smiles"])
            source_smiles = set(source_df["smiles"])

            missing_in_fp = source_smiles - fp_smiles
            extra_in_fp = fp_smiles - source_smiles

            if missing_in_fp:
                errors.append(
                    f"{split}: {len(missing_in_fp)} SMILES in source but not in fingerprint file"
                )
            if extra_in_fp:
                errors.append(
                    f"{split}: {len(extra_in_fp)} SMILES in fingerprint file but not in source"
                )

        if not checked_any:
            pytest.skip(f"No matching {fp_type} fingerprint and source files found")

        assert not errors, "\n".join(errors)


class TestFingerprintDeterminism:
    """Test that fingerprint generation is deterministic."""

    def test_ecfp4_identical_smiles_produce_identical_fingerprints(self):
        """Verify same SMILES always produces same ECFP4 fingerprint."""
        test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC(C(=O)O)C(=O)CC",  # Complex PKS-like
            "C=CCC(C(=O)O)C(=O)CCC(=O)C(CC=C)C(=O)CC",  # Larger molecule
        ]

        for smiles in test_smiles:
            fp1 = compute_ecfp4(smiles)
            fp2 = compute_ecfp4(smiles)

            assert fp1 is not None, f"Failed to compute ECFP4 fingerprint for {smiles}"
            assert np.array_equal(fp1, fp2), (
                f"ECFP4 fingerprints differ for same SMILES: {smiles}"
            )

    def test_atompair_identical_smiles_produce_identical_fingerprints(self):
        """Verify same SMILES always produces same Atom Pair fingerprint."""
        test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC(C(=O)O)C(=O)CC",  # Complex PKS-like
            "C=CCC(C(=O)O)C(=O)CCC(=O)C(CC=C)C(=O)CC",  # Larger molecule
        ]

        for smiles in test_smiles:
            fp1 = compute_atompair(smiles)
            fp2 = compute_atompair(smiles)

            assert fp1 is not None, f"Failed to compute Atom Pair fingerprint for {smiles}"
            assert np.array_equal(fp1, fp2), (
                f"Atom Pair fingerprints differ for same SMILES: {smiles}"
            )

    def test_ecfp4_different_smiles_produce_different_fingerprints(self):
        """Verify structurally different molecules have different ECFP4 fingerprints."""
        pairs = [
            ("CCO", "CCCO"),  # Ethanol vs Propanol
            ("c1ccccc1", "c1ccc(O)cc1"),  # Benzene vs Phenol
            ("CC(=O)O", "CCC(=O)O"),  # Acetic vs Propionic acid
        ]

        for smi1, smi2 in pairs:
            fp1 = compute_ecfp4(smi1)
            fp2 = compute_ecfp4(smi2)

            assert fp1 is not None and fp2 is not None
            assert not np.array_equal(fp1, fp2), (
                f"Different molecules have identical ECFP4 fingerprints: {smi1} vs {smi2}"
            )

    def test_atompair_different_smiles_produce_different_fingerprints(self):
        """Verify structurally different molecules have different Atom Pair fingerprints."""
        pairs = [
            ("CCO", "CCCO"),  # Ethanol vs Propanol
            ("c1ccccc1", "c1ccc(O)cc1"),  # Benzene vs Phenol
            ("CC(=O)O", "CCC(=O)O"),  # Acetic vs Propionic acid
        ]

        for smi1, smi2 in pairs:
            fp1 = compute_atompair(smi1)
            fp2 = compute_atompair(smi2)

            assert fp1 is not None and fp2 is not None
            assert not np.array_equal(fp1, fp2), (
                f"Different molecules have identical Atom Pair fingerprints: {smi1} vs {smi2}"
            )


class TestFingerprintNonZero:
    """Test that fingerprints are not all zeros."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    @pytest.mark.parametrize("fp_type", ["ecfp4", "atompair"])
    def test_fingerprints_are_not_all_zeros(self, data_dir: Path, fp_type: str):
        """Verify no fingerprint is entirely zeros (would indicate error)."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, fp_type)
            if fp_path is None:
                continue

            checked_any = True
            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                continue

            fp_data = df[fp_cols].values
            row_sums = fp_data.sum(axis=1)
            zero_rows = np.sum(row_sums == 0)

            if zero_rows > 0:
                zero_indices = np.where(row_sums == 0)[0][:5]
                zero_smiles = df.iloc[zero_indices]["smiles"].tolist()
                errors.append(
                    f"{split}: {zero_rows} molecules have all-zero fingerprints. "
                    f"Examples: {zero_smiles}"
                )

        if not checked_any:
            pytest.skip(f"No {fp_type} fingerprint files found")

        assert not errors, "\n".join(errors)

    @pytest.mark.parametrize("fp_type", ["ecfp4", "atompair"])
    def test_fingerprints_have_reasonable_density(self, data_dir: Path, fp_type: str):
        """
        Verify fingerprints have reasonable bit density.
        Fingerprints typically have 1-20% of bits set.
        """
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            fp_path = _find_fp_file(data_dir, split, fp_type)
            if fp_path is None:
                continue

            checked_any = True
            df = pd.read_parquet(fp_path)
            fp_cols = _get_fp_columns(df)

            if len(fp_cols) == 0:
                continue

            fp_data = df[fp_cols].values
            mean_density = fp_data.mean()

            # Fingerprints typically have 0.5-20% bits set
            if mean_density < 0.005 or mean_density > 0.20:
                errors.append(
                    f"{split} ({fp_type}): Unusual fingerprint density {mean_density:.4f}. "
                    f"Expected 0.5-20% of bits set."
                )
            else:
                print(f"{split} ({fp_type}): Mean fingerprint density = {mean_density:.4f} ({mean_density*100:.2f}% bits set)")

        if not checked_any:
            pytest.skip(f"No {fp_type} fingerprint files found")

        assert not errors, "\n".join(errors)


class TestECFP4vsAtomPairDifference:
    """Test that ECFP4 and Atom Pair produce different fingerprints."""

    def test_different_fingerprint_types_produce_different_results(self):
        """Verify ECFP4 and Atom Pair fingerprints are different for same molecule."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)CC(C(=O)O)C(=O)CC",
        ]

        for smiles in test_smiles:
            ecfp4_fp = compute_ecfp4(smiles)
            atompair_fp = compute_atompair(smiles)

            assert ecfp4_fp is not None and atompair_fp is not None
            assert not np.array_equal(ecfp4_fp, atompair_fp), (
                f"ECFP4 and Atom Pair fingerprints should differ for: {smiles}"
            )


class TestFingerprintConsistencyAcrossTypes:
    """Test consistency between ECFP4 and Atom Pair fingerprint files."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_same_smiles_in_both_fingerprint_types(self, data_dir: Path):
        """Verify both fingerprint types have the same SMILES."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            ecfp4_path = _find_fp_file(data_dir, split, "ecfp4")
            atompair_path = _find_fp_file(data_dir, split, "atompair")

            if ecfp4_path is None or atompair_path is None:
                continue

            checked_any = True
            ecfp4_df = pd.read_parquet(ecfp4_path)
            atompair_df = pd.read_parquet(atompair_path)

            ecfp4_smiles = set(ecfp4_df["smiles"])
            atompair_smiles = set(atompair_df["smiles"])

            only_in_ecfp4 = ecfp4_smiles - atompair_smiles
            only_in_atompair = atompair_smiles - ecfp4_smiles

            if only_in_ecfp4:
                errors.append(
                    f"{split}: {len(only_in_ecfp4)} SMILES only in ECFP4 file"
                )
            if only_in_atompair:
                errors.append(
                    f"{split}: {len(only_in_atompair)} SMILES only in Atom Pair file"
                )

        if not checked_any:
            pytest.skip("No splits have both ECFP4 and Atom Pair fingerprint files")

        assert not errors, "\n".join(errors)

    def test_same_row_count_in_both_fingerprint_types(self, data_dir: Path):
        """Verify both fingerprint types have the same number of rows."""
        splits = ["train", "val", "test"]
        errors = []
        checked_any = False

        for split in splits:
            ecfp4_path = _find_fp_file(data_dir, split, "ecfp4")
            atompair_path = _find_fp_file(data_dir, split, "atompair")

            if ecfp4_path is None or atompair_path is None:
                continue

            checked_any = True
            ecfp4_df = pd.read_parquet(ecfp4_path)
            atompair_df = pd.read_parquet(atompair_path)

            if len(ecfp4_df) != len(atompair_df):
                errors.append(
                    f"{split}: ECFP4 has {len(ecfp4_df)} rows, "
                    f"Atom Pair has {len(atompair_df)} rows"
                )

        if not checked_any:
            pytest.skip("No splits have both ECFP4 and Atom Pair fingerprint files")

        assert not errors, "\n".join(errors)
