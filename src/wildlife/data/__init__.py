"""Data collection, license verification, cleaning, manifest, and split modules."""

from wildlife.data.licenses import (
    ALLOWED_PHOTO_LICENSES,
    LicenseViolationError,
    audit_manifest_licenses,
    normalize_license_code,
    validate_photo_license,
    write_attribution_and_audit,
)
from wildlife.data.clean import (
    CorruptImageError,
    DecompressionBombError,
    ImageValidationError,
    UnsupportedImageFormatError,
    compute_phash,
    decode_image_bytes,
    encode_rgb_to_png_bytes,
    phash_hamming_distance,
    strip_exif_and_validate,
)
from wildlife.data.manifest import (
    build_manifest_rows,
    read_manifest_parquet,
    validate_manifest_schema,
    write_manifest_parquet,
)
from wildlife.data.split import (
    SplitLeakageError,
    create_observation_grouped_splits,
    export_split_csv,
    load_split_csv,
    verify_no_split_leakage,
)

__all__ = [
    "ALLOWED_PHOTO_LICENSES",
    "LicenseViolationError",
    "audit_manifest_licenses",
    "normalize_license_code",
    "validate_photo_license",
    "write_attribution_and_audit",
    "CorruptImageError",
    "DecompressionBombError",
    "ImageValidationError",
    "UnsupportedImageFormatError",
    "compute_phash",
    "decode_image_bytes",
    "encode_rgb_to_png_bytes",
    "phash_hamming_distance",
    "strip_exif_and_validate",
    "build_manifest_rows",
    "read_manifest_parquet",
    "validate_manifest_schema",
    "write_manifest_parquet",
    "SplitLeakageError",
    "create_observation_grouped_splits",
    "export_split_csv",
    "load_split_csv",
    "verify_no_split_leakage",
]
