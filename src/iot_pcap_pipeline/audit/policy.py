"""Phase 1B.2 audit policy constants and issue codes."""

from __future__ import annotations

from iot_pcap_pipeline.pcap.decode import DLT_EN10MB

AUDIT_STRATEGY_VERSION = "phase1b2_v1"

SUPPORTED_LINKTYPES = frozenset({DLT_EN10MB})
EXPECTED_LINKTYPE = DLT_EN10MB

DEFAULT_ISSUE_CAP_PER_CODE = 5
DEFAULT_MALFORMED_HIGH_WARNING_RATE = 0.01
DEFAULT_MALFORMED_CATASTROPHIC_RATE = 0.80

SEVERITY_WARNING = "warning"
SEVERITY_HIGH_WARNING = "high_warning"
SEVERITY_HARD_FAILURE = "hard_failure"

SCOPE_CORPUS = "corpus"
SCOPE_FILE = "file"
SCOPE_PACKET = "packet"

# Corpus / file contract codes
ISSUE_MANIFEST_DUPLICATE = "manifest:duplicate_path"
ISSUE_MANIFEST_MISSING = "manifest:missing_pcap"
ISSUE_MANIFEST_EXTRA = "manifest:extra_pcap"
ISSUE_MANIFEST_SPLIT_MISMATCH = "manifest:split_mismatch"
ISSUE_MANIFEST_SIZE_MISMATCH = "manifest:file_size_mismatch"
ISSUE_OPEN_FAILURE = "file:open_failure"
ISSUE_UNSUPPORTED_LINKTYPE = "file:unsupported_linktype"
ISSUE_DECODER_ERROR = "file:decoder_error"
ISSUE_ACCOUNTING_INVARIANT = "file:accounting_invariant"
ISSUE_MALFORMED_CATASTROPHIC = "file:malformed_catastrophic"
ISSUE_MALFORMED_HIGH = "file:malformed_high"
ISSUE_MALFORMED_PRESENT = "file:malformed_present"
ISSUE_PARTIAL_PRESENT = "file:partial_present"
ISSUE_UNSUPPORTED_PRESENT = "file:unsupported_present"
ISSUE_TIMESTAMP_DUPLICATE = "file:timestamp_duplicate"
ISSUE_TIMESTAMP_REVERSAL = "file:timestamp_reversal"
ISSUE_ZERO_DURATION = "file:zero_timestamp_span"
ISSUE_IP_CARDINALITY_CAPPED = "file:ip_cardinality_capped"
ISSUE_ZERO_PACKETS = "file:zero_packets"
ISSUE_WORKER_CRASH = "file:worker_crash"

HARD_FAILURE_CODES = frozenset(
    {
        ISSUE_MANIFEST_DUPLICATE,
        ISSUE_MANIFEST_MISSING,
        ISSUE_MANIFEST_EXTRA,
        ISSUE_MANIFEST_SPLIT_MISMATCH,
        ISSUE_MANIFEST_SIZE_MISMATCH,
        ISSUE_OPEN_FAILURE,
        ISSUE_UNSUPPORTED_LINKTYPE,
        ISSUE_DECODER_ERROR,
        ISSUE_ACCOUNTING_INVARIANT,
        ISSUE_MALFORMED_CATASTROPHIC,
        ISSUE_WORKER_CRASH,
    }
)

DEFAULT_WORKERS = 4
