"""Version-controlled exact taxonomies for CICIoMT2024 Wi-Fi/MQTT PCAPs.

No fuzzy matching. Unknown spellings must remain unresolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TAXONOMY_VERSION = "phase1a_v1"

# Exact alias map: raw token found in path/filename -> canonical device id.
DEVICE_ALIASES: dict[str, str] = {
    "Blink_Camera": "Blink_Camera",
    "Blink_Mini_Camera": "Blink_Camera",
    "Ecobee_Camera": "Ecobee_Camera",
    "M1T_Camera": "M1T_Camera",
    "Owltron_Camera": "Owltron_Camera",
    "Multifunctional_Pager": "Multifunctional_Pager",
    "SenseU": "SenseU",
    "SenseUBaby": "SenseU",
    "Singcall": "Singcall",
}

# Exact stem (without _train/_test suffix and optional trailing replicate digit)
# maps to (attack_family, attack_type).
# Numbered TCP/IP pieces share the same attack_type.
_ATTACK_LABELS: dict[str, tuple[str, str]] = {
    "ARP_Spoofing": ("Spoofing", "ARP_Spoofing"),
    "Recon-OS_Scan": ("Recon", "OS_Scan"),
    "Recon-Ping_Sweep": ("Recon", "Ping_Sweep"),
    "Recon-Port_Scan": ("Recon", "Port_Scan"),
    "Recon-VulScan": ("Recon", "VulScan"),
    "MQTT-DDoS-Connect_Flood": ("MQTT", "MQTT_DDoS_Connect_Flood"),
    "MQTT-DDoS-Publish_Flood": ("MQTT", "MQTT_DDoS_Publish_Flood"),
    "MQTT-DoS-Connect_Flood": ("MQTT", "MQTT_DoS_Connect_Flood"),
    "MQTT-DoS-Publish_Flood": ("MQTT", "MQTT_DoS_Publish_Flood"),
    "MQTT-Malformed_Data": ("MQTT", "MQTT_Malformed_Data"),
    "TCP_IP-DDoS-ICMP": ("DDoS", "DDoS_ICMP"),
    "TCP_IP-DDoS-SYN": ("DDoS", "DDoS_SYN"),
    "TCP_IP-DDoS-TCP": ("DDoS", "DDoS_TCP"),  
    "TCP_IP-DDoS-UDP": ("DDoS", "DDoS_UDP"),
    "TCP_IP-DoS-ICMP": ("DoS", "DoS_ICMP"),
    "TCP_IP-DoS-SYN": ("DoS", "DoS_SYN"),
    "TCP_IP-DoS-TCP": ("DoS", "DoS_TCP"),
    "TCP_IP-DoS-UDP": ("DoS", "DoS_UDP"),
}

_SPLIT_SUFFIX_RE = re.compile(r"_(train|test)$")
_TCPIP_REPLICATE_RE = re.compile(
    r"^(TCP_IP-(?:DDoS|DoS)-(?:ICMP|SYN|TCP|UDP))(\d+)$"
)


@dataclass(frozen=True)
class AttackTaxonomy:
    family: str
    attack_type: str
    capture_session: str
    base_label: str
    replicate: int | None


def strip_split_suffix(stem: str) -> tuple[str, str | None]:
    """Strip trailing _train/_test from a filename stem."""
    match = _SPLIT_SUFFIX_RE.search(stem)
    if not match:
        return stem, None
    return stem[: match.start()], match.group(1)


def resolve_device_alias(raw_name: str) -> str | None:
    """Return canonical device id for an exact known alias, else None."""
    return DEVICE_ALIASES.get(raw_name)


def classify_attack_stem(stem: str) -> AttackTaxonomy | None:
    """Classify an attack filename stem (with or without _train/_test).

    Returns None for Benign or unrecognized labels.
    """
    base, _ = strip_split_suffix(stem)
    if base == "Benign":
        return None

    replicate: int | None = None
    lookup = base
    replicate_match = _TCPIP_REPLICATE_RE.match(base)
    if replicate_match:
        lookup = replicate_match.group(1)
        replicate = int(replicate_match.group(2))

    mapped = _ATTACK_LABELS.get(lookup)
    if mapped is None:
        return None

    family, attack_type = mapped
    if replicate is not None:
        capture_session = f"{attack_type}_{replicate}"
    else:
        capture_session = attack_type

    return AttackTaxonomy(
        family=family,
        attack_type=attack_type,
        capture_session=capture_session,
        base_label=lookup,
        replicate=replicate,
    )


def is_publisher_benign_stem(stem: str) -> bool:
    base, _ = strip_split_suffix(stem)
    return base == "Benign"
