"""Frame decoding helpers built on DPKT (no application-layer parsing)."""

from __future__ import annotations

import socket
from dataclasses import replace
from typing import Any

import dpkt

from iot_pcap_pipeline.pcap.packet import PacketRecord, ParseStatus

# BSD/libpcap DLT values we care about for Phase 1B.1.
DLT_EN10MB = 1  # Ethernet
DLT_RAW = 101
DLT_IPV4 = 228
DLT_IPV6 = 229

_ETH_TYPE_NAMES = {
    dpkt.ethernet.ETH_TYPE_IP: "ipv4",
    dpkt.ethernet.ETH_TYPE_IP6: "ipv6",
    dpkt.ethernet.ETH_TYPE_ARP: "arp",
}


def _mac_to_str(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw)


def _ip4_to_str(raw: bytes) -> str | None:
    try:
        return socket.inet_ntop(socket.AF_INET, raw)
    except (OSError, ValueError):
        return None


def _ip6_to_str(raw: bytes) -> str | None:
    try:
        return socket.inet_ntop(socket.AF_INET6, raw)
    except (OSError, ValueError):
        return None


def _vlan_ids_from_ethernet(eth: dpkt.ethernet.Ethernet) -> tuple[int, ...]:
    tags = getattr(eth, "vlan_tags", None) or []
    ids: list[int] = []
    for tag in tags:
        # dpkt.ethernet.VLANtag8021Q exposes .id
        vid = getattr(tag, "id", None)
        if vid is None and isinstance(tag, int):
            vid = tag & 0x0FFF
        if vid is not None:
            ids.append(int(vid))
    return tuple(ids)


def _effective_ethertype(eth: dpkt.ethernet.Ethernet) -> int:
    """Return ethertype after VLAN unwrap when tags are present."""
    tags = getattr(eth, "vlan_tags", None) or []
    if tags:
        inner = getattr(tags[-1], "type", None)
        if inner is not None:
            return int(inner)
    return int(eth.type)


def _base_record(
    *,
    packet_index: int,
    timestamp: float,
    frame_length: int,
    linktype: int,
    parse_status: ParseStatus,
    parse_detail: str | None = None,
) -> PacketRecord:
    return PacketRecord(
        packet_index=packet_index,
        timestamp=float(timestamp),
        frame_length=int(frame_length),
        linktype=int(linktype),
        parse_status=parse_status,
        parse_detail=parse_detail,
    )


def _with_tcp_flags(record: PacketRecord, tcp: dpkt.tcp.TCP) -> PacketRecord:
    flags = int(tcp.flags)
    return replace(
        record,
        is_tcp=True,
        src_port=int(tcp.sport),
        dst_port=int(tcp.dport),
        tcp_flags=flags,
        tcp_flag_fin=bool(flags & dpkt.tcp.TH_FIN),
        tcp_flag_syn=bool(flags & dpkt.tcp.TH_SYN),
        tcp_flag_rst=bool(flags & dpkt.tcp.TH_RST),
        tcp_flag_psh=bool(flags & dpkt.tcp.TH_PUSH),
        tcp_flag_ack=bool(flags & dpkt.tcp.TH_ACK),
        tcp_flag_urg=bool(flags & dpkt.tcp.TH_URG),
        protocol_name="tcp",
    )


def _decode_l4(record: PacketRecord, ip_proto: int, payload: Any) -> PacketRecord:
    if ip_proto == dpkt.ip.IP_PROTO_TCP:
        if not isinstance(payload, dpkt.tcp.TCP):
            try:
                payload = dpkt.tcp.TCP(payload)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError) as exc:
                return replace(
                    record,
                    parse_status=ParseStatus.PARTIAL,
                    parse_detail=f"tcp decode failed: {exc}",
                    protocol_name="ipv4" if record.is_ipv4 else "ipv6",
                )
        return _with_tcp_flags(record, payload)

    if ip_proto == dpkt.ip.IP_PROTO_UDP:
        if not isinstance(payload, dpkt.udp.UDP):
            try:
                payload = dpkt.udp.UDP(payload)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError) as exc:
                return replace(
                    record,
                    parse_status=ParseStatus.PARTIAL,
                    parse_detail=f"udp decode failed: {exc}",
                    protocol_name="ipv4" if record.is_ipv4 else "ipv6",
                )
        return replace(
            record,
            is_udp=True,
            src_port=int(payload.sport),
            dst_port=int(payload.dport),
            protocol_name="udp",
        )

    if ip_proto == dpkt.ip.IP_PROTO_ICMP:
        return replace(record, is_icmp=True, protocol_name="icmp")

    if ip_proto == dpkt.ip.IP_PROTO_ICMP6:
        return replace(record, is_icmpv6=True, protocol_name="icmpv6")

    if ip_proto == dpkt.ip.IP_PROTO_IGMP:
        return replace(record, is_igmp=True, protocol_name="igmp")

    return replace(
        record,
        parse_status=ParseStatus.UNSUPPORTED_PROTOCOL,
        parse_detail=f"unsupported ip protocol: {ip_proto}",
        protocol_name=f"ip_proto_{ip_proto}",
    )


def _decode_ipv4(record: PacketRecord, ip: dpkt.ip.IP) -> PacketRecord:
    record = replace(
        record,
        is_ipv4=True,
        ip_version=4,
        ip_protocol=int(ip.p),
        src_ip=_ip4_to_str(ip.src),
        dst_ip=_ip4_to_str(ip.dst),
        protocol_name="ipv4",
    )
    return _decode_l4(record, int(ip.p), ip.data)


def _decode_ipv6(record: PacketRecord, ip6: dpkt.ip6.IP6) -> PacketRecord:
    # dpkt exposes the last/next header in .nxt after extension walking when possible.
    nxt = int(getattr(ip6, "nxt", 0))
    record = replace(
        record,
        is_ipv6=True,
        ip_version=6,
        ip_protocol=nxt,
        src_ip=_ip6_to_str(ip6.src),
        dst_ip=_ip6_to_str(ip6.dst),
        protocol_name="ipv6",
    )
    return _decode_l4(record, nxt, ip6.data)


def _decode_arp(record: PacketRecord, arp: dpkt.arp.ARP) -> PacketRecord:
    src_ip = None
    dst_ip = None
    if len(arp.spa) == 4:
        src_ip = _ip4_to_str(arp.spa)
    if len(arp.tpa) == 4:
        dst_ip = _ip4_to_str(arp.tpa)
    return replace(
        record,
        is_arp=True,
        src_ip=src_ip,
        dst_ip=dst_ip,
        protocol_name="arp",
    )


def _decode_ethernet_payload(record: PacketRecord, eth: dpkt.ethernet.Ethernet) -> PacketRecord:
    vlan_ids = _vlan_ids_from_ethernet(eth)
    ethertype = _effective_ethertype(eth)
    record = replace(
        record,
        ethertype=ethertype,
        vlan_ids=vlan_ids,
        extra={
            **record.extra,
            "src_mac": _mac_to_str(eth.src),
            "dst_mac": _mac_to_str(eth.dst),
        },
    )

    data = eth.data

    if ethertype == dpkt.ethernet.ETH_TYPE_ARP or isinstance(data, dpkt.arp.ARP):
        if not isinstance(data, dpkt.arp.ARP):
            try:
                data = dpkt.arp.ARP(data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError) as exc:
                return replace(
                    record,
                    parse_status=ParseStatus.PARTIAL,
                    parse_detail=f"arp decode failed: {exc}",
                    protocol_name="ethernet",
                )
        return _decode_arp(record, data)

    if ethertype == dpkt.ethernet.ETH_TYPE_IP or isinstance(data, dpkt.ip.IP):
        if not isinstance(data, dpkt.ip.IP):
            try:
                data = dpkt.ip.IP(data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError) as exc:
                return replace(
                    record,
                    parse_status=ParseStatus.PARTIAL,
                    parse_detail=f"ipv4 decode failed: {exc}",
                    protocol_name="ethernet",
                )
        return _decode_ipv4(record, data)

    if ethertype == dpkt.ethernet.ETH_TYPE_IP6 or isinstance(data, dpkt.ip6.IP6):
        if not isinstance(data, dpkt.ip6.IP6):
            try:
                data = dpkt.ip6.IP6(data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError) as exc:
                return replace(
                    record,
                    parse_status=ParseStatus.PARTIAL,
                    parse_detail=f"ipv6 decode failed: {exc}",
                    protocol_name="ethernet",
                )
        return _decode_ipv6(record, data)

    name = _ETH_TYPE_NAMES.get(ethertype, f"ethertype_0x{ethertype:04x}")
    return replace(
        record,
        parse_status=ParseStatus.UNSUPPORTED_PROTOCOL,
        parse_detail=f"unsupported ethertype: 0x{ethertype:04x}",
        protocol_name=name,
    )


def decode_frame(
    buf: bytes,
    *,
    packet_index: int,
    timestamp: float,
    linktype: int,
) -> PacketRecord:
    """Decode one captured frame into a PacketRecord.

    Always preserves timestamp and captured frame length. Never consults labels,
    filenames, or split metadata.
    """
    frame_length = len(buf)
    base = _base_record(
        packet_index=packet_index,
        timestamp=timestamp,
        frame_length=frame_length,
        linktype=linktype,
        parse_status=ParseStatus.OK,
    )

    try:
        if linktype == DLT_EN10MB:
            if frame_length < 14:
                return replace(
                    base,
                    parse_status=ParseStatus.MALFORMED,
                    parse_detail="ethernet frame shorter than 14 bytes",
                    protocol_name="ethernet",
                )
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError) as exc:
                return replace(
                    base,
                    parse_status=ParseStatus.MALFORMED,
                    parse_detail=f"ethernet unpack failed: {exc}",
                    protocol_name="ethernet",
                )
            return _decode_ethernet_payload(base, eth)

        if linktype in {DLT_RAW, DLT_IPV4}:
            try:
                ip = dpkt.ip.IP(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError) as exc:
                return replace(
                    base,
                    parse_status=ParseStatus.MALFORMED,
                    parse_detail=f"raw ipv4 unpack failed: {exc}",
                )
            return _decode_ipv4(base, ip)

        if linktype == DLT_IPV6:
            try:
                ip6 = dpkt.ip6.IP6(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError) as exc:
                return replace(
                    base,
                    parse_status=ParseStatus.MALFORMED,
                    parse_detail=f"raw ipv6 unpack failed: {exc}",
                )
            return _decode_ipv6(base, ip6)

        return replace(
            base,
            parse_status=ParseStatus.UNSUPPORTED_LINKTYPE,
            parse_detail=f"unsupported linktype: {linktype}",
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the streamer
        return replace(
            base,
            parse_status=ParseStatus.ERROR,
            parse_detail=f"unexpected decode error: {type(exc).__name__}: {exc}",
        )
