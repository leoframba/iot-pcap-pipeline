"""Helpers to synthesize PCAPs for decoder tests."""

from __future__ import annotations

import socket
from pathlib import Path

import dpkt


def _ip4(addr: str) -> bytes:
    return socket.inet_pton(socket.AF_INET, addr)


def _ip6(addr: str) -> bytes:
    return socket.inet_pton(socket.AF_INET6, addr)


def write_pcap(path: Path, packets: list[tuple[float, bytes]], linktype: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=linktype)
        for ts, buf in packets:
            writer.writepkt(buf, ts=ts)
    return path


def eth_ip_tcp(
    *,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    sport: int = 12345,
    dport: int = 80,
    flags: int = dpkt.tcp.TH_SYN,
) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, seq=1, ack=0, win=8192)
    # Empty payload — do not invent application data.
    tcp.data = b""
    ip = dpkt.ip.IP(
        src=_ip4(src),
        dst=_ip4(dst),
        p=dpkt.ip.IP_PROTO_TCP,
        data=tcp,
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\xaa\xbb\xcc\xdd\xee\xff",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def eth_ip_udp(
    *,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    sport: int = 53,
    dport: int = 53,
) -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport)
    udp.data = b""
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(
        src=_ip4(src),
        dst=_ip4(dst),
        p=dpkt.ip.IP_PROTO_UDP,
        data=udp,
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\xaa\xbb\xcc\xdd\xee\xff",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def eth_ip_icmp(*, src: str = "10.0.0.1", dst: str = "10.0.0.2") -> bytes:
    icmp = dpkt.icmp.ICMP(type=8, code=0, data=dpkt.icmp.ICMP.Echo(id=1, seq=1, data=b"ping"))
    ip = dpkt.ip.IP(
        src=_ip4(src),
        dst=_ip4(dst),
        p=dpkt.ip.IP_PROTO_ICMP,
        data=icmp,
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\xaa\xbb\xcc\xdd\xee\xff",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def eth_ip_igmp(*, src: str = "10.0.0.1", dst: str = "224.0.0.1") -> bytes:
    igmp = dpkt.igmp.IGMP(type=0x11, maxresp=10, group=_ip4("0.0.0.0"))
    ip = dpkt.ip.IP(
        src=_ip4(src),
        dst=_ip4(dst),
        p=dpkt.ip.IP_PROTO_IGMP,
        data=igmp,
    )
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\x01\x00\x5e\x00\x00\x01",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def eth_ipv6_tcp(
    *,
    src: str = "2001:db8::1",
    dst: str = "2001:db8::2",
    sport: int = 1234,
    dport: int = 443,
) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=dpkt.tcp.TH_ACK, seq=1, ack=1)
    tcp.data = b""
    ip6 = dpkt.ip6.IP6(
        src=_ip6(src),
        dst=_ip6(dst),
        nxt=dpkt.ip.IP_PROTO_TCP,
        data=tcp,
    )
    ip6.plen = len(tcp)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\xaa\xbb\xcc\xdd\xee\xff",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP6,
        data=ip6,
    )
    return bytes(eth)


def eth_ipv6_icmp(*, src: str = "2001:db8::1", dst: str = "2001:db8::2") -> bytes:
    icmp6 = dpkt.icmp6.ICMP6(type=128, code=0, data=b"\x00" * 4)
    ip6 = dpkt.ip6.IP6(
        src=_ip6(src),
        dst=_ip6(dst),
        nxt=dpkt.ip.IP_PROTO_ICMP6,
        data=icmp6,
    )
    ip6.plen = len(icmp6)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\xaa\xbb\xcc\xdd\xee\xff",
        src=b"\x11\x22\x33\x44\x55\x66",
        type=dpkt.ethernet.ETH_TYPE_IP6,
        data=ip6,
    )
    return bytes(eth)


def _mac6(addr: str | bytes) -> bytes:
    if isinstance(addr, (bytes, bytearray)):
        raw = bytes(addr)
        if len(raw) != 6:
            raise ValueError(f"MAC must be 6 bytes, got {len(raw)}")
        return raw
    parts = addr.split(":")
    if len(parts) != 6:
        raise ValueError(f"MAC must have 6 octets, got {addr!r}")
    return bytes(int(p, 16) for p in parts)


def eth_arp(
    *,
    spa: str = "10.0.0.1",
    tpa: str = "10.0.0.2",
    op: int = dpkt.arp.ARP_OP_REQUEST,
    sha: str | bytes = "11:22:33:44:55:66",
    tha: str | bytes = "00:00:00:00:00:00",
    eth_src: str | bytes | None = None,
    eth_dst: str | bytes = "ff:ff:ff:ff:ff:ff",
) -> bytes:
    sha_b = _mac6(sha)
    tha_b = _mac6(tha)
    eth_src_b = sha_b if eth_src is None else _mac6(eth_src)
    arp = dpkt.arp.ARP(
        sha=sha_b,
        spa=_ip4(spa),
        tha=tha_b,
        tpa=_ip4(tpa),
        op=op,
    )
    eth = dpkt.ethernet.Ethernet(
        dst=_mac6(eth_dst),
        src=eth_src_b,
        type=dpkt.ethernet.ETH_TYPE_ARP,
        data=arp,
    )
    return bytes(eth)


def eth_arp_truncated(*, spa: str = "10.0.0.1", tpa: str = "10.0.0.2") -> bytes:
    """Ethernet ARP frame with a truncated ARP body (NeedData / PARTIAL)."""
    dest = b"\xff\xff\xff\xff\xff\xff"
    src = b"\x11\x22\x33\x44\x55\x66"
    ethertype = dpkt.ethernet.ETH_TYPE_ARP.to_bytes(2, "big")
    # hrd=1, pro=0x0800, hln=6, pln=4, op=request, then only 3 SHA bytes.
    arp_hdr = b"\x00\x01\x08\x00\x06\x04\x00\x01" + b"\xaa\xbb\xcc"
    return dest + src + ethertype + arp_hdr


def eth_vlan_ip_udp(*, vlan_id: int = 100) -> bytes:
    """Build an 802.1Q Ethernet frame carrying IPv4/UDP."""
    udp = dpkt.udp.UDP(sport=4000, dport=5000, data=b"")
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(
        src=_ip4("10.1.0.1"),
        dst=_ip4("10.1.0.2"),
        p=dpkt.ip.IP_PROTO_UDP,
        data=udp,
    )
    ip.len = len(ip)
    # Manual VLAN tag: dest(6)+src(6)+802.1Q(4)+type(2)+payload
    dest = b"\xaa\xbb\xcc\xdd\xee\xff"
    src = b"\x11\x22\x33\x44\x55\x66"
    tci = vlan_id & 0x0FFF
    vlan_hdr = b"\x81\x00" + tci.to_bytes(2, "big")
    ethertype = dpkt.ethernet.ETH_TYPE_IP.to_bytes(2, "big")
    return dest + src + vlan_hdr + ethertype + bytes(ip)


def eth_unknown_ethertype() -> bytes:
    dest = b"\xaa\xbb\xcc\xdd\xee\xff"
    src = b"\x11\x22\x33\x44\x55\x66"
    # Experimental ethertype
    return dest + src + b"\x88\xb5" + b"\x00" * 20


def eth_ieee8023_llc(*, length: int = 6) -> bytes:
    """Minimal IEEE 802.3 + LLC frame matching SenseU/Singcall pattern."""
    dest = b"\xff\xff\xff\xff\xff\xff"
    src = b"\x34\x94\x54\xf0\xdb\xf0"
    # length field < 0x0600
    type_or_len = int(length).to_bytes(2, "big")
    # LLC: DSAP=0, SSAP=1, Control=0xAF, plus 3 payload bytes (total LLC PDU len 6)
    llc_pdu = b"\x00\x01\xaf\x81\x01\x02"
    frame = dest + src + type_or_len + llc_pdu
    # Pad to minimum Ethernet frame size
    if len(frame) < 60:
        frame = frame + b"\x00" * (60 - len(frame))
    return frame


def eth_lldp() -> bytes:
    dest = b"\x01\x80\xc2\x00\x00\x0e"
    src = b"\x11\x22\x33\x44\x55\x66"
    return dest + src + b"\x88\xcc" + b"\x00" * 20
