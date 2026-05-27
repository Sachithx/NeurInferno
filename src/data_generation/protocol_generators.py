"""
Protocol message generators for Tier 2.

Each generator yields Scapy packets covering one protocol.
Packets are varied in field values to maximise semantic diversity.
"""

from __future__ import annotations

import random
import socket
import struct
from typing import Iterator

# ── Scapy imports ────────────────────────────────────────────────────────────
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP, Ether
from scapy.layers.dns import DNS, DNSQR, DNSRR
from scapy.layers.dhcp import DHCP, BOOTP
from scapy.layers.ntp import NTP
from scapy.layers.snmp import SNMP, SNMPget, SNMPnext, SNMPresponse, SNMPvarbind
from scapy.contrib.modbus import (
    ModbusADURequest, ModbusADUResponse,
    ModbusPDU01ReadCoilsRequest, ModbusPDU01ReadCoilsResponse,
    ModbusPDU03ReadHoldingRegistersRequest,
    ModbusPDU03ReadHoldingRegistersResponse,
    ModbusPDU06WriteSingleRegisterRequest,
    ModbusPDU10WriteMultipleRegistersRequest,
)
from scapy.contrib.igmp import IGMP
from scapy.contrib.ospf import OSPF_Hdr, OSPF_Hello
from scapy.contrib.coap import CoAP
from scapy.contrib.mqtt import MQTT, MQTTConnect, MQTTPublish, MQTTSubscribe
from scapy.layers.tls.record import TLS, TLSApplicationData

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rip() -> str:
    """Random routable IPv4 address."""
    return f"{random.randint(1,223)}.{random.randint(0,255)}." \
           f"{random.randint(0,255)}.{random.randint(1,254)}"


def _rmac() -> str:
    return ":".join(f"{random.randint(0,255):02x}" for _ in range(6))


def _rport() -> int:
    return random.randint(1024, 65535)


_TCP_FLAG_COMBOS = ["S", "SA", "A", "FA", "F", "R", "PA", "P"]

_DNS_QTYPES = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "PTR", "SOA"]
_DNS_NAMES = ["example.com", "google.com", "test.local", "a.b.c.d.e",
              "mail.example.org", "ns1.domain.net", "x.com"]

_DHCP_MSG_TYPES = [1, 2, 3, 4, 5, 6, 7, 8]  # DISCOVER…INFORM

_ICMP_TYPE_CODES = [(0, 0), (8, 0), (3, 0), (3, 1), (3, 3),
                    (11, 0), (11, 1), (5, 0), (13, 0), (14, 0)]

_OSPF_TYPES = [1, 2, 3, 4, 5]

_COAP_CODES = [1, 2, 3, 4, 65, 68, 69, 132, 133, 160, 161, 162, 163]
_COAP_TYPES = [0, 1, 2, 3]


# ── 1. IP ─────────────────────────────────────────────────────────────────────

def gen_ip(n: int = 1000) -> list:
    pkts = []
    protos = [6, 17, 1, 89, 132, 0, 41]
    for _ in range(n):
        pkts.append(IP(
            src=_rip(), dst=_rip(),
            ttl=random.randint(1, 255),
            tos=random.choice([0, 0x10, 0x20, 0x28, 0x48, 0xb8]),
            flags=random.choice([0, 2]),
            id=random.randint(0, 0xFFFF),
            proto=random.choice(protos),
        ))
    return pkts


# ── 2. ICMP ───────────────────────────────────────────────────────────────────

def gen_icmp(n: int = 1000) -> list:
    pkts = []
    for _ in range(n):
        t, c = random.choice(_ICMP_TYPE_CODES)
        pkt = IP(src=_rip(), dst=_rip()) / ICMP(type=t, code=c,
              id=random.randint(0, 0xFFFF),
              seq=random.randint(0, 0xFFFF))
        if t in (8, 0):
            pkt = pkt / (b"\x00" * random.randint(0, 32))
        pkts.append(pkt)
    return pkts


# ── 3. TCP ────────────────────────────────────────────────────────────────────

def gen_tcp(n: int = 1000) -> list:
    pkts = []
    for _ in range(n):
        flags = random.choice(_TCP_FLAG_COMBOS)
        pkt = IP(src=_rip(), dst=_rip()) / TCP(
            sport=_rport(), dport=random.choice([80, 443, 22, 25, 21, _rport()]),
            flags=flags,
            seq=random.randint(0, 2**32 - 1),
            ack=random.randint(0, 2**32 - 1) if "A" in flags else 0,
            window=random.choice([1024, 4096, 8192, 16384, 65535]),
        )
        if "P" in flags:
            pkt = pkt / (b"GET / HTTP/1.1\r\n" if random.random() < 0.5 else
                         bytes(random.randint(1, 40) for _ in range(random.randint(4, 32))))
        pkts.append(pkt)
    return pkts


# ── 4. UDP ────────────────────────────────────────────────────────────────────

def gen_udp(n: int = 1000) -> list:
    pkts = []
    for _ in range(n):
        payload_len = random.randint(0, 64)
        pkts.append(IP(src=_rip(), dst=_rip()) / UDP(
            sport=_rport(),
            dport=random.choice([53, 67, 68, 123, 161, 162, _rport()]),
        ) / bytes(random.randint(0, 255) for _ in range(payload_len)))
    return pkts


# ── 5. ARP ────────────────────────────────────────────────────────────────────

def gen_arp(n: int = 1000) -> list:
    pkts = []
    for _ in range(n):
        op = random.choice([1, 2])
        pkts.append(ARP(
            op=op,
            hwsrc=_rmac(), psrc=_rip(),
            hwdst=_rmac() if op == 2 else "00:00:00:00:00:00",
            pdst=_rip(),
        ))
    return pkts


# ── 6. DNS ────────────────────────────────────────────────────────────────────

def gen_dns(n: int = 1500) -> list:
    pkts = []
    for _ in range(n):
        qname = random.choice(_DNS_NAMES)
        qtype = random.choice(_DNS_QTYPES)
        is_response = random.random() < 0.5
        qd = DNSQR(qname=qname, qtype=qtype)
        an = None
        if is_response:
            if qtype == "A":
                an = DNSRR(rrname=qname, type="A", ttl=random.randint(60, 86400),
                           rdata=_rip())
            elif qtype == "MX":
                an = DNSRR(rrname=qname, type="MX", ttl=300,
                           rdata=f"mail.{qname}")
            elif qtype == "NS":
                an = DNSRR(rrname=qname, type="NS", ttl=3600,
                           rdata=f"ns1.{qname}")
            else:
                # For AAAA, TXT, CNAME, PTR, SOA — use A record to avoid type issues
                an = DNSRR(rrname=qname, type="A", ttl=300, rdata=_rip())

        dns_kw = dict(
            id=random.randint(0, 0xFFFF),
            qr=int(is_response),
            opcode=random.choice([0, 0, 0, 1, 2]),
            rd=random.randint(0, 1),
            ra=int(is_response),
            qdcount=1,
            qd=qd,
        )
        if an:
            dns_kw["ancount"] = 1
            dns_kw["an"] = an
        pkt = IP(src=_rip(), dst=_rip()) / UDP(sport=_rport(), dport=53) / DNS(**dns_kw)
        pkts.append(pkt)
    return pkts


# ── 7. DHCP ───────────────────────────────────────────────────────────────────

def gen_dhcp(n: int = 1000) -> list:
    pkts = []
    msg_types = [1, 2, 3, 4, 5, 6, 7, 8]  # discover, offer, request, …
    for _ in range(n):
        mt = random.choice(msg_types)
        xid = random.randint(0, 0xFFFFFFFF)
        ciaddr = _rip() if mt >= 3 else "0.0.0.0"
        yiaddr = _rip() if mt in (2, 5) else "0.0.0.0"
        opts = [("message-type", mt),
                ("subnet_mask", "255.255.255.0"),
                ("router", _rip()),
                ("name_server", _rip()),
                ("lease_time", random.randint(3600, 86400)),
                "end"]
        pkt = IP(src=_rip(), dst="255.255.255.255") / UDP(sport=68, dport=67) / \
              BOOTP(op=1 if mt in (1, 3, 7, 8) else 2,
                    xid=xid, ciaddr=ciaddr, yiaddr=yiaddr,
                    chaddr=bytes.fromhex(_rmac().replace(":", ""))) / \
              DHCP(options=opts)
        pkts.append(pkt)
    return pkts


# ── 8. NTP ────────────────────────────────────────────────────────────────────

def gen_ntp(n: int = 1000) -> list:
    pkts = []
    for _ in range(n):
        pkts.append(NTP(
            leap=random.randint(0, 3),
            version=random.choice([3, 4]),
            mode=random.choice([1, 2, 3, 4, 5, 6]),
            stratum=random.randint(0, 15),
            poll=random.randint(3, 10),
            precision=random.randint(-20, 0),
        ))
    return pkts


# ── 9. Modbus ─────────────────────────────────────────────────────────────────

def gen_modbus(n: int = 1000) -> list:
    pkts = []
    func_codes = [1, 3, 6, 16]
    for _ in range(n):
        fc = random.choice(func_codes)
        tid = random.randint(0, 0xFFFF)
        uid = random.randint(1, 247)
        adu = ModbusADURequest(transId=tid, unitId=uid)
        if fc == 1:
            pdu = ModbusPDU01ReadCoilsRequest(
                startAddr=random.randint(0, 0xFFFF),
                quantity=random.randint(1, 2000))
        elif fc == 3:
            pdu = ModbusPDU03ReadHoldingRegistersRequest(
                startAddr=random.randint(0, 0xFFFF),
                quantity=random.randint(1, 125))
        elif fc == 6:
            pdu = ModbusPDU06WriteSingleRegisterRequest(
                registerAddr=random.randint(0, 0xFFFF),
                registerValue=random.randint(0, 0xFFFF))
        else:
            n_regs = random.randint(1, 10)
            regs = bytes(random.randint(0, 255) for _ in range(n_regs * 2))
            pdu = ModbusPDU10WriteMultipleRegistersRequest(
                startAddr=random.randint(0, 0xFFFF),
                quantityRegisters=n_regs,
                byteCount=n_regs * 2,
                outputsValue=regs)
        pkts.append(adu / pdu)
    return pkts


# ── 10. IGMP ──────────────────────────────────────────────────────────────────

def gen_igmp(n: int = 500) -> list:
    pkts = []
    types = [0x11, 0x12, 0x16, 0x17, 0x22]
    for _ in range(n):
        pkts.append(IP(src=_rip(), dst="224.0.0.1") / IGMP(
            type=random.choice(types),
            mrcode=random.randint(0, 255),
            gaddr=f"224.{random.randint(0,255)}.{random.randint(0,255)}."
                  f"{random.randint(0,255)}",
        ))
    return pkts


# ── 11. OSPF ──────────────────────────────────────────────────────────────────

def gen_ospf(n: int = 500) -> list:
    pkts = []
    for _ in range(n):
        pkts.append(IP(src=_rip(), dst="224.0.0.5", proto=89) / OSPF_Hdr(
            type=1,
            src=_rip(),
            area=f"{random.randint(0,255)}.{random.randint(0,255)}."
                 f"{random.randint(0,255)}.{random.randint(0,255)}",
        ) / OSPF_Hello(
            mask=f"255.255.{random.choice([0,255])}.0",
            hellointerval=random.choice([10, 30]),
            deadinterval=random.choice([40, 120]),
            router=_rip(),
        ))
    return pkts


# ── 12. SNMP ──────────────────────────────────────────────────────────────────

_OID_SAMPLES = [
    "1.3.6.1.2.1.1.1.0",  # sysDescr
    "1.3.6.1.2.1.1.3.0",  # sysUpTime
    "1.3.6.1.2.1.2.1.0",  # ifNumber
    "1.3.6.1.2.1.2.2.1.2.1",  # ifDescr
    "1.3.6.1.2.1.4.1.0",  # ipForwarding
]

def gen_snmp(n: int = 500) -> list:
    pkts = []
    for _ in range(n):
        oid = random.choice(_OID_SAMPLES)
        community = random.choice([b"public", b"private", b"community"])
        pdu_type = random.choice(["get", "next", "response"])
        vb = SNMPvarbind(oid=oid)
        if pdu_type == "get":
            pdu = SNMPget(varbindlist=[vb])
        elif pdu_type == "next":
            pdu = SNMPnext(varbindlist=[vb])
        else:
            pdu = SNMPresponse(varbindlist=[vb])
        pkts.append(IP(src=_rip(), dst=_rip()) / UDP(sport=_rport(), dport=161) /
                    SNMP(community=community, PDU=pdu))
    return pkts


# ── 13. TLS record ────────────────────────────────────────────────────────────

def gen_tls(n: int = 500) -> list:
    pkts = []
    # TLS content types: 20=ChangeCipherSpec, 21=Alert, 22=Handshake, 23=AppData
    content_types = [20, 21, 22, 23]
    versions = [0x0301, 0x0302, 0x0303, 0x0304]
    for _ in range(n):
        ct = random.choice(content_types)
        ver = random.choice(versions)
        payload_len = random.randint(4, 64)
        payload = bytes(random.randint(0, 255) for _ in range(payload_len))
        pkts.append(TLS(type=ct, version=ver) / TLSApplicationData(data=payload))
    return pkts


# ── 14. CoAP ──────────────────────────────────────────────────────────────────

def gen_coap(n: int = 500) -> list:
    pkts = []
    for _ in range(n):
        code = random.choice(_COAP_CODES)
        token_len = random.randint(0, 8)
        token = bytes(random.randint(0, 255) for _ in range(token_len))
        pkts.append(CoAP(
            ver=1,
            type=random.choice(_COAP_TYPES),
            code=code,
            msg_id=random.randint(0, 0xFFFF),
            token=token,
        ))
    return pkts


# ── 15. MQTT ──────────────────────────────────────────────────────────────────

def gen_mqtt(n: int = 500) -> list:
    pkts = []
    topics = [b"sensor/temp", b"home/light", b"status/device", b"test/data"]
    for _ in range(n):
        pkt_type = random.choice(["connect", "publish", "subscribe"])
        if pkt_type == "connect":
            pkts.append(MQTT() / MQTTConnect(
                clientId=b"client" + bytes([random.randint(48, 57)] * 4),
                klive=random.choice([10, 30, 60, 120]),
            ))
        elif pkt_type == "publish":
            topic = random.choice(topics)
            payload = bytes(random.randint(0, 255) for _ in range(random.randint(1, 16)))
            pkts.append(MQTT() / MQTTPublish(topic=topic, value=payload))
        else:
            topic = random.choice(topics)
            pkts.append(MQTT() / MQTTSubscribe(
                msgid=random.randint(1, 0xFFFF),
                topics=[topic],
            ))
    return pkts


# ── 16. ICMP6 / IPv6 ──────────────────────────────────────────────────────────

def _r6ip():
    return ":".join(f"{random.randint(0,0xffff):04x}" for _ in range(8))

def gen_ipv6(n: int = 500) -> list:
    from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
    pkts = []
    for _ in range(n):
        is_echo = random.random() < 0.6
        if is_echo:
            pkts.append(IPv6(src=_r6ip(), dst=_r6ip(),
                             hlim=random.randint(1, 255),
                             tc=random.randint(0, 255),
                             fl=random.randint(0, 0xFFFFF)) /
                        ICMPv6EchoRequest(id=random.randint(0, 0xFFFF),
                                          seq=random.randint(0, 0xFFFF)))
        else:
            pkts.append(IPv6(src=_r6ip(), dst=_r6ip(),
                             hlim=random.randint(1, 255),
                             nh=random.choice([6, 17, 43, 44, 59, 60])))
    return pkts


# ── 17. BGP (manual — scapy BGP contrib not always available) ─────────────────

def gen_bgp_raw(n: int = 500) -> list:
    """
    Hand-craft minimal BGP OPEN and KEEPALIVE messages.
    Marker (16x 0xff) + length (2B) + type (1B) + payload.
    """
    pkts = []
    for _ in range(n):
        msg_type = random.choice([1, 4])  # OPEN or KEEPALIVE
        if msg_type == 4:  # KEEPALIVE: just marker+len+type
            length = 19
            raw = b"\xff" * 16 + struct.pack(">HB", length, msg_type)
        else:  # OPEN
            version = 4
            asn = random.randint(1, 65535)
            hold_time = random.choice([90, 180, 240])
            bgp_id = socket.inet_aton(_rip())
            opt_len = 0
            payload = struct.pack(">BHH4sB",
                                  version, asn, hold_time, bgp_id, opt_len)
            length = 19 + len(payload)
            raw = b"\xff" * 16 + struct.pack(">HB", length, msg_type) + payload
        pkts.append(raw)
    return pkts


# ── Registry ─────────────────────────────────────────────────────────────────

SCAPY_PROTOCOLS: dict[str, tuple] = {
    # (generator_fn, n_messages, has_scapy_pkt)
    "ip":      (gen_ip,      4000, True),
    "icmp":    (gen_icmp,    4000, True),
    "tcp":     (gen_tcp,     4000, True),
    "udp":     (gen_udp,     4000, True),
    "arp":     (gen_arp,     4000, True),
    "dns":     (gen_dns,     6000, True),
    "dhcp":    (gen_dhcp,    4000, True),
    "ntp":     (gen_ntp,     4000, True),
    "modbus":  (gen_modbus,  2000, True),
    "igmp":    (gen_igmp,    2000, True),
    "ospf":    (gen_ospf,    2000, True),
    "snmp":    (gen_snmp,    2000, True),
    "tls":     (gen_tls,     2000, True),
    "coap":    (gen_coap,    2000, True),
    "mqtt":    (gen_mqtt,    2000, True),
    "ipv6":    (gen_ipv6,    2000, True),
    "bgp_raw": (gen_bgp_raw, 2000, False),  # raw bytes, no Scapy dissect
}
