import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pymc_core.node.handlers.ack import AckHandler
from pymc_core.node.handlers.text import TextMessageHandler
from pymc_core.node.events.events import MeshEvents
from pymc_core.protocol import PacketBuilder, PacketTimingUtils
from pymc_core.protocol.constants import (
    ADVERT_FLAG_HAS_LOCATION,
    ADVERT_FLAG_HAS_NAME,
    ADVERT_FLAG_IS_CHAT_NODE,
    ADVERT_FLAG_IS_REPEATER,
    ADVERT_FLAG_IS_ROOM_SERVER,
    PAYLOAD_TYPE_ACK,
    PAYLOAD_TYPE_ADVERT,
    PAYLOAD_TYPE_PATH,
    PAYLOAD_TYPE_TXT_MSG,
)
from pymc_core.protocol.utils import decode_appdata, parse_advert_payload
from repeater.config import save_config

logger = logging.getLogger("WifiCompanion")


class CompanionProtocol:
    FRAME_OUTGOING = 0x3C  # "<" app->radio
    FRAME_INCOMING = 0x3E  # ">" radio->app

    class CommandCodes:
        AppStart = 1
        SendTxtMsg = 2
        SendChannelTxtMsg = 3
        GetContacts = 4
        GetDeviceTime = 5
        SetDeviceTime = 6
        SendSelfAdvert = 7
        SetAdvertName = 8
        AddUpdateContact = 9
        SyncNextMessage = 10
        SetRadioParams = 11
        SetTxPower = 12
        ResetPath = 13
        SetAdvertLatLon = 14
        RemoveContact = 15
        ShareContact = 16
        ExportContact = 17
        ImportContact = 18
        Reboot = 19
        GetBatteryVoltage = 20
        SetTuningParams = 21
        DeviceQuery = 22
        ExportPrivateKey = 23
        ImportPrivateKey = 24
        SendRawData = 25
        SendLogin = 26
        SendStatusReq = 27
        GetChannel = 31
        SetChannel = 32
        SignStart = 33
        SignData = 34
        SignFinish = 35
        SendTracePath = 36
        SetOtherParams = 38
        SendTelemetryReq = 39
        SendBinaryReq = 50

    class ResponseCodes:
        Ok = 0
        Err = 1
        ContactsStart = 2
        Contact = 3
        EndOfContacts = 4
        SelfInfo = 5
        Sent = 6
        ContactMsgRecv = 7
        ChannelMsgRecv = 8
        CurrTime = 9
        NoMoreMessages = 10
        ExportContact = 11
        BatteryVoltage = 12
        DeviceInfo = 13
        PrivateKey = 14
        Disabled = 15
        ChannelInfo = 18
        SignStart = 19
        Signature = 20

    class PushCodes:
        Advert = 0x80
        PathUpdated = 0x81
        SendConfirmed = 0x82
        MsgWaiting = 0x83
        RawData = 0x84
        LoginSuccess = 0x85
        LoginFail = 0x86
        StatusResponse = 0x87
        LogRxData = 0x88
        TraceData = 0x89
        NewAdvert = 0x8A
        TelemetryResponse = 0x8B
        BinaryResponse = 0x8C

    class ErrorCodes:
        UnsupportedCmd = 1
        NotFound = 2
        TableFull = 3
        BadState = 4
        FileIoError = 5
        IllegalArg = 6

    class AdvType:
        NoneType = 0
        Chat = 1
        Repeater = 2
        Room = 3


class ByteReader:
    def __init__(self, data: bytes):
        self._buf = memoryview(data)
        self._pos = 0

    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def read_bytes(self, count: int) -> bytes:
        data = self._buf[self._pos : self._pos + count].tobytes()
        self._pos += count
        return data

    def read_u8(self) -> int:
        return self.read_bytes(1)[0]

    def read_i8(self) -> int:
        return struct.unpack("<b", self.read_bytes(1))[0]

    def read_u16le(self) -> int:
        return struct.unpack("<H", self.read_bytes(2))[0]

    def read_u32le(self) -> int:
        return struct.unpack("<I", self.read_bytes(4))[0]

    def read_i32le(self) -> int:
        return struct.unpack("<i", self.read_bytes(4))[0]

    def read_string(self) -> str:
        return self.read_bytes(self.remaining()).decode("utf-8", errors="replace")

    def read_cstring(self, max_len: int) -> str:
        raw = self.read_bytes(max_len)
        if b"\x00" in raw:
            raw = raw.split(b"\x00", 1)[0]
        return raw.decode("utf-8", errors="replace")


class ByteWriter:
    def __init__(self):
        self._buf = bytearray()

    def to_bytes(self) -> bytes:
        return bytes(self._buf)

    def write_bytes(self, data: bytes):
        self._buf.extend(data)

    def write_u8(self, value: int):
        self._buf.extend(struct.pack("<B", value & 0xFF))

    def write_i8(self, value: int):
        self._buf.extend(struct.pack("<b", int(value)))

    def write_u16le(self, value: int):
        self._buf.extend(struct.pack("<H", value & 0xFFFF))

    def write_u32le(self, value: int):
        self._buf.extend(struct.pack("<I", value & 0xFFFFFFFF))

    def write_i32le(self, value: int):
        self._buf.extend(struct.pack("<i", int(value)))

    def write_string(self, value: str):
        self._buf.extend(value.encode("utf-8"))

    def write_cstring(self, value: str, max_len: int):
        raw = value.encode("utf-8")[: max_len - 1]
        buf = bytearray(max_len)
        buf[: len(raw)] = raw
        buf[-1] = 0
        self._buf.extend(buf)


@dataclass
class CompanionContact:
    public_key: str
    name: str
    contact_type: int = CompanionProtocol.AdvType.Chat
    flags: int = 0
    out_path: bytearray = field(default_factory=bytearray)
    out_path_len: int = 0
    last_advert: int = 0
    adv_lat: int = 0
    adv_lon: int = 0
    last_mod: int = 0

    @property
    def pubkey_prefix(self) -> bytes:
        return bytes.fromhex(self.public_key)[:6]


class ContactStore:
    def __init__(self):
        self._contacts: Dict[str, CompanionContact] = {}

    @property
    def contacts(self) -> List[CompanionContact]:
        return list(self._contacts.values())

    def get(self, pubkey: str) -> Optional[CompanionContact]:
        return self._contacts.get(pubkey.lower())

    def get_by_prefix(self, prefix: bytes) -> Optional[CompanionContact]:
        for contact in self._contacts.values():
            if contact.pubkey_prefix == prefix:
                return contact
        return None

    def upsert(self, contact: CompanionContact) -> Tuple[CompanionContact, bool]:
        key = contact.public_key.lower()
        existing = self._contacts.get(key)
        if existing:
            self._contacts[key] = contact
            return contact, False
        self._contacts[key] = contact
        return contact, True

    def remove(self, pubkey: str) -> bool:
        return self._contacts.pop(pubkey.lower(), None) is not None

    def list_contacts(self) -> List[CompanionContact]:
        return self.contacts


class WifiEventService:
    def __init__(self, server):
        self._server = server

    def publish_sync(self, event_name: str, data: dict):
        self._server.handle_event(event_name, data)

    def publish(self, event_name: str, data: dict):
        self.publish_sync(event_name, data)


class CompanionSession:
    def __init__(self, server, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, session_id: int):
        self.server = server
        self.reader = reader
        self.writer = writer
        self.session_id = session_id
        self.connected_at = time.time()
        self.remote = writer.get_extra_info("peername")
        self._buffer = bytearray()

    async def run(self):
        try:
            while True:
                data = await self.reader.read(4096)
                if not data:
                    break
                self._buffer.extend(data)
                await self._process_buffer()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(f"Session {self.session_id} error: {exc}")
        finally:
            await self.close()

    async def _process_buffer(self):
        header_len = 3
        while len(self._buffer) >= header_len:
            frame_type = self._buffer[0]
            frame_len = int.from_bytes(self._buffer[1:3], "little")
            if frame_len == 0:
                self._buffer = self._buffer[1:]
                continue
            total_len = header_len + frame_len
            if len(self._buffer) < total_len:
                return
            frame_data = bytes(self._buffer[header_len:total_len])
            self._buffer = self._buffer[total_len:]
            await self.server.handle_frame(self, frame_type, frame_data)

    async def send_frame(self, frame_data: bytes):
        frame = bytearray()
        frame.append(CompanionProtocol.FRAME_INCOMING)
        frame.extend(struct.pack("<H", len(frame_data)))
        frame.extend(frame_data)
        self.writer.write(frame)
        await self.writer.drain()

    async def close(self):
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        self.server.unregister_session(self)

    def to_dict(self) -> dict:
        return {
            "id": self.session_id,
            "remote": str(self.remote),
            "connected_at": int(self.connected_at),
        }


class WiFiCompanionServer:
    def __init__(self, daemon, config: dict):
        self.daemon = daemon
        self.config = config or {}
        wifi_cfg = self.config.get("wifi_client", {})
        self.enabled = wifi_cfg.get("enabled", True)
        self.host = wifi_cfg.get("host", "0.0.0.0")
        self.port = int(wifi_cfg.get("port", 5000))
        self.manual_add_contacts = bool(wifi_cfg.get("manual_add_contacts", False))
        self.include_neighbors = bool(wifi_cfg.get("include_neighbors", True))
        self.max_sessions = int(wifi_cfg.get("max_sessions", 5))

        self.contact_store = ContactStore()
        self.sessions: Dict[int, CompanionSession] = {}
        self._next_session_id = 1
        self._server = None

        self._pending_acks: Dict[int, float] = {}
        self._event_service = WifiEventService(self)

        self.text_handler = TextMessageHandler(
            local_identity=self.daemon.local_identity,
            contacts=self.contact_store,
            log_fn=logger.info,
            send_packet_fn=self._send_packet,
            event_service=self._event_service,
            radio_config=self._get_radio_config(),
        )

        self.ack_handler = AckHandler(logger.info, dispatcher=self.daemon.dispatcher)

    def _get_radio_config(self) -> dict:
        radio = getattr(self.daemon, "radio", None)
        if not radio:
            return self.config.get("radio", {})
        return {
            "spreading_factor": getattr(radio, "spreading_factor", 8),
            "bandwidth": getattr(radio, "bandwidth", 125000),
            "coding_rate": getattr(radio, "coding_rate", 8),
            "preamble_length": getattr(radio, "preamble_length", 17),
            "frequency": getattr(radio, "frequency", 915000000),
            "tx_power": getattr(radio, "tx_power", 14),
        }

    async def start(self):
        if not self.enabled:
            logger.info("WiFi companion disabled via config")
            return
        if self.daemon.dispatcher:
            self.daemon.dispatcher.set_contact_book(self.contact_store)
        self._refresh_contacts_from_storage()
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        logger.info(f"WiFi companion listening on {self.host}:{self.port}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        sessions = list(self.sessions.values())
        for session in sessions:
            await session.close()

    def unregister_session(self, session: CompanionSession):
        self.sessions.pop(session.session_id, None)

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "session_count": len(self.sessions),
            "sessions": [s.to_dict() for s in self.sessions.values()],
            "contacts": len(self.contact_store.contacts),
            "manual_add_contacts": self.manual_add_contacts,
            "include_neighbors": self.include_neighbors,
        }

    def get_contacts_snapshot(self) -> List[dict]:
        return [
            {
                "public_key": c.public_key,
                "name": c.name,
                "type": c.contact_type,
                "last_advert": c.last_advert,
                "last_mod": c.last_mod,
            }
            for c in self.contact_store.contacts
        ]

    def disconnect_session(self, session_id: int) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False
        asyncio.create_task(session.close())
        return True

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if len(self.sessions) >= self.max_sessions:
            writer.close()
            await writer.wait_closed()
            return
        session_id = self._next_session_id
        self._next_session_id += 1
        session = CompanionSession(self, reader, writer, session_id)
        self.sessions[session_id] = session
        logger.info(f"WiFi companion client connected: {session.remote} (id={session_id})")
        await session.run()
        logger.info(f"WiFi companion client disconnected: id={session_id}")

    async def handle_frame(self, session: CompanionSession, frame_type: int, data: bytes):
        if frame_type != CompanionProtocol.FRAME_OUTGOING:
            return
        await self._handle_command(session, data)

    async def _handle_command(self, session: CompanionSession, data: bytes):
        if not data:
            return
        reader = ByteReader(data)
        cmd = reader.read_u8()
        if cmd == CompanionProtocol.CommandCodes.AppStart:
            await self._send_self_info(session)
        elif cmd == CompanionProtocol.CommandCodes.DeviceQuery:
            await self._send_device_info(session)
        elif cmd == CompanionProtocol.CommandCodes.GetDeviceTime:
            await self._send_curr_time(session)
        elif cmd == CompanionProtocol.CommandCodes.SetDeviceTime:
            await self._send_ok(session)
        elif cmd == CompanionProtocol.CommandCodes.GetBatteryVoltage:
            await self._send_battery_voltage(session)
        elif cmd == CompanionProtocol.CommandCodes.GetContacts:
            since = reader.read_u32le() if reader.remaining() >= 4 else None
            await self._send_contacts(session, since)
        elif cmd == CompanionProtocol.CommandCodes.AddUpdateContact:
            await self._handle_add_update_contact(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.RemoveContact:
            await self._handle_remove_contact(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.SendTxtMsg:
            await self._handle_send_txt_msg(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.SendChannelTxtMsg:
            await self._send_disabled(session)
        elif cmd == CompanionProtocol.CommandCodes.SyncNextMessage:
            await self._send_no_more_messages(session)
        elif cmd == CompanionProtocol.CommandCodes.SendSelfAdvert:
            await self._handle_send_self_advert(session)
        elif cmd == CompanionProtocol.CommandCodes.SetAdvertName:
            await self._handle_set_advert_name(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.SetAdvertLatLon:
            await self._handle_set_advert_latlon(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.ExportContact:
            await self._handle_export_contact(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.ImportContact:
            await self._handle_import_contact(session, reader)
        elif cmd == CompanionProtocol.CommandCodes.SetOtherParams:
            await self._handle_set_other_params(session, reader)
        else:
            await self._send_err(session, CompanionProtocol.ErrorCodes.UnsupportedCmd)

    async def _send_ok(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.Ok)
        await session.send_frame(payload.to_bytes())

    async def _send_err(self, session: CompanionSession, err_code: int):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.Err)
        payload.write_u8(err_code)
        await session.send_frame(payload.to_bytes())

    async def _send_disabled(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.Disabled)
        await session.send_frame(payload.to_bytes())

    async def _send_device_info(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.DeviceInfo)
        payload.write_i8(1)  # firmwareVer
        payload.write_bytes(b"\x00" * 6)
        build_date = time.strftime("%d %b %Y")
        payload.write_cstring(build_date, 12)
        payload.write_string("pyMC Repeater WiFi Companion")
        await session.send_frame(payload.to_bytes())

    async def _send_self_info(self, session: CompanionSession):
        pubkey = self.daemon.local_identity.get_public_key() if self.daemon.local_identity else b"\x00" * 32
        repeater_cfg = self.config.get("repeater", {})
        radio_cfg = self.config.get("radio", {})
        adv_lat = int(repeater_cfg.get("latitude", 0.0) * 1_000_000)
        adv_lon = int(repeater_cfg.get("longitude", 0.0) * 1_000_000)
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.SelfInfo)
        payload.write_u8(CompanionProtocol.AdvType.Repeater)
        payload.write_u8(int(radio_cfg.get("tx_power", 14)))
        payload.write_u8(int(radio_cfg.get("tx_power", 14)))
        payload.write_bytes(pubkey[:32])
        payload.write_i32le(adv_lat)
        payload.write_i32le(adv_lon)
        payload.write_bytes(b"\x00" * 3)
        payload.write_u8(1 if self.manual_add_contacts else 0)
        payload.write_u32le(int(radio_cfg.get("frequency", 0)))
        payload.write_u32le(int(radio_cfg.get("bandwidth", 0)))
        payload.write_u8(int(radio_cfg.get("spreading_factor", 8)))
        payload.write_u8(int(radio_cfg.get("coding_rate", 8)))
        payload.write_string(repeater_cfg.get("node_name", "Repeater"))
        await session.send_frame(payload.to_bytes())

    async def _send_curr_time(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.CurrTime)
        payload.write_u32le(int(time.time()))
        await session.send_frame(payload.to_bytes())

    async def _send_battery_voltage(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.BatteryVoltage)
        payload.write_u16le(0)
        await session.send_frame(payload.to_bytes())

    async def _send_no_more_messages(self, session: CompanionSession):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.NoMoreMessages)
        await session.send_frame(payload.to_bytes())

    async def _send_contacts(self, session: CompanionSession, since: Optional[int]):
        self._refresh_contacts_from_storage()
        contacts = self.contact_store.contacts
        if since is not None:
            contacts = [c for c in contacts if c.last_mod > since]
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.ContactsStart)
        payload.write_u32le(len(contacts))
        await session.send_frame(payload.to_bytes())

        for contact in contacts:
            await session.send_frame(self._build_contact_frame(contact))

        most_recent = max((c.last_mod for c in contacts), default=0)
        end_payload = ByteWriter()
        end_payload.write_u8(CompanionProtocol.ResponseCodes.EndOfContacts)
        end_payload.write_u32le(most_recent)
        await session.send_frame(end_payload.to_bytes())

    def _build_contact_frame(self, contact: CompanionContact) -> bytes:
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.Contact)
        payload.write_bytes(bytes.fromhex(contact.public_key))
        payload.write_u8(contact.contact_type)
        payload.write_u8(contact.flags)
        payload.write_i8(contact.out_path_len)
        out_path = bytes(contact.out_path)
        if len(out_path) < 64:
            out_path = out_path + b"\x00" * (64 - len(out_path))
        payload.write_bytes(out_path[:64])
        payload.write_cstring(contact.name or "", 32)
        payload.write_u32le(int(contact.last_advert))
        payload.write_i32le(int(contact.adv_lat))
        payload.write_i32le(int(contact.adv_lon))
        payload.write_u32le(int(contact.last_mod))
        return payload.to_bytes()

    async def _handle_add_update_contact(self, session: CompanionSession, reader: ByteReader):
        pubkey = reader.read_bytes(32).hex()
        contact_type = reader.read_u8()
        flags = reader.read_u8()
        out_path_len = reader.read_i8()
        out_path = reader.read_bytes(64)
        name = reader.read_cstring(32)
        last_advert = reader.read_u32le()
        adv_lat = reader.read_i32le()
        adv_lon = reader.read_i32le()
        now = int(time.time())
        contact = CompanionContact(
            public_key=pubkey,
            name=name or f"{pubkey[:8]}",
            contact_type=contact_type,
            flags=flags,
            out_path=bytearray(out_path[: max(0, out_path_len)]),
            out_path_len=max(0, out_path_len),
            last_advert=last_advert,
            adv_lat=adv_lat,
            adv_lon=adv_lon,
            last_mod=now,
        )
        self.contact_store.upsert(contact)
        await self._send_ok(session)

    async def _handle_remove_contact(self, session: CompanionSession, reader: ByteReader):
        pubkey = reader.read_bytes(32).hex()
        if self.contact_store.remove(pubkey):
            await self._send_ok(session)
        else:
            await self._send_err(session, CompanionProtocol.ErrorCodes.NotFound)

    async def _handle_send_txt_msg(self, session: CompanionSession, reader: ByteReader):
        txt_type = reader.read_u8()
        attempt = reader.read_u8()
        sender_timestamp = reader.read_u32le()
        prefix = reader.read_bytes(6)
        message = reader.read_string().rstrip("\x00")

        contact = self.contact_store.get_by_prefix(prefix)
        if not contact:
            await self._send_err(session, CompanionProtocol.ErrorCodes.NotFound)
            return

        message_type = "direct" if contact.out_path_len > 0 else "flood"
        packet, ack_crc = PacketBuilder.create_text_message(
            contact=contact,
            local_identity=self.daemon.local_identity,
            message=message,
            attempt=attempt,
            message_type=message_type,
            out_path=list(contact.out_path) if contact.out_path_len > 0 else None,
        )

        await self._send_packet(packet, wait_for_ack=False)

        packet_bytes = packet.write_to()
        airtime_ms = PacketTimingUtils.estimate_airtime_ms(len(packet_bytes), self._get_radio_config())
        if message_type == "flood":
            timeout_ms = PacketTimingUtils.calc_flood_timeout_ms(airtime_ms)
        else:
            timeout_ms = PacketTimingUtils.calc_direct_timeout_ms(airtime_ms, contact.out_path_len)

        self._pending_acks[ack_crc] = time.time()

        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.Sent)
        payload.write_i8(0)  # result ok
        payload.write_u32le(ack_crc)
        payload.write_u32le(int(timeout_ms))
        await session.send_frame(payload.to_bytes())

    async def _handle_send_self_advert(self, session: CompanionSession):
        if not self.daemon.send_advert:
            await self._send_err(session, CompanionProtocol.ErrorCodes.BadState)
            return
        try:
            await self.daemon.send_advert()
            await self._send_ok(session)
        except Exception:
            await self._send_err(session, CompanionProtocol.ErrorCodes.BadState)

    async def _handle_set_advert_name(self, session: CompanionSession, reader: ByteReader):
        name = reader.read_string().strip() or "Repeater"
        if "repeater" not in self.config:
            self.config["repeater"] = {}
        self.config["repeater"]["node_name"] = name
        self._persist_config()
        await self._send_ok(session)

    async def _handle_set_advert_latlon(self, session: CompanionSession, reader: ByteReader):
        lat = reader.read_i32le() / 1_000_000
        lon = reader.read_i32le() / 1_000_000
        if "repeater" not in self.config:
            self.config["repeater"] = {}
        self.config["repeater"]["latitude"] = lat
        self.config["repeater"]["longitude"] = lon
        self._persist_config()
        await self._send_ok(session)

    async def _handle_export_contact(self, session: CompanionSession, reader: ByteReader):
        pubkey = reader.read_bytes(32) if reader.remaining() >= 32 else None
        local_pubkey = self.daemon.local_identity.get_public_key() if self.daemon.local_identity else b""
        if pubkey and pubkey != local_pubkey:
            await self._send_err(session, CompanionProtocol.ErrorCodes.NotFound)
            return
        repeater_cfg = self.config.get("repeater", {})
        packet = PacketBuilder.create_advert(
            local_identity=self.daemon.local_identity,
            name=repeater_cfg.get("node_name", "Repeater"),
            lat=repeater_cfg.get("latitude", 0.0),
            lon=repeater_cfg.get("longitude", 0.0),
            feature1=0,
            feature2=0,
            flags=0,
            route_type="flood",
        )
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.ExportContact)
        payload.write_bytes(packet.payload)
        await session.send_frame(payload.to_bytes())

    async def _handle_import_contact(self, session: CompanionSession, reader: ByteReader):
        advert_bytes = reader.read_bytes(reader.remaining())
        try:
            advert = parse_advert_payload(advert_bytes)
            appdata = decode_appdata(advert["appdata"])
            contact = self._contact_from_advert_data(
                advert["pubkey"],
                appdata,
                advert.get("timestamp", int(time.time())),
            )
            self.contact_store.upsert(contact)
            await self._send_ok(session)
        except Exception:
            await self._send_err(session, CompanionProtocol.ErrorCodes.IllegalArg)

    async def _handle_set_other_params(self, session: CompanionSession, reader: ByteReader):
        manual = reader.read_u8()
        self.manual_add_contacts = bool(manual)
        await self._send_ok(session)

    async def _send_packet(self, packet, wait_for_ack: bool = False):
        if not self.daemon.router:
            return False
        return await self.daemon.router.inject_packet(packet, wait_for_ack=wait_for_ack)

    def _persist_config(self):
        config_path = getattr(self.daemon, "config_path", None)
        if not config_path:
            return
        try:
            save_config(self.config, config_path)
        except Exception as exc:
            logger.warning(f"Failed to persist WiFi companion config changes: {exc}")

    def handle_event(self, event_name: str, data: dict):
        if event_name != MeshEvents.NEW_MESSAGE:
            return
        if data.get("is_outgoing"):
            return
        contact_pubkey = data.get("contact_pubkey")
        if not contact_pubkey:
            return
        contact = self.contact_store.get(contact_pubkey)
        if not contact:
            return
        self._send_contact_message_push(contact, data)

    def _send_contact_message_push(self, contact: CompanionContact, data: dict):
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.ResponseCodes.ContactMsgRecv)
        payload.write_bytes(contact.pubkey_prefix)
        hops = data.get("network_info", {}).get("hops", 0)
        payload.write_u8(int(hops))
        payload.write_u8(0)  # txt_type plain
        payload.write_u32le(int(data.get("timestamp", time.time())))
        payload.write_string(data.get("message_text", ""))
        asyncio.create_task(self._broadcast(payload.to_bytes()))

    async def _broadcast(self, frame_data: bytes):
        for session in list(self.sessions.values()):
            try:
                await session.send_frame(frame_data)
            except Exception:
                continue

    def _refresh_contacts_from_storage(self):
        if not self.include_neighbors:
            return
        storage = getattr(self.daemon.repeater_handler, "storage", None)
        if not storage:
            return
        neighbors = storage.get_neighbors()
        for pubkey, info in neighbors.items():
            contact = self._contact_from_neighbor(pubkey, info)
            self.contact_store.upsert(contact)

    def _contact_from_neighbor(self, pubkey: str, info: dict) -> CompanionContact:
        contact_type = self._map_contact_type(info.get("contact_type"), info.get("is_repeater"))
        flags = 0
        if info.get("latitude") is not None and info.get("longitude") is not None:
            flags |= ADVERT_FLAG_HAS_LOCATION
        if info.get("node_name"):
            flags |= ADVERT_FLAG_HAS_NAME
        if contact_type == CompanionProtocol.AdvType.Chat:
            flags |= ADVERT_FLAG_IS_CHAT_NODE
        if contact_type == CompanionProtocol.AdvType.Repeater:
            flags |= ADVERT_FLAG_IS_REPEATER
        if contact_type == CompanionProtocol.AdvType.Room:
            flags |= ADVERT_FLAG_IS_ROOM_SERVER

        last_seen = int(info.get("last_seen", time.time()))
        lat = info.get("latitude") or 0.0
        lon = info.get("longitude") or 0.0
        return CompanionContact(
            public_key=pubkey,
            name=info.get("node_name") or f"{pubkey[:8]}",
            contact_type=contact_type,
            flags=flags,
            out_path=bytearray(),
            out_path_len=0,
            last_advert=last_seen,
            adv_lat=int(lat * 1_000_000),
            adv_lon=int(lon * 1_000_000),
            last_mod=last_seen,
        )

    def _contact_from_advert_data(self, pubkey: str, appdata: dict, timestamp: int) -> CompanionContact:
        flags = appdata.get("flags", 0)
        contact_type = self._contact_type_from_flags(flags)
        name = appdata.get("node_name") or f"{pubkey[:8]}"
        lat = appdata.get("latitude", 0.0)
        lon = appdata.get("longitude", 0.0)
        return CompanionContact(
            public_key=pubkey,
            name=name,
            contact_type=contact_type,
            flags=flags,
            out_path=bytearray(),
            out_path_len=0,
            last_advert=timestamp,
            adv_lat=int(lat * 1_000_000),
            adv_lon=int(lon * 1_000_000),
            last_mod=timestamp,
        )

    def _map_contact_type(self, contact_type: Optional[str], is_repeater: Optional[bool]) -> int:
        if contact_type:
            lowered = contact_type.lower()
            if "room" in lowered:
                return CompanionProtocol.AdvType.Room
            if "repeater" in lowered:
                return CompanionProtocol.AdvType.Repeater
            if "chat" in lowered:
                return CompanionProtocol.AdvType.Chat
        if is_repeater:
            return CompanionProtocol.AdvType.Repeater
        return CompanionProtocol.AdvType.Chat

    def _contact_type_from_flags(self, flags: int) -> int:
        if flags & ADVERT_FLAG_IS_ROOM_SERVER:
            return CompanionProtocol.AdvType.Room
        if flags & ADVERT_FLAG_IS_REPEATER:
            return CompanionProtocol.AdvType.Repeater
        if flags & ADVERT_FLAG_IS_CHAT_NODE:
            return CompanionProtocol.AdvType.Chat
        return CompanionProtocol.AdvType.Chat

    async def process_packet(self, packet):
        if not self.enabled:
            return
        payload_type = packet.get_payload_type()
        if payload_type == PAYLOAD_TYPE_TXT_MSG:
            await self.text_handler(packet)
        elif payload_type == PAYLOAD_TYPE_ACK:
            await self._handle_ack_packet(packet)
        elif payload_type == PAYLOAD_TYPE_PATH:
            await self._handle_path_packet(packet)
        elif payload_type == PAYLOAD_TYPE_ADVERT:
            await self._handle_advert_packet(packet)

    async def _handle_ack_packet(self, packet):
        ack_crc = await self.ack_handler.process_discrete_ack(packet)
        if ack_crc is None:
            return
        self._notify_ack_received(ack_crc)

    async def _handle_path_packet(self, packet):
        ack_crc = await self.ack_handler.process_path_ack_variants(packet)
        if ack_crc is None:
            return
        self._notify_ack_received(ack_crc)

    async def _handle_advert_packet(self, packet):
        try:
            advert = parse_advert_payload(packet.payload)
            appdata = decode_appdata(advert["appdata"])
            contact = self._contact_from_advert_data(
                advert["pubkey"],
                appdata,
                advert.get("timestamp", int(time.time())),
            )
            _, is_new = self.contact_store.upsert(contact)
            if is_new:
                await self._send_advert_push(contact)
        except Exception as exc:
            logger.debug(f"Failed to parse advert for WiFi companion: {exc}")

    def _notify_ack_received(self, ack_crc: int):
        sent_at = self._pending_acks.pop(ack_crc, None)
        if sent_at is None:
            return
        round_trip = int((time.time() - sent_at) * 1000)
        payload = ByteWriter()
        payload.write_u8(CompanionProtocol.PushCodes.SendConfirmed)
        payload.write_u32le(ack_crc)
        payload.write_u32le(round_trip)
        asyncio.create_task(self._broadcast(payload.to_bytes()))

    async def _send_advert_push(self, contact: CompanionContact):
        if self.manual_add_contacts:
            payload = ByteWriter()
            payload.write_u8(CompanionProtocol.PushCodes.NewAdvert)
            payload.write_bytes(bytes.fromhex(contact.public_key))
            payload.write_u8(contact.contact_type)
            payload.write_u8(contact.flags)
            payload.write_i8(contact.out_path_len)
            out_path = bytes(contact.out_path)
            if len(out_path) < 64:
                out_path = out_path + b"\x00" * (64 - len(out_path))
            payload.write_bytes(out_path[:64])
            payload.write_cstring(contact.name, 32)
            payload.write_u32le(int(contact.last_advert))
            payload.write_i32le(int(contact.adv_lat))
            payload.write_i32le(int(contact.adv_lon))
            payload.write_u32le(int(contact.last_mod))
            await self._broadcast(payload.to_bytes())
        else:
            payload = ByteWriter()
            payload.write_u8(CompanionProtocol.PushCodes.Advert)
            payload.write_bytes(bytes.fromhex(contact.public_key))
            await self._broadcast(payload.to_bytes())
