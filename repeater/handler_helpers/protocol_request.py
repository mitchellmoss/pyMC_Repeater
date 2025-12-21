"""
Protocol request (REQ) handling helper for pyMC Repeater.

Provides repeater-specific callbacks for status and telemetry requests.
"""

import asyncio
import logging
import struct
import time

from pymc_core.node.handlers.protocol_request import (
    ProtocolRequestHandler,
    REQ_TYPE_GET_STATUS,
    REQ_TYPE_GET_TELEMETRY_DATA,
    REQ_TYPE_GET_ACCESS_LIST,
    REQ_TYPE_GET_NEIGHBOURS,
    SERVER_RESPONSE_DELAY_MS
)

logger = logging.getLogger("ProtocolRequestHelper")

TELEM_CHANNEL_SELF = 1
LPP_PERCENTAGE = 120
UPS_BATCAP_CACHE = "/var/lib/ups-lowpower/last_batcap"


class ProtocolRequestHelper:
    """Provides repeater-specific protocol request handlers."""
    
    def __init__(self, identity_manager, packet_injector=None, acl_dict=None, radio=None, engine=None, neighbor_tracker=None):

        self.identity_manager = identity_manager
        self.packet_injector = packet_injector
        self.acl_dict = acl_dict or {}
        self.radio = radio
        self.engine = engine
        self.neighbor_tracker = neighbor_tracker
        
        # Dictionary of core handlers keyed by dest_hash
        self.handlers = {}
        
    def register_identity(self, name: str, identity, identity_type: str = "repeater"):

        hash_byte = identity.get_public_key()[0]
        
        # Get ACL for this identity
        identity_acl = self.acl_dict.get(hash_byte)
        if not identity_acl:
            logger.warning(f"Cannot register identity '{name}': no ACL for hash 0x{hash_byte:02X}")
            return
        
        # Create ACL contacts wrapper
        acl_contacts = self._create_acl_contacts_wrapper(identity_acl)
        
        # Build request handlers dict
        request_handlers = {
            REQ_TYPE_GET_STATUS: self._handle_get_status,
            REQ_TYPE_GET_TELEMETRY_DATA: self._handle_get_telemetry_data,
        }
        
        # Create core handler
        handler = ProtocolRequestHandler(
            local_identity=identity,
            contacts=acl_contacts,
            get_client_fn=lambda src_hash: self._get_client_from_acl(identity_acl, src_hash),
            request_handlers=request_handlers,
            log_fn=logger.info,
        )
        
        self.handlers[hash_byte] = {
            "handler": handler,
            "identity": identity,
            "name": name,
            "type": identity_type,
        }
        
        logger.info(f"Registered protocol request handler for '{name}': hash=0x{hash_byte:02X}")
    
    def _create_acl_contacts_wrapper(self, acl):
        """Create contacts wrapper from ACL."""
        class ACLContactsWrapper:
            def __init__(self, identity_acl):
                self._acl = identity_acl
            
            @property
            def contacts(self):
                return self._acl.get_all_clients()
        
        return ACLContactsWrapper(acl)
    
    def _get_client_from_acl(self, acl, src_hash: int):
        """Get client from ACL by source hash."""
        for client_info in acl.get_all_clients():
            if client_info.id.get_public_key()[0] == src_hash:
                return client_info
        return None
    
    async def process_request_packet(self, packet):

        try:
            if len(packet.payload) < 2:
                return False
            
            dest_hash = packet.payload[0]
            
            handler_info = self.handlers.get(dest_hash)
            if not handler_info:
                return False
            
            # Let core handler build response
            response_packet = await handler_info["handler"](packet)
            
            # Send response after delay
            if response_packet and self.packet_injector:
                await asyncio.sleep(SERVER_RESPONSE_DELAY_MS / 1000.0)
                await self.packet_injector(response_packet, wait_for_ack=False)
            
            packet.mark_do_not_retransmit()
            return True
            
        except Exception as e:
            logger.error(f"Error processing protocol request: {e}", exc_info=True)
            return False
    
    def _handle_get_status(self, client, timestamp: int, req_data: bytes):

        # C++ struct RepeaterStats (44 bytes total):
        # uint16_t batt_milli_volts;
        # uint16_t curr_tx_queue_len;
        # int16_t noise_floor;
        # int16_t last_rssi;
        # uint32_t n_packets_recv;
        # uint32_t n_packets_sent;
        # uint32_t total_air_time_secs;
        # uint32_t total_up_time_secs;
        # uint32_t n_sent_flood;
        # uint32_t n_sent_direct;
        # uint32_t n_recv_flood;
        # uint32_t n_recv_direct;
        # uint32_t err_events;
        # int16_t last_snr;
        # uint32_t n_direct_dups;
        # uint32_t n_flood_dups;
        # uint32_t total_rx_air_time_secs;
        
        # Get stats from radio/engine
        noise_floor = int(self.radio.get_noise_floor() * 1.0) if self.radio else -120
        last_rssi = int(self.radio.last_rssi) if self.radio and hasattr(self.radio, 'last_rssi') else -120
        last_snr = int((self.radio.last_snr * 4.0) if self.radio and hasattr(self.radio, 'last_snr') else 0)
        
        # Get packet counts
        n_packets_recv = self.radio.packets_received if self.radio and hasattr(self.radio, 'packets_received') else 0
        n_packets_sent = self.radio.packets_sent if self.radio and hasattr(self.radio, 'packets_sent') else 0
        
        # Get airtime stats
        total_air_time_secs = 0
        total_rx_air_time_secs = 0
        if self.engine and hasattr(self.engine, 'airtime_manager'):
            total_air_time_secs = int(self.engine.airtime_manager.total_tx_airtime_ms / 1000)
        
        # Get routing stats
        n_sent_flood = 0
        n_sent_direct = 0
        n_recv_flood = 0
        n_recv_direct = 0
        n_direct_dups = 0
        n_flood_dups = 0
        
        if self.engine:
            n_sent_flood = getattr(self.engine, 'sent_flood_count', 0)
            n_sent_direct = getattr(self.engine, 'sent_direct_count', 0)
            n_recv_flood = getattr(self.engine, 'recv_flood_count', 0)
            n_recv_direct = getattr(self.engine, 'recv_direct_count', 0)
            n_direct_dups = getattr(self.engine, 'direct_dup_count', 0)
            n_flood_dups = getattr(self.engine, 'flood_dup_count', 0)
        
        # Pack struct (little-endian)
        stats = struct.pack(
            '<HHhhIIIIIIIIIhIII',
            0,  # batt_milli_volts (not available on Pi)
            0,  # curr_tx_queue_len (TODO)
            noise_floor,
            last_rssi,
            n_packets_recv,
            n_packets_sent,
            total_air_time_secs,
            int(time.time()),  # total_up_time_secs
            n_sent_flood,
            n_sent_direct,
            n_recv_flood,
            n_recv_direct,
            0,  # err_events
            last_snr,
            n_direct_dups,
            n_flood_dups,
            total_rx_air_time_secs,
        )
        
        logger.debug(f"GET_STATUS: noise={noise_floor}dBm, rssi={last_rssi}dBm, snr={last_snr/4}dB")
        
        return stats

    def _handle_get_telemetry_data(self, client, timestamp: int, req_data: bytes):
        percent = self._read_ups_battery_percent()
        if percent is None:
            logger.info("REQ_TYPE_GET_TELEMETRY_DATA no UPS battery data available")
            return b""

        percent = max(0, min(100, int(percent)))
        return bytes((TELEM_CHANNEL_SELF, LPP_PERCENTAGE, percent))

    def _read_ups_battery_percent(self):
        try:
            with open(UPS_BATCAP_CACHE, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except FileNotFoundError:
            logger.debug("UPS battery cache missing at %s", UPS_BATCAP_CACHE)
            return None
        except OSError as exc:
            logger.debug("UPS battery cache read failed: %s", exc)
            return None

        for line in lines:
            if line.startswith("batcap="):
                value = line.split("=", 1)[1].strip()
                try:
                    return int(float(value))
                except ValueError:
                    logger.debug("UPS battery cache parse failed: %s", line)
                    return None

        logger.debug("UPS battery cache missing batcap entry")
        return None
