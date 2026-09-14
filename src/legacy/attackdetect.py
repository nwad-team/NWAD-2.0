from scapy.all import *
import logging
import sys
import signal

logging.basicConfig(filename='attackdetect.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger()
handler = logging.FileHandler('attackdetect.log')
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

syn_count = {}
packet_count = 0
THRESHOLD = 100
DDOS_THRESHOLD = 100

attack_detected = False
sniffer = None

def capture_traffic(interface="Wi-Fi"):
    global sniffer
    print(f"Starting sniffer on interface: {interface}")
    sniffer = AsyncSniffer(prn=analyze_packet, store=0, iface=interface)
    print("Sniffer started")
    sniffer.start()
    sniffer.join()

def analyze_packet(packet):
    print(f"Packet captured: {packet.summary()} (Length: {len(packet)})")
    if packet.haslayer(IP):
        if packet.haslayer(TCP) and packet[TCP].flags == "S":
            detect_syn_flood(packet)
        detect_ddos(packet)
    elif packet.haslayer(ICMP):
        print(f"ICMP packet detected: {packet.summary()} (Length: {len(packet)})")
    else:
        print(f"Non-IP packet detected: {packet.summary()}")


def detect_syn_flood(packet):
    global attack_detected
    src_ip = packet[IP].src
    if src_ip not in syn_count:
        syn_count[src_ip] = 0
    syn_count[src_ip] += 1
    if syn_count[src_ip] > THRESHOLD:
        log_attack(f"SYN Flood attack detected from IP: {src_ip}")
        attack_detected = True

def detect_ddos(packet):
    global packet_count
    global attack_detected
    packet_count += 1
    if packet_count > DDOS_THRESHOLD:
        log_attack("Potential DDoS attack detected")
        attack_detected = True

def log_attack(message):
    logging.info(message)
    print(f"Logging attack: {message}")
    stop_sniffing()
    print("Attack logged and processed")
    logger.handlers[0].flush()

def handle_signal(signal, frame):
    print("Script stopped by user.")
    stop_sniffing()
    sys.exit(0)

def stop_sniffing():
    global sniffer
    if sniffer and sniffer.running:
        sniffer.stop()
    print("Sniffer stopped")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    try:
        capture_traffic(interface="Wi-Fi")
    except KeyboardInterrupt:
        handle_signal(None, None)