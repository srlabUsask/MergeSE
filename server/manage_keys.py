#!/usr/bin/env python3
"""Offline admin CLI for MergeSE API keys.

Mint, list, and revoke API keys directly against the auth database, without the
server running. The plaintext key is printed exactly once at mint time; only
its hash is stored.

Examples
--------
    # Mint a standard researcher key
    python server/manage_keys.py mint --email alice@uni.edu

    # Mint a higher-quota collaborator key
    python server/manage_keys.py mint --tier approved --daily-limit 500

    # List and revoke
    python server/manage_keys.py list
    python server/manage_keys.py revoke cli_1a2b3c4d5e6f7a8b

The database path and signing secret follow the same env vars the server uses
(MERGESE_AUTH_DB, MERGESE_AUTH_SECRET), so point them at the same values.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auth  # noqa: E402


def _store() -> auth.AuthStore:
    db = Path(os.environ.get(
        "MERGESE_AUTH_DB",
        str(Path(__file__).resolve().parent.parent / "artifacts" / "_auth" / "auth.db")))
    secret = auth.load_secret(os.environ.get("MERGESE_AUTH_SECRET"),
                              db.parent / "signing.secret")
    return auth.AuthStore(db, secret)


def cmd_mint(args) -> None:
    store = _store()
    client_id, plaintext = store.mint_key(args.email, args.tier, args.daily_limit)
    print("client_id :", client_id)
    print("api_key   :", plaintext)
    print("tier      :", args.tier)
    print("\nStore the api_key now - it is not recoverable.")


def cmd_revoke(args) -> None:
    ok = _store().revoke_key(args.client_id)
    print("revoked" if ok else "no such client_id")
    sys.exit(0 if ok else 1)


def cmd_list(args) -> None:
    store = _store()
    rows = store._conn.execute(
        "SELECT client_id, email, tier, daily_limit, created_at, last_used_at, "
        "disabled FROM api_clients ORDER BY created_at DESC").fetchall()
    if not rows:
        print("(no keys)")
        return
    for r in rows:
        flag = "DISABLED" if r["disabled"] else "active"
        print(f"{r['client_id']}  {flag:8}  tier={r['tier']:8}  "
              f"limit={r['daily_limit']}  email={r['email']}")


def main() -> None:
    p = argparse.ArgumentParser(description="MergeSE API key admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mint", help="mint a new API key")
    m.add_argument("--email", default=None)
    m.add_argument("--tier", default="key", choices=sorted(auth.TIERS))
    m.add_argument("--daily-limit", type=int, default=None,
                   help="override the tier's daily job budget")
    m.set_defaults(func=cmd_mint)

    r = sub.add_parser("revoke", help="revoke a key by client_id")
    r.add_argument("client_id")
    r.set_defaults(func=cmd_revoke)

    l = sub.add_parser("list", help="list keys")
    l.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
