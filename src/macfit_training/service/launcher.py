"""Do not execute GPU code until the supervisor durably records our PID identity."""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-fd", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        permission = os.read(arguments.gate_fd, 1)
    finally:
        os.close(arguments.gate_fd)
    if permission != b"1" or not command:
        raise SystemExit(125)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
