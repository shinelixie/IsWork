#!/usr/bin/env python3
"""Bidirectional TCP-over-ttyd tunnel.

One long-lived ttyd WebSocket carries a multiplexed, full-duplex byte stream.
A small Python agent runs on the remote side inside the raw PTY; it dials
remote targets for --forward rules and listens on remote ports for --reverse
rules.  All listeners bind to 127.0.0.1 by default.

Usage:
    python3 -I /tmp/ttyd_tunnel.py \
        --forward 127.0.0.1:18888=127.0.0.1:8888 \
        --reverse 127.0.0.1:18000=127.0.0.1:19999

Frame layout (all integers big-endian):
    type:1  conn_id:4  length:4  payload:length
    O open      (payload JSON {"host","port"} for forward, {"rule":N} for reverse)
    D data
    C close
    E error     (payload UTF-8 message)
    R ready     (agent -> local, payload JSON listener status)
"""
import argparse
import asyncio
import base64
import json
import os
import struct
import sys

import websockets

DEFAULT_WS = "ws://210.38.241.24:11026/ws"
DEFAULT_ORIGIN = "http://210.38.241.24:11026"
HDR = struct.Struct("!BII")
MAX_FRAME = 16384
CHUNK = 8192

O, D, C, E, R = (ord(x) for x in "ODCER")

AGENT_SRC = r'''
import errno, json, os, selectors, socket, struct, termios, tty

CFG = __CFG__

HDR = struct.Struct("!BII")
O, D, C, E, R = (ord(x) for x in "ODCER")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
MAX_DATA = 16384

# Bash is gone (exec), but a startup file or pty setting may still leave the
# line discipline non-raw.  Re-apply it and verify the flags we care about.
try:
    tty.setraw(0, termios.TCSANOW)
except Exception:
    pass

def raw_ok():
    try:
        a = termios.tcgetattr(0)
    except Exception:
        return False
    ifl, ofl, lfl = a[0], a[1], a[3]
    bad = ifl & (termios.IXON | termios.ICRNL | termios.ISTRIP |
                 termios.INLCR | termios.IGNCR | termios.BRKINT | termios.PARMRK)
    bad |= ofl & termios.OPOST
    bad |= lfl & (termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
    return not bad

def pack(t, cid, payload=b""):
    return HDR.pack(t, cid, len(payload)) + payload

class Conn:
    __slots__ = ("cid", "sock", "connected", "out", "rule", "closing")
    def __init__(self, cid, sock, connected):
        self.cid, self.sock, self.connected = cid, sock, connected
        self.out = bytearray()
        self.rule = None
        self.closing = False

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sel = selectors.DefaultSelector()
        self.out = bytearray()
        self.inbuf = bytearray()
        self.conns = {}
        self.next_id = 0x80000000
        self.stdout_on = False
        self.running = True
        os.set_blocking(0, False)
        os.set_blocking(1, False)
        self.sel.register(0, selectors.EVENT_READ, ("stdin", None))

    def sync_stdout(self):
        if self.out and not self.stdout_on:
            self.sel.register(1, selectors.EVENT_WRITE, ("stdout", None))
            self.stdout_on = True
        elif not self.out and self.stdout_on:
            self.sel.unregister(1)
            self.stdout_on = False

    def send(self, t, cid, payload=b""):
        self.out += pack(t, cid, payload)
        self.sync_stdout()

    def flush(self):
        while self.out:
            try:
                n = os.write(1, self.out)
            except BlockingIOError:
                break
            except OSError:
                self.running = False
                break
            if n <= 0:
                self.running = False
                break
            del self.out[:n]
        self.sync_stdout()

    def feed_stdin(self):
        while True:
            try:
                data = os.read(0, 65536)
            except BlockingIOError:
                return
            except OSError:
                self.running = False
                return
            if not data:
                self.running = False
                return
            self.inbuf += data
            self.parse()

    def parse(self):
        while len(self.inbuf) >= HDR.size:
            t, cid, ln = HDR.unpack_from(self.inbuf)
            if ln > MAX_DATA:
                self.running = False
                return
            if len(self.inbuf) < HDR.size + ln:
                return
            payload = bytes(self.inbuf[HDR.size:HDR.size + ln])
            del self.inbuf[:HDR.size + ln]
            self.dispatch(t, cid, payload)

    def dispatch(self, t, cid, payload):
        if t == O:
            self.open_forward(cid, payload)
        elif t == D:
            c = self.conns.get(cid)
            if c is not None:
                # Queue even while a non-blocking connect is still in flight.
                c.out += payload
                self.sync_conn(c)
        elif t == C:
            c = self.conns.get(cid)
            if c is None:
                return
            if c.out:
                # The response bytes may still be queued: flush, then close.
                c.closing = True
                self.sync_conn(c)
            else:
                self.conns.pop(cid, None)
                self.close_conn(c)

    def open_forward(self, cid, payload):
        try:
            req = json.loads(payload.decode())
            host, port = req["host"], int(req["port"])
        except Exception:
            self.send(E, cid, b"bad open frame")
            return
        if host not in LOOPBACK and host not in self.cfg.get("allow_hosts", []):
            self.send(E, cid, ("target not allowed: %s" % host).encode())
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setblocking(False)
            err = s.connect_ex((host, port))
        except OSError as exc:
            self.send(E, cid, ("connect: %s" % exc).encode())
            return
        c = Conn(cid, s, err == 0)
        self.conns[cid] = c
        if err not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK):
            self.send(E, cid, ("connect failed: %s" % os.strerror(err)).encode())
            self.conns.pop(cid, None)
            self.close_conn(c)
            return
        self.sync_conn(c)

    def sync_conn(self, c):
        try:
            self.sel.unregister(c.sock)
        except (KeyError, ValueError):
            pass
        events = 0
        if c.connected and not c.closing:
            events |= selectors.EVENT_READ
        if (not c.connected) or c.out:
            events |= selectors.EVENT_WRITE
        if events:
            self.sel.register(c.sock, events, ("sock", c))

    def close_conn(self, c):
        try:
            self.sel.unregister(c.sock)
        except (KeyError, ValueError):
            pass
        try:
            c.sock.close()
        except OSError:
            pass

    def handle_sock(self, c, mask):
        if not c.connected:
            err = c.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err:
                self.send(E, c.cid, ("connect: %s" % os.strerror(err)).encode())
                self.conns.pop(c.cid, None)
                self.close_conn(c)
                return
            c.connected = True
        if mask & selectors.EVENT_READ:
            try:
                data = c.sock.recv(8192)
            except BlockingIOError:
                data = None
            except OSError:
                data = b""
            if data == b"":
                self.conns.pop(c.cid, None)
                self.send(C, c.cid)
                self.close_conn(c)
                return
            if data:
                self.send(D, c.cid, data)
        if c.out and (mask & selectors.EVENT_WRITE):
            try:
                n = c.sock.send(c.out)
            except BlockingIOError:
                n = 0
            except OSError:
                n = -1
            if n < 0:
                self.conns.pop(c.cid, None)
                self.send(E, c.cid, b"send failed")
                self.close_conn(c)
                return
            del c.out[:n]
            if c.closing and not c.out:
                self.conns.pop(c.cid, None)
                self.close_conn(c)
                return
        self.sync_conn(c)

    def handle_listen(self, rule, lsock):
        while True:
            try:
                s, _ = lsock.accept()
            except (BlockingIOError, OSError):
                return
            s.setblocking(False)
            cid = self.next_id
            self.next_id += 1
            if self.next_id >= 0xFFFFFFFF:
                self.next_id = 0x80000000
            c = Conn(cid, s, True)
            c.rule = rule
            self.conns[cid] = c
            self.send(O, cid, json.dumps({"rule": rule}).encode())
            self.sync_conn(c)

    def start(self):
        infos = []
        for i, r in enumerate(self.cfg.get("reverse", [])):
            host = r.get("bind_host", "127.0.0.1")
            port = int(r["bind_port"])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                s.listen(128)
                s.setblocking(False)
                self.sel.register(s, selectors.EVENT_READ, ("listen", (i, s)))
                infos.append({"rule": i, "ok": True, "addr": "%s:%d" % (host, port)})
            except OSError as exc:
                infos.append({"rule": i, "ok": False, "error": str(exc)})
                s.close()
        self.send(R, 0, json.dumps(
            {"listeners": infos, "pid": os.getpid(), "raw": raw_ok()}).encode())

def main():
    agent = Agent(CFG)
    agent.start()
    while agent.running:
        try:
            events = agent.sel.select(1.0)
        except OSError:
            break
        for key, mask in events:
            kind, obj = key.data
            if kind == "stdin":
                agent.feed_stdin()
            elif kind == "stdout":
                agent.flush()
            elif kind == "listen":
                agent.handle_listen(obj[0], obj[1])
            else:
                agent.handle_sock(obj, mask)
            if not agent.running:
                break
        agent.flush()

main()
'''


def log(*a):
    print("[ttyd-tunnel]", *a, file=sys.stderr, flush=True)


def parse_addr(text):
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("expected host:port, got %r" % text)
    return host, int(port)


def parse_rule(text):
    bind, sep, target = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected LOCAL=REMOTE, got %r" % text)
    return parse_addr(bind), parse_addr(target)


class LocalConn:
    __slots__ = ("cid", "reader", "writer", "queue", "closed", "removed", "tasks")

    def __init__(self, cid, reader=None, writer=None):
        self.cid = cid
        self.reader = reader
        self.writer = writer
        self.queue = asyncio.Queue()
        self.closed = False
        self.removed = False
        self.tasks = []


class Controller:
    def __init__(self, args):
        self.args = args
        self.ws = None
        self.send_lock = asyncio.Lock()
        self.conns = {}
        self.next_cid = 1
        self.inbuf = bytearray()
        self.ready = asyncio.Event()
        self.agent_raw = None
        self.reader_task = None
        self.servers = []
        self.forward_rules = args.forward or []
        self.reverse_rules = args.reverse or []

    def alloc_cid(self):
        cid = self.next_cid
        self.next_cid += 1
        if self.next_cid >= 0x7FFFFFFF:
            self.next_cid = 1
        return cid

    async def send_frame(self, t, cid, payload=b""):
        frame = HDR.pack(t, cid, len(payload)) + payload
        async with self.send_lock:
            await self.ws.send(b"0" + frame)

    async def connect_and_start_agent(self):
        nonce = os.urandom(6).hex()
        self.ws = await websockets.connect(
            self.args.ws, subprotocols=["tty"], origin=self.args.origin,
            proxy=None, compression=None, max_size=2 * 1024 * 1024,
            open_timeout=10, ping_interval=20, ping_timeout=20)
        await self.ws.send(json.dumps({"AuthToken": "", "columns": 200, "rows": 50}))
        await asyncio.sleep(0.25)

        # The echoed command must not contain the full marker: printf does the join.
        det = ("stty -echo 2>/dev/null; "
               "command -v python3 >/dev/null 2>&1 && printf '__HAS_PY_%s__\\n' " + nonce +
               "; printf '__DET_%s__\\n' " + nonce + "\n")
        await self.ws.send(b"0" + det.encode())
        out = await self.read_until(("__DET_%s__" % nonce).encode())
        if ("__HAS_PY_%s__" % nonce).encode() not in out:
            raise RuntimeError("remote has no python3:\n" + out.decode("utf-8", "replace")[-1500:])

        cfg = {
            "reverse": [
                {"bind_host": bind[0], "bind_port": bind[1], "rule": i}
                for i, (bind, _t) in enumerate(self.reverse_rules)
            ],
            "allow_hosts": ["127.0.0.1", "localhost", "::1"],
        }
        tmp_b64 = "/tmp/.ttyd_agent.b64"
        tmp_py = "/tmp/.ttyd_agent.py"
        src = AGENT_SRC.replace("__CFG__", repr(cfg))
        # The agent removes its own temporary files as soon as it starts.
        src = ("import os\n"
               "for _p in (%r, %r):\n"
               "    try:\n"
               "        os.unlink(_p)\n"
               "    except OSError:\n"
               "        pass\n" % (tmp_py, tmp_b64)) + src
        b64 = base64.b64encode(src.encode()).decode()

        # Upload in short lines: a canonical-mode PTY truncates lines at ~4096.
        await self.send_shell("rm -f %s %s" % (tmp_b64, tmp_py))
        await asyncio.sleep(0.15)
        for i in range(0, len(b64), 2000):
            await self.send_shell("printf '%%s' '%s' >> %s" % (b64[i:i + 2000], tmp_b64))
            await asyncio.sleep(0.03)
        await self.send_shell("base64 -d %s > %s && rm -f %s" % (tmp_b64, tmp_py, tmp_b64))
        await asyncio.sleep(0.4)

        mark = "__TUNNEL_%s__" % nonce
        fail = "__TUNNELFAIL_%s__" % nonce
        line = ("if [ -s " + tmp_py + " ]; then stty raw -echo 2>/dev/null; "
                "printf '__TUNNEL_%s__\\n' " + nonce + "; exec python3 -u " + tmp_py +
                " 2>/dev/null; else printf '__TUNNELFAIL_%s__\\n' " + nonce + "; fi")
        await self.send_shell(line)
        out = await self.read_until_any([mark.encode(), fail.encode()], timeout=20.0)
        if fail.encode() in out:
            raise RuntimeError("remote agent upload failed; tail=%r" % out[-800:])
        idx = out.find(mark.encode())
        rest = out[idx + len(mark):].lstrip(b"\r\n")
        self.inbuf += rest
        self.parse()

        # The agent's R frame may arrive after the marker, so the reader must
        # be running before we wait for it.
        self.reader_task = asyncio.create_task(self.ws_reader())
        try:
            await asyncio.wait_for(self.ready.wait(), timeout=20)
        except asyncio.TimeoutError:
            raise RuntimeError("remote agent did not become ready (no R frame)")
        if self.agent_raw is False:
            raise RuntimeError("remote PTY is not in raw mode; refusing to tunnel binary data")

    async def send_shell(self, line):
        await self.ws.send(b"0" + (line + "\n").encode())

    async def read_until(self, marker, timeout=10.0):
        return await self.read_until_any([marker], timeout)

    async def read_until_any(self, markers, timeout=10.0):
        buf = b""
        end = asyncio.get_running_loop().time() + timeout
        while not any(m in buf for m in markers):
            left = end - asyncio.get_running_loop().time()
            if left <= 0:
                raise TimeoutError("none of %r seen; tail=%r"
                                   % (markers, bytes(buf[-600:])))
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=left)
            except asyncio.TimeoutError:
                raise TimeoutError("none of %r seen; tail=%r"
                                   % (markers, bytes(buf[-600:])))
            raw = msg.encode() if isinstance(msg, str) else msg
            if raw[:1] == b"0":
                buf += raw[1:]
        return buf

    def parse(self):
        while len(self.inbuf) >= HDR.size:
            t, cid, ln = HDR.unpack_from(self.inbuf)
            if ln > MAX_FRAME:
                log("frame too large (%d) - dropping buffer" % ln)
                self.inbuf.clear()
                return
            if len(self.inbuf) < HDR.size + ln:
                return
            payload = bytes(self.inbuf[HDR.size:HDR.size + ln])
            del self.inbuf[:HDR.size + ln]
            self.on_frame(t, cid, payload)

    def on_frame(self, t, cid, payload):
        if t == R:
            try:
                info = json.loads(payload.decode())
            except Exception:
                info = {}
            self.agent_raw = info.get("raw")
            log("agent ready: pid=%s raw=%s listeners=%s"
                % (info.get("pid"), self.agent_raw, info.get("listeners")))
            self.ready.set()
            return
        if t == O:
            try:
                rule = int(json.loads(payload.decode())["rule"])
            except Exception:
                log("bad reverse open for cid %d" % cid)
                return
            conn = LocalConn(cid)
            self.conns[cid] = conn
            conn.tasks.append(asyncio.create_task(self.reverse_open(conn, rule)))
            return
        conn = self.conns.get(cid)
        if conn is None:
            return
        if t == D:
            conn.queue.put_nowait(payload)
        elif t == C:
            conn.queue.put_nowait(None)
        elif t == E:
            log("remote error cid=%d: %s" % (cid, payload.decode("utf-8", "replace")))
            conn.queue.put_nowait(None)

    async def reverse_open(self, conn, rule):
        if rule < 0 or rule >= len(self.reverse_rules):
            log("unknown reverse rule %d" % rule)
            await self.abort(conn)
            return
        host, port = self.reverse_rules[rule][1]
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except Exception as exc:
            log("reverse connect %s:%d failed: %s" % (host, port, exc))
            await self.abort(conn)
            return
        conn.reader, conn.writer = reader, writer
        conn.tasks.append(asyncio.create_task(self.pump_local(conn)))
        conn.tasks.append(asyncio.create_task(self.drain_conn(conn)))

    async def abort(self, conn):
        try:
            await self.send_frame(C, conn.cid)
        except Exception:
            pass
        self.teardown(conn)

    def teardown(self, conn):
        if conn.removed:
            return
        conn.removed = True
        self.conns.pop(conn.cid, None)
        if conn.writer is not None:
            try:
                conn.writer.close()
            except Exception:
                pass
        current = asyncio.current_task()
        for task in conn.tasks:
            if task is not current and not task.done():
                task.cancel()

    async def pump_local(self, conn):
        try:
            while True:
                data = await conn.reader.read(CHUNK)
                if not data:
                    break
                await self.send_frame(D, conn.cid, data)
        except (asyncio.CancelledError, ConnectionError):
            raise
        except Exception as exc:
            log("local read cid=%d: %s" % (conn.cid, exc))
        finally:
            if not conn.closed:
                conn.closed = True
                try:
                    await self.send_frame(C, conn.cid)
                except Exception:
                    pass
                self.teardown(conn)

    async def drain_conn(self, conn):
        try:
            while True:
                item = await conn.queue.get()
                if item is None:
                    break
                conn.writer.write(item)
                await conn.writer.drain()
        except (asyncio.CancelledError, ConnectionError):
            raise
        except Exception as exc:
            log("local write cid=%d: %s" % (conn.cid, exc))
        finally:
            if not conn.closed:
                conn.closed = True
                try:
                    await self.send_frame(C, conn.cid)
                except Exception:
                    pass
                self.teardown(conn)

    async def forward_accept(self, reader, writer, target):
        cid = self.alloc_cid()
        conn = LocalConn(cid, reader, writer)
        self.conns[cid] = conn
        try:
            await self.send_frame(O, cid, json.dumps(
                {"host": target[0], "port": target[1]}).encode())
        except Exception as exc:
            log("forward open failed: %s" % exc)
            self.teardown(conn)
            return
        conn.tasks.append(asyncio.create_task(self.pump_local(conn)))
        conn.tasks.append(asyncio.create_task(self.drain_conn(conn)))

    async def ws_reader(self):
        async for msg in self.ws:
            raw = msg.encode() if isinstance(msg, str) else msg
            if raw[:1] == b"0":
                self.inbuf += raw[1:]
                self.parse()
            elif raw[:1] == b"1":
                log("remote title: %s" % raw[1:].decode("utf-8", "replace"))

    async def run(self):
        log("connecting %s" % self.args.ws)
        await self.connect_and_start_agent()
        reader_task = self.reader_task
        for (bind, target) in self.forward_rules:
            srv = await asyncio.start_server(
                lambda r, w, t=target: self.forward_accept(r, w, t),
                bind[0], bind[1], reuse_address=True)
            self.servers.append(srv)
            log("forward %s:%d -> remote %s:%d" % (bind[0], bind[1], target[0], target[1]))
        for (bind, target) in self.reverse_rules:
            log("reverse remote %s:%d -> local %s:%d" % (bind[0], bind[1], target[0], target[1]))
        log("running")
        done, pending = await asyncio.wait(
            {reader_task}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for srv in self.servers:
            srv.close()


def main():
    ap = argparse.ArgumentParser(description="TCP-over-ttyd tunnel (forward + reverse)")
    ap.add_argument("--forward", action="append", type=parse_rule, metavar="L_BIND=REMOTE",
                    help="local 127.0.0.1:LPORT=remote_host:RPORT")
    ap.add_argument("--reverse", action="append", type=parse_rule, metavar="R_BIND=LOCAL",
                    help="remote bind_host:RPORT=local_host:LPORT")
    ap.add_argument("--ws", default=DEFAULT_WS)
    ap.add_argument("--origin", default=DEFAULT_ORIGIN)
    args = ap.parse_args()
    if not args.forward and not args.reverse:
        ap.error("at least one --forward or --reverse rule is required")
    try:
        asyncio.run(Controller(args).run())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log("fatal: %s: %s" % (type(exc).__name__, exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
