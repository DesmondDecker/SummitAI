"""Load test a running server.
   python tools/loadtest.py --port 8080 --clients 8 --requests 200 --endpoint translate --repeat 0.5
--repeat = share of requests that reuse an earlier phrase (served from cache). Reports throughput and latency percentiles."""
import argparse, http.client, json, random, threading, time


def run(host, port, clients, requests, endpoint, repeat, phrases, key=None, seed=0):
    lat, codes, lock = [], {}, threading.Lock()
    hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

    def worker(i):
        rng = random.Random(seed + i); c = http.client.HTTPConnection(host, port, timeout=60); mine = []
        for n in range(requests):
            base = rng.choice(phrases[:20]) if rng.random() < repeat else rng.choice(phrases) + f" {rng.randint(0, 10**6)}"
            body = {"text": base, "src": "krio", "tgt": "en"} if endpoint == "translate" else {"text": base}
            t = time.perf_counter()
            try:
                c.request("POST", f"/v1/{endpoint}", json.dumps(body), hdr); r = c.getresponse(); r.read(); code = r.status
            except Exception:
                code = "error"; c = http.client.HTTPConnection(host, port, timeout=60)
            mine.append((time.perf_counter() - t) * 1000)
            with lock:
                codes[code] = codes.get(code, 0) + 1
        with lock:
            lat.extend(mine)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0; lat.sort(); q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    return dict(requests=len(lat), seconds=round(wall, 2), rps=round(len(lat) / wall, 1), p50_ms=round(q(.5), 1),
                p95_ms=round(q(.95), 1), p99_ms=round(q(.99), 1), status=codes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--clients", type=int, default=8); ap.add_argument("--requests", type=int, default=100, help="per client")
    ap.add_argument("--endpoint", default="translate", choices=["translate", "identify"]); ap.add_argument("--repeat", type=float, default=0.5)
    ap.add_argument("--key"); ap.add_argument("--phrases", default=None, help="text file, one phrase per line")
    a = ap.parse_args()
    phrases = [l.strip() for l in open(a.phrases, encoding="utf-8")] if a.phrases else ["aw yu du", "a dey fayn tenki", "wetin na yu nem", "usai yu dey go"]
    print(json.dumps(run(a.host, a.port, a.clients, a.requests, a.endpoint, a.repeat, phrases, a.key), indent=2))
