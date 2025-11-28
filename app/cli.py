# app/cli.py
import argparse, json, time, sys
from app.orchestrator import run_pipeline
from utils.metrics import init_metrics

def main():
    p = argparse.ArgumentParser(description="MVP: pipeline de voz afectiva y segura")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Ejecuta el pipeline una vez")
    runp.add_argument("--audio", required=True)
    runp.add_argument("--pretty", action="store_true")
    runp.add_argument("--metrics-port", type=int, default=None)
    runp.add_argument("--run-id", type=str, default=None)
    runp.add_argument("--hold", type=int, default=0)

    servep = sub.add_parser("serve", help="Levanta /metrics y queda corriendo (sin ejecutar pipeline)")
    servep.add_argument("--metrics-port", type=int, default=9000)

    args = p.parse_args()

    if args.cmd == "run":
        out = run_pipeline(args.audio, metrics_port=args.metrics_port, run_id=args.run_id)
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.pretty else json.dumps(out, ensure_ascii=False))
        if args.metrics_port is not None and args.hold > 0:
            url = f"http://localhost:{args.metrics_port}/metrics"
            print(f"\n[Métricas] Servidor expuesto en {url} durante {args.hold} segundos...")
            try:
                time.sleep(args.hold)
            except KeyboardInterrupt:
                print("\n[Salida por Ctrl+C]")
                sys.exit(0)

    elif args.cmd == "serve":
        init_metrics(args.metrics_port)
        url = f"http://localhost:{args.metrics_port}/metrics"
        print(f"[Métricas] Servidor expuesto en {url}. Ctrl+C para salir.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Salida por Ctrl+C]")
            sys.exit(0)

if __name__ == "__main__":
    main()
