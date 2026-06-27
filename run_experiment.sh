#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$SCRIPT_DIR/runs}"

usage() {
    cat <<EOF
Usage: ./run_experiment.sh <command> [options]

Commands:
  train     -c <config.yaml> [overrides...]
  generate  -m <model.pt> -p "<prompt>" [-t temp] [-n chars] [-c config.yaml]
  sweep     -c <sweep.yaml>
  pipeline  -c <pipeline.yaml> [overrides...]
  test      [pytest args...]
  list
  tensorboard-only

Examples:
  ./run_experiment.sh train -c configs/gqa_swiglu.yaml
  ./run_experiment.sh train -c configs/gqa_swiglu.yaml model.attention.type=gqa
  ./run_experiment.sh train model.attention.type=mamba model.ffn.type=swiglu training.num_epochs=10
  ./run_experiment.sh generate -m best_model.pt -p "ROMEO:" -t 0.7 -n 300 -c configs/gqa_swiglu.yaml
  ./run_experiment.sh sweep -c configs/sweep_example.yaml
  ./run_experiment.sh pipeline -c configs/pipeline_full.yaml
  ./run_experiment.sh pipeline -c configs/pipeline_sft_dpo.yaml training.batch_size=16
  ./run_experiment.sh test
  ./run_experiment.sh test -v -k dpo
  ./run_experiment.sh list

Environment variables:
  TENSORBOARD_PORT   Port for TensorBoard (default: 6006)
  TENSORBOARD_DIR    Directory for logs (default: ./runs)
  NO_TENSORBOARD     Set to "1" to skip launching TensorBoard
EOF
    exit 1
}

cleanup() {
    if [ -n "${TB_PID:-}" ]; then
        echo "Deteniendo TensorBoard (PID $TB_PID)..."
        kill "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_tensorboard() {
    if [ "${NO_TENSORBOARD:-}" = "1" ]; then
        echo "[SKIP] TensorBoard desactivado via NO_TENSORBOARD=1"
        return
    fi
    mkdir -p "$TENSORBOARD_DIR"
    echo "Iniciando TensorBoard en http://localhost:$TENSORBOARD_PORT ..."
    tensorboard --logdir="$TENSORBOARD_DIR" --port="$TENSORBOARD_PORT" --host=0.0.0.0 &>/dev/null &
    TB_PID=$!
    sleep 2
    if kill -0 "$TB_PID" 2>/dev/null; then
        echo "  → TensorBoard corriendo (PID $TB_PID)"
        echo "  → Abre http://localhost:$TENSORBOARD_PORT en tu navegador"
    else
        echo "  ⚠ No se pudo iniciar TensorBoard"
    fi
}

run_python() {
    if [ -f "$VENV_PYTHON" ]; then
        PYTHON="$VENV_PYTHON"
    else
        PYTHON="python3"
    fi
    cd "$SCRIPT_DIR"
    echo "Ejecutando: $PYTHON run.py $*"
    echo ""
    $PYTHON run.py "$@"
}

if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    train)
        start_tensorboard
        run_python train "$@"
        ;;
    generate)
        run_python generate "$@"
        ;;
    sweep)
        start_tensorboard
        run_python sweep "$@"
        ;;
    pipeline)
        start_tensorboard
        run_python pipeline "$@"
        ;;
    test)
        .venv/bin/python -m pytest tests/ "$@"
        ;;
    list)
        run_python list
        ;;
    tensorboard-only)
        start_tensorboard
        echo "TensorBoard corriendo. Presiona Ctrl+C para detener."
        wait "${TB_PID:-}"
        ;;
    *)
        usage
        ;;
esac
