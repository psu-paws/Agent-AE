#!/bin/bash
trap 'echo "Interrupted. Exiting..."; exit 1' INT TERM

OUTDIR=0521
mkdir -p "$OUTDIR"

run_agent() {
    local l=$1
    local i=$2
    echo "Processing Level $l - Task $i..."

    docker run -it --rm \
        --ipc=host \
        --network=host \
        -e PYTHONUNBUFFERED=1 \
        -e no_proxy="localhost,127.0.0.1" \
        -e NO_PROXY="localhost,127.0.0.1" \
        -v "$(pwd)/run_gaia_workforce_vllm_planner.py:/app/run_gaia_workforce_vllm_planner.py" \
        agent-image \
        -t "$i" -l "$l" 2>&1 | tee "$OUTDIR/L${l}_${i}_0.txt"
}

# ----------------------------------------
# Process Level 1
# ----------------------------------------
l=1
for i in {41,52}
do
    run_agent "$l" "$i"
done

# ----------------------------------------
# Process Level 2
# ----------------------------------------
l=2
for i in {7,8,14,17,22,28,69}
do
    run_agent "$l" "$i"
done

# ----------------------------------------
# Process Level 3
# ----------------------------------------
l=3
for i in {3,4,5,10,17,24}
do
    run_agent "$l" "$i"
done
