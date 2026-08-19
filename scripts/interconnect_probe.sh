#!/usr/bin/env bash
# Measure what a per-layer collective would cost between two nodes.
#
# The layer split in src/bin/inkling_forward.rs exists partly because a
# within-layer split needs an all-reduce per layer. Whether that is affordable
# is a property of the link AND of the transport, and on this hardware the two
# answers differ by more than an order of magnitude on the same wire, so
# measure both rather than reasoning from a link rate.
#
# Run on BOTH boxes, rank 0 first:
#   scripts/interconnect_probe.sh 0 <peer-ip>      # on the first box
#   scripts/interconnect_probe.sh 1 <rank0-ip>     # on the second
#
# Override the fabric with IFACE and HCA when autodetection picks the wrong
# one. Measuring the management NIC by accident is exactly the mistake this
# script exists to stop, so it prints the interface it chose and the speed that
# interface negotiated.
#
# Needs: perftest (ib_write_lat), nvcc, libnccl-dev.
set -euo pipefail

RANK="${1:?usage: interconnect_probe.sh <rank 0|1> <peer-ip> [port]}"
PEER="${2:?peer ip}"
PORT="${3:-19700}"
WORK="${WORK:-/tmp/interconnect_probe}"
mkdir -p "$WORK"

IFACE="${IFACE:-$(ip -o route get "$PEER" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)}"
[ -n "$IFACE" ] || { echo "no route to $PEER; set IFACE=" >&2; exit 1; }
HCA="${HCA:-$(ls -1 "/sys/class/net/$IFACE/device/infiniband" 2>/dev/null | head -1)}"

echo "=== link ==="
echo "peer      : $PEER"
echo "interface : $IFACE"
echo "hca       : ${HCA:-none (no RDMA on this interface)}"
ethtool "$IFACE" 2>/dev/null | grep -E "Speed:|Duplex:|Port:" || true
ip -o link show "$IFACE" | sed -n 's/.*\(mtu [0-9]*\).*/\1/p'
echo

# --- RDMA reference: what the wire does with the kernel out of the way ------
if [ -n "$HCA" ]; then
  echo "=== RDMA write latency, one way ==="
  for S in 1024 4096 8192 16384 65536; do
    if [ "$RANK" = 0 ]; then
      ib_write_lat -d "$HCA" -x 3 -s "$S" -n 3000 -p $((PORT + 1)) >/dev/null 2>&1 || true
    else
      sleep 1
      ib_write_lat -d "$HCA" -x 3 -s "$S" -n 3000 -p $((PORT + 1)) "$PEER" 2>/dev/null |
        awk -v s="$S" '$1 == s { printf "  %7d B  %8.2f us\n", $1, $4 }'
      sleep 1
    fi
  done
  echo
fi

# --- The number that decides it: a GPU-resident all-reduce -----------------
cat >"$WORK/nccl_allreduce.cu" <<'CU_EOF'
// Two-rank cross-machine all-reduce, timed at the sizes a decoder layer uses.
// Bootstraps the NCCL unique id over a plain socket, so no MPI is needed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include <nccl.h>
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { printf("cuda err %d L%d\n", (int)e, __LINE__); exit(1); } } while (0)
#define NK(x) do { auto e = (x); if (e != ncclSuccess) { printf("nccl err %s L%d\n", ncclGetErrorString(e), __LINE__); exit(1); } } while (0)
static void rdall(int f, char *b, int n) { int g = 0; while (g < n) { int r = read(f, b + g, n - g); if (r <= 0) exit(2); g += r; } }
static void wrall(int f, char *b, int n) { int g = 0; while (g < n) { int r = write(f, b + g, n - g); if (r <= 0) exit(2); g += r; } }
int main(int argc, char **argv) {
  int rank = atoi(argv[1]); const char *peer = argv[2]; int port = atoi(argv[3]);
  ncclUniqueId id; int sock; int one = 1;
  if (rank == 0) {
    NK(ncclGetUniqueId(&id));
    int l = socket(AF_INET, SOCK_STREAM, 0); setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, 4);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
    bind(l, (sockaddr *)&a, sizeof a); listen(l, 1); sock = accept(l, 0, 0);
    wrall(sock, (char *)&id, sizeof id);
  } else {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, peer, &a.sin_addr);
    while (connect(sock, (sockaddr *)&a, sizeof a)) usleep(200000);
    rdall(sock, (char *)&id, sizeof id);
  }
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, 4);
  CK(cudaSetDevice(0));
  ncclComm_t comm; NK(ncclCommInitRank(&comm, 2, id, rank));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  // 4096 f32 is one token of the residual stream on the 42-layer model; the
  // rest bracket it, up to a prefill-sized payload.
  const int sizes[] = {1024, 2048, 4096, 8192, 16384, 65536, 262144, 4194304, 16777216};
  if (rank == 0)
    printf("%10s %12s %14s %12s   %s\n", "elems", "bytes(f32)", "allreduce_us", "GB/s", "84x (ms/token)");
  for (int si = 0; si < 9; si++) {
    int N = sizes[si]; float *db; CK(cudaMalloc(&db, (size_t)N * 4)); CK(cudaMemset(db, 0, (size_t)N * 4));
    int warm = N > 1000000 ? 10 : 200, IT = N > 1000000 ? 50 : 1000;
    for (int i = 0; i < warm; i++) NK(ncclAllReduce(db, db, N, ncclFloat, ncclSum, comm, s));
    CK(cudaStreamSynchronize(s));
    char sync = 1;
    if (rank == 0) { wrall(sock, &sync, 1); rdall(sock, &sync, 1); }
    else { rdall(sock, &sync, 1); wrall(sock, &sync, 1); }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < IT; i++) { NK(ncclAllReduce(db, db, N, ncclFloat, ncclSum, comm, s)); CK(cudaStreamSynchronize(s)); }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / IT;
    if (rank == 0)
      printf("%10d %12zu %14.2f %12.2f   %12.2f\n", N, (size_t)N * 4, us, ((size_t)N * 4) / us / 1000.0, 84 * us / 1000.0);
    CK(cudaFree(db));
  }
  ncclCommDestroy(comm); return 0;
}
CU_EOF

echo "=== NCCL all-reduce, GPU-resident, 2 ranks ==="
echo "(84 collectives per token = 42 layers x 2, after attention out-proj and after MoE down-proj)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_121}"
"$NVCC" -O2 -arch="$ARCH" -o "$WORK/nccl_allreduce" "$WORK/nccl_allreduce.cu" -lnccl
export NCCL_SOCKET_IFNAME="$IFACE"
[ -n "$HCA" ] && export NCCL_IB_HCA="$HCA"
"$WORK/nccl_allreduce" "$RANK" "$PEER" "$PORT"
