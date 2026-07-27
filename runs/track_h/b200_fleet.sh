#!/bin/bash
# B200 4노드 제어기 — **컨테이너 4대(yong-1..4) × GPU 1장**, 노드끼리 SSH 메시가 뚫려 있다.
#   레인 n ↔ yong-n ↔ queue_b200_lane<n>.txt ↔ $BATCH/runlogs/logs_lane<n>/
#
#   ⚠ 정본 런처 run_b200_batch.sh 는 **단일 서버 GPU 0–3** 을 가정한다(GPUS="0 1 2 3" + 큐 1개).
#     이 서버는 그 구성이 아니다 — 그대로 띄우면 슬롯 1·2·3 셀이 즉사하며 큐가 소진된다.
#     그래서 노드마다 run_b200_lane.sh 로 GPUS="0" + 자기 레인 큐를 돌린다.
#
#   홈(/home/edgeai_lab)은 노드-로컬(overlay)이고 공유되는 건 lustre 뿐이라, 런처가
#   HOME=$BATCH/home 으로 갈아끼우는 게 필수다(노드-로컬 홈이면 HF 캐시가 갈린다).
#
#   사용:
#     bash runs/track_h/b200_fleet.sh nodes        # 매핑·도달성·GPU
#     bash runs/track_h/b200_fleet.sh status       # 레인별 드라이버/진행셀/큐진도/남은시간
#     bash runs/track_h/b200_fleet.sh up [1 2 3 4] # 기동(SSH 끊겨도 생존; 인자 없으면 4대)
#     bash runs/track_h/b200_fleet.sh tail 1 [n]   # 진행 중 셀 로그
#     bash runs/track_h/b200_fleet.sh drain 1      # 안전 정지 = 남은 큐 주석(진행 셀은 완주)
#     bash runs/track_h/b200_fleet.sh seal [1..4]  # 컨테이너 교체 전: 완료 셀을 레포 큐에서 주석
#     bash runs/track_h/b200_fleet.sh kill 1       # 강제 종료 (⚠ 진행 셀 전손)
#     bash runs/track_h/b200_fleet.sh exec 'cmd'   # 전 노드 임의 명령
#
#   ⚠ BATCH 을 export 한 채로 부르지 말 것 — 러너가 batch-size 노브로 읽어 전 셀이 즉사한다
#     (07-25 실사례). 이 스크립트는 `env -u BATCH` 로 자식에게서 지운 뒤 런처를 부른다.
set -u

REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds}
BATCH_DIR=${BATCH_DIR:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch}
# 레인 n → NODES[n]. 노드 이름이 바뀌면 FLEET_NODES 로 덮어쓴다(순서 = 레인 1,2,3,4).
read -ra NODE_ARR <<< "${FLEET_NODES:-yong-1 yong-2 yong-3 yong-4}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
LANES_ALL="1 2 3 4"
DONE_RE="MATRIX DONE|TRACK G DONE|TRACK D DONE|\[persist\]"
EXPIRE_EPOCH=${EXPIRE_EPOCH:-$(date -d "2026-07-27 21:11:06" +%s)}   # 컨테이너 48h 동기 만료

node_of() { echo "${NODE_ARR[$(( $1 - 1 ))]}"; }
out_of()  { echo "$BATCH_DIR/runlogs/_b200_lane$1.out"; }
pid_of()  { echo "$BATCH_DIR/runlogs/_b200_lane$1.pid"; }
qsrc_of() { echo "$REPO/runs/track_h/queue_b200_lane$1.txt"; }
qrun_of() { echo "$BATCH_DIR/runlogs/logs_lane$1/queue.run.txt"; }
ldir_of() { echo "$BATCH_DIR/runlogs/logs_lane$1"; }

# 드라이버 생존 판정 — RUNNING | idle | UNREACHABLE
#   ⚠ `pgrep -f run_multi_driver.sh` 를 그냥 쓰면 **SSH 가 띄운 자기 래퍼**의 명령줄이
#     패턴을 포함해 항상 RUNNING 이 된다(07-25 확인). ①PID 파일이 1차 근거이고,
#     ②보조 pgrep 은 `[r]un…` 브래킷 패턴 — 자기 명령줄엔 `[r]` 이 그대로 있어 매칭되지 않는다.
alive_of() {
  local c=$1 h; h=$(node_of "$c")
  local pf; pf=$(pid_of "$c")
  local r
  r=$($SSH "$h" "
    pid=\$(cat $pf 2>/dev/null)
    if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then echo RUNNING; exit 0; fi
    pgrep -f '[r]un_multi_driver' >/dev/null 2>&1 && { echo RUNNING; exit 0; }
    pgrep -f '[e]xperiments/.*\.py' >/dev/null 2>&1 && { echo RUNNING; exit 0; }
    echo idle" 2>/dev/null)
  echo "${r:-UNREACHABLE}"
}

cmd_nodes() {
  printf "%-6s %-9s %-6s %-22s %s\n" LANE NODE SSH GPU "MEM used/total · util"
  for c in $LANES_ALL; do
    h=$(node_of "$c")
    info=$($SSH "$h" 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null' 2>/dev/null)
    if [ -z "$info" ]; then printf "%-6s %-9s %-6s %s\n" "$c" "$h" "DOWN" "-"; else
      printf "%-6s %-9s %-6s %-22s %s\n" "$c" "$h" "OK" "$(echo "$info" | cut -d, -f1)" \
             "$(echo "$info" | cut -d, -f2,3,4 | tr -d ' ')"
    fi
  done
  local left=$(( (EXPIRE_EPOCH - $(date +%s)) / 60 ))
  printf "\n컨테이너 만료까지 %d시간 %d분  (%s)\n" $((left/60)) $((left%60)) \
         "$(date -d "@$EXPIRE_EPOCH" '+%F %H:%M')"
}

cmd_status() {
  local now; now=$(date +%s)
  for c in $LANES_ALL; do
    h=$(node_of "$c"); out=$(out_of "$c"); ldir=$(ldir_of "$c")
    alive=$(alive_of "$c")
    gpu=$($SSH "$h" 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null' 2>/dev/null | tr -d ' ')
    last_start=$(grep -a 'start:' "$out" 2>/dev/null | tail -1)
    cell=$(echo "$last_start" | sed -n 's/.*start: \([^ ]*\) .*/\1/p')
    prog=$(echo "$last_start" | sed -n 's/.*\[\([0-9]*\/[0-9]*\)\].*/\1/p')
    log=""
    if [ -n "$cell" ]; then log="$ldir/$cell.log"
    else log=$(ls -t "$ldir"/*.log 2>/dev/null | head -1); [ -n "$log" ] && cell=$(basename "$log" .log); fi
    echo "── 레인 $c  ($h)  driver=$alive  gpu=${gpu:-?}"
    if [ -n "$cell" ] && [ -f "$log" ]; then
      mt=$(stat -c %Y "$log" 2>/dev/null)
      age=$(( (now - mt) / 60 ))
      st=$(grep -a "start: $cell " "$out" 2>/dev/null | tail -1 | sed -n 's/^\[\([0-9:]*\)\].*/\1/p')
      echo "    cell=$cell  큐=${prog:-?}  시작=${st:-?}  로그갱신=${age}분 전"
      echo "    last: $(tail -1 "$log" 2>/dev/null | cut -c1-120)"
    else
      echo "    (기동 이력 없음)"
    fi
    q=$(qrun_of "$c"); [ -f "$q" ] && echo "    런타임 큐 남음=$(grep -c '^[^#]' "$q" 2>/dev/null)"
  done
  echo
  printf "레포 레인 큐 활성: "; for c in $LANES_ALL; do printf "L%s=%s " "$c" "$(grep -c '^[^#]' "$(qsrc_of "$c")" 2>/dev/null)"; done; echo
  local left=$(( (EXPIRE_EPOCH - now) / 60 ))
  printf "컨테이너 만료까지 %d시간 %d분\n" $((left/60)) $((left%60))
}

cmd_up() {
  local lanes="${*:-$LANES_ALL}"
  for c in $lanes; do
    h=$(node_of "$c"); out=$(out_of "$c"); pf=$(pid_of "$c")
    if ! $SSH "$h" true 2>/dev/null; then echo "[L$c/$h] SKIP — SSH 불가"; continue; fi
    st=$(alive_of "$c")
    if [ "$st" = "RUNNING" ]; then
      echo "[L$c/$h] SKIP — 이미 돌고 있다 (중복 기동은 같은 GPU 를 두 셀이 물어 OOM)"; continue
    fi
    if [ ! -f "$(qsrc_of "$c")" ]; then echo "[L$c/$h] SKIP — 레인 큐 없음: $(qsrc_of "$c")"; continue; fi
    mkdir -p "$BATCH_DIR/runlogs"
    [ -f "$out" ] && mv -f "$out" "$out.prev"
    # bash -c 안에서 자기 PID 를 적고 exec 로 런처가 되므로, PID 파일이 런처를 정확히 가리킨다.
    $SSH "$h" "setsid nohup bash -c 'echo \$\$ > $pf; exec env -u BATCH LANE=$c REPO=$REPO bash $REPO/runs/track_h/run_b200_lane.sh' \
                 > $out 2>&1 < /dev/null & echo started" >/dev/null 2>&1
    echo "[L$c/$h] 기동 → $out"
  done
  echo; echo "몇 초 뒤 확인:  bash runs/track_h/b200_fleet.sh status"
}

cmd_tail() {
  local c=${1:?LANE}; local n=${2:-40}
  out=$(out_of "$c"); ldir=$(ldir_of "$c")
  cell=$(grep -a 'start:' "$out" 2>/dev/null | tail -1 | sed -n 's/.*start: \([^ ]*\) .*/\1/p')
  log="$ldir/$cell.log"
  if [ -z "$cell" ] || [ ! -f "$log" ]; then
    log=$(ls -t "$ldir"/*.log 2>/dev/null | head -1)
    [ -z "$log" ] && { echo "셀 로그 없음 — 런처 출력:"; tail -"$n" "$out" 2>&1; return; }
    cell=$(basename "$log" .log)
  fi
  echo "── L$c cell=$cell  ($log)"; tail -"$n" "$log" 2>&1
}

cmd_drain() {
  local c=${1:?LANE}; q=$(qrun_of "$c")
  [ -f "$q" ] || { echo "런타임 큐 없음: $q"; return 1; }
  before=$(grep -c '^[^#]' "$q")
  sed -i 's/^\([^#]\)/#\1/' "$q"          # 줄 수 보존 = 드라이버 인덱스 안 밀림
  echo "[L$c] drain: 활성 $before → $(grep -c '^[^#]' "$q") — 진행 중 셀은 완주 후 드라이버 종료"
}

# 컨테이너 교체 전: **완료가 확인된 셀만** 레포 레인 큐에서 주석 처리한다.
#   완료 판정 = 셀 로그에 MATRIX DONE / TRACK G DONE / TRACK D DONE / [persist]
#   (문서 규약과 동일). 미완료 줄은 건드리지 않으므로 재기동하면 거기서 이어간다.
cmd_seal() {
  local lanes="${*:-$LANES_ALL}"
  for c in $lanes; do
    src=$(qsrc_of "$c"); ldir=$(ldir_of "$c")
    [ -f "$src" ] || { echo "[L$c] 레인 큐 없음"; continue; }
    local sealed=0 pending=0
    while IFS= read -r line; do
      case "$line" in ''|'#'*) continue ;; esac
      name=$(echo "$line" | cut -d'|' -f2)
      # ★ 로그를 **전 레인**에서 찾는다 — 07-26 부하 균형으로 셀 4개가 레포 큐와 다른
      #   레인에서 돌았다(런타임 큐만 수정, 레포 파일은 원배치 유지). 자기 레인만 보면
      #   그 4개가 미완료로 남아 재기동 때 중복 실행된다.
      found=""
      for _c in $LANES_ALL; do
        _l="$(ldir_of "$_c")/$name.log"
        [ -f "$_l" ] && grep -qaE "$DONE_RE" "$_l" 2>/dev/null && { found="L$_c"; break; }
      done
      if [ -n "$found" ]; then
        # 줄 삭제 금지 — 맨 앞에 '#' 만 붙인다(드라이버 인덱스 보존)
        python3 - "$src" "$line" <<'PY'
import sys
path, target = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
out = ["#" + l if l == target else l for l in lines]
open(path, "w").write("\n".join(out))
PY
        if [ "$found" = "L$c" ]; then echo "  [L$c] sealed: $name"
        else echo "  [L$c] sealed: $name   (← $found 에서 실행됨)"; fi
        sealed=$((sealed+1))
      else
        pending=$((pending+1))
      fi
    done < "$src"
    echo "[L$c] 완료 $sealed 줄 주석 · 미완료 $pending 줄 유지  → 재기동하면 미완료분부터"
  done
  echo; echo "교체 후 재기동:  bash runs/track_h/b200_fleet.sh up"
}

cmd_kill() {
  local c=${1:?LANE}; h=$(node_of "$c"); pf=$(pid_of "$c")
  echo "⚠ [L$c/$h] 진행 중 셀은 전손된다(G1 은 cell-end 1회 persist = 19.6h). 정상 정지는 drain 이다."
  $SSH "$h" "
    pid=\$(cat $pf 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null
    pkill -f '[r]un_multi_driver'; pkill -f '[e]xperiments/.*\.py'
    rm -f $pf
    pgrep -f '[r]un_multi_driver' >/dev/null 2>&1 && echo 'still alive — 재확인 필요' || echo 'driver stopped'" 2>&1
}

cmd_exec() {
  local cmdline="${*:?명령}"
  for c in $LANES_ALL; do h=$(node_of "$c"); echo "===== L$c / $h ====="; $SSH "$h" "$cmdline" 2>&1; done
}

case "${1:-status}" in
  nodes)  cmd_nodes ;;
  status) cmd_status ;;
  up)     shift; cmd_up "$@" ;;
  tail)   shift; cmd_tail "$@" ;;
  drain)  shift; cmd_drain "$@" ;;
  seal)   shift; cmd_seal "$@" ;;
  kill)   shift; cmd_kill "$@" ;;
  exec)   shift; cmd_exec "$@" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
