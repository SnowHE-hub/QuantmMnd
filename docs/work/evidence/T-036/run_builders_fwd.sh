#!/bin/bash
# T-036: run the frozen T-030 builders unchanged against the forward shadow root. Stop at the first failure.
FR=${1:-/home/lenovo/qm_fwd_root/20261020}      # 影子数据根（2026-10-21 起作为参数传入）
M=$FR/main; A=$FR/a158
L=/home/lenovo/t030_scratch/logs/t036/$(basename $FR); mkdir -p $L; S=$L/builders.status
cd /home/lenovo/projects/qm-t035
G=/home/lenovo/miniforge3/envs/qm-gpu/bin/python
Q=/home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python
E=docs/work/evidence/T-030
run() {  # name, command...
  local n=$1; shift
  echo "$n start $(date '+%F %T')" >> $S
  PYTHONPATH=. WRITE_MODE=parquet_only "$@" > $L/b_$n.log 2>&1
  local rc=$?
  echo "$n exit $rc $(date '+%F %T')" >> $S
  [ $rc -eq 0 ] || { echo "STOP at $n" >> $S; exit $rc; }
}
run base   $G $E/build_daily_base.py $M --workers 2
run fnd    $G $E/build_daily_fnd.py $M --workers 2
run sh     $G $E/build_daily_sh.py $M
run seq    $G $E/build_daily_seq.py $M
run event  $G $E/build_daily_events.py $M
run sector $G $E/build_daily_sector.py $M
run a158   $Q $E/build_daily_a158.py $A
for f in $A/data/gpu_v1/panel/a158/*.parquet; do
  [ -L "$f" ] && continue          # frozen months are links; only new outputs are regular files
  ln -s "$f" "$M/data/gpu_v1/panel/a158/$(basename $f)"
done
echo "a158 linked $(ls $M/data/gpu_v1/panel/a158 | wc -l) months $(date '+%F %T')" >> $S
run model_ready $G $E/build_model_ready.py $M
echo "ALL DONE $(date '+%F %T')" >> $S
