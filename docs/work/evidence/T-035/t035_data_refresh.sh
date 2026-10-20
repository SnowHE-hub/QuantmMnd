#!/bin/bash
# T-035 P0 data refresh (in-place append approved 2026-10-19/21): market series + event tables to 2026-10-20,
# then the three new tables (pledge_stat, fina_audit, dividend), compact to new *_20261020 files, then an F0 snapshot.
L=/home/lenovo/t030_scratch/logs/t035; mkdir -p $L
S=$L/data_refresh.status
cd /home/lenovo/projects/qm-t035
P=/home/lenovo/miniforge3/envs/quantmind/bin/python
R=/home/lenovo/projects/quantmind
step() { echo "$1 start $(date '+%F %T')" >> $S; }
done_() { echo "$1 exit $2 $(date '+%F %T')" >> $S; }
step lake;  WRITE_MODE=parquet_only $P -u scripts/data_pipeline/update_lake_latest.py --end 2026-10-20 --data-root $R > $L/lake.log 2>&1; done_ lake $?
EV=stk_limit,suspend_d,namechange,forecast,express,disclosure_date,stk_holdernumber,stk_holdertrade,repurchase,share_float,moneyflow,top_list,top_inst,block_trade,index_weight
step events; WRITE_MODE=parquet_only $P -u scripts/data_pipeline/download_events.py --data-root $R --apis $EV --start 2018-01-01 --end 2026-10-20 > $L/events.log 2>&1; done_ events $?
step events_compact; WRITE_MODE=parquet_only $P -u scripts/data_pipeline/download_events.py --data-root $R --apis $EV --end 2026-10-20 --compact > $L/events_compact.log 2>&1; done_ events_compact $?
NEW=pledge_stat,fina_audit,dividend
step new; WRITE_MODE=parquet_only $P -u scripts/data_pipeline/download_events.py --data-root $R --apis $NEW --start 2018-01-01 --end 2026-10-20 > $L/new.log 2>&1; done_ new $?
step new_compact; WRITE_MODE=parquet_only $P -u scripts/data_pipeline/download_events.py --data-root $R --apis $NEW --end 2026-10-20 --compact > $L/new_compact.log 2>&1; done_ new_compact $?
step snapshot; PYTHONPATH=. $P -m quantmind.line_a.snapshot --data-root $R --as-of 2026-10-20 > $L/snapshot.log 2>&1; done_ snapshot $?
