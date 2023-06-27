#!/usr/bin/env bash

log_names=(
  "bpic2011_f1"
  "bpic2011_f2"
  "bpic2011_f3"
  "bpic2011_f4"
  "bpic2012_accepted"
  "bpic2012_cancelled"
  "bpic2012_declined"
  "bpic2015_1_f2"
  "bpic2015_2_f2"
  "bpic2015_3_f2"
  "bpic2015_4_f2"
  "bpic2015_5_f2"
  "bpic2017_accepted"
  "bpic2017_cancelled"
  "bpic2017_refused"
  "hospital_billing_2"
  "hospital_billing_3"
  "Production"
  "sepsis_cases_1"
  "sepsis_cases_2"
  "sepsis_cases_4"
  "traffic_fines_1"
        )

for log_name in "${log_names[@]}"
do
    python experiments_runner.py --log=${log_name} &
done