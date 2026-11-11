# Feature inventory - UNSW - NB15   

49 columns. Categorized for leakage risk and shortcut learning risk per Dr. Chuprovs Proposal feedback.

## Excluded
| Column | Reason |
|---|---|
| Label | Target variable (binary) |
| attack_cat | Target variable (multi-class) |
| srcip | Raw IP address, identifies host/capture scenario rather than attack behavior |
| dstip | Same as srcip |
| Stime | Timestamp, could leak which capture period an attack came from |
| Ltime | Same as Stime |

## Flagged for shortcut learning check

These will be tested with a standalone basline model (ports/proto/service/state only) to check whether they predict attack_cat too easily on their own, which would suggest the model is learning dataset construction artifacts rather than attack behavior. 

| Column | Reason |
|---|---|
| sport | Source port, attacks may cluster on staged ports specific to this capture |
| dsport | Destination port, same risk |
| proto | Protocol type, may correlate too strongly with specific categories in this dataset |
| service | Same risk as proto |
| state | Connection state, worth testing in isolation |
| is_sm_ips_ports | Derived from IP/port equality, indirectly tied to excluded address fields |


## Kept as behavorial features - 37

dur, sbytes, dbytes, sttl, dttl, sloss, dloss, Sload, Dload, Spkts,
Dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth,
res_bdy_len, Sjit, Djit, Sintpkt, Dintpkt, tcprtt, synack, ackdat,
ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src,
ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm,
ct_dst_src_ltm

Notes: stcpb/dtcpb are TCP sequence numbers. Not identifiers in the usual sense, but closer to noise than behavior. Some papers drop them, so this is open for team discussion. 

## Missing values

Only two non target columns have missing values, both explained by flow type rather than data quality issues 

| Column | Missing | Reason |
|---|---|---|
| is_ftp_login | 1,429,879 | Only populated for FTP sessions |
| ct_flw_http_mthd | 1,348,145 | Only populated for HTTP sessions |

Handling is a preprocessing.py decision, not yet impemented.