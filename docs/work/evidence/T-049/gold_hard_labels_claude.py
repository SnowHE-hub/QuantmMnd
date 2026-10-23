"""Claude 盲标 gold-hard（X4 v2.1，指南 v2）：编号 → 类型（空 = 不属于 14 类）。标注时只看了标题。"""
import json
import pandas as pd
R, U, RP, ST, CC, RL, LI, PF, PE, DV, HI = ("restructuring", "unlock", "repurchase", "st_change", "control_change",
                                             "regulatory_letter", "litigation", "performance_forecast",
                                             "performance_express", "dividend", "holder_increase")
L = {1: [R], 2: [ST], 3: [R], 5: [U], 13: [R], 15: [R], 16: [U], 17: [U], 23: [R], 27: [RL], 28: [U], 34: [R], 39: [U],
     40: [PF, PE], 45: [U], 47: [R], 48: [R], 49: [R], 50: [R], 51: [U], 52: [R], 53: [CC], 59: [ST], 60: [HI], 65: [R],
     67: [R], 78: [U], 81: [CC], 84: [R], 86: [R], 95: [R], 96: [R], 98: [U], 100: [U], 102: [U], 103: [U], 104: [PF],
     112: [R], 113: [RP], 114: [R], 115: [U], 121: [U], 124: [RP], 127: [R], 129: [R], 133: [R], 137: [R], 138: [R],
     140: [RP], 143: [R], 146: [U], 148: [U], 149: [U],
     151: [U], 154: [U], 158: [U], 160: [CC], 162: [ST], 168: [LI], 171: [U], 172: [U], 173: [R], 179: [DV], 187: [R],
     188: [R], 189: [ST], 190: [U], 191: [U], 192: [U], 193: [U], 194: [ST], 198: [U], 202: [DV], 208: [U], 217: [R],
     221: [U], 222: [U], 223: [CC], 225: [R], 226: [R], 227: [R], 228: [U], 231: [RL], 234: [R], 240: [U], 241: [U],
     242: [U], 245: [R], 247: [U], 258: [ST], 262: [RP], 265: [R], 267: [RP], 270: [U], 271: [RP], 275: [U], 276: [ST],
     278: [LI], 280: [U], 281: [R], 284: [U], 285: [U], 287: [U], 288: [U], 289: [U], 293: [R], 294: [R], 296: [U],
     297: [RP], 298: [U], 299: [U]}
D = "/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v21"
g = pd.read_parquet(f"{D}/gold_hard_index.parquet")
assert len(g) == 300
g["types"] = [L.get(i, []) for i in range(300)]
g.to_parquet(f"{D}/gold_hard_labeled.parquet")
print(len(g), sum(len(t) > 0 for t in g["types"]), json.dumps(pd.Series([t for ts in g["types"] for t in ts]).value_counts().to_dict()))
