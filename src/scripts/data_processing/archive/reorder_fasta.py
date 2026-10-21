from Bio import SeqIO

TEMP_MESO_MAX=40
TEMP_THERMO_MIN=60

file_OGT = "/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/THOR_BIG/test_OGT_09_05.fasta" 

file_name_meso = "/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/THOR_BIG/test_OGT_09_05_meso.fasta"
file_name_thermo = "/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/THOR_BIG/test_OGT_09_05_thermo.fasta"

df_OGT = {}

for rec in SeqIO.parse(file_OGT, "fasta"):
    try:
        df_OGT[rec.id] = {"seq": str(rec.seq), "Temperature": float(rec.description.split()[-1])}
    except:
        df_OGT[rec.id] = {"seq": str(rec.seq), "Temperature": float(rec.description.split()[-2])}


# Separate out thermophiles and mesophiles

## Mesophiles 
cnt = 0
with open(file_name_meso, "w") as f:
    for item in df_OGT.items():
        if item[1]["Temperature"] <= TEMP_MESO_MAX:
            f.write(f">{item[0]} {item[1]['Temperature']}\n{item[1]['seq']}\n")
            cnt += 1
print(f"Wrote: {cnt} Mesophiles to {file_name_meso}")
## Thermophiles
cnt = 0
with open(file_name_thermo, "w") as f:
    for item in df_OGT.items():
        if item[1]["Temperature"] >= TEMP_THERMO_MIN:
            f.write(f">{item[0]} {item[1]['Temperature']}\n{item[1]['seq']}\n")
            cnt += 1
print(f"Wrote: {cnt} Thermophiles to {file_name_thermo}")