from Bio import SeqIO
import os
import sys

def extract_temp(description):
    """Extract temperature (assumed to be after space) from FASTA header."""
    try:
        return float(description.strip().split()[-1])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract temperature from: {description}")

def sort_and_split_fasta(input_file, num_parts=50):
    # Read all records from input fasta
    records = list(SeqIO.parse(input_file, "fasta"))

    # Attach temperature to each record
    records_with_temp = []
    for record in records:
        try:
            temp = extract_temp(record.description)
            records_with_temp.append((temp, record))
        except ValueError as e:
            print(e)

    # Sort by temperature
    records_sorted = [r for _, r in sorted(records_with_temp, key=lambda x: x[0])]

    total = len(records_sorted)
    print(f"Total sorted sequences: {total}")

    # Calculate chunk size
    chunk_size = (total + num_parts - 1) // num_parts  # ceiling division
    base_name = os.path.splitext(input_file)[0]

    for i in range(num_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        part_records = records_sorted[start:end]

        if not part_records:
            break

        output_file = f"{base_name}_{i}.fasta"
        SeqIO.write(part_records, output_file, "fasta")
        print(f"Written {len(part_records)} sequences to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sort_and_split_fasta.py input.fasta")
        sys.exit(1)

    input_fasta = sys.argv[1]
    print(f"processing file: {input_fasta}")
    sort_and_split_fasta(input_fasta)