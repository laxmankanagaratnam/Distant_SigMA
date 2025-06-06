from astropy.table import Table


CSV_FILENAME = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Data/OrionBox/OrionRectangle_plx_oerror_bigger_4p5_fidelity_bigger_0p5.csv"  # Your VOTable with source_id column
# --- STEP 1: Convert CSV to VOTable (only source_id) ---
print(f"Reading {CSV_FILENAME} and converting to VOTable...")
csv_table = Table.read(CSV_FILENAME, format="csv")
# Keep only source_id column (assuming it's named correctly)
source_table = Table({'source_id': csv_table['source_id']})
VOT_FILENAME = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Data/OrionBox/OrionRectangle_plx_oerror_bigger_4p5_fidelity_bigger_0p5.vot"
source_table.write(VOT_FILENAME, format="votable", overwrite=True)
# print(f"Written temporary VOTable to {VOT_FILENAME}")