import pyvo
from astropy.table import Table

# ---- CONFIGURATION ----
TAP_URL = "https://dc.g-vo.org/tap"  # GAVO TAP service with gedr3spur

VOT_FILENAME = "/Users/alena/Downloads/OrionRectangle_plx_oerror.vot"  # Your VOTable with source_id column
UPLOAD_TABLE_NAME = "t1"            # The name to refer to in ADQL
OUTPUT_FILENAME = "/Users/alena/Downloads/OrionRectangle_plx_oerror_bigger_4p5_fidelity.vot"  # Output file

# ---- LOAD VOTABLE ----
upload_table = Table.read(VOT_FILENAME)
# ---- CONNECT TO TAP SERVICE ----
tap_service = pyvo.dal.TAPService(TAP_URL)

# ---- SUBMIT QUERY (with upload) ----
job = tap_service.submit_job(
    f"""
    SELECT gaia.source_id, gaia.fidelity_v2
    FROM gedr3spur.main AS gaia
    JOIN TAP_UPLOAD.{UPLOAD_TABLE_NAME} AS mine
    USING (source_id)
    """,
    uploads={UPLOAD_TABLE_NAME: upload_table},
    language="ADQL",
    maxrec=2000000
)

job.run()
job.wait()

# ---- FETCH RESULT ----
results = job.fetch_result()
results.to_table().write(OUTPUT_FILENAME, format="votable")

print(f"Saved {len(results)} rows to {OUTPUT_FILENAME}")
