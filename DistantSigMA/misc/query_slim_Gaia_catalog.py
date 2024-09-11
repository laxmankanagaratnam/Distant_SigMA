from DistantSigMA.DistantSigMA.cluster_simulations import slim_sampling_data

from astroquery.gaia import Gaia
import pandas as pd

# Define the query to randomly sample 1% of stars within 1 kpc
query = """
SELECT *
FROM gaiadr3.gaia_source
WHERE parallax > 0 AND 1000/parallax <= 1000
AND random_index < 0.01
"""

# Gaia does not have a built-in RAND() function, so we use a workaround by using a random_index column
# Launch the asynchronous job
job = Gaia.launch_job_async(query, dump_to_file=True)

# Retrieve the results
results = job.get_results()

# Convert results to a Pandas DataFrame for easier manipulation
df = results.to_pandas()

# Display the first few rows of the DataFrame
print(df.head())

# Save results to a CSV file
df.to_csv("/Users/alena/PycharmProjects/Distant_SigMA/Data/Gaia/Gaia_DR3_1kpc_10percent.csv", index=False)

