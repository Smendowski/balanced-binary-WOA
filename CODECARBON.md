| Field              | Description                                                                                                                                                  |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **timestamp**       | Time of the experiment in `%Y-%m-%dT%H:%M:%S` format                                                                                                         |
| **project_name**    | Name of the project, defaults to `codecarbon`                                                                                                                |
| **run-id**          | ID of the run                                                                                                                                               |
| **duration**        | Duration of the compute, in seconds                                                                                                                          |
| **emissions**       | Emissions as CO₂-equivalents [CO₂eq], in kg                                                                                                                  |
| **emissions_rate**  | Emissions divided per duration, in Kg/s                                                                                                                     |
| **cpu_power**       | CPU power (W)                                                                                                                                               |
| **gpu_power**       | GPU power (W)                                                                                                                                               |
| **ram_power**       | RAM power (W)                                                                                                                                               |
| **cpu_energy**      | Energy used per CPU (kWh)                                                                                                                                   |
| **gpu_energy**      | Energy used per GPU (kWh)                                                                                                                                   |
| **ram_energy**      | Energy used per RAM (kWh)                                                                                                                                   |
| **energy_consumed** | Sum of `cpu_energy`, `gpu_energy`, and `ram_energy` (kWh)                                                                                                  |
| **country_name**    | Name of the country where the infrastructure is hosted                                                                                                      |
| **country_iso_code**| 3-letter alphabet ISO Code of the respective country                                                                                                        |
| **region**          | Province/State/City where the compute infrastructure is hosted                                                                                             |
| **on_cloud**        | `Y` if the infrastructure is on cloud, `N` in case of private infrastructure                                                                                |
| **cloud_provider**  | One of the 3 major cloud providers, `aws/azure/gcp`                                                                                                         |
| **cloud_region**    | Geographical Region for respective cloud provider, examples: `us-east-2` for aws, `brazilsouth` for azure, `asia-east1` for gcp                               |
| **os**              | OS on the device (e.g., `Windows-10-10.0.19044-SP0`)                                                                                                         |
| **python_version**  | Python version (e.g., `3.8.10`)                                                                                                                              |
| **cpu_count**       | Number of CPU                                                                                                                                              |
| **cpu_model**       | Example: `Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz`                                                                                                        |
| **gpu_count**       | Number of GPU                                                                                                                                              |
| **gpu_model**       | Example: `1 x NVIDIA GeForce GTX 1080 Ti`                                                                                                                   |
| **longitude**       | Longitude, with reduced precision to a range of 11.1 km / 123 km². This is done for privacy protection.                                                      |
| **latitude**        | Latitude, with reduced precision to a range of 11.1 km / 123 km². This is done for privacy protection.                                                      |
| **ram_total_size**  | Total RAM available (GB)                                                                                                                                     |
| **tracking_mode**   | `machine` or `process` (default is `machine`)                                                                                                                |
