Retail Sales Forecasting

A pipeline that predicts how many units each product will sell, at each store, for the next few weeks. It reads past sales from Snowflake, trains a model per product category, and writes weekly forecasts to db.

What it does
For every product-store-channel combination, the pipeline produces a weekly sales forecast for the next H weeks (default: 4).

Two models work together:

Sales model — predicts how many units will sell in a given week.
Occurrence model — predicts the chance a product sells at all that week.
Final forecast = sales prediction × occurrence probability. Splitting the two lets the sales model focus on the size of the sale while the occurrence model handles products that only sell occasionally.

How it works
For each product category, the pipeline runs four steps:

Pull data from two Snowflake tables:
fct_sales — the transactions.
dim_work_unit — the list of active products with their category.
Build features — weekly sales lags (last 10 weeks), holiday counts for the next few weeks, seasonality (week / month / quarter), product age, and a couple of signals for products with lots of zero-sales weeks.
Tune and train — Optuna searches for good LightGBM settings using time-based validation, then trains the final model.
Predict and save — forecasts are written to forecast_regression/<tenant>_<start_date>/*.csv.
Categories run one at a time so each one can use all CPU cores. The upstream data preparation for many categories runs in parallel.

Repo layout
driver_regression.py       # entry point for the sales model
driver_do.py               # entry point for the occurrence model
src/                       # tuning, training, inference code
utils/                     # data prep, database, parallelism helpers
core_utils/                # config loader, Snowflake connection, logger
sql/                       # SQL templates read by the pipeline
config/                    # YAML config (Snowflake creds, model defaults)
dag_scripts/               # scheduler that decides which job to run per tenant
Dockerfile
pyproject.toml
Run it
Requires Python 3.9 and Poetry.

poetry install
Set Snowflake credentials in config/snowflake_config.yaml (or via environment variables USER, PASSWORD, ACCOUNT, DB, SCHEMA, WAREHOUSE).

Edit the tenant / date / horizon at the bottom of the driver, then:

poetry run python driver_regression.py
poetry run python driver_do.py
Or with Docker:

docker build -t sales-forecasting .
docker run --env-file .env sales-forecasting --tenant_id <tenant_name> --job_id <job_to_run>
Outputs
prepared_data_regression/<tenant>/*.parquet.gzip — feature tables.
tuning_regression/<tenant>/<category>/params_1.pickle — best model settings.
training_regression/<tenant>/<category>/model_1_0.5.txt — trained model.
forecast_regression/<tenant>_<start_date>/*.csv — the actual forecasts.
Timing and memory info are appended to output.txt while the job runs.

Config
Set at the top of each driver:

Key	What it does
tenant_id	Which customer to run for.
env_name	DEV or PROD — picks credentials.
start_date	How far back to pull training data.
infer_start_date	The Sunday the forecast starts from.
horizon	Number of weeks to forecast.
sales_lags	How many past weeks of sales to use as features (default 10).
tuning_n_trials	How hard Optuna searches for good settings.
data_save_method	local (Parquet) or db (Snowflake table).
Scheduled runs
dag_scripts/dag_runner.py reads dag_scripts/schedule_interval.yaml to decide when each job runs per tenant:

run-only-tuning: 30      # re-tune every 30 days
run-wo-tuning: 14        # retrain + forecast every 14 days
It checks the last-run date in a scheduler table in Snowflake and only fires jobs whose interval has passed.

Tech stack
Python 3.9, LightGBM, Optuna, pandas, Snowflake, joblib, Docker.

Notes
The pipeline treats item$#$loc$#$channel as a single product ID. Split on $#$ to get the parts back.
Categories with too few active products end up in no_data_cat_<tenant>.txt and are skipped.
Only two Snowflake tables are used — no pricing, promo, or inventory data is involved.
