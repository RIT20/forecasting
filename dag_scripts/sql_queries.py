last_run_date_query = """select last_run_date from {table} 
                        where tenant_id='{tenantid}' and 
                            module_name= '{module}' and 
                            job_name = '{jobname}';"""

latest_etl_query = """select max(updated_ts) as latest_date from dim_work_unit
                        where tenant_id='{tenantid}'"""

insert_qry = """insert into {table} 
                    (tenant_id, module_name, job_name, interval_in_days, last_run_date)
                values('{tenantid}', '{module}', '{job}', {interval}, '{curr_date}')"""

update_qry = """update {table}
                set last_run_date='{curr_date}'
                where   tenant_id = '{tenantid}' and  module_name='{module}' and job_name='{jobname}'"""
