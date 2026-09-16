select
    distinct a.loc as store,
    b.uda_category as category
from
    (
        select
            *
        from
            { db_name }.ml.fct_sales
        where
            tenant_id = '{tenant_id}'
    ) a
    inner join (
        select
            *
        from
            { db_name }.ml.dim_work_unit
        where
            tenant_id = '{tenant_id}'
            and status = 1
    ) b on a.item = b.item
    and a.loc = b.loc
    and a.channel = b.channel;