with t1 as (
    select
        -- item as product_id,
        item || '$#$' || loc || '$#$' || channel as product_id,
        date_trunc('week', effective_date) as week_effective_date,
        sum(quantity) as sales
    from
        -- {db_name}.ml.fct_sales
        (
            SELECT
                *
            FROM
                {db_name}.ml.fct_sales
            where
                (item, loc, channel) in (
                    select
                        distinct item,
                        loc,
                        channel
                    from
                        {db_name}.ml.dim_work_unit
                    where
                        status = 1
                        and tenant_id = '{tenant_id}'
                        and uda_category='{category}'
                ) --QUALIFY ROW_NUMBER() OVER (PARTITION BY tenant_id, item, loc, channel, effective_date ORDER BY quantity desc) = 1
        )
    where
        item is not null
        and item != ''
        and effective_date between '{start_date}'
        and '{end_date}'
        and tenant_id = '{tenant_id}'
        and net_sales >= 0
        and item != 'SALES TAX'
    group by
        product_id,
        week_effective_date
),
t2 as (
    select
        -- item as product_id,
        item || '$#$' || loc || '$#$' || channel as product_id,
        min(effective_date) as first_sale_date,
        max(effective_date) as last_sale_date
    from
        -- {db_name}.ml.fct_sales
        (
            SELECT
                *
            FROM
                {db_name}.ml.fct_sales
            where
                (item, loc, channel) in (
                    select
                        distinct item,
                        loc,
                        channel
                    from
                        {db_name}.ml.dim_work_unit
                    where
                        status = 1
                        and tenant_id = '{tenant_id}'
                        and uda_category='{category}'
                ) --QUALIFY ROW_NUMBER() OVER (PARTITION BY tenant_id, item, loc, channel, effective_date ORDER BY quantity desc) = 1
        )
    where
        item is not null
        and item != ''
        and effective_date between '{start_date}'
        and '{end_date}'
        and tenant_id = '{tenant_id}'
        and net_sales >= 0
    group by
        product_id
) -- t3 as
-- (
-- select
--     item as product_id
--     ,avg(weight) as weight
--     ,mode(weight_uom) as weight_uom
--     ,mode(uda_level1) as uda_level1
--     ,mode(uda_level2) as uda_level2
--     ,mode(uda_level3) as uda_level3
--     ,mode(uda_level4) as uda_level4
--     ,mode(uda_level5) as uda_level5
--     ,mode(uda_level6) as uda_level6
--     ,mode(uda_attribute1) as uda_attribute1
--     ,mode(uda_attribute2) as uda_attribute2
--     ,mode(uda_attribute3) as uda_attribute3
--     ,mode(uda_attribute4) as uda_attribute4
--     ,mode(uda_attribute5) as uda_attribute5
-- from {db_name}.ml.dim_work_unit
-- where
--     tenant_id = '{tenant_id}'
-- group by product_id
-- ),
-- t4 as
-- (
-- select
--     product_id,
--     any_value(new_cluster) as cluster_id
-- from
--     {db_name}.ml.ecom_cluster
-- where
--     tenant_id = '{tenant_id}' and
--     running_id = '{cluster_run_id}'
-- group by product_id
-- )
select
    t1.product_id,
    t1.week_effective_date,
    t1.sales,
    t2.first_sale_date,
    datediff('day', t2.first_sale_date, select dateadd('day',6,(select max(week_effective_date) from t1)) as age_in_days -- t3.weight,
    -- t3.weight_uom,
    -- t3.uda_level1,
    -- t3.uda_level2,
    -- t3.uda_level3,
    -- t3.uda_level4,
    -- t3.uda_level5,
    -- t3.uda_level6,
    -- t3.uda_attribute1,
    -- t3.uda_attribute2,
    -- t3.uda_attribute3,
    -- t3.uda_attribute4,
    -- t3.uda_attribute5,
    -- t4.cluster_id
from
    t1
    left join t2 on t1.product_id = t2.product_id
where
    age_in_days > 70 and
    t1.product_id in (
        select
            distinct product_id
        from
            t2
        where
            last_sale_date >= '{active_products_start_date}'
    ) -- left join t3 on t1.product_id = t3.product_id
    -- left join t4 on t1.product_id = t4.product_id