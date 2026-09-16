with t1 as (
    select
        item || '$#$' || loc || '$#$' || channel as product_id,
        date_trunc('week', effective_date) as week_effective_date,
        sum(quantity) as sales
    from
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
                        and uda_category = '{category}'
                )
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
                        and uda_category = '{category}'
                )
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
)
select
    t1.product_id,
    t1.week_effective_date,
    t1.sales,
    t2.first_sale_date,
    datediff(days, t2.first_sale_date, t1.week_effective_date) as age_in_days
from
    t1
    left join t2 on t1.product_id = t2.product_id
where
    t1.product_id in (
        select
            distinct product_id
        from
            t2
        where
            last_sale_date >= '{active_products_start_date}'
    )