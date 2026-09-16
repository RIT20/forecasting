with t1 as (
    select
        item || '$#$' || loc || '$#$' || channel as product_id,
        dateadd(day, - DAYOFWEEK(effective_date), effective_date) as week_sunday,
        sum(quantity) as sales
    from
        (
            SELECT
                *
            FROM
                { db_name }.ml.fct_sales QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY tenant_id,
                    item,
                    loc,
                    channel,
                    effective_date
                    ORDER BY
                        quantity desc
                ) = 1
        )
    where
        item is not null
        and item != ''
        and effective_date <= '{end_date}'
        and tenant_id = '{tenant_id}'
        and net_sales > 0
    group by
        product_id,
        week_sunday
)
select
    t1.product_id,
    sum(t1.sales) as sum_sales,
    count(t1.sales) as count_sales,
    (1 + sum_sales / count_sales) as scaling_factor
from
    t1
group by
    product_id