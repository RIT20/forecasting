
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
                        and {category_col} ='{category}'
                        --('28 Ice(Purch/Bagged From Vndr)')
                         --'01 Propane Prefills','23 Other Publications')
                        --('14 Packaged Bread')
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
        max(effective_date) as last_sale_date,
        category
    from
        (
            SELECT
                a1.effective_date,a1.item,a1.loc,a1.channel,a1.net_Sales,a2.{category_col} as category
            FROM
                {db_name}.ml.fct_sales as a1
                inner join
                {db_name}.ml.dim_work_unit as a2
                on a1.item=a2.item
                and a1.loc=a2.loc
                and a1.channel = a2.channel
                and a1.tenant_id = a2.tenant_id
            where
            a1.tenant_id = '{tenant_id}'
            and a2.status = 1
            and a2.{category_col} ='{category}'

        )
    where
        item is not null
        and item != ''
        and effective_date between '{start_date}'
        and '{end_date}'
        and net_sales >= 0
    group by
        product_id,category
),
final as (
select
    t1.product_id,
    t1.week_effective_date,
    t1.sales,
    t2.first_sale_date,
    datediff(days, first_sale_date , '{end_date}') as age_in_days
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
    )

     )
    select  {select_col}
    --product_id,week_effective_date,sales
    from final
    order by product_id asc,week_effective_date asc, sales asc
    {limit_query}