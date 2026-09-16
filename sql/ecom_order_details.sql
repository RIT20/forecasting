SELECT tenant_id, item, channel, loc,
	price_zone,
	<product_hierarchy>
	net_cost AS base_cost,
	price AS effective_price,
	quantity,
	net_sales,
	order_date as date
FROM DIM_ORDERS_DETAILS t1
LEFT JOIN DIM_WORK_UNIT b USING (tenant_id, item, channel, loc)
where t1.tenant_id = '<tenant_id>'
and t1.price > 0.1
and t1.quantity > 0.1
and t1.net_sales > 0.1