"""SQL query constants for network_spec persistence."""

FLOW_INSERT_NETWORK_SPEC: str = (
    "INSERT INTO network_specs (id, name, architecture_type, layer_count, parameter_budget, status) "
    "VALUES (:id, :name, :architecture_type, :layer_count, :parameter_budget, :status);"
)

FLOW_SELECT_NETWORK_SPEC_BY_ID: str = (
    "SELECT id, name, architecture_type, layer_count, parameter_budget, status "
    "FROM network_specs WHERE id = :id;"
)
