CREATE TABLE IF NOT EXISTS network_specs (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    architecture_type VARCHAR(64) NOT NULL,
    layer_count INTEGER NOT NULL,
    parameter_budget BIGINT NOT NULL,
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
