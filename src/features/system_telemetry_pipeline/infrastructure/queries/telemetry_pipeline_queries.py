FLOW_INSERT_PIPELINE_RECORD: str = (
    "INSERT INTO telemetry_pipeline_records (pipeline_id, task_name, status, majority_class, accuracy, f1_score, created_at) "
    "VALUES (:pipeline_id, :task_name, :status, :majority_class, :accuracy, :f1_score, :created_at);"
)
