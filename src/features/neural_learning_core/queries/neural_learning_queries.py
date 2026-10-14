FLOW_INSERT_TRAINING_RUN: str = (
    "INSERT INTO neural_training_runs (model_name, initial_loss, final_loss, steps_taken, status, created_at) "
    "VALUES (:model_name, :initial_loss, :final_loss, :steps_taken, :status, :created_at);"
)
