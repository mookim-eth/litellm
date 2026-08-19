-- Track daily global original GPT spend without user-facing margin.
CREATE TABLE IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend" (
    "id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend_original" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyGlobalOriginalSpend_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_date_model_model_group_custom_llm_provider_endpoint_key"
ON "LiteLLM_DailyGlobalOriginalSpend"("date", "model", "model_group", "custom_llm_provider", "endpoint");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_date_idx" ON "LiteLLM_DailyGlobalOriginalSpend"("date");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_model_idx" ON "LiteLLM_DailyGlobalOriginalSpend"("model");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_model_group_idx" ON "LiteLLM_DailyGlobalOriginalSpend"("model_group");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_custom_llm_provider_idx" ON "LiteLLM_DailyGlobalOriginalSpend"("custom_llm_provider");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyGlobalOriginalSpend_endpoint_idx" ON "LiteLLM_DailyGlobalOriginalSpend"("endpoint");
