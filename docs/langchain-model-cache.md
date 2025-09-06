LangChain Model Cache Key

- Summary: The `LangChainOpenAIRequestAdapter` now caches `ChatOpenAI` / `ChatOpenAIWithCustomFields` instances by `(ProviderConfig, model_name)` instead of a string composed from `base_url`.
- Rationale: Different providers can share a `base_url` but differ in other fields (e.g., `adapter`, `api_key_env`, `timeouts_ms`). Using the full provider configuration prevents collisions and ensures correct behavior across multiple providers.
- Implementation Notes:
  - `ProviderConfig` is a Pydantic model with `model_config = ConfigDict(frozen=True)`, making it immutable and hashable.
  - `TimeoutsConfig` is also frozen to allow `ProviderConfig` hashing when the `timeouts_ms` field is set.
  - The cache field in `LangChainOpenAIRequestAdapter` is now typed as:
    - `dict[tuple[ProviderConfig, str], RunnableSerializable[Any, BaseMessage]]`
  - No public API changes; tests continue to pass.

