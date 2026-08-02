# SPDX-License-Identifier: Apache-2.0
"""Additive OpenAI chat protocol extension for KV retention directives.

The image build preserves the exact vendor 0.21.1 protocol as
``protocol_base.py``.  Executing it here avoids replacing unrelated DeepSeek V4
protocol changes, then subclasses only ``ChatCompletionRequest`` with the
RFC-37003 request fields and validation.
"""

from pathlib import Path

_base = Path(__file__).with_name("protocol_base.py")
exec(compile(_base.read_bytes(), str(_base), "exec"), globals(), globals())

_BaseChatCompletionRequest = ChatCompletionRequest


class ChatCompletionRequest(_BaseChatCompletionRequest):
    retention_directives: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Priority KV-cache retention directives. A directive selects an "
            "explicit token range or uses covers_prompt/covers_output."
        ),
    )
    retention_scope: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Opaque owner scope for retention refresh/downgrade.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_retention_directives(cls, data):
        if not isinstance(data, dict):
            return data
        directives = data.get("retention_directives")
        if directives is None:
            return data
        if not isinstance(directives, list) or len(directives) > 16:
            raise VLLMValidationError(
                "`retention_directives` must be a list of at most 16 items.",
                parameter="retention_directives",
            )

        ordered: list[tuple[float, int, int]] = []
        for index, directive in enumerate(directives):
            if not isinstance(directive, dict):
                raise VLLMValidationError(
                    "Each retention directive must be an object.",
                    parameter="retention_directives",
                )
            covers_prompt = directive.get("covers_prompt") is True
            covers_output = directive.get("covers_output") is True
            if covers_prompt and covers_output:
                raise VLLMValidationError(
                    "A retention directive cannot cover both prompt and output.",
                    parameter="retention_directives",
                )
            if (covers_prompt or covers_output) and (
                "start" in directive or "end" in directive
            ):
                raise VLLMValidationError(
                    "covers_prompt/covers_output cannot be combined with start/end.",
                    parameter="retention_directives",
                )

            start = directive.get("start", 0)
            end = directive.get("end")
            priority = directive.get("priority", 0)
            duration = directive.get("duration")
            if (
                not isinstance(priority, int)
                or isinstance(priority, bool)
                or not 0 <= priority <= 100
            ):
                raise VLLMValidationError(
                    "Retention priority must be an integer from 0 through 100.",
                    parameter="retention_directives",
                )
            if duration is not None and (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not 0 < float(duration) <= 86400
            ):
                raise VLLMValidationError(
                    "Retention duration must be null or in (0, 86400] seconds.",
                    parameter="retention_directives",
                )
            if not covers_prompt and not covers_output:
                if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                    raise VLLMValidationError(
                        "Retention start must be a non-negative integer.",
                        parameter="retention_directives",
                    )
                if end is not None and (
                    not isinstance(end, int) or isinstance(end, bool) or end <= start
                ):
                    raise VLLMValidationError(
                        "Retention end must be null or an integer greater than start.",
                        parameter="retention_directives",
                    )
            position = 0 if covers_prompt else float("inf") if covers_output else start
            ordered.append((position, index, priority))

        previous_priority: int | None = None
        for _position, _index, priority in sorted(ordered):
            if previous_priority is not None and priority > previous_priority:
                raise VLLMValidationError(
                    "Retention priorities must be non-increasing with token position.",
                    parameter="retention_directives",
                )
            previous_priority = priority
        return data

    def to_sampling_params(self, *args, **kwargs):
        params = super().to_sampling_params(*args, **kwargs)
        if self.retention_directives is not None or self.retention_scope is not None:
            extra_args = dict(params.extra_args or {})
            if self.retention_directives is not None:
                extra_args["retention_directives"] = self.retention_directives
            if self.retention_scope is not None:
                extra_args["retention_scope"] = self.retention_scope
            params.extra_args = extra_args
        return params
