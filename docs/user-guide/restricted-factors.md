# Restricted Factors

PLP includes restricted execution helpers for compiling and running sandboxed scoring code. This is useful when a factor should be provided as a small expression or script, but arbitrary Python execution would be too dangerous.

The intended use is narrow:

- evaluate simple domain scores;
- keep scoring logic auditable;
- avoid giving generated code access to the full Python runtime.

Restricted execution is not a substitute for a full security boundary. Treat it as a safer scoring helper inside a trusted application, and review any capability exposed to the sandbox.
