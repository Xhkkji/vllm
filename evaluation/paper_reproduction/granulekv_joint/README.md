# GranuleKV Joint Experiment

The joint experiment composes an already validated selector, layer/speculative
prefetch plan, and request scheduler. It is the final stage; it must not be
used to hide a failure in an individual component.
