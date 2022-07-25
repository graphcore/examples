## Testing PopART BERT Advanced API

All tests must:

* Be deterministic, set seeds etc.

### Unit

* Test app specific code
* Requires **no** hardware to run.

**Examples:**

* Checking a learning rate function produces the correct values.
* Testing a layer constructs the correct graph.

### Integration

* Can require hardware
* Use OnDemand device acquisition.
* Compare against a reference model written in PyTorch.

**Examples:**

* Testing a layer for Layer testing
* Checking a profile for correct vertices
* 