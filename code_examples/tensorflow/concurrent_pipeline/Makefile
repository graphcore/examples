TARGETS := all clean

DIRS := custom_ops

$(TARGETS): $(DIRS)
$(DIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: $(TARGETS) $(DIRS)

