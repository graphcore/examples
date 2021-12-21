SUBDIRS = IPU/custom_ops/nms IPU/custom_ops/roi_align

.PHONY: all
all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
		echo "make in $$subdir";\
		$(MAKE) -C $$subdir;\
	done

.PHONY: clean
clean:
	@echo Making clean
	@list='$(SUBDIRS)'; for subdir in $$list; do \
		echo "make in $$subdir";\
		$(MAKE) -C $$subdir clean;\
	done