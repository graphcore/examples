SUBDIRS = nms roi_align

cur_makefile_path := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all
all:
	@echo "// Copyright (c) 2021 Graphcore Ltd. All rights reserved." > include/customop.h
	@echo "#define CUSTOM_OPS_PATH \"$(cur_makefile_path)\"" >> include/customop.h
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
