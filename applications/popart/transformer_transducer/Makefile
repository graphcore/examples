BUILD_DIR = ./build

all: $(BUILD_DIR) rnnt_loss exp_avg feat_augmentation

$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)

.PHONY: rnnt_loss
rnnt_loss:
	$(MAKE) -C custom_ops/rnnt_loss

.PHONY: exp_avg
exp_avg:
	$(MAKE) -C custom_ops/exp_avg

.PHONY: feat_augmentation
feat_augmentation:
	$(MAKE) -C custom_ops/feat_augmentation

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
