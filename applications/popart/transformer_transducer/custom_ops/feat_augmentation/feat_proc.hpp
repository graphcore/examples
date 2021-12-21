// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include "gen_random.hpp"

extern "C" {

uint32_t featsOutSize(uint32_t* featsInShape, uint32_t stacking, uint32_t maxSeqLenAfterStacking);

// Synchronous API
void featProcess(float* featsIn, uint32_t* featsInShape, uint16_t* featsOut, uint32_t* featsOutShape, int32_t* featLens, uint32_t* featLensShape,
                 bool doSpecaugm, uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks0, float minTime0, float maxTime0,
                 uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking);

// Asynchronous API
void featProcInit(bool doSpecaugm, uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks, float minTime0, float maxTime,
                  uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking);
void featProcSubmit(int64_t tag,
                    float* featsIn, uint32_t* featsInShape,
                    uint16_t* featsOut, uint32_t* featsOutShape,
                    int32_t* featLens, uint32_t* featLensShape);
int64_t featProcGet();
void featProcStop();
uint32_t featProcCurrentQueueLen();

// Blocks of processing logic
uint32_t stackSubsampleSize(uint32_t* featsInShape, uint32_t stacking, uint32_t maxSeqLenAfterStacking);
void stackSubsample(float* featsIn, uint32_t* featsInShape, uint16_t* featsOut, uint32_t* featsOutShape, int32_t* featLens, uint32_t* featLensShape,
                    uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking);
void specAugment(float* feats, uint32_t* featsShape, int32_t* featLens,
                 uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks0, float minTime0, float maxTime0);

// Randomness support
void setRandomSeed(uint64_t seed);
void setRandomGen(IRndGenerator* ir);
int32_t getRandom(int32_t from, int32_t to);

}