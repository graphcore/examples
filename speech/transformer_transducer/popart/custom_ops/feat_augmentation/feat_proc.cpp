// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "feat_proc.hpp"
#include "np_fp16_convert.hpp"
#include "torch_random.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <assert.h>

extern "C" {

static const size_t FEATS_RANK = 3;

uint32_t stackSubsampleSize(uint32_t* featsInShape, uint32_t stacking, uint32_t maxSeqLenAfterStacking) {
    uint32_t featsOutSize = featsInShape[0] * (featsInShape[1] * stacking) * maxSeqLenAfterStacking;
    return featsOutSize;
}

/*
Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

Truncates/pads frames in a time dimension up to maximum sequence length

Converts features to float16 format

See also rnnt_speech_recognition/common/data/features.py

    params:
    
    featsIn: [in] input features
    featsInShape: [in] input features shape
    featsOut: [out] output features
    featsOutShape: [out] output features shape
    featLens: [inout] features lengths
    featLensShape: [in] features lengths shape
    stacking: [in] stacking level
    subsampling: [in] subsampling level
    maxSeqLenAfterStacking: [in] maximum sequence length
*/
void stackSubsample(float* featsIn, uint32_t* featsInShape, uint16_t* featsOut, uint32_t* featsOutShape, int32_t* featLens, uint32_t* featLensShape,
                    uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking) { 
    assert(stacking > 0);
    assert(subsampling > 0);
    for (unsigned i = 0; i < static_cast<unsigned>(featLensShape[0]); ++i) {
        featLens[i] = (featLens[i] + subsampling - 1) / subsampling;
    }

    featsOutShape[0] = featsInShape[0];
    featsOutShape[1] = featsInShape[1] * stacking;
    featsOutShape[2] = maxSeqLenAfterStacking;

    size_t sizeInJK = featsInShape[1] * featsInShape[2];
    unsigned sizeInK = featsInShape[2];
    size_t sizeOutJK = featsOutShape[1] * featsOutShape[2];
    unsigned sizeOutK = featsOutShape[2];

    size_t featsOutSize = featsOutShape[0] * featsOutShape[1] * featsOutShape[2];
    memset(featsOut, 0, featsOutSize * sizeof(uint16_t));
    size_t offsetInI = 0;
    size_t offsetOutI = 0;
    for (unsigned i = 0; i < featsInShape[0]; ++i, offsetInI += sizeInJK, offsetOutI += sizeOutJK) {
        unsigned maxLenK = std::min(sizeOutK, static_cast<unsigned>(featLens[i]));
        for (unsigned idxStack = 0, offsetOutJ = offsetOutI; idxStack < stacking; ++idxStack) {
            size_t offsetInJ = offsetInI;
            for (unsigned jIn = 0; jIn < featsInShape[1]; ++jIn, offsetInJ += sizeInK, offsetOutJ += sizeOutK) {
                size_t ijkOut = offsetOutJ;
                for (unsigned kIn = idxStack, kOut = 0; kIn < sizeInK && kOut < maxLenK; kIn += subsampling, ++kOut, ++ijkOut) {
                    size_t ijkIn = offsetInJ + kIn;
                    featsOut[ijkOut] = npy_float_to_half(featsIn[ijkIn]);
                }
            }
        }
    }
#ifdef TRACE
    printf("featsOut:\n");
    for (unsigned i = 0, idx = 0; i < featsOutShape[0]; ++i) {
        for (unsigned j = 0; j < featsOutShape[1]; ++j) {
            for (unsigned k = 0; k < featsOutShape[2]; ++k, ++idx) {
                printf("%f ", featsOut[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
}

/*
Regularize by masking entire time steps/frequency bands.

    Implementes SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

See also rnnt_speech_recognition/common/data/features.py

    params:
    
    feats: [inout] features
    featsShape: [in] features shape
    featLens: [in] features lengths
    freqMasks: [in] number of masks for frequency bands
    featsShape: [in] minimum number of frequencies in a single mask
    maxFreq: [in] maximum number of frequencies in a single mask
    timeMasks0: [in] number of masks or adaptive percentage
    minTime0: [in] minimum number of masked time steps per mask; applies
        only if max is non-adaptive
    maxTime0: [in] maximum number of masked time steps per mask,
        value 0 < 1 then denotes adaptive percentage

*/
void specAugment(float* feats, uint32_t* featsShape, int32_t* featLens,
                 uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks0, float minTime0, float maxTime0) {
    
    std::fesetround(FE_TONEAREST);

    assert(maxFreq >= minFreq);

    size_t idx = 0;
    for (unsigned i = 0; i < featsShape[0]; ++i) {
        // Adaptive frquency masking
        std::vector<unsigned char> maskFreq(featsShape[1], 1);
        
        for (unsigned iFm = 0; iFm < freqMasks; ++iFm) {
            unsigned freqWnd = getRandom(minFreq, maxFreq + 1);
            unsigned freqStart = getRandom(0, std::max(1U, featsShape[1] - freqWnd + 1));
            assert(freqStart + freqWnd <= static_cast<unsigned>(maskFreq.size()));

            unsigned char* ptrMask = maskFreq.data() + freqStart;
            size_t lenMaskByte = freqWnd * sizeof(unsigned char);
            memset(ptrMask, 0, lenMaskByte);
        }

        // Adaptive time masking
        std::vector<unsigned char> maskTime(featsShape[2], 1);

        unsigned timeMasks;
        if (0.0f < timeMasks0 && timeMasks0 < 1.0f) {
            timeMasks = std::nearbyint(featLens[i] * timeMasks0);
        } else {
            timeMasks = static_cast<unsigned>(timeMasks0);
        }

        unsigned maxTime;
        if (0.0f < maxTime0 && maxTime0 < 1.0f) {
            maxTime = std::nearbyint(featLens[i] * maxTime0);
        } else {
            maxTime = static_cast<unsigned>(maxTime0);
        }
        unsigned minTime = minTime0;
        assert(maxTime >= minTime);

        for (unsigned iTm = 0; iTm < timeMasks; ++iTm) {
            unsigned timeWnd = getRandom(minTime, maxTime + 1);
            unsigned timeStart = getRandom(0, std::max(1U, featsShape[2] - timeWnd + 1));
            assert(timeStart + timeWnd <= static_cast<unsigned>(maskTime.size()));

            unsigned char* ptrMask = maskTime.data() + timeStart;
            size_t lenMaskByte = timeWnd * sizeof(unsigned char);
            memset(ptrMask, 0, lenMaskByte);
        }

        for (unsigned j = 0; j < featsShape[1]; ++j) {
            for (unsigned k = 0; k < featsShape[2]; ++k, ++idx) {
                feats[idx] *= (maskFreq[j] * maskTime[k]);
            }
        }
    }

#ifdef TRACE
    printf("feats:\n");
    for (unsigned i = 0, idx = 0; i < featsShape[0]; ++i) {
        for (unsigned j = 0; j < featsShape[1]; ++j) {
            for (unsigned k = 0; k < featsShape[2]; ++k, ++idx) {
                printf("%f ", feats[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
}

static IRndGenerator* getDefaultRng() {
    static at::RndGenerator rng;
    return &rng;
}

static IRndGenerator* iRng = getDefaultRng();

int32_t getRandom(int32_t from, int32_t to) {
    return iRng->getRandom(from, to);
}

void setRandomSeed(uint64_t seed) {
    iRng->setRandomSeed(seed);
}

void setRandomGen(IRndGenerator* ir) {
    iRng = ir;
}

uint32_t featsOutSize(uint32_t* featsInShape, uint32_t stacking, uint32_t maxSeqLenAfterStacking) {
    return stackSubsampleSize(featsInShape, stacking, maxSeqLenAfterStacking);
}

void featProcess(float* featsIn, uint32_t* featsInShape, uint16_t* featsOut, uint32_t* featsOutShape, int32_t* featLens, uint32_t* featLensShape,
                 bool doSpecaugm, uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks0, float minTime0, float maxTime0,
                 uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking) {
    if (doSpecaugm) {
        specAugment(featsIn, featsInShape, featLens, freqMasks, minFreq, maxFreq, timeMasks0, minTime0, maxTime0);
    }
    stackSubsample(featsIn, featsInShape, featsOut, featsOutShape, featLens, featLensShape, stacking, subsampling, maxSeqLenAfterStacking);
}

}
