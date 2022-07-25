// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <stdio.h>
#include <memory.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <stdlib.h>
#include <string.h>

#include "feat_proc.hpp"

extern "C" {

class FeatProcAsync1Thread {
public:
    static FeatProcAsync1Thread& getInstance() {
        static FeatProcAsync1Thread instance;
        return instance;
    }

    struct Params {
        bool doSpecaugm;
        uint32_t freqMasks;
        uint32_t minFreq;
        uint32_t maxFreq;
        float timeMasks;
        float minTime;
        float maxTime;
        uint32_t stacking;
        uint32_t subsampling;
        uint32_t maxSeqLenAfterStacking;
    };

    void init(const Params& params_) {
        assert(!initialized);
        if (initialized) {
            return;
        }
        params = params_;
        
        initialized = true;

        for (size_t i = 0; i < numThreads; ++i) {
            threadPool.push_back(std::move(std::thread([&] { run(); })));
        }

        char* rnntLogLevel = getenv("RNNT_LOG_LEVEL");
        if (rnntLogLevel) {
            debugLog = (strcmp(rnntLogLevel, "DEBUG") == 0) ||
                       (strcmp(rnntLogLevel, "TRACE") == 0);
        }
    }

    void submit(int64_t tag,
                float* featsIn, uint32_t* featsInShape,
                uint16_t* featsOut, uint32_t* featsOutShape,
                int32_t* featLens, uint32_t* featLensShape) {
        assert(initialized);

        std::shared_ptr<InputArgs> spInputArgs = std::make_shared<InputArgs>();
        InputArgs &inputArgs = *spInputArgs;

        inputArgs.tag = tag;
        inputArgs.featsIn = featsIn;
        inputArgs.featsInShape = featsInShape;
        inputArgs.featLens = featLens;
        inputArgs.featLensShape = featLensShape;
        inputArgs.featsOut = featsOut;
        inputArgs.featsOutShape = featsOutShape;

        std::unique_lock<std::mutex> lk(mtxSubmit);
        inputQueue.push(spInputArgs);
        lk.unlock();
        cvSubmit.notify_one();
    }

private:
    struct InputArgs {
        int64_t tag;
        float* featsIn;
        uint32_t* featsInShape;
        int32_t* featLens;
        uint32_t* featLensShape;
        uint16_t* featsOut;
        uint32_t* featsOutShape;
    };

    struct OutputArgs {
        int64_t tag;
    };

    void run() {
        while (true) {
            std::unique_lock<std::mutex> lkSubmit(mtxSubmit);
            cvSubmit.wait(lkSubmit, [&]{ return !inputQueue.empty() || stopSignal; });

            if (stopSignal) {
                break;
            }

            std::shared_ptr<InputArgs> spInputArgs = inputQueue.front();
            inputQueue.pop();
            lkSubmit.unlock();

            std::shared_ptr<OutputArgs> spOutputArgs = execute(spInputArgs);

            std::unique_lock<std::mutex> lkGet(mtxGet);
            outputQueue.push(spOutputArgs);
            lkGet.unlock();
            cvGet.notify_one();
        }
    }

    std::shared_ptr<OutputArgs> execute(std::shared_ptr<InputArgs> spInputArgs) {
        InputArgs &inputArgs = *spInputArgs;

        auto start = std::chrono::system_clock::now();

        assert(inputArgs.featsIn);
        assert(inputArgs.featsInShape);
        assert(inputArgs.featsOut);
        assert(inputArgs.featsOutShape);
        assert(inputArgs.featLens);
        assert(inputArgs.featLensShape);

        featProcess(inputArgs.featsIn, inputArgs.featsInShape, inputArgs.featsOut, inputArgs.featsOutShape, inputArgs.featLens, inputArgs.featLensShape,
                    params.doSpecaugm, params.freqMasks, params.minFreq, params.maxFreq, params.timeMasks, params.minTime, params.maxTime,
                    params.stacking, params.subsampling, params.maxSeqLenAfterStacking);

        if (debugLog) {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsedSeconds = end - start;
            std::cout << "Feature processing time = " << std::fixed << std::setprecision(3) << elapsedSeconds.count() << std::endl;
        }

        std::shared_ptr<OutputArgs> spOutputArgs = std::make_shared<OutputArgs>();
        OutputArgs &outputArgs = *spOutputArgs;
        outputArgs.tag = inputArgs.tag;

        return spOutputArgs;
    }

public:
    int64_t get() {
        assert(initialized);

        std::unique_lock<std::mutex> lk(mtxGet);
        cvGet.wait(lk, [&]{ return !outputQueue.empty() || stopSignal; });

        if (stopSignal) {
            return 0;
        }

        std::shared_ptr<OutputArgs> spOutputArgs = outputQueue.front();
        outputQueue.pop();
        lk.unlock();

        OutputArgs &outputArgs = *spOutputArgs;
        return outputArgs.tag;
    }

    uint32_t currentQueueLen() {
        std::unique_lock<std::mutex> lk(mtxSubmit);
        int32_t ql = inputQueue.size();
        lk.unlock();
        return ql;
    }

    void stop() {
        stopSignal = true;
        cvSubmit.notify_all();

        for (size_t i = 0; i < threadPool.size(); ++i) {
            threadPool[i].join();
        }
    }

protected:
    unsigned numThreads = 1;

private:
    bool initialized = false;
    Params params;

    std::queue<std::shared_ptr<InputArgs>> inputQueue;
    std::queue<std::shared_ptr<OutputArgs>> outputQueue;
    std::vector<std::thread> threadPool;
    std::condition_variable cvSubmit;
    std::condition_variable cvGet;
    std::mutex mtxSubmit;
    std::mutex mtxGet;
    std::atomic<bool> stopSignal;
    bool debugLog = false;
};

void featProcInit(bool doSpecaugm, uint32_t freqMasks, uint32_t minFreq, uint32_t maxFreq, float timeMasks, float minTime, float maxTime,
                  uint32_t stacking, uint32_t subsampling, uint32_t maxSeqLenAfterStacking) {
    FeatProcAsync1Thread::Params params;
    params.doSpecaugm = doSpecaugm;
    params.freqMasks = freqMasks;
    params.minFreq = minFreq;
    params.maxFreq = maxFreq;
    params.timeMasks = timeMasks;
    params.minTime = minTime;
    params.maxTime = maxTime;
    params.stacking = stacking;
    params.subsampling = subsampling;
    params.maxSeqLenAfterStacking = maxSeqLenAfterStacking;
    FeatProcAsync1Thread::getInstance().init(params);
}

void featProcSubmit(int64_t tag,
                    float* featsIn, uint32_t* featsInShape,
                    uint16_t* featsOut, uint32_t* featsOutShape,
                    int32_t* featLens, uint32_t* featLensShape) {
    FeatProcAsync1Thread::getInstance().submit(tag, featsIn, featsInShape, featsOut, featsOutShape, featLens, featLensShape);
}

int64_t featProcGet() {
    return FeatProcAsync1Thread::getInstance().get();
}

void featProcStop() {
    FeatProcAsync1Thread::getInstance().stop();
}

uint32_t featProcCurrentQueueLen() {
    return FeatProcAsync1Thread::getInstance().currentQueueLen();
}

}