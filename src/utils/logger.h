#ifndef SIMPLE_VIO_UTILS_LOGGER_H_
#define SIMPLE_VIO_UTILS_LOGGER_H_

#if defined(ANDROID)

#include <android/log.h>

#define LOGV(TAG, ...) \
    __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGI(TAG, ...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGD(TAG, ...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGW(TAG, ...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(TAG, ...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGF(TAG, ...) __android_log_print(ANDROID_LOG_FATAL, TAG, __VA_ARGS__)

#else  // Android

#include <stdio.h>

#define LOGV(TAG, ...) printf("V/" TAG ": " __VA_ARGS__), printf("\n")
#define LOGI(TAG, ...) printf("I/" TAG ": " __VA_ARGS__), printf("\n")
#define LOGD(TAG, ...) printf("D/" TAG ": " __VA_ARGS__), printf("\n")
#define LOGW(TAG, ...) printf("W/" TAG ": " __VA_ARGS__), printf("\n")
#define LOGE(TAG, ...) printf("E/" TAG ": " __VA_ARGS__), printf("\n")
#define LOGF(TAG, ...) printf("F/" TAG ": " __VA_ARGS__), printf("\n")

#endif  // ANDROID

#endif  // SIMPLE_VIO_UTILS_LOGGER_H_
