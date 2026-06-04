/* mid_common_string.h — generated from mid-common ffi/string.rs */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct { uint8_t buf[32];  size_t len; } CMidFixedStr32;
typedef struct { uint8_t buf[64];  size_t len; } CMidFixedStr64;
typedef struct { uint8_t buf[256]; size_t len; } CMidFixedStr256;
typedef struct { uint32_t index; uint32_t score; } CSearchResult;
typedef struct MidStringSearch MidStringSearch;
typedef bool (*MidIsNameTaken)(const char* name, void* userdata);

/* FixedStr32 */
CMidFixedStr32  mid_fixed_str32_new(void);
CMidFixedStr32  mid_fixed_str32_from_cstr(const char* s);
size_t          mid_fixed_str32_push_str(CMidFixedStr32* str, const char* s);
size_t          mid_fixed_str32_set(CMidFixedStr32* str, const char* s);
void            mid_fixed_str32_clear(CMidFixedStr32* str);
const char*     mid_fixed_str32_as_ptr(const CMidFixedStr32* str);
size_t          mid_fixed_str32_len(const CMidFixedStr32* str);
bool            mid_fixed_str32_is_empty(const CMidFixedStr32* str);
bool            mid_fixed_str32_is_full(const CMidFixedStr32* str);

/* FixedStr64 */
CMidFixedStr64  mid_fixed_str64_new(void);
CMidFixedStr64  mid_fixed_str64_from_cstr(const char* s);
size_t          mid_fixed_str64_push_str(CMidFixedStr64* str, const char* s);
size_t          mid_fixed_str64_set(CMidFixedStr64* str, const char* s);
void            mid_fixed_str64_clear(CMidFixedStr64* str);
const char*     mid_fixed_str64_as_ptr(const CMidFixedStr64* str);
size_t          mid_fixed_str64_len(const CMidFixedStr64* str);
bool            mid_fixed_str64_is_empty(const CMidFixedStr64* str);
bool            mid_fixed_str64_is_full(const CMidFixedStr64* str);

/* FixedStr256 */
CMidFixedStr256 mid_fixed_str256_new(void);
CMidFixedStr256 mid_fixed_str256_from_cstr(const char* s);
size_t          mid_fixed_str256_push_str(CMidFixedStr256* str, const char* s);
size_t          mid_fixed_str256_set(CMidFixedStr256* str, const char* s);
void            mid_fixed_str256_clear(CMidFixedStr256* str);
const char*     mid_fixed_str256_as_ptr(const CMidFixedStr256* str);
size_t          mid_fixed_str256_len(const CMidFixedStr256* str);
bool            mid_fixed_str256_is_empty(const CMidFixedStr256* str);
bool            mid_fixed_str256_is_full(const CMidFixedStr256* str);

/* Utilities */
size_t   mid_flip_side_name(const char* name, char* out, size_t out_len);
uint32_t mid_split_name_number(const char* name, char delim, char* out_base, size_t out_base_len);
size_t   mid_uniquename(const char* name, char delim, MidIsNameTaken is_taken, void* userdata, char* out, size_t out_len);
uint64_t mid_damerau_levenshtein_distance(const char* a, const char* b);
bool     mid_fuzzy_match_score(const char* query, const char* full, uint64_t* out_score);

/* StringSearch */
MidStringSearch* mid_string_search_create(void);
void             mid_string_search_destroy(MidStringSearch* handle);
void             mid_string_search_add(MidStringSearch* handle, const char* name, uint32_t index, float weight);
void             mid_string_search_mark_recent(MidStringSearch* handle, const char* name);
size_t           mid_string_search_query(const MidStringSearch* handle, const char* query, CSearchResult* out, size_t max_results);
void             mid_string_search_clear(MidStringSearch* handle);
size_t           mid_string_search_len(const MidStringSearch* handle);
