# 1 "CMakeCUDACompilerId.cu"
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **);__attribute__((unused))  static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#define __nv_is_extended_device_lambda_with_preserved_return_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)&& defined(__nv_is_extended_device_lambda_with_preserved_return_type)
#endif

# 1
# 8 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/_cplus_preinclude.h" 3
struct __va_list_tag { 
# 9
unsigned gp_offset; 
# 10
unsigned fp_offset; 
# 11
char *overflow_arg_area; 
# 12
char *reg_save_area; 
# 13
}; 
# 28 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/_cplus_preinclude.h" 3
typedef __va_list_tag __pgi_va_list[1]; 
# 9 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/openacc_predef.h" 3
#pragma acc routine seq
extern "C" void __cxa_vec_ctor(void * __array_address, unsigned long __element_count, unsigned long __element_size, void (* __constructor)(void *), void (* __destructor)(void *)); 
# 16
#pragma acc routine seq
extern "C" void __cxa_vec_cctor(void * __destination_array, void * __source_array, unsigned long __element_count, unsigned long __element_size, void (* __constructor)(void *, void *), void (* __destructor)(void *)); 
# 24
#pragma acc routine seq
extern "C" void __cxa_vec_dtor(void * __array_address, unsigned long __element_count, unsigned long __element_size, void (* __destructor)(void *)); 
# 30
#pragma acc routine seq
extern "C" void *__cxa_vec_new(unsigned long __element_count, unsigned long __element_size, unsigned long __padding_size, void (* __constructor)(void *), void (* __destructor)(void *)); 
# 37
#pragma acc routine seq
extern "C" void *__cxa_vec_new2(unsigned long __element_count, unsigned long __element_size, unsigned long __padding_size, void (* __constructor)(void *), void (* __destructor)(void *), void *(* __allocator)(unsigned long), void (* __deallocator)(void *)); 
# 46
#pragma acc routine seq
extern "C" void *__cxa_vec_new3(unsigned long __element_count, unsigned long __element_size, unsigned long __padding_size, void (* __constructor)(void *), void (* __destructor)(void *), void *(* __allocator)(unsigned long), void (* __deallocator)(void *, unsigned long)); 
# 55
#pragma acc routine seq
extern "C" void __cxa_vec_delete(void * __array_address, unsigned long __element_size, unsigned long __padding_size, void (* __destructor)(void *)); 
# 61
#pragma acc routine seq
extern "C" void __cxa_vec_delete2(void * __array_address, unsigned long __element_size, unsigned long __padding_size, void (* __destructor)(void *), void (* __deallocator)(void *)); 
# 68
#pragma acc routine seq
extern "C" void __cxa_vec_delete3(void * __array_address, unsigned long __element_size, unsigned long __padding_size, void (* __destructor)(void *), void (* __deallocator)(void *, unsigned long)); 
# 31 "/usr/include/bits/types.h" 3
typedef unsigned char __u_char; 
# 32
typedef unsigned short __u_short; 
# 33
typedef unsigned __u_int; 
# 34
typedef unsigned long __u_long; 
# 37
typedef signed char __int8_t; 
# 38
typedef unsigned char __uint8_t; 
# 39
typedef signed short __int16_t; 
# 40
typedef unsigned short __uint16_t; 
# 41
typedef signed int __int32_t; 
# 42
typedef unsigned __uint32_t; 
# 44
typedef signed long __int64_t; 
# 45
typedef unsigned long __uint64_t; 
# 52
typedef __int8_t __int_least8_t; 
# 53
typedef __uint8_t __uint_least8_t; 
# 54
typedef __int16_t __int_least16_t; 
# 55
typedef __uint16_t __uint_least16_t; 
# 56
typedef __int32_t __int_least32_t; 
# 57
typedef __uint32_t __uint_least32_t; 
# 58
typedef __int64_t __int_least64_t; 
# 59
typedef __uint64_t __uint_least64_t; 
# 63
typedef long __quad_t; 
# 64
typedef unsigned long __u_quad_t; 
# 72
typedef long __intmax_t; 
# 73
typedef unsigned long __uintmax_t; 
# 145 "/usr/include/bits/types.h" 3
typedef unsigned long __dev_t; 
# 146
typedef unsigned __uid_t; 
# 147
typedef unsigned __gid_t; 
# 148
typedef unsigned long __ino_t; 
# 149
typedef unsigned long __ino64_t; 
# 150
typedef unsigned __mode_t; 
# 151
typedef unsigned long __nlink_t; 
# 152
typedef long __off_t; 
# 153
typedef long __off64_t; 
# 154
typedef int __pid_t; 
# 155
typedef struct { int __val[2]; } __fsid_t; 
# 156
typedef long __clock_t; 
# 157
typedef unsigned long __rlim_t; 
# 158
typedef unsigned long __rlim64_t; 
# 159
typedef unsigned __id_t; 
# 160
typedef long __time_t; 
# 161
typedef unsigned __useconds_t; 
# 162
typedef long __suseconds_t; 
# 164
typedef int __daddr_t; 
# 165
typedef int __key_t; 
# 168
typedef int __clockid_t; 
# 171
typedef void *__timer_t; 
# 174
typedef long __blksize_t; 
# 179
typedef long __blkcnt_t; 
# 180
typedef long __blkcnt64_t; 
# 183
typedef unsigned long __fsblkcnt_t; 
# 184
typedef unsigned long __fsblkcnt64_t; 
# 187
typedef unsigned long __fsfilcnt_t; 
# 188
typedef unsigned long __fsfilcnt64_t; 
# 191
typedef long __fsword_t; 
# 193
typedef long __ssize_t; 
# 196
typedef long __syscall_slong_t; 
# 198
typedef unsigned long __syscall_ulong_t; 
# 202
typedef __off64_t __loff_t; 
# 203
typedef char *__caddr_t; 
# 206
typedef long __intptr_t; 
# 209
typedef unsigned __socklen_t; 
# 214
typedef int __sig_atomic_t; 
# 28 "/usr/include/ctype.h" 3
extern "C" {
# 47 "/usr/include/ctype.h" 3
enum { 
# 48
_ISupper = ((0 < 8) ? (1 << 0) << 8 : ((1 << 0) >> 8)), 
# 49
_ISlower = ((1 < 8) ? (1 << 1) << 8 : ((1 << 1) >> 8)), 
# 50
_ISalpha = ((2 < 8) ? (1 << 2) << 8 : ((1 << 2) >> 8)), 
# 51
_ISdigit = ((3 < 8) ? (1 << 3) << 8 : ((1 << 3) >> 8)), 
# 52
_ISxdigit = ((4 < 8) ? (1 << 4) << 8 : ((1 << 4) >> 8)), 
# 53
_ISspace = ((5 < 8) ? (1 << 5) << 8 : ((1 << 5) >> 8)), 
# 54
_ISprint = ((6 < 8) ? (1 << 6) << 8 : ((1 << 6) >> 8)), 
# 55
_ISgraph = ((7 < 8) ? (1 << 7) << 8 : ((1 << 7) >> 8)), 
# 56
_ISblank = ((8 < 8) ? (1 << 8) << 8 : ((1 << 8) >> 8)), 
# 57
_IScntrl, 
# 58
_ISpunct = ((10 < 8) ? (1 << 10) << 8 : ((1 << 10) >> 8)), 
# 59
_ISalnum = ((11 < 8) ? (1 << 11) << 8 : ((1 << 11) >> 8))
# 60
}; 
# 79
extern const unsigned short **__ctype_b_loc() throw()
# 80
 __attribute((const)); 
# 81
extern const __int32_t **__ctype_tolower_loc() throw()
# 82
 __attribute((const)); 
# 83
extern const __int32_t **__ctype_toupper_loc() throw()
# 84
 __attribute((const)); 
# 108 "/usr/include/ctype.h" 3
extern int isalnum(int) throw(); 
# 109
extern int isalpha(int) throw(); 
# 110
extern int iscntrl(int) throw(); 
# 111
extern int isdigit(int) throw(); 
# 112
extern int islower(int) throw(); 
# 113
extern int isgraph(int) throw(); 
# 114
extern int isprint(int) throw(); 
# 115
extern int ispunct(int) throw(); 
# 116
extern int isspace(int) throw(); 
# 117
extern int isupper(int) throw(); 
# 118
extern int isxdigit(int) throw(); 
# 122
extern int tolower(int __c) throw(); 
# 125
extern int toupper(int __c) throw(); 
# 130
extern int isblank(int) throw(); 
# 135
extern int isctype(int __c, int __mask) throw(); 
# 142
extern int isascii(int __c) throw(); 
# 146
extern int toascii(int __c) throw(); 
# 150
extern int _toupper(int) throw(); 
# 151
extern int _tolower(int) throw(); 
# 28 "/usr/include/bits/types/__locale_t.h" 3
struct __locale_struct { 
# 31
struct __locale_data *__locales[13]; 
# 34
const unsigned short *__ctype_b; 
# 35
const int *__ctype_tolower; 
# 36
const int *__ctype_toupper; 
# 39
const char *__names[13]; 
# 40
}; 
# 42
typedef __locale_struct *__locale_t; 
# 24 "/usr/include/bits/types/locale_t.h" 3
typedef __locale_t locale_t; 
# 251 "/usr/include/ctype.h" 3
extern int isalnum_l(int, locale_t) throw(); 
# 252
extern int isalpha_l(int, locale_t) throw(); 
# 253
extern int iscntrl_l(int, locale_t) throw(); 
# 254
extern int isdigit_l(int, locale_t) throw(); 
# 255
extern int islower_l(int, locale_t) throw(); 
# 256
extern int isgraph_l(int, locale_t) throw(); 
# 257
extern int isprint_l(int, locale_t) throw(); 
# 258
extern int ispunct_l(int, locale_t) throw(); 
# 259
extern int isspace_l(int, locale_t) throw(); 
# 260
extern int isupper_l(int, locale_t) throw(); 
# 261
extern int isxdigit_l(int, locale_t) throw(); 
# 263
extern int isblank_l(int, locale_t) throw(); 
# 267
extern int __tolower_l(int __c, locale_t __l) throw(); 
# 268
extern int tolower_l(int __c, locale_t __l) throw(); 
# 271
extern int __toupper_l(int __c, locale_t __l) throw(); 
# 272
extern int toupper_l(int __c, locale_t __l) throw(); 
# 327 "/usr/include/ctype.h" 3
}
# 68 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_types.h"
#if 0
# 68
enum cudaRoundMode { 
# 70
cudaRoundNearest, 
# 71
cudaRoundZero, 
# 72
cudaRoundPosInf, 
# 73
cudaRoundMinInf
# 74
}; 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 104
struct char1 { 
# 106
signed char x; 
# 107
}; 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 109
struct uchar1 { 
# 111
unsigned char x; 
# 112
}; 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 115
struct __attribute((aligned(2))) char2 { 
# 117
signed char x, y; 
# 118
}; 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 120
struct __attribute((aligned(2))) uchar2 { 
# 122
unsigned char x, y; 
# 123
}; 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 125
struct char3 { 
# 127
signed char x, y, z; 
# 128
}; 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 130
struct uchar3 { 
# 132
unsigned char x, y, z; 
# 133
}; 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 135
struct __attribute((aligned(4))) char4 { 
# 137
signed char x, y, z, w; 
# 138
}; 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 140
struct __attribute((aligned(4))) uchar4 { 
# 142
unsigned char x, y, z, w; 
# 143
}; 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 145
struct short1 { 
# 147
short x; 
# 148
}; 
#endif
# 150 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 150
struct ushort1 { 
# 152
unsigned short x; 
# 153
}; 
#endif
# 155 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 155
struct __attribute((aligned(4))) short2 { 
# 157
short x, y; 
# 158
}; 
#endif
# 160 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 160
struct __attribute((aligned(4))) ushort2 { 
# 162
unsigned short x, y; 
# 163
}; 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 165
struct short3 { 
# 167
short x, y, z; 
# 168
}; 
#endif
# 170 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 170
struct ushort3 { 
# 172
unsigned short x, y, z; 
# 173
}; 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 175
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 176
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 178
struct int1 { 
# 180
int x; 
# 181
}; 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 183
struct uint1 { 
# 185
unsigned x; 
# 186
}; 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 188
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 189
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 191 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 191
struct int3 { 
# 193
int x, y, z; 
# 194
}; 
#endif
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 196
struct uint3 { 
# 198
unsigned x, y, z; 
# 199
}; 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 201
struct __attribute((aligned(16))) int4 { 
# 203
int x, y, z, w; 
# 204
}; 
#endif
# 206 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 206
struct __attribute((aligned(16))) uint4 { 
# 208
unsigned x, y, z, w; 
# 209
}; 
#endif
# 211 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 211
struct long1 { 
# 213
long x; 
# 214
}; 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 216
struct ulong1 { 
# 218
unsigned long x; 
# 219
}; 
#endif
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 226
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 228
long x, y; 
# 229
}; 
#endif
# 231 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 231
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 233
unsigned long x, y; 
# 234
}; 
#endif
# 238 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 238
struct long3 { 
# 240
long x, y, z; 
# 241
}; 
#endif
# 243 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 243
struct ulong3 { 
# 245
unsigned long x, y, z; 
# 246
}; 
#endif
# 248 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 248
struct __attribute((aligned(16))) long4 { 
# 250
long x, y, z, w; 
# 251
}; 
#endif
# 253 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 253
struct __attribute((aligned(16))) ulong4 { 
# 255
unsigned long x, y, z, w; 
# 256
}; 
#endif
# 258 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 258
struct float1 { 
# 260
float x; 
# 261
}; 
#endif
# 280 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 280
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 285 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 285
struct float3 { 
# 287
float x, y, z; 
# 288
}; 
#endif
# 290 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 290
struct __attribute((aligned(16))) float4 { 
# 292
float x, y, z, w; 
# 293
}; 
#endif
# 295 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 295
struct longlong1 { 
# 297
long long x; 
# 298
}; 
#endif
# 300 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 300
struct ulonglong1 { 
# 302
unsigned long long x; 
# 303
}; 
#endif
# 305 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 305
struct __attribute((aligned(16))) longlong2 { 
# 307
long long x, y; 
# 308
}; 
#endif
# 310 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 310
struct __attribute((aligned(16))) ulonglong2 { 
# 312
unsigned long long x, y; 
# 313
}; 
#endif
# 315 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 315
struct longlong3 { 
# 317
long long x, y, z; 
# 318
}; 
#endif
# 320 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 320
struct ulonglong3 { 
# 322
unsigned long long x, y, z; 
# 323
}; 
#endif
# 325 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 325
struct __attribute((aligned(16))) longlong4 { 
# 327
long long x, y, z, w; 
# 328
}; 
#endif
# 330 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 330
struct __attribute((aligned(16))) ulonglong4 { 
# 332
unsigned long long x, y, z, w; 
# 333
}; 
#endif
# 335 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 335
struct double1 { 
# 337
double x; 
# 338
}; 
#endif
# 340 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 340
struct __attribute((aligned(16))) double2 { 
# 342
double x, y; 
# 343
}; 
#endif
# 345 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 345
struct double3 { 
# 347
double x, y, z; 
# 348
}; 
#endif
# 350 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 350
struct __attribute((aligned(16))) double4 { 
# 352
double x, y, z, w; 
# 353
}; 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char1 
# 367
char1; 
#endif
# 368 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar1 
# 368
uchar1; 
#endif
# 369 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char2 
# 369
char2; 
#endif
# 370 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar2 
# 370
uchar2; 
#endif
# 371 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char3 
# 371
char3; 
#endif
# 372 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar3 
# 372
uchar3; 
#endif
# 373 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char4 
# 373
char4; 
#endif
# 374 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar4 
# 374
uchar4; 
#endif
# 375 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short1 
# 375
short1; 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort1 
# 376
ushort1; 
#endif
# 377 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short2 
# 377
short2; 
#endif
# 378 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort2 
# 378
ushort2; 
#endif
# 379 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short3 
# 379
short3; 
#endif
# 380 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort3 
# 380
ushort3; 
#endif
# 381 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short4 
# 381
short4; 
#endif
# 382 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort4 
# 382
ushort4; 
#endif
# 383 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int1 
# 383
int1; 
#endif
# 384 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint1 
# 384
uint1; 
#endif
# 385 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int2 
# 385
int2; 
#endif
# 386 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint2 
# 386
uint2; 
#endif
# 387 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int3 
# 387
int3; 
#endif
# 388 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint3 
# 388
uint3; 
#endif
# 389 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int4 
# 389
int4; 
#endif
# 390 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint4 
# 390
uint4; 
#endif
# 391 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long1 
# 391
long1; 
#endif
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong1 
# 392
ulong1; 
#endif
# 393 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long2 
# 393
long2; 
#endif
# 394 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong2 
# 394
ulong2; 
#endif
# 395 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long3 
# 395
long3; 
#endif
# 396 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong3 
# 396
ulong3; 
#endif
# 397 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long4 
# 397
long4; 
#endif
# 398 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong4 
# 398
ulong4; 
#endif
# 399 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float1 
# 399
float1; 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float2 
# 400
float2; 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float3 
# 401
float3; 
#endif
# 402 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float4 
# 402
float4; 
#endif
# 403 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong1 
# 403
longlong1; 
#endif
# 404 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong1 
# 404
ulonglong1; 
#endif
# 405 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong2 
# 405
longlong2; 
#endif
# 406 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong2 
# 406
ulonglong2; 
#endif
# 407 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong3 
# 407
longlong3; 
#endif
# 408 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong3 
# 408
ulonglong3; 
#endif
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong4 
# 409
longlong4; 
#endif
# 410 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong4 
# 410
ulonglong4; 
#endif
# 411 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double1 
# 411
double1; 
#endif
# 412 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double2 
# 412
double2; 
#endif
# 413 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double3 
# 413
double3; 
#endif
# 414 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double4 
# 414
double4; 
#endif
# 426 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 426
struct dim3 { 
# 428
unsigned x, y, z; 
# 440 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
}; 
#endif
# 442 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef dim3 
# 442
dim3; 
#endif
# 145 "/usr/lib64/gcc/x86_64-suse-linux/13/include/stddef.h" 3
typedef long ptrdiff_t; 
# 214 "/usr/lib64/gcc/x86_64-suse-linux/13/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 436 "/usr/lib64/gcc/x86_64-suse-linux/13/include/stddef.h" 3
typedef 
# 425
struct { 
# 426
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 427
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 436
} max_align_t; 
# 443
typedef __decltype((nullptr)) nullptr_t; 
# 205 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 205
enum cudaError { 
# 212
cudaSuccess, 
# 218
cudaErrorInvalidValue, 
# 224
cudaErrorMemoryAllocation, 
# 230
cudaErrorInitializationError, 
# 237
cudaErrorCudartUnloading, 
# 244
cudaErrorProfilerDisabled, 
# 252
cudaErrorProfilerNotInitialized, 
# 259
cudaErrorProfilerAlreadyStarted, 
# 266
cudaErrorProfilerAlreadyStopped, 
# 274
cudaErrorInvalidConfiguration, 
# 280
cudaErrorInvalidPitchValue = 12, 
# 286
cudaErrorInvalidSymbol, 
# 294
cudaErrorInvalidHostPointer = 16, 
# 302
cudaErrorInvalidDevicePointer, 
# 307
cudaErrorInvalidTexture, 
# 313
cudaErrorInvalidTextureBinding, 
# 320
cudaErrorInvalidChannelDescriptor, 
# 326
cudaErrorInvalidMemcpyDirection, 
# 336
cudaErrorAddressOfConstant, 
# 345
cudaErrorTextureFetchFailed, 
# 354
cudaErrorTextureNotBound, 
# 363
cudaErrorSynchronizationError, 
# 368
cudaErrorInvalidFilterSetting, 
# 374
cudaErrorInvalidNormSetting, 
# 382
cudaErrorMixedDeviceExecution, 
# 390
cudaErrorNotYetImplemented = 31, 
# 399
cudaErrorMemoryValueTooLarge, 
# 405
cudaErrorStubLibrary = 34, 
# 412
cudaErrorInsufficientDriver, 
# 419
cudaErrorCallRequiresNewerDriver, 
# 425
cudaErrorInvalidSurface, 
# 431
cudaErrorDuplicateVariableName = 43, 
# 437
cudaErrorDuplicateTextureName, 
# 443
cudaErrorDuplicateSurfaceName, 
# 453
cudaErrorDevicesUnavailable, 
# 466
cudaErrorIncompatibleDriverContext = 49, 
# 472
cudaErrorMissingConfiguration = 52, 
# 481
cudaErrorPriorLaunchFailure, 
# 487
cudaErrorLaunchMaxDepthExceeded = 65, 
# 495
cudaErrorLaunchFileScopedTex, 
# 503
cudaErrorLaunchFileScopedSurf, 
# 519
cudaErrorSyncDepthExceeded, 
# 531
cudaErrorLaunchPendingCountExceeded, 
# 537
cudaErrorInvalidDeviceFunction = 98, 
# 543
cudaErrorNoDevice = 100, 
# 550
cudaErrorInvalidDevice, 
# 555
cudaErrorDeviceNotLicensed, 
# 564
cudaErrorSoftwareValidityNotEstablished, 
# 569
cudaErrorStartupFailure = 127, 
# 574
cudaErrorInvalidKernelImage = 200, 
# 584
cudaErrorDeviceUninitialized, 
# 589
cudaErrorMapBufferObjectFailed = 205, 
# 594
cudaErrorUnmapBufferObjectFailed, 
# 600
cudaErrorArrayIsMapped, 
# 605
cudaErrorAlreadyMapped, 
# 613
cudaErrorNoKernelImageForDevice, 
# 618
cudaErrorAlreadyAcquired, 
# 623
cudaErrorNotMapped, 
# 629
cudaErrorNotMappedAsArray, 
# 635
cudaErrorNotMappedAsPointer, 
# 641
cudaErrorECCUncorrectable, 
# 647
cudaErrorUnsupportedLimit, 
# 653
cudaErrorDeviceAlreadyInUse, 
# 659
cudaErrorPeerAccessUnsupported, 
# 665
cudaErrorInvalidPtx, 
# 670
cudaErrorInvalidGraphicsContext, 
# 676
cudaErrorNvlinkUncorrectable, 
# 683
cudaErrorJitCompilerNotFound, 
# 690
cudaErrorUnsupportedPtxVersion, 
# 697
cudaErrorJitCompilationDisabled, 
# 702
cudaErrorUnsupportedExecAffinity, 
# 708
cudaErrorUnsupportedDevSideSync, 
# 713
cudaErrorInvalidSource = 300, 
# 718
cudaErrorFileNotFound, 
# 723
cudaErrorSharedObjectSymbolNotFound, 
# 728
cudaErrorSharedObjectInitFailed, 
# 733
cudaErrorOperatingSystem, 
# 740
cudaErrorInvalidResourceHandle = 400, 
# 746
cudaErrorIllegalState, 
# 754
cudaErrorLossyQuery, 
# 761
cudaErrorSymbolNotFound = 500, 
# 769
cudaErrorNotReady = 600, 
# 777
cudaErrorIllegalAddress = 700, 
# 786
cudaErrorLaunchOutOfResources, 
# 797
cudaErrorLaunchTimeout, 
# 803
cudaErrorLaunchIncompatibleTexturing, 
# 810
cudaErrorPeerAccessAlreadyEnabled, 
# 817
cudaErrorPeerAccessNotEnabled, 
# 830
cudaErrorSetOnActiveProcess = 708, 
# 837
cudaErrorContextIsDestroyed, 
# 844
cudaErrorAssert, 
# 851
cudaErrorTooManyPeers, 
# 857
cudaErrorHostMemoryAlreadyRegistered, 
# 863
cudaErrorHostMemoryNotRegistered, 
# 872
cudaErrorHardwareStackError, 
# 880
cudaErrorIllegalInstruction, 
# 889
cudaErrorMisalignedAddress, 
# 900
cudaErrorInvalidAddressSpace, 
# 908
cudaErrorInvalidPc, 
# 919
cudaErrorLaunchFailure, 
# 928
cudaErrorCooperativeLaunchTooLarge, 
# 933
cudaErrorNotPermitted = 800, 
# 939
cudaErrorNotSupported, 
# 948
cudaErrorSystemNotReady, 
# 955
cudaErrorSystemDriverMismatch, 
# 964
cudaErrorCompatNotSupportedOnDevice, 
# 969
cudaErrorMpsConnectionFailed, 
# 974
cudaErrorMpsRpcFailure, 
# 980
cudaErrorMpsServerNotReady, 
# 985
cudaErrorMpsMaxClientsReached, 
# 990
cudaErrorMpsMaxConnectionsReached, 
# 995
cudaErrorMpsClientTerminated, 
# 1000
cudaErrorCdpNotSupported, 
# 1005
cudaErrorCdpVersionMismatch, 
# 1010
cudaErrorStreamCaptureUnsupported = 900, 
# 1016
cudaErrorStreamCaptureInvalidated, 
# 1022
cudaErrorStreamCaptureMerge, 
# 1027
cudaErrorStreamCaptureUnmatched, 
# 1033
cudaErrorStreamCaptureUnjoined, 
# 1040
cudaErrorStreamCaptureIsolation, 
# 1046
cudaErrorStreamCaptureImplicit, 
# 1052
cudaErrorCapturedEvent, 
# 1059
cudaErrorStreamCaptureWrongThread, 
# 1064
cudaErrorTimeout, 
# 1070
cudaErrorGraphExecUpdateFailure, 
# 1080
cudaErrorExternalDevice, 
# 1086
cudaErrorInvalidClusterSize, 
# 1091
cudaErrorUnknown = 999, 
# 1099
cudaErrorApiFailureBase = 10000
# 1100
}; 
#endif
# 1105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1105
enum cudaChannelFormatKind { 
# 1107
cudaChannelFormatKindSigned, 
# 1108
cudaChannelFormatKindUnsigned, 
# 1109
cudaChannelFormatKindFloat, 
# 1110
cudaChannelFormatKindNone, 
# 1111
cudaChannelFormatKindNV12, 
# 1112
cudaChannelFormatKindUnsignedNormalized8X1, 
# 1113
cudaChannelFormatKindUnsignedNormalized8X2, 
# 1114
cudaChannelFormatKindUnsignedNormalized8X4, 
# 1115
cudaChannelFormatKindUnsignedNormalized16X1, 
# 1116
cudaChannelFormatKindUnsignedNormalized16X2, 
# 1117
cudaChannelFormatKindUnsignedNormalized16X4, 
# 1118
cudaChannelFormatKindSignedNormalized8X1, 
# 1119
cudaChannelFormatKindSignedNormalized8X2, 
# 1120
cudaChannelFormatKindSignedNormalized8X4, 
# 1121
cudaChannelFormatKindSignedNormalized16X1, 
# 1122
cudaChannelFormatKindSignedNormalized16X2, 
# 1123
cudaChannelFormatKindSignedNormalized16X4, 
# 1124
cudaChannelFormatKindUnsignedBlockCompressed1, 
# 1125
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
# 1126
cudaChannelFormatKindUnsignedBlockCompressed2, 
# 1127
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
# 1128
cudaChannelFormatKindUnsignedBlockCompressed3, 
# 1129
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
# 1130
cudaChannelFormatKindUnsignedBlockCompressed4, 
# 1131
cudaChannelFormatKindSignedBlockCompressed4, 
# 1132
cudaChannelFormatKindUnsignedBlockCompressed5, 
# 1133
cudaChannelFormatKindSignedBlockCompressed5, 
# 1134
cudaChannelFormatKindUnsignedBlockCompressed6H, 
# 1135
cudaChannelFormatKindSignedBlockCompressed6H, 
# 1136
cudaChannelFormatKindUnsignedBlockCompressed7, 
# 1137
cudaChannelFormatKindUnsignedBlockCompressed7SRGB
# 1138
}; 
#endif
# 1143 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1143
struct cudaChannelFormatDesc { 
# 1145
int x; 
# 1146
int y; 
# 1147
int z; 
# 1148
int w; 
# 1149
cudaChannelFormatKind f; 
# 1150
}; 
#endif
# 1155 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 1160
typedef const cudaArray *cudaArray_const_t; 
# 1162
struct cudaArray; 
# 1167
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1172
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1174
struct cudaMipmappedArray; 
# 1184
#if 0
# 1184
struct cudaArraySparseProperties { 
# 1185
struct { 
# 1186
unsigned width; 
# 1187
unsigned height; 
# 1188
unsigned depth; 
# 1189
} tileExtent; 
# 1190
unsigned miptailFirstLevel; 
# 1191
unsigned long long miptailSize; 
# 1192
unsigned flags; 
# 1193
unsigned reserved[4]; 
# 1194
}; 
#endif
# 1199 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1199
struct cudaArrayMemoryRequirements { 
# 1200
size_t size; 
# 1201
size_t alignment; 
# 1202
unsigned reserved[4]; 
# 1203
}; 
#endif
# 1208 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1208
enum cudaMemoryType { 
# 1210
cudaMemoryTypeUnregistered, 
# 1211
cudaMemoryTypeHost, 
# 1212
cudaMemoryTypeDevice, 
# 1213
cudaMemoryTypeManaged
# 1214
}; 
#endif
# 1219 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1219
enum cudaMemcpyKind { 
# 1221
cudaMemcpyHostToHost, 
# 1222
cudaMemcpyHostToDevice, 
# 1223
cudaMemcpyDeviceToHost, 
# 1224
cudaMemcpyDeviceToDevice, 
# 1225
cudaMemcpyDefault
# 1226
}; 
#endif
# 1233 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1233
struct cudaPitchedPtr { 
# 1235
void *ptr; 
# 1236
size_t pitch; 
# 1237
size_t xsize; 
# 1238
size_t ysize; 
# 1239
}; 
#endif
# 1246 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1246
struct cudaExtent { 
# 1248
size_t width; 
# 1249
size_t height; 
# 1250
size_t depth; 
# 1251
}; 
#endif
# 1258 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1258
struct cudaPos { 
# 1260
size_t x; 
# 1261
size_t y; 
# 1262
size_t z; 
# 1263
}; 
#endif
# 1268 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1268
struct cudaMemcpy3DParms { 
# 1270
cudaArray_t srcArray; 
# 1271
cudaPos srcPos; 
# 1272
cudaPitchedPtr srcPtr; 
# 1274
cudaArray_t dstArray; 
# 1275
cudaPos dstPos; 
# 1276
cudaPitchedPtr dstPtr; 
# 1278
cudaExtent extent; 
# 1279
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1280
}; 
#endif
# 1285 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1285
struct cudaMemcpyNodeParams { 
# 1286
int flags; 
# 1287
int reserved[3]; 
# 1288
cudaMemcpy3DParms copyParams; 
# 1289
}; 
#endif
# 1294 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1294
struct cudaMemcpy3DPeerParms { 
# 1296
cudaArray_t srcArray; 
# 1297
cudaPos srcPos; 
# 1298
cudaPitchedPtr srcPtr; 
# 1299
int srcDevice; 
# 1301
cudaArray_t dstArray; 
# 1302
cudaPos dstPos; 
# 1303
cudaPitchedPtr dstPtr; 
# 1304
int dstDevice; 
# 1306
cudaExtent extent; 
# 1307
}; 
#endif
# 1312 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1312
struct cudaMemsetParams { 
# 1313
void *dst; 
# 1314
size_t pitch; 
# 1315
unsigned value; 
# 1316
unsigned elementSize; 
# 1317
size_t width; 
# 1318
size_t height; 
# 1319
}; 
#endif
# 1324 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1324
struct cudaMemsetParamsV2 { 
# 1325
void *dst; 
# 1326
size_t pitch; 
# 1327
unsigned value; 
# 1328
unsigned elementSize; 
# 1329
size_t width; 
# 1330
size_t height; 
# 1331
}; 
#endif
# 1336 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1336
enum cudaAccessProperty { 
# 1337
cudaAccessPropertyNormal, 
# 1338
cudaAccessPropertyStreaming, 
# 1339
cudaAccessPropertyPersisting
# 1340
}; 
#endif
# 1353 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1353
struct cudaAccessPolicyWindow { 
# 1354
void *base_ptr; 
# 1355
size_t num_bytes; 
# 1356
float hitRatio; 
# 1357
cudaAccessProperty hitProp; 
# 1358
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1359
}; 
#endif
# 1371 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef void (*cudaHostFn_t)(void * userData); 
# 1376
#if 0
# 1376
struct cudaHostNodeParams { 
# 1377
cudaHostFn_t fn; 
# 1378
void *userData; 
# 1379
}; 
#endif
# 1384 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1384
struct cudaHostNodeParamsV2 { 
# 1385
cudaHostFn_t fn; 
# 1386
void *userData; 
# 1387
}; 
#endif
# 1392 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1392
enum cudaStreamCaptureStatus { 
# 1393
cudaStreamCaptureStatusNone, 
# 1394
cudaStreamCaptureStatusActive, 
# 1395
cudaStreamCaptureStatusInvalidated
# 1397
}; 
#endif
# 1403 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1403
enum cudaStreamCaptureMode { 
# 1404
cudaStreamCaptureModeGlobal, 
# 1405
cudaStreamCaptureModeThreadLocal, 
# 1406
cudaStreamCaptureModeRelaxed
# 1407
}; 
#endif
# 1409 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1409
enum cudaSynchronizationPolicy { 
# 1410
cudaSyncPolicyAuto = 1, 
# 1411
cudaSyncPolicySpin, 
# 1412
cudaSyncPolicyYield, 
# 1413
cudaSyncPolicyBlockingSync
# 1414
}; 
#endif
# 1419 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1419
enum cudaClusterSchedulingPolicy { 
# 1420
cudaClusterSchedulingPolicyDefault, 
# 1421
cudaClusterSchedulingPolicySpread, 
# 1422
cudaClusterSchedulingPolicyLoadBalancing
# 1423
}; 
#endif
# 1428 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1428
enum cudaStreamUpdateCaptureDependenciesFlags { 
# 1429
cudaStreamAddCaptureDependencies, 
# 1430
cudaStreamSetCaptureDependencies
# 1431
}; 
#endif
# 1436 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1436
enum cudaUserObjectFlags { 
# 1437
cudaUserObjectNoDestructorSync = 1
# 1438
}; 
#endif
# 1443 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1443
enum cudaUserObjectRetainFlags { 
# 1444
cudaGraphUserObjectMove = 1
# 1445
}; 
#endif
# 1450 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
struct cudaGraphicsResource; 
# 1455
#if 0
# 1455
enum cudaGraphicsRegisterFlags { 
# 1457
cudaGraphicsRegisterFlagsNone, 
# 1458
cudaGraphicsRegisterFlagsReadOnly, 
# 1459
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1460
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1461
cudaGraphicsRegisterFlagsTextureGather = 8
# 1462
}; 
#endif
# 1467 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1467
enum cudaGraphicsMapFlags { 
# 1469
cudaGraphicsMapFlagsNone, 
# 1470
cudaGraphicsMapFlagsReadOnly, 
# 1471
cudaGraphicsMapFlagsWriteDiscard
# 1472
}; 
#endif
# 1477 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1477
enum cudaGraphicsCubeFace { 
# 1479
cudaGraphicsCubeFacePositiveX, 
# 1480
cudaGraphicsCubeFaceNegativeX, 
# 1481
cudaGraphicsCubeFacePositiveY, 
# 1482
cudaGraphicsCubeFaceNegativeY, 
# 1483
cudaGraphicsCubeFacePositiveZ, 
# 1484
cudaGraphicsCubeFaceNegativeZ
# 1485
}; 
#endif
# 1490 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1490
enum cudaResourceType { 
# 1492
cudaResourceTypeArray, 
# 1493
cudaResourceTypeMipmappedArray, 
# 1494
cudaResourceTypeLinear, 
# 1495
cudaResourceTypePitch2D
# 1496
}; 
#endif
# 1501 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1501
enum cudaResourceViewFormat { 
# 1503
cudaResViewFormatNone, 
# 1504
cudaResViewFormatUnsignedChar1, 
# 1505
cudaResViewFormatUnsignedChar2, 
# 1506
cudaResViewFormatUnsignedChar4, 
# 1507
cudaResViewFormatSignedChar1, 
# 1508
cudaResViewFormatSignedChar2, 
# 1509
cudaResViewFormatSignedChar4, 
# 1510
cudaResViewFormatUnsignedShort1, 
# 1511
cudaResViewFormatUnsignedShort2, 
# 1512
cudaResViewFormatUnsignedShort4, 
# 1513
cudaResViewFormatSignedShort1, 
# 1514
cudaResViewFormatSignedShort2, 
# 1515
cudaResViewFormatSignedShort4, 
# 1516
cudaResViewFormatUnsignedInt1, 
# 1517
cudaResViewFormatUnsignedInt2, 
# 1518
cudaResViewFormatUnsignedInt4, 
# 1519
cudaResViewFormatSignedInt1, 
# 1520
cudaResViewFormatSignedInt2, 
# 1521
cudaResViewFormatSignedInt4, 
# 1522
cudaResViewFormatHalf1, 
# 1523
cudaResViewFormatHalf2, 
# 1524
cudaResViewFormatHalf4, 
# 1525
cudaResViewFormatFloat1, 
# 1526
cudaResViewFormatFloat2, 
# 1527
cudaResViewFormatFloat4, 
# 1528
cudaResViewFormatUnsignedBlockCompressed1, 
# 1529
cudaResViewFormatUnsignedBlockCompressed2, 
# 1530
cudaResViewFormatUnsignedBlockCompressed3, 
# 1531
cudaResViewFormatUnsignedBlockCompressed4, 
# 1532
cudaResViewFormatSignedBlockCompressed4, 
# 1533
cudaResViewFormatUnsignedBlockCompressed5, 
# 1534
cudaResViewFormatSignedBlockCompressed5, 
# 1535
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1536
cudaResViewFormatSignedBlockCompressed6H, 
# 1537
cudaResViewFormatUnsignedBlockCompressed7
# 1538
}; 
#endif
# 1543 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1543
struct cudaResourceDesc { 
# 1544
cudaResourceType resType; 
# 1546
union { 
# 1547
struct { 
# 1548
cudaArray_t array; 
# 1549
} array; 
# 1550
struct { 
# 1551
cudaMipmappedArray_t mipmap; 
# 1552
} mipmap; 
# 1553
struct { 
# 1554
void *devPtr; 
# 1555
cudaChannelFormatDesc desc; 
# 1556
size_t sizeInBytes; 
# 1557
} linear; 
# 1558
struct { 
# 1559
void *devPtr; 
# 1560
cudaChannelFormatDesc desc; 
# 1561
size_t width; 
# 1562
size_t height; 
# 1563
size_t pitchInBytes; 
# 1564
} pitch2D; 
# 1565
} res; 
# 1566
}; 
#endif
# 1571 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1571
struct cudaResourceViewDesc { 
# 1573
cudaResourceViewFormat format; 
# 1574
size_t width; 
# 1575
size_t height; 
# 1576
size_t depth; 
# 1577
unsigned firstMipmapLevel; 
# 1578
unsigned lastMipmapLevel; 
# 1579
unsigned firstLayer; 
# 1580
unsigned lastLayer; 
# 1581
}; 
#endif
# 1586 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1586
struct cudaPointerAttributes { 
# 1592
cudaMemoryType type; 
# 1603
int device; 
# 1609
void *devicePointer; 
# 1618
void *hostPointer; 
# 1619
}; 
#endif
# 1624 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1624
struct cudaFuncAttributes { 
# 1631
size_t sharedSizeBytes; 
# 1637
size_t constSizeBytes; 
# 1642
size_t localSizeBytes; 
# 1649
int maxThreadsPerBlock; 
# 1654
int numRegs; 
# 1661
int ptxVersion; 
# 1668
int binaryVersion; 
# 1674
int cacheModeCA; 
# 1681
int maxDynamicSharedSizeBytes; 
# 1690
int preferredShmemCarveout; 
# 1696
int clusterDimMustBeSet; 
# 1707
int requiredClusterWidth; 
# 1708
int requiredClusterHeight; 
# 1709
int requiredClusterDepth; 
# 1715
int clusterSchedulingPolicyPreference; 
# 1737
int nonPortableClusterSizeAllowed; 
# 1742
int reserved[16]; 
# 1743
}; 
#endif
# 1748 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1748
enum cudaFuncAttribute { 
# 1750
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1751
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1752
cudaFuncAttributeClusterDimMustBeSet, 
# 1753
cudaFuncAttributeRequiredClusterWidth, 
# 1754
cudaFuncAttributeRequiredClusterHeight, 
# 1755
cudaFuncAttributeRequiredClusterDepth, 
# 1756
cudaFuncAttributeNonPortableClusterSizeAllowed, 
# 1757
cudaFuncAttributeClusterSchedulingPolicyPreference, 
# 1758
cudaFuncAttributeMax
# 1759
}; 
#endif
# 1764 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1764
enum cudaFuncCache { 
# 1766
cudaFuncCachePreferNone, 
# 1767
cudaFuncCachePreferShared, 
# 1768
cudaFuncCachePreferL1, 
# 1769
cudaFuncCachePreferEqual
# 1770
}; 
#endif
# 1776 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1776
enum cudaSharedMemConfig { 
# 1778
cudaSharedMemBankSizeDefault, 
# 1779
cudaSharedMemBankSizeFourByte, 
# 1780
cudaSharedMemBankSizeEightByte
# 1781
}; 
#endif
# 1786 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1786
enum cudaSharedCarveout { 
# 1787
cudaSharedmemCarveoutDefault = (-1), 
# 1788
cudaSharedmemCarveoutMaxShared = 100, 
# 1789
cudaSharedmemCarveoutMaxL1 = 0
# 1790
}; 
#endif
# 1795 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1795
enum cudaComputeMode { 
# 1797
cudaComputeModeDefault, 
# 1798
cudaComputeModeExclusive, 
# 1799
cudaComputeModeProhibited, 
# 1800
cudaComputeModeExclusiveProcess
# 1801
}; 
#endif
# 1806 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1806
enum cudaLimit { 
# 1808
cudaLimitStackSize, 
# 1809
cudaLimitPrintfFifoSize, 
# 1810
cudaLimitMallocHeapSize, 
# 1811
cudaLimitDevRuntimeSyncDepth, 
# 1812
cudaLimitDevRuntimePendingLaunchCount, 
# 1813
cudaLimitMaxL2FetchGranularity, 
# 1814
cudaLimitPersistingL2CacheSize
# 1815
}; 
#endif
# 1820 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1820
enum cudaMemoryAdvise { 
# 1822
cudaMemAdviseSetReadMostly = 1, 
# 1823
cudaMemAdviseUnsetReadMostly, 
# 1824
cudaMemAdviseSetPreferredLocation, 
# 1825
cudaMemAdviseUnsetPreferredLocation, 
# 1826
cudaMemAdviseSetAccessedBy, 
# 1827
cudaMemAdviseUnsetAccessedBy
# 1828
}; 
#endif
# 1833 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1833
enum cudaMemRangeAttribute { 
# 1835
cudaMemRangeAttributeReadMostly = 1, 
# 1836
cudaMemRangeAttributePreferredLocation, 
# 1837
cudaMemRangeAttributeAccessedBy, 
# 1838
cudaMemRangeAttributeLastPrefetchLocation, 
# 1839
cudaMemRangeAttributePreferredLocationType, 
# 1840
cudaMemRangeAttributePreferredLocationId, 
# 1841
cudaMemRangeAttributeLastPrefetchLocationType, 
# 1842
cudaMemRangeAttributeLastPrefetchLocationId
# 1843
}; 
#endif
# 1848 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1848
enum cudaFlushGPUDirectRDMAWritesOptions { 
# 1849
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
# 1850
cudaFlushGPUDirectRDMAWritesOptionMemOps
# 1851
}; 
#endif
# 1856 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1856
enum cudaGPUDirectRDMAWritesOrdering { 
# 1857
cudaGPUDirectRDMAWritesOrderingNone, 
# 1858
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
# 1859
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
# 1860
}; 
#endif
# 1865 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1865
enum cudaFlushGPUDirectRDMAWritesScope { 
# 1866
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
# 1867
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
# 1868
}; 
#endif
# 1873 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1873
enum cudaFlushGPUDirectRDMAWritesTarget { 
# 1874
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
# 1875
}; 
#endif
# 1881 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1881
enum cudaDeviceAttr { 
# 1883
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1884
cudaDevAttrMaxBlockDimX, 
# 1885
cudaDevAttrMaxBlockDimY, 
# 1886
cudaDevAttrMaxBlockDimZ, 
# 1887
cudaDevAttrMaxGridDimX, 
# 1888
cudaDevAttrMaxGridDimY, 
# 1889
cudaDevAttrMaxGridDimZ, 
# 1890
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1891
cudaDevAttrTotalConstantMemory, 
# 1892
cudaDevAttrWarpSize, 
# 1893
cudaDevAttrMaxPitch, 
# 1894
cudaDevAttrMaxRegistersPerBlock, 
# 1895
cudaDevAttrClockRate, 
# 1896
cudaDevAttrTextureAlignment, 
# 1897
cudaDevAttrGpuOverlap, 
# 1898
cudaDevAttrMultiProcessorCount, 
# 1899
cudaDevAttrKernelExecTimeout, 
# 1900
cudaDevAttrIntegrated, 
# 1901
cudaDevAttrCanMapHostMemory, 
# 1902
cudaDevAttrComputeMode, 
# 1903
cudaDevAttrMaxTexture1DWidth, 
# 1904
cudaDevAttrMaxTexture2DWidth, 
# 1905
cudaDevAttrMaxTexture2DHeight, 
# 1906
cudaDevAttrMaxTexture3DWidth, 
# 1907
cudaDevAttrMaxTexture3DHeight, 
# 1908
cudaDevAttrMaxTexture3DDepth, 
# 1909
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1910
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1911
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1912
cudaDevAttrSurfaceAlignment, 
# 1913
cudaDevAttrConcurrentKernels, 
# 1914
cudaDevAttrEccEnabled, 
# 1915
cudaDevAttrPciBusId, 
# 1916
cudaDevAttrPciDeviceId, 
# 1917
cudaDevAttrTccDriver, 
# 1918
cudaDevAttrMemoryClockRate, 
# 1919
cudaDevAttrGlobalMemoryBusWidth, 
# 1920
cudaDevAttrL2CacheSize, 
# 1921
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1922
cudaDevAttrAsyncEngineCount, 
# 1923
cudaDevAttrUnifiedAddressing, 
# 1924
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1925
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1926
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1927
cudaDevAttrMaxTexture2DGatherHeight, 
# 1928
cudaDevAttrMaxTexture3DWidthAlt, 
# 1929
cudaDevAttrMaxTexture3DHeightAlt, 
# 1930
cudaDevAttrMaxTexture3DDepthAlt, 
# 1931
cudaDevAttrPciDomainId, 
# 1932
cudaDevAttrTexturePitchAlignment, 
# 1933
cudaDevAttrMaxTextureCubemapWidth, 
# 1934
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1935
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1936
cudaDevAttrMaxSurface1DWidth, 
# 1937
cudaDevAttrMaxSurface2DWidth, 
# 1938
cudaDevAttrMaxSurface2DHeight, 
# 1939
cudaDevAttrMaxSurface3DWidth, 
# 1940
cudaDevAttrMaxSurface3DHeight, 
# 1941
cudaDevAttrMaxSurface3DDepth, 
# 1942
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1943
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1944
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1945
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1946
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1947
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1948
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1949
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1950
cudaDevAttrMaxTexture1DLinearWidth, 
# 1951
cudaDevAttrMaxTexture2DLinearWidth, 
# 1952
cudaDevAttrMaxTexture2DLinearHeight, 
# 1953
cudaDevAttrMaxTexture2DLinearPitch, 
# 1954
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1955
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1956
cudaDevAttrComputeCapabilityMajor, 
# 1957
cudaDevAttrComputeCapabilityMinor, 
# 1958
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1959
cudaDevAttrStreamPrioritiesSupported, 
# 1960
cudaDevAttrGlobalL1CacheSupported, 
# 1961
cudaDevAttrLocalL1CacheSupported, 
# 1962
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1963
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1964
cudaDevAttrManagedMemory, 
# 1965
cudaDevAttrIsMultiGpuBoard, 
# 1966
cudaDevAttrMultiGpuBoardGroupID, 
# 1967
cudaDevAttrHostNativeAtomicSupported, 
# 1968
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1969
cudaDevAttrPageableMemoryAccess, 
# 1970
cudaDevAttrConcurrentManagedAccess, 
# 1971
cudaDevAttrComputePreemptionSupported, 
# 1972
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1973
cudaDevAttrReserved92, 
# 1974
cudaDevAttrReserved93, 
# 1975
cudaDevAttrReserved94, 
# 1976
cudaDevAttrCooperativeLaunch, 
# 1977
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1978
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1979
cudaDevAttrCanFlushRemoteWrites, 
# 1980
cudaDevAttrHostRegisterSupported, 
# 1981
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1982
cudaDevAttrDirectManagedMemAccessFromHost, 
# 1983
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
# 1984
cudaDevAttrMaxPersistingL2CacheSize = 108, 
# 1985
cudaDevAttrMaxAccessPolicyWindowSize, 
# 1986
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
# 1987
cudaDevAttrSparseCudaArraySupported, 
# 1988
cudaDevAttrHostRegisterReadOnlySupported, 
# 1989
cudaDevAttrTimelineSemaphoreInteropSupported, 
# 1990
cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114, 
# 1991
cudaDevAttrMemoryPoolsSupported, 
# 1992
cudaDevAttrGPUDirectRDMASupported, 
# 1993
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
# 1994
cudaDevAttrGPUDirectRDMAWritesOrdering, 
# 1995
cudaDevAttrMemoryPoolSupportedHandleTypes, 
# 1996
cudaDevAttrClusterLaunch, 
# 1997
cudaDevAttrDeferredMappingCudaArraySupported, 
# 1998
cudaDevAttrReserved122, 
# 1999
cudaDevAttrReserved123, 
# 2000
cudaDevAttrReserved124, 
# 2001
cudaDevAttrIpcEventSupport, 
# 2002
cudaDevAttrMemSyncDomainCount, 
# 2003
cudaDevAttrReserved127, 
# 2004
cudaDevAttrReserved128, 
# 2005
cudaDevAttrReserved129, 
# 2006
cudaDevAttrNumaConfig, 
# 2007
cudaDevAttrNumaId, 
# 2008
cudaDevAttrReserved132, 
# 2009
cudaDevAttrMpsEnabled, 
# 2010
cudaDevAttrHostNumaId, 
# 2011
cudaDevAttrMax
# 2012
}; 
#endif
# 2017 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2017
enum cudaMemPoolAttr { 
# 2027
cudaMemPoolReuseFollowEventDependencies = 1, 
# 2034
cudaMemPoolReuseAllowOpportunistic, 
# 2042
cudaMemPoolReuseAllowInternalDependencies, 
# 2053
cudaMemPoolAttrReleaseThreshold, 
# 2059
cudaMemPoolAttrReservedMemCurrent, 
# 2066
cudaMemPoolAttrReservedMemHigh, 
# 2072
cudaMemPoolAttrUsedMemCurrent, 
# 2079
cudaMemPoolAttrUsedMemHigh
# 2080
}; 
#endif
# 2085 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2085
enum cudaMemLocationType { 
# 2086
cudaMemLocationTypeInvalid, 
# 2087
cudaMemLocationTypeDevice, 
# 2088
cudaMemLocationTypeHost, 
# 2089
cudaMemLocationTypeHostNuma, 
# 2090
cudaMemLocationTypeHostNumaCurrent
# 2091
}; 
#endif
# 2099 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2099
struct cudaMemLocation { 
# 2100
cudaMemLocationType type; 
# 2101
int id; 
# 2102
}; 
#endif
# 2107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2107
enum cudaMemAccessFlags { 
# 2108
cudaMemAccessFlagsProtNone, 
# 2109
cudaMemAccessFlagsProtRead, 
# 2110
cudaMemAccessFlagsProtReadWrite = 3
# 2111
}; 
#endif
# 2116 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2116
struct cudaMemAccessDesc { 
# 2117
cudaMemLocation location; 
# 2118
cudaMemAccessFlags flags; 
# 2119
}; 
#endif
# 2124 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2124
enum cudaMemAllocationType { 
# 2125
cudaMemAllocationTypeInvalid, 
# 2129
cudaMemAllocationTypePinned, 
# 2130
cudaMemAllocationTypeMax = 2147483647
# 2131
}; 
#endif
# 2136 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2136
enum cudaMemAllocationHandleType { 
# 2137
cudaMemHandleTypeNone, 
# 2138
cudaMemHandleTypePosixFileDescriptor, 
# 2139
cudaMemHandleTypeWin32, 
# 2140
cudaMemHandleTypeWin32Kmt = 4, 
# 2141
cudaMemHandleTypeFabric = 8
# 2142
}; 
#endif
# 2147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2147
struct cudaMemPoolProps { 
# 2148
cudaMemAllocationType allocType; 
# 2149
cudaMemAllocationHandleType handleTypes; 
# 2150
cudaMemLocation location; 
# 2157
void *win32SecurityAttributes; 
# 2158
size_t maxSize; 
# 2159
unsigned char reserved[56]; 
# 2160
}; 
#endif
# 2165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2165
struct cudaMemPoolPtrExportData { 
# 2166
unsigned char reserved[64]; 
# 2167
}; 
#endif
# 2172 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2172
struct cudaMemAllocNodeParams { 
# 2177
cudaMemPoolProps poolProps; 
# 2178
const cudaMemAccessDesc *accessDescs; 
# 2179
size_t accessDescCount; 
# 2180
size_t bytesize; 
# 2181
void *dptr; 
# 2182
}; 
#endif
# 2187 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2187
struct cudaMemAllocNodeParamsV2 { 
# 2192
cudaMemPoolProps poolProps; 
# 2193
const cudaMemAccessDesc *accessDescs; 
# 2194
size_t accessDescCount; 
# 2195
size_t bytesize; 
# 2196
void *dptr; 
# 2197
}; 
#endif
# 2202 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2202
struct cudaMemFreeNodeParams { 
# 2203
void *dptr; 
# 2204
}; 
#endif
# 2209 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2209
enum cudaGraphMemAttributeType { 
# 2214
cudaGraphMemAttrUsedMemCurrent, 
# 2221
cudaGraphMemAttrUsedMemHigh, 
# 2228
cudaGraphMemAttrReservedMemCurrent, 
# 2235
cudaGraphMemAttrReservedMemHigh
# 2236
}; 
#endif
# 2242 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2242
enum cudaDeviceP2PAttr { 
# 2243
cudaDevP2PAttrPerformanceRank = 1, 
# 2244
cudaDevP2PAttrAccessSupported, 
# 2245
cudaDevP2PAttrNativeAtomicSupported, 
# 2246
cudaDevP2PAttrCudaArrayAccessSupported
# 2247
}; 
#endif
# 2254 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2254
struct CUuuid_st { 
# 2255
char bytes[16]; 
# 2256
}; 
#endif
# 2257 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2257
CUuuid; 
#endif
# 2259 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2259
cudaUUID_t; 
#endif
# 2264 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2264
struct cudaDeviceProp { 
# 2266
char name[256]; 
# 2267
cudaUUID_t uuid; 
# 2268
char luid[8]; 
# 2269
unsigned luidDeviceNodeMask; 
# 2270
size_t totalGlobalMem; 
# 2271
size_t sharedMemPerBlock; 
# 2272
int regsPerBlock; 
# 2273
int warpSize; 
# 2274
size_t memPitch; 
# 2275
int maxThreadsPerBlock; 
# 2276
int maxThreadsDim[3]; 
# 2277
int maxGridSize[3]; 
# 2278
int clockRate; 
# 2279
size_t totalConstMem; 
# 2280
int major; 
# 2281
int minor; 
# 2282
size_t textureAlignment; 
# 2283
size_t texturePitchAlignment; 
# 2284
int deviceOverlap; 
# 2285
int multiProcessorCount; 
# 2286
int kernelExecTimeoutEnabled; 
# 2287
int integrated; 
# 2288
int canMapHostMemory; 
# 2289
int computeMode; 
# 2290
int maxTexture1D; 
# 2291
int maxTexture1DMipmap; 
# 2292
int maxTexture1DLinear; 
# 2293
int maxTexture2D[2]; 
# 2294
int maxTexture2DMipmap[2]; 
# 2295
int maxTexture2DLinear[3]; 
# 2296
int maxTexture2DGather[2]; 
# 2297
int maxTexture3D[3]; 
# 2298
int maxTexture3DAlt[3]; 
# 2299
int maxTextureCubemap; 
# 2300
int maxTexture1DLayered[2]; 
# 2301
int maxTexture2DLayered[3]; 
# 2302
int maxTextureCubemapLayered[2]; 
# 2303
int maxSurface1D; 
# 2304
int maxSurface2D[2]; 
# 2305
int maxSurface3D[3]; 
# 2306
int maxSurface1DLayered[2]; 
# 2307
int maxSurface2DLayered[3]; 
# 2308
int maxSurfaceCubemap; 
# 2309
int maxSurfaceCubemapLayered[2]; 
# 2310
size_t surfaceAlignment; 
# 2311
int concurrentKernels; 
# 2312
int ECCEnabled; 
# 2313
int pciBusID; 
# 2314
int pciDeviceID; 
# 2315
int pciDomainID; 
# 2316
int tccDriver; 
# 2317
int asyncEngineCount; 
# 2318
int unifiedAddressing; 
# 2319
int memoryClockRate; 
# 2320
int memoryBusWidth; 
# 2321
int l2CacheSize; 
# 2322
int persistingL2CacheMaxSize; 
# 2323
int maxThreadsPerMultiProcessor; 
# 2324
int streamPrioritiesSupported; 
# 2325
int globalL1CacheSupported; 
# 2326
int localL1CacheSupported; 
# 2327
size_t sharedMemPerMultiprocessor; 
# 2328
int regsPerMultiprocessor; 
# 2329
int managedMemory; 
# 2330
int isMultiGpuBoard; 
# 2331
int multiGpuBoardGroupID; 
# 2332
int hostNativeAtomicSupported; 
# 2333
int singleToDoublePrecisionPerfRatio; 
# 2334
int pageableMemoryAccess; 
# 2335
int concurrentManagedAccess; 
# 2336
int computePreemptionSupported; 
# 2337
int canUseHostPointerForRegisteredMem; 
# 2338
int cooperativeLaunch; 
# 2339
int cooperativeMultiDeviceLaunch; 
# 2340
size_t sharedMemPerBlockOptin; 
# 2341
int pageableMemoryAccessUsesHostPageTables; 
# 2342
int directManagedMemAccessFromHost; 
# 2343
int maxBlocksPerMultiProcessor; 
# 2344
int accessPolicyMaxWindowSize; 
# 2345
size_t reservedSharedMemPerBlock; 
# 2346
int hostRegisterSupported; 
# 2347
int sparseCudaArraySupported; 
# 2348
int hostRegisterReadOnlySupported; 
# 2349
int timelineSemaphoreInteropSupported; 
# 2350
int memoryPoolsSupported; 
# 2351
int gpuDirectRDMASupported; 
# 2352
unsigned gpuDirectRDMAFlushWritesOptions; 
# 2353
int gpuDirectRDMAWritesOrdering; 
# 2354
unsigned memoryPoolSupportedHandleTypes; 
# 2355
int deferredMappingCudaArraySupported; 
# 2356
int ipcEventSupported; 
# 2357
int clusterLaunch; 
# 2358
int unifiedFunctionPointers; 
# 2359
int reserved2[2]; 
# 2360
int reserved1[1]; 
# 2361
int reserved[60]; 
# 2362
}; 
#endif
# 2375 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 2372
struct cudaIpcEventHandle_st { 
# 2374
char reserved[64]; 
# 2375
} cudaIpcEventHandle_t; 
#endif
# 2383 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 2380
struct cudaIpcMemHandle_st { 
# 2382
char reserved[64]; 
# 2383
} cudaIpcMemHandle_t; 
#endif
# 2391 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 2388
struct cudaMemFabricHandle_st { 
# 2390
char reserved[64]; 
# 2391
} cudaMemFabricHandle_t; 
#endif
# 2396 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2396
enum cudaExternalMemoryHandleType { 
# 2400
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 2404
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 2408
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 2412
cudaExternalMemoryHandleTypeD3D12Heap, 
# 2416
cudaExternalMemoryHandleTypeD3D12Resource, 
# 2420
cudaExternalMemoryHandleTypeD3D11Resource, 
# 2424
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
# 2428
cudaExternalMemoryHandleTypeNvSciBuf
# 2429
}; 
#endif
# 2471 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2471
struct cudaExternalMemoryHandleDesc { 
# 2475
cudaExternalMemoryHandleType type; 
# 2476
union { 
# 2482
int fd; 
# 2498
struct { 
# 2502
void *handle; 
# 2507
const void *name; 
# 2508
} win32; 
# 2513
const void *nvSciBufObject; 
# 2514
} handle; 
# 2518
unsigned long long size; 
# 2522
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2523
}; 
#endif
# 2528 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2528
struct cudaExternalMemoryBufferDesc { 
# 2532
unsigned long long offset; 
# 2536
unsigned long long size; 
# 2540
unsigned flags; 
# 2541
}; 
#endif
# 2546 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2546
struct cudaExternalMemoryMipmappedArrayDesc { 
# 2551
unsigned long long offset; 
# 2555
cudaChannelFormatDesc formatDesc; 
# 2559
cudaExtent extent; 
# 2564
unsigned flags; 
# 2568
unsigned numLevels; 
# 2569
}; 
#endif
# 2574 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2574
enum cudaExternalSemaphoreHandleType { 
# 2578
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 2582
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 2586
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 2590
cudaExternalSemaphoreHandleTypeD3D12Fence, 
# 2594
cudaExternalSemaphoreHandleTypeD3D11Fence, 
# 2598
cudaExternalSemaphoreHandleTypeNvSciSync, 
# 2602
cudaExternalSemaphoreHandleTypeKeyedMutex, 
# 2606
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
# 2610
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
# 2614
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
# 2615
}; 
#endif
# 2620 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2620
struct cudaExternalSemaphoreHandleDesc { 
# 2624
cudaExternalSemaphoreHandleType type; 
# 2625
union { 
# 2632
int fd; 
# 2648
struct { 
# 2652
void *handle; 
# 2657
const void *name; 
# 2658
} win32; 
# 2662
const void *nvSciSyncObj; 
# 2663
} handle; 
# 2667
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2668
}; 
#endif
# 2673 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2673
struct cudaExternalSemaphoreSignalParams_v1 { 
# 2674
struct { 
# 2678
struct { 
# 2682
unsigned long long value; 
# 2683
} fence; 
# 2684
union { 
# 2689
void *fence; 
# 2690
unsigned long long reserved; 
# 2691
} nvSciSync; 
# 2695
struct { 
# 2699
unsigned long long key; 
# 2700
} keyedMutex; 
# 2701
} params; 
# 2712
unsigned flags; 
# 2713
}; 
#endif
# 2718 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2718
struct cudaExternalSemaphoreWaitParams_v1 { 
# 2719
struct { 
# 2723
struct { 
# 2727
unsigned long long value; 
# 2728
} fence; 
# 2729
union { 
# 2734
void *fence; 
# 2735
unsigned long long reserved; 
# 2736
} nvSciSync; 
# 2740
struct { 
# 2744
unsigned long long key; 
# 2748
unsigned timeoutMs; 
# 2749
} keyedMutex; 
# 2750
} params; 
# 2761
unsigned flags; 
# 2762
}; 
#endif
# 2767 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2767
struct cudaExternalSemaphoreSignalParams { 
# 2768
struct { 
# 2772
struct { 
# 2776
unsigned long long value; 
# 2777
} fence; 
# 2778
union { 
# 2783
void *fence; 
# 2784
unsigned long long reserved; 
# 2785
} nvSciSync; 
# 2789
struct { 
# 2793
unsigned long long key; 
# 2794
} keyedMutex; 
# 2795
unsigned reserved[12]; 
# 2796
} params; 
# 2807
unsigned flags; 
# 2808
unsigned reserved[16]; 
# 2809
}; 
#endif
# 2814 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2814
struct cudaExternalSemaphoreWaitParams { 
# 2815
struct { 
# 2819
struct { 
# 2823
unsigned long long value; 
# 2824
} fence; 
# 2825
union { 
# 2830
void *fence; 
# 2831
unsigned long long reserved; 
# 2832
} nvSciSync; 
# 2836
struct { 
# 2840
unsigned long long key; 
# 2844
unsigned timeoutMs; 
# 2845
} keyedMutex; 
# 2846
unsigned reserved[10]; 
# 2847
} params; 
# 2858
unsigned flags; 
# 2859
unsigned reserved[16]; 
# 2860
}; 
#endif
# 2871 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaError 
# 2871
cudaError_t; 
#endif
# 2876 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 2876
cudaStream_t; 
#endif
# 2881 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 2881
cudaEvent_t; 
#endif
# 2886 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 2886
cudaGraphicsResource_t; 
#endif
# 2891 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
# 2891
cudaExternalMemory_t; 
#endif
# 2896 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
# 2896
cudaExternalSemaphore_t; 
#endif
# 2901 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraph_st *
# 2901
cudaGraph_t; 
#endif
# 2906 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraphNode_st *
# 2906
cudaGraphNode_t; 
#endif
# 2911 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUuserObject_st *
# 2911
cudaUserObject_t; 
#endif
# 2916 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef unsigned long long 
# 2916
cudaGraphConditionalHandle; 
#endif
# 2921 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUfunc_st *
# 2921
cudaFunction_t; 
#endif
# 2926 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUkern_st *
# 2926
cudaKernel_t; 
#endif
# 2931 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
# 2931
cudaMemPool_t; 
#endif
# 2936 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2936
enum cudaCGScope { 
# 2937
cudaCGScopeInvalid, 
# 2938
cudaCGScopeGrid, 
# 2939
cudaCGScopeMultiGrid
# 2940
}; 
#endif
# 2945 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2945
struct cudaLaunchParams { 
# 2947
void *func; 
# 2948
dim3 gridDim; 
# 2949
dim3 blockDim; 
# 2950
void **args; 
# 2951
size_t sharedMem; 
# 2952
cudaStream_t stream; 
# 2953
}; 
#endif
# 2958 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2958
struct cudaKernelNodeParams { 
# 2959
void *func; 
# 2960
dim3 gridDim; 
# 2961
dim3 blockDim; 
# 2962
unsigned sharedMemBytes; 
# 2963
void **kernelParams; 
# 2964
void **extra; 
# 2965
}; 
#endif
# 2970 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2970
struct cudaKernelNodeParamsV2 { 
# 2971
void *func; 
# 2973
dim3 gridDim; 
# 2974
dim3 blockDim; 
# 2980
unsigned sharedMemBytes; 
# 2981
void **kernelParams; 
# 2982
void **extra; 
# 2983
}; 
#endif
# 2988 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2988
struct cudaExternalSemaphoreSignalNodeParams { 
# 2989
cudaExternalSemaphore_t *extSemArray; 
# 2990
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 2991
unsigned numExtSems; 
# 2992
}; 
#endif
# 2997 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2997
struct cudaExternalSemaphoreSignalNodeParamsV2 { 
# 2998
cudaExternalSemaphore_t *extSemArray; 
# 2999
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 3000
unsigned numExtSems; 
# 3001
}; 
#endif
# 3006 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3006
struct cudaExternalSemaphoreWaitNodeParams { 
# 3007
cudaExternalSemaphore_t *extSemArray; 
# 3008
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 3009
unsigned numExtSems; 
# 3010
}; 
#endif
# 3015 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3015
struct cudaExternalSemaphoreWaitNodeParamsV2 { 
# 3016
cudaExternalSemaphore_t *extSemArray; 
# 3017
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 3018
unsigned numExtSems; 
# 3019
}; 
#endif
# 3021 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3021
enum cudaGraphConditionalHandleFlags { 
# 3022
cudaGraphCondAssignDefault = 1
# 3023
}; 
#endif
# 3028 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3028
enum cudaGraphConditionalNodeType { 
# 3029
cudaGraphCondTypeIf, 
# 3030
cudaGraphCondTypeWhile
# 3031
}; 
#endif
# 3036 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3036
struct cudaConditionalNodeParams { 
# 3037
cudaGraphConditionalHandle handle; 
# 3040
cudaGraphConditionalNodeType type; 
# 3041
unsigned size; 
# 3042
cudaGraph_t *phGraph_out; 
# 3052
}; 
#endif
# 3057 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3057
enum cudaGraphNodeType { 
# 3058
cudaGraphNodeTypeKernel, 
# 3059
cudaGraphNodeTypeMemcpy, 
# 3060
cudaGraphNodeTypeMemset, 
# 3061
cudaGraphNodeTypeHost, 
# 3062
cudaGraphNodeTypeGraph, 
# 3063
cudaGraphNodeTypeEmpty, 
# 3064
cudaGraphNodeTypeWaitEvent, 
# 3065
cudaGraphNodeTypeEventRecord, 
# 3066
cudaGraphNodeTypeExtSemaphoreSignal, 
# 3067
cudaGraphNodeTypeExtSemaphoreWait, 
# 3068
cudaGraphNodeTypeMemAlloc, 
# 3069
cudaGraphNodeTypeMemFree, 
# 3070
cudaGraphNodeTypeConditional = 13, 
# 3087
cudaGraphNodeTypeCount
# 3088
}; 
#endif
# 3093 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3093
struct cudaChildGraphNodeParams { 
# 3094
cudaGraph_t graph; 
# 3096
}; 
#endif
# 3101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3101
struct cudaEventRecordNodeParams { 
# 3102
cudaEvent_t event; 
# 3103
}; 
#endif
# 3108 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3108
struct cudaEventWaitNodeParams { 
# 3109
cudaEvent_t event; 
# 3110
}; 
#endif
# 3115 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3115
struct cudaGraphNodeParams { 
# 3116
cudaGraphNodeType type; 
# 3117
int reserved0[3]; 
# 3119
union { 
# 3120
long long reserved1[29]; 
# 3121
cudaKernelNodeParamsV2 kernel; 
# 3122
cudaMemcpyNodeParams memcpy; 
# 3123
cudaMemsetParamsV2 memset; 
# 3124
cudaHostNodeParamsV2 host; 
# 3125
cudaChildGraphNodeParams graph; 
# 3126
cudaEventWaitNodeParams eventWait; 
# 3127
cudaEventRecordNodeParams eventRecord; 
# 3128
cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal; 
# 3129
cudaExternalSemaphoreWaitNodeParamsV2 extSemWait; 
# 3130
cudaMemAllocNodeParamsV2 alloc; 
# 3131
cudaMemFreeNodeParams free; 
# 3132
cudaConditionalNodeParams conditional; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3133
}; 
# 3135
long long reserved2; 
# 3136
}; 
#endif
# 3148 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3141
enum cudaGraphDependencyType_enum { 
# 3142
cudaGraphDependencyTypeDefault, 
# 3143
cudaGraphDependencyTypeProgrammatic
# 3148
} cudaGraphDependencyType; 
#endif
# 3178 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3155
struct cudaGraphEdgeData_st { 
# 3156
unsigned char from_port; 
# 3166
unsigned char to_port; 
# 3173
unsigned char type; 
# 3176
unsigned char reserved[5]; 
# 3178
} cudaGraphEdgeData; 
#endif
# 3199 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 3204
#if 0
# 3204
enum cudaGraphExecUpdateResult { 
# 3205
cudaGraphExecUpdateSuccess, 
# 3206
cudaGraphExecUpdateError, 
# 3207
cudaGraphExecUpdateErrorTopologyChanged, 
# 3208
cudaGraphExecUpdateErrorNodeTypeChanged, 
# 3209
cudaGraphExecUpdateErrorFunctionChanged, 
# 3210
cudaGraphExecUpdateErrorParametersChanged, 
# 3211
cudaGraphExecUpdateErrorNotSupported, 
# 3212
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
# 3213
cudaGraphExecUpdateErrorAttributesChanged
# 3214
}; 
#endif
# 3225 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3219
enum cudaGraphInstantiateResult { 
# 3220
cudaGraphInstantiateSuccess, 
# 3221
cudaGraphInstantiateError, 
# 3222
cudaGraphInstantiateInvalidStructure, 
# 3223
cudaGraphInstantiateNodeOperationNotSupported, 
# 3224
cudaGraphInstantiateMultipleDevicesNotSupported
# 3225
} cudaGraphInstantiateResult; 
#endif
# 3236 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3230
struct cudaGraphInstantiateParams_st { 
# 3232
unsigned long long flags; 
# 3233
cudaStream_t uploadStream; 
# 3234
cudaGraphNode_t errNode_out; 
# 3235
cudaGraphInstantiateResult result_out; 
# 3236
} cudaGraphInstantiateParams; 
#endif
# 3258 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3241
struct cudaGraphExecUpdateResultInfo_st { 
# 3245
cudaGraphExecUpdateResult result; 
# 3252
cudaGraphNode_t errorNode; 
# 3257
cudaGraphNode_t errorFromNode; 
# 3258
} cudaGraphExecUpdateResultInfo; 
#endif
# 3263 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct CUgraphDeviceUpdatableNode_st *cudaGraphDeviceNode_t; 
# 3268
#if 0
# 3268
enum cudaGraphKernelNodeField { 
# 3270
cudaGraphKernelNodeFieldInvalid, 
# 3271
cudaGraphKernelNodeFieldGridDim, 
# 3272
cudaGraphKernelNodeFieldParam, 
# 3273
cudaGraphKernelNodeFieldEnabled
# 3274
}; 
#endif
# 3279 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3279
struct cudaGraphKernelNodeUpdate { 
# 3280
cudaGraphDeviceNode_t node; 
# 3281
cudaGraphKernelNodeField field; 
# 3282
union { 
# 3284
dim3 gridDim; 
# 3289
struct { 
# 3290
const void *pValue; 
# 3291
size_t offset; 
# 3292
size_t size; 
# 3293
} param; 
# 3294
unsigned isEnabled; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3295
} updateData; 
# 3296
}; 
#endif
# 3302 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3302
enum cudaGetDriverEntryPointFlags { 
# 3303
cudaEnableDefault, 
# 3304
cudaEnableLegacyStream, 
# 3305
cudaEnablePerThreadDefaultStream
# 3306
}; 
#endif
# 3311 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3311
enum cudaDriverEntryPointQueryResult { 
# 3312
cudaDriverEntryPointSuccess, 
# 3313
cudaDriverEntryPointSymbolNotFound, 
# 3314
cudaDriverEntryPointVersionNotSufficent
# 3315
}; 
#endif
# 3320 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3320
enum cudaGraphDebugDotFlags { 
# 3321
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
# 3322
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
# 3323
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
# 3324
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
# 3325
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
# 3326
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
# 3327
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
# 3328
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
# 3329
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
# 3330
cudaGraphDebugDotFlagsHandles = (1 << 10), 
# 3331
cudaGraphDebugDotFlagsConditionalNodeParams = (1 << 15)
# 3332
}; 
#endif
# 3337 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3337
enum cudaGraphInstantiateFlags { 
# 3338
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
# 3339
cudaGraphInstantiateFlagUpload, 
# 3342
cudaGraphInstantiateFlagDeviceLaunch = 4, 
# 3345
cudaGraphInstantiateFlagUseNodePriority = 8
# 3347
}; 
#endif
# 3368 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3365
enum cudaLaunchMemSyncDomain { 
# 3366
cudaLaunchMemSyncDomainDefault, 
# 3367
cudaLaunchMemSyncDomainRemote
# 3368
} cudaLaunchMemSyncDomain; 
#endif
# 3384 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3381
struct cudaLaunchMemSyncDomainMap_st { 
# 3382
unsigned char default_; 
# 3383
unsigned char remote; 
# 3384
} cudaLaunchMemSyncDomainMap; 
#endif
# 3493 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3389
enum cudaLaunchAttributeID { 
# 3390
cudaLaunchAttributeIgnore, 
# 3391
cudaLaunchAttributeAccessPolicyWindow, 
# 3393
cudaLaunchAttributeCooperative, 
# 3395
cudaLaunchAttributeSynchronizationPolicy, 
# 3396
cudaLaunchAttributeClusterDimension, 
# 3398
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
# 3400
cudaLaunchAttributeProgrammaticStreamSerialization, 
# 3411
cudaLaunchAttributeProgrammaticEvent, 
# 3437
cudaLaunchAttributePriority, 
# 3439
cudaLaunchAttributeMemSyncDomainMap, 
# 3441
cudaLaunchAttributeMemSyncDomain, 
# 3443
cudaLaunchAttributeLaunchCompletionEvent = 12, 
# 3465
cudaLaunchAttributeDeviceUpdatableKernelNode
# 3493
} cudaLaunchAttributeID; 
#endif
# 3549 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3498
union cudaLaunchAttributeValue { 
# 3499
char pad[64]; 
# 3500
cudaAccessPolicyWindow accessPolicyWindow; 
# 3501
int cooperative; 
# 3503
cudaSynchronizationPolicy syncPolicy; 
# 3517
struct { 
# 3518
unsigned x; 
# 3519
unsigned y; 
# 3520
unsigned z; 
# 3521
} clusterDim; 
# 3522
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
# 3525
int programmaticStreamSerializationAllowed; 
# 3527
struct { 
# 3528
cudaEvent_t event; 
# 3529
int flags; 
# 3531
int triggerAtBlockStart; 
# 3532
} programmaticEvent; 
# 3533
int priority; 
# 3534
cudaLaunchMemSyncDomainMap memSyncDomainMap; 
# 3537
cudaLaunchMemSyncDomain memSyncDomain; 
# 3539
struct { 
# 3540
cudaEvent_t event; 
# 3541
int flags; 
# 3543
} launchCompletionEvent; 
# 3545
struct { 
# 3546
int deviceUpdatable; 
# 3547
cudaGraphDeviceNode_t devNode; 
# 3548
} deviceUpdatableKernelNode; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3549
} cudaLaunchAttributeValue; 
#endif
# 3558 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3554
struct cudaLaunchAttribute_st { 
# 3555
cudaLaunchAttributeID id; 
# 3556
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
# 3557
cudaLaunchAttributeValue val; 
# 3558
} cudaLaunchAttribute; 
#endif
# 3570 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3563
struct cudaLaunchConfig_st { 
# 3564
dim3 gridDim; 
# 3565
dim3 blockDim; 
# 3566
size_t dynamicSmemBytes; 
# 3567
cudaStream_t stream; 
# 3568
cudaLaunchAttribute *attrs; 
# 3569
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3570
} cudaLaunchConfig_t; 
#endif
# 3593 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 3593
enum cudaDeviceNumaConfig { 
# 3594
cudaDeviceNumaConfigNone, 
# 3595
cudaDeviceNumaConfigNumaNode
# 3596
}; 
#endif
# 3601 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct cudaAsyncCallbackEntry *cudaAsyncCallbackHandle_t; 
# 3603
struct cudaAsyncCallbackEntry; 
# 3610
#if 0
typedef 
# 3608
enum cudaAsyncNotificationType_enum { 
# 3609
cudaAsyncNotificationTypeOverBudget = 1
# 3610
} cudaAsyncNotificationType; 
#endif
# 3623 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 3615
struct cudaAsyncNotificationInfo { 
# 3617
cudaAsyncNotificationType type; 
# 3618
union { 
# 3619
struct { 
# 3620
unsigned long long bytesOverBudget; 
# 3621
} overBudget; 
# 3622
} info; 
# 3623
} cudaAsyncNotificationInfo_t; 
#endif
# 3625 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_types.h"
typedef void (*cudaAsyncCallback)(cudaAsyncNotificationInfo_t *, void *, cudaAsyncCallbackHandle_t); 
# 86 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 86
enum cudaSurfaceBoundaryMode { 
# 88
cudaBoundaryModeZero, 
# 89
cudaBoundaryModeClamp, 
# 90
cudaBoundaryModeTrap
# 91
}; 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 96
enum cudaSurfaceFormatMode { 
# 98
cudaFormatModeForced, 
# 99
cudaFormatModeAuto
# 100
}; 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
typedef unsigned long long 
# 105
cudaSurfaceObject_t; 
#endif
# 86 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 86
enum cudaTextureAddressMode { 
# 88
cudaAddressModeWrap, 
# 89
cudaAddressModeClamp, 
# 90
cudaAddressModeMirror, 
# 91
cudaAddressModeBorder
# 92
}; 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 97
enum cudaTextureFilterMode { 
# 99
cudaFilterModePoint, 
# 100
cudaFilterModeLinear
# 101
}; 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 106
enum cudaTextureReadMode { 
# 108
cudaReadModeElementType, 
# 109
cudaReadModeNormalizedFloat
# 110
}; 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 115
struct cudaTextureDesc { 
# 120
cudaTextureAddressMode addressMode[3]; 
# 124
cudaTextureFilterMode filterMode; 
# 128
cudaTextureReadMode readMode; 
# 132
int sRGB; 
# 136
float borderColor[4]; 
# 140
int normalizedCoords; 
# 144
unsigned maxAnisotropy; 
# 148
cudaTextureFilterMode mipmapFilterMode; 
# 152
float mipmapLevelBias; 
# 156
float minMipmapLevelClamp; 
# 160
float maxMipmapLevelClamp; 
# 164
int disableTrilinearOptimization; 
# 168
int seamlessCubemap; 
# 169
}; 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
typedef unsigned long long 
# 174
cudaTextureObject_t; 
#endif
# 89 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/library_types.h"
typedef 
# 57
enum cudaDataType_t { 
# 59
CUDA_R_16F = 2, 
# 60
CUDA_C_16F = 6, 
# 61
CUDA_R_16BF = 14, 
# 62
CUDA_C_16BF, 
# 63
CUDA_R_32F = 0, 
# 64
CUDA_C_32F = 4, 
# 65
CUDA_R_64F = 1, 
# 66
CUDA_C_64F = 5, 
# 67
CUDA_R_4I = 16, 
# 68
CUDA_C_4I, 
# 69
CUDA_R_4U, 
# 70
CUDA_C_4U, 
# 71
CUDA_R_8I = 3, 
# 72
CUDA_C_8I = 7, 
# 73
CUDA_R_8U, 
# 74
CUDA_C_8U, 
# 75
CUDA_R_16I = 20, 
# 76
CUDA_C_16I, 
# 77
CUDA_R_16U, 
# 78
CUDA_C_16U, 
# 79
CUDA_R_32I = 10, 
# 80
CUDA_C_32I, 
# 81
CUDA_R_32U, 
# 82
CUDA_C_32U, 
# 83
CUDA_R_64I = 24, 
# 84
CUDA_C_64I, 
# 85
CUDA_R_64U, 
# 86
CUDA_C_64U, 
# 87
CUDA_R_8F_E4M3, 
# 88
CUDA_R_8F_E5M2
# 89
} cudaDataType; 
# 97
typedef 
# 92
enum libraryPropertyType_t { 
# 94
MAJOR_VERSION, 
# 95
MINOR_VERSION, 
# 96
PATCH_LEVEL
# 97
} libraryPropertyType; 
# 2603 "/usr/include/c++/13/x86_64-suse-linux/bits/c++config.h" 3
namespace std { 
# 2605
typedef unsigned long size_t; 
# 2606
typedef long ptrdiff_t; 
# 2609
typedef __decltype((nullptr)) nullptr_t; 
# 2612
#pragma GCC visibility push ( default )
# 2615
__attribute((__noreturn__, __always_inline__)) inline void 
# 2616
__terminate() noexcept 
# 2617
{ 
# 2618
void terminate() noexcept __attribute((__noreturn__)); 
# 2619
terminate(); 
# 2620
} 
#pragma GCC visibility pop
}
# 2636
namespace std { 
# 2638
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 2639
}
# 2640
namespace __gnu_cxx { 
# 2642
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 2643
}
# 2829 "/usr/include/c++/13/x86_64-suse-linux/bits/c++config.h" 3
namespace std { 
# 2831
#pragma GCC visibility push ( default )
# 2837
constexpr bool __is_constant_evaluated() noexcept 
# 2838
{ 
# 2844
return __builtin_is_constant_evaluated(); 
# 2848
} 
#pragma GCC visibility pop
}
# 33 "/usr/include/stdlib.h" 3
extern "C" {
# 62 "/usr/include/stdlib.h" 3
typedef 
# 59
struct { 
# 60
int quot; 
# 61
int rem; 
# 62
} div_t; 
# 70
typedef 
# 67
struct { 
# 68
long quot; 
# 69
long rem; 
# 70
} ldiv_t; 
# 80
typedef 
# 77
struct { 
# 78
long long quot; 
# 79
long long rem; 
# 80
} lldiv_t; 
# 97
extern size_t __ctype_get_mb_cur_max() throw(); 
# 101
extern double atof(const char * __nptr) throw()
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 104
extern int atoi(const char * __nptr) throw()
# 105
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 107
extern long atol(const char * __nptr) throw()
# 108
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern long long atoll(const char * __nptr) throw()
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 117
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 119
 __attribute((__nonnull__(1))); 
# 123
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 124
 __attribute((__nonnull__(1))); 
# 126
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 128
 __attribute((__nonnull__(1))); 
# 176
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 178
 __attribute((__nonnull__(1))); 
# 180
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 182
 __attribute((__nonnull__(1))); 
# 187
extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 192
extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 194
 __attribute((__nonnull__(1))); 
# 200
extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 205
extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 207
 __attribute((__nonnull__(1))); 
# 212
extern int strfromd(char * __dest, size_t __size, const char * __format, double __f) throw()
# 214
 __attribute((__nonnull__(3))); 
# 216
extern int strfromf(char * __dest, size_t __size, const char * __format, float __f) throw()
# 218
 __attribute((__nonnull__(3))); 
# 220
extern int strfroml(char * __dest, size_t __size, const char * __format, long double __f) throw()
# 222
 __attribute((__nonnull__(3))); 
# 274
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 276
 __attribute((__nonnull__(1, 4))); 
# 278
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 281
 __attribute((__nonnull__(1, 4))); 
# 284
extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 287
 __attribute((__nonnull__(1, 4))); 
# 290
extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 293
 __attribute((__nonnull__(1, 4))); 
# 295
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 297
 __attribute((__nonnull__(1, 3))); 
# 299
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 301
 __attribute((__nonnull__(1, 3))); 
# 303
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 306
 __attribute((__nonnull__(1, 3))); 
# 385 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 388
extern long a64l(const char * __s) throw()
# 389
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 42
typedef __loff_t loff_t; 
# 47
typedef __ino_t ino_t; 
# 54 "/usr/include/sys/types.h" 3
typedef __ino64_t ino64_t; 
# 59
typedef __dev_t dev_t; 
# 64
typedef __gid_t gid_t; 
# 69
typedef __mode_t mode_t; 
# 74
typedef __nlink_t nlink_t; 
# 79
typedef __uid_t uid_t; 
# 85
typedef __off_t off_t; 
# 92 "/usr/include/sys/types.h" 3
typedef __off64_t off64_t; 
# 97
typedef __pid_t pid_t; 
# 103
typedef __id_t id_t; 
# 108
typedef __ssize_t ssize_t; 
# 114
typedef __daddr_t daddr_t; 
# 115
typedef __caddr_t caddr_t; 
# 121
typedef __key_t key_t; 
# 7 "/usr/include/bits/types/clock_t.h" 3
typedef __clock_t clock_t; 
# 7 "/usr/include/bits/types/clockid_t.h" 3
typedef __clockid_t clockid_t; 
# 7 "/usr/include/bits/types/time_t.h" 3
typedef __time_t time_t; 
# 7 "/usr/include/bits/types/timer_t.h" 3
typedef __timer_t timer_t; 
# 134 "/usr/include/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 138
typedef __suseconds_t suseconds_t; 
# 148 "/usr/include/sys/types.h" 3
typedef unsigned long ulong; 
# 149
typedef unsigned short ushort; 
# 150
typedef unsigned uint; 
# 24 "/usr/include/bits/stdint-intn.h" 3
typedef __int8_t int8_t; 
# 25
typedef __int16_t int16_t; 
# 26
typedef __int32_t int32_t; 
# 27
typedef __int64_t int64_t; 
# 158 "/usr/include/sys/types.h" 3
typedef __uint8_t u_int8_t; 
# 159
typedef __uint16_t u_int16_t; 
# 160
typedef __uint32_t u_int32_t; 
# 161
typedef __uint64_t u_int64_t; 
# 164
typedef long register_t __attribute((__mode__(__word__))); 
# 34 "/usr/include/bits/byteswap.h" 3
static inline __uint16_t __bswap_16(__uint16_t __bsx) 
# 35
{ 
# 37
return __builtin_bswap16(__bsx); 
# 41
} 
# 49
static inline __uint32_t __bswap_32(__uint32_t __bsx) 
# 50
{ 
# 52
return __builtin_bswap32(__bsx); 
# 56
} 
# 70 "/usr/include/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 71
{ 
# 73
return __builtin_bswap64(__bsx); 
# 77
} 
# 33 "/usr/include/bits/uintn-identity.h" 3
static inline __uint16_t __uint16_identity(__uint16_t __x) 
# 34
{ 
# 35
return __x; 
# 36
} 
# 39
static inline __uint32_t __uint32_identity(__uint32_t __x) 
# 40
{ 
# 41
return __x; 
# 42
} 
# 45
static inline __uint64_t __uint64_identity(__uint64_t __x) 
# 46
{ 
# 47
return __x; 
# 48
} 
# 8 "/usr/include/bits/types/__sigset_t.h" 3
typedef 
# 6
struct { 
# 7
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 8
} __sigset_t; 
# 7 "/usr/include/bits/types/sigset_t.h" 3
typedef __sigset_t sigset_t; 
# 8 "/usr/include/bits/types/struct_timeval.h" 3
struct timeval { 
# 10
__time_t tv_sec; 
# 11
__suseconds_t tv_usec; 
# 12
}; 
# 10 "/usr/include/bits/types/struct_timespec.h" 3
struct timespec { 
# 12
__time_t tv_sec; 
# 16
__syscall_slong_t tv_nsec; 
# 26 "/usr/include/bits/types/struct_timespec.h" 3
}; 
# 49 "/usr/include/sys/select.h" 3
typedef long __fd_mask; 
# 70
typedef 
# 60
struct { 
# 64
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 70
} fd_set; 
# 77
typedef __fd_mask fd_mask; 
# 91
extern "C" {
# 101
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 113
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 126
}
# 185 "/usr/include/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 192
typedef __blkcnt_t blkcnt_t; 
# 196
typedef __fsblkcnt_t fsblkcnt_t; 
# 200
typedef __fsfilcnt_t fsfilcnt_t; 
# 219 "/usr/include/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 220
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 221
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 53 "/usr/include/bits/thread-shared-types.h" 3
typedef 
# 49
struct __pthread_internal_list { 
# 51
__pthread_internal_list *__prev; 
# 52
__pthread_internal_list *__next; 
# 53
} __pthread_list_t; 
# 58
typedef 
# 55
struct __pthread_internal_slist { 
# 57
__pthread_internal_slist *__next; 
# 58
} __pthread_slist_t; 
# 22 "/usr/include/bits/struct_mutex.h" 3
struct __pthread_mutex_s { 
# 24
int __lock; 
# 25
unsigned __count; 
# 26
int __owner; 
# 28
unsigned __nusers; 
# 32
int __kind; 
# 34
short __spins; 
# 35
short __elision; 
# 36
__pthread_list_t __list; 
# 53 "/usr/include/bits/struct_mutex.h" 3
}; 
# 23 "/usr/include/bits/struct_rwlock.h" 3
struct __pthread_rwlock_arch_t { 
# 25
unsigned __readers; 
# 26
unsigned __writers; 
# 27
unsigned __wrphase_futex; 
# 28
unsigned __writers_futex; 
# 29
unsigned __pad3; 
# 30
unsigned __pad4; 
# 32
int __cur_writer; 
# 33
int __shared; 
# 34
signed char __rwelision; 
# 39
unsigned char __pad1[7]; 
# 42
unsigned long __pad2; 
# 45
unsigned __flags; 
# 55 "/usr/include/bits/struct_rwlock.h" 3
}; 
# 92 "/usr/include/bits/thread-shared-types.h" 3
struct __pthread_cond_s { 
# 95
union { 
# 96
unsigned long long __wseq; 
# 98
struct { 
# 99
unsigned __low; 
# 100
unsigned __high; 
# 101
} __wseq32; 
# 102
}; 
# 104
union { 
# 105
unsigned long long __g1_start; 
# 107
struct { 
# 108
unsigned __low; 
# 109
unsigned __high; 
# 110
} __g1_start32; 
# 111
}; 
# 112
unsigned __g_refs[2]; 
# 113
unsigned __g_size[2]; 
# 114
unsigned __g1_orig_size; 
# 115
unsigned __wrefs; 
# 116
unsigned __g_signals[2]; 
# 117
}; 
# 27 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 36
typedef 
# 33
union { 
# 34
char __size[4]; 
# 35
int __align; 
# 36
} pthread_mutexattr_t; 
# 45
typedef 
# 42
union { 
# 43
char __size[4]; 
# 44
int __align; 
# 45
} pthread_condattr_t; 
# 49
typedef unsigned pthread_key_t; 
# 53
typedef int pthread_once_t; 
# 56
union pthread_attr_t { 
# 58
char __size[56]; 
# 59
long __align; 
# 60
}; 
# 62
typedef pthread_attr_t pthread_attr_t; 
# 72
typedef 
# 68
union { 
# 69
__pthread_mutex_s __data; 
# 70
char __size[40]; 
# 71
long __align; 
# 72
} pthread_mutex_t; 
# 80
typedef 
# 76
union { 
# 77
__pthread_cond_s __data; 
# 78
char __size[48]; 
# 79
long long __align; 
# 80
} pthread_cond_t; 
# 91
typedef 
# 87
union { 
# 88
__pthread_rwlock_arch_t __data; 
# 89
char __size[56]; 
# 90
long __align; 
# 91
} pthread_rwlock_t; 
# 97
typedef 
# 94
union { 
# 95
char __size[8]; 
# 96
long __align; 
# 97
} pthread_rwlockattr_t; 
# 103
typedef volatile int pthread_spinlock_t; 
# 112
typedef 
# 109
union { 
# 110
char __size[32]; 
# 111
long __align; 
# 112
} pthread_barrier_t; 
# 118
typedef 
# 115
union { 
# 116
char __size[4]; 
# 117
int __align; 
# 118
} pthread_barrierattr_t; 
# 230 "/usr/include/sys/types.h" 3
}
# 401 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 404
extern void srandom(unsigned __seed) throw(); 
# 410
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 411
 __attribute((__nonnull__(2))); 
# 415
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 423
struct random_data { 
# 425
int32_t *fptr; 
# 426
int32_t *rptr; 
# 427
int32_t *state; 
# 428
int rand_type; 
# 429
int rand_deg; 
# 430
int rand_sep; 
# 431
int32_t *end_ptr; 
# 432
}; 
# 434
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 435
 __attribute((__nonnull__(1, 2))); 
# 437
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 438
 __attribute((__nonnull__(2))); 
# 440
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 443
 __attribute((__nonnull__(2, 4))); 
# 445
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 447
 __attribute((__nonnull__(1, 2))); 
# 453
extern int rand() throw(); 
# 455
extern void srand(unsigned __seed) throw(); 
# 459
extern int rand_r(unsigned * __seed) throw(); 
# 467
extern double drand48() throw(); 
# 468
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 471
extern long lrand48() throw(); 
# 472
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 473
 __attribute((__nonnull__(1))); 
# 476
extern long mrand48() throw(); 
# 477
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 478
 __attribute((__nonnull__(1))); 
# 481
extern void srand48(long __seedval) throw(); 
# 482
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 483
 __attribute((__nonnull__(1))); 
# 484
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 490
struct drand48_data { 
# 492
unsigned short __x[3]; 
# 493
unsigned short __old_x[3]; 
# 494
unsigned short __c; 
# 495
unsigned short __init; 
# 496
unsigned long long __a; 
# 498
}; 
# 501
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 502
 __attribute((__nonnull__(1, 2))); 
# 503
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 505
 __attribute((__nonnull__(1, 2))); 
# 508
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 510
 __attribute((__nonnull__(1, 2))); 
# 511
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 514
 __attribute((__nonnull__(1, 2))); 
# 517
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 519
 __attribute((__nonnull__(1, 2))); 
# 520
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 523
 __attribute((__nonnull__(1, 2))); 
# 526
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 527
 __attribute((__nonnull__(2))); 
# 529
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 530
 __attribute((__nonnull__(1, 2))); 
# 532
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 534
 __attribute((__nonnull__(1, 2))); 
# 539
extern void *malloc(size_t __size) throw() __attribute((__malloc__))
# 540
 __attribute((__alloc_size__(1))); 
# 542
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 543
 __attribute((__malloc__)) __attribute((__alloc_size__(1, 2))); 
# 550
extern void *realloc(void * __ptr, size_t __size) throw()
# 551
 __attribute((__warn_unused_result__)) __attribute((__alloc_size__(2))); 
# 559
extern void *reallocarray(void * __ptr, size_t __nmemb, size_t __size) throw()
# 560
 __attribute((__warn_unused_result__))
# 561
 __attribute((__alloc_size__(2, 3))); 
# 565
extern void free(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 23 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/alloca.h" 3
extern "C" {
# 25
extern void *__alloca(size_t); 
# 30
}
# 574 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__))
# 575
 __attribute((__alloc_size__(1))); 
# 580
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 581
 __attribute((__nonnull__(1))); 
# 586
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 587
 __attribute((__malloc__)) __attribute((__alloc_size__(2))); 
# 591
extern void abort() throw() __attribute((__noreturn__)); 
# 595
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 600
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 601
 __attribute((__nonnull__(1))); 
# 610
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 611
 __attribute((__nonnull__(1))); 
# 617
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 623
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 629
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 634
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 639
extern char *secure_getenv(const char * __name) throw()
# 640
 __attribute((__nonnull__(1))); 
# 647
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 653
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 654
 __attribute((__nonnull__(2))); 
# 657
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 664
extern int clearenv() throw(); 
# 675
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 688
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 698 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 710
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 720 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 721
 __attribute((__nonnull__(1))); 
# 731
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 742
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 752 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 762
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 763
 __attribute((__nonnull__(1))); 
# 774 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 775
 __attribute((__nonnull__(1))); 
# 784
extern int system(const char * __command); 
# 790
extern char *canonicalize_file_name(const char * __name) throw()
# 791
 __attribute((__nonnull__(1))); 
# 800
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 808
typedef int (*__compar_fn_t)(const void *, const void *); 
# 811
typedef __compar_fn_t comparison_fn_t; 
# 815
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 820
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 822
 __attribute((__nonnull__(1, 2, 5))); 
# 830
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 831
 __attribute((__nonnull__(1, 4))); 
# 833
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 835
 __attribute((__nonnull__(1, 4))); 
# 840
extern int abs(int __x) throw() __attribute((const)); 
# 841
extern long labs(long __x) throw() __attribute((const)); 
# 844
extern long long llabs(long long __x) throw()
# 845
 __attribute((const)); 
# 852
extern div_t div(int __numer, int __denom) throw()
# 853
 __attribute((const)); 
# 854
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 855
 __attribute((const)); 
# 858
extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 860
 __attribute((const)); 
# 872
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 873
 __attribute((__nonnull__(3, 4))); 
# 878
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 879
 __attribute((__nonnull__(3, 4))); 
# 884
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 885
 __attribute((__nonnull__(3))); 
# 890
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 892
 __attribute((__nonnull__(3, 4))); 
# 893
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 895
 __attribute((__nonnull__(3, 4))); 
# 896
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 897
 __attribute((__nonnull__(3))); 
# 902
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 904
 __attribute((__nonnull__(3, 4, 5))); 
# 905
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 907
 __attribute((__nonnull__(3, 4, 5))); 
# 909
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 912
 __attribute((__nonnull__(3, 4, 5))); 
# 913
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 916
 __attribute((__nonnull__(3, 4, 5))); 
# 922
extern int mblen(const char * __s, size_t __n) throw(); 
# 925
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 929
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 933
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 936
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 946
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 957
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 960
 __attribute((__nonnull__(1, 2, 3))); 
# 968
extern int posix_openpt(int __oflag); 
# 976
extern int grantpt(int __fd) throw(); 
# 980
extern int unlockpt(int __fd) throw(); 
# 985
extern char *ptsname(int __fd) throw(); 
# 992
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 993
 __attribute((__nonnull__(2))); 
# 996
extern int getpt(); 
# 1003
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 1004
 __attribute((__nonnull__(1))); 
# 1023 "/usr/include/stdlib.h" 3
}
# 46 "/usr/include/c++/13/bits/std_abs.h" 3
extern "C++" {
# 48
namespace std __attribute((__visibility__("default"))) { 
# 52
using ::abs;
# 56
inline long abs(long __i) { return __builtin_labs(__i); } 
# 61
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 71
constexpr double abs(double __x) 
# 72
{ return __builtin_fabs(__x); } 
# 75
constexpr float abs(float __x) 
# 76
{ return __builtin_fabsf(__x); } 
# 79
constexpr long double abs(long double __x) 
# 80
{ return __builtin_fabsl(__x); } 
# 85
constexpr __int128 abs(__int128 __x) { return (__x >= (0)) ? __x : (-__x); } 
# 151 "/usr/include/c++/13/bits/std_abs.h" 3
}
# 152
}
# 125 "/usr/include/c++/13/cstdlib" 3
extern "C++" {
# 127
namespace std __attribute((__visibility__("default"))) { 
# 131
using ::div_t;
# 132
using ::ldiv_t;
# 134
using ::abort;
# 136
using ::aligned_alloc;
# 138
using ::atexit;
# 141
using ::at_quick_exit;
# 144
using ::atof;
# 145
using ::atoi;
# 146
using ::atol;
# 147
using ::bsearch;
# 148
using ::calloc;
# 149
using ::div;
# 150
using ::exit;
# 151
using ::free;
# 152
using ::getenv;
# 153
using ::labs;
# 154
using ::ldiv;
# 155
using ::malloc;
# 157
using ::mblen;
# 158
using ::mbstowcs;
# 159
using ::mbtowc;
# 161
using ::qsort;
# 164
using ::quick_exit;
# 167
using ::rand;
# 168
using ::realloc;
# 169
using ::srand;
# 170
using ::strtod;
# 171
using ::strtol;
# 172
using ::strtoul;
# 173
using ::system;
# 175
using ::wcstombs;
# 176
using ::wctomb;
# 181
inline ldiv_t div(long __i, long __j) noexcept { return ldiv(__i, __j); } 
# 186
}
# 199 "/usr/include/c++/13/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 204
using ::lldiv_t;
# 210
using ::_Exit;
# 214
using ::llabs;
# 217
inline lldiv_t div(long long __n, long long __d) 
# 218
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 220
using ::lldiv;
# 231 "/usr/include/c++/13/cstdlib" 3
using ::atoll;
# 232
using ::strtoll;
# 233
using ::strtoull;
# 235
using ::strtof;
# 236
using ::strtold;
# 239
}
# 241
namespace std { 
# 244
using __gnu_cxx::lldiv_t;
# 246
using __gnu_cxx::_Exit;
# 248
using __gnu_cxx::llabs;
# 249
using __gnu_cxx::div;
# 250
using __gnu_cxx::lldiv;
# 252
using __gnu_cxx::atoll;
# 253
using __gnu_cxx::strtof;
# 254
using __gnu_cxx::strtoll;
# 255
using __gnu_cxx::strtoull;
# 256
using __gnu_cxx::strtold;
# 257
}
# 261
}
# 15 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/cstdlib" 3
namespace std { 
# 16
extern "C" void *__host_malloc(size_t); 
# 17
extern "C" void __host_free(void *); 
# 18
}
# 19
extern "C" void *__host_malloc(std::size_t); 
# 20
extern "C" void __host_free(void *); 
# 38 "/usr/include/c++/13/stdlib.h" 3
using std::abort;
# 39
using std::atexit;
# 40
using std::exit;
# 43
using std::at_quick_exit;
# 46
using std::quick_exit;
# 49
using std::_Exit;
# 57
using std::abs;
# 58
using std::atof;
# 59
using std::atoi;
# 60
using std::atol;
# 61
using std::bsearch;
# 62
using std::calloc;
# 63
using std::div;
# 64
using std::free;
# 65
using std::getenv;
# 66
using std::labs;
# 67
using std::ldiv;
# 68
using std::malloc;
# 70
using std::mblen;
# 71
using std::mbstowcs;
# 72
using std::mbtowc;
# 74
using std::qsort;
# 75
using std::rand;
# 76
using std::realloc;
# 77
using std::srand;
# 78
using std::strtod;
# 79
using std::strtol;
# 80
using std::strtoul;
# 81
using std::system;
# 83
using std::wcstombs;
# 84
using std::wctomb;
# 18 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/stdlib.h" 3
extern "C" void *__host_malloc(size_t); 
# 23
extern "C" void __host_free(void *); 
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern "C" {
# 187
__attribute__((unused)) extern cudaError_t __cudaDeviceSynchronizeDeprecationAvoidance(); 
# 236 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 237
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 238
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 239
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 240
__attribute__((unused)) extern cudaError_t __cudaCDP2GetLastError(); 
# 241
__attribute__((unused)) extern cudaError_t __cudaCDP2PeekAtLastError(); 
# 242
__attribute__((unused)) extern const char *__cudaCDP2GetErrorString(cudaError_t error); 
# 243
__attribute__((unused)) extern const char *__cudaCDP2GetErrorName(cudaError_t error); 
# 244
__attribute__((unused)) extern cudaError_t __cudaCDP2GetDeviceCount(int * count); 
# 245
__attribute__((unused)) extern cudaError_t __cudaCDP2GetDevice(int * device); 
# 246
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 247
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamDestroy(cudaStream_t stream); 
# 248
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 249
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 250
__attribute__((unused)) extern cudaError_t __cudaCDP2EventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 251
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecord(cudaEvent_t event, cudaStream_t stream); 
# 252
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 253
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 254
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 255
__attribute__((unused)) extern cudaError_t __cudaCDP2EventDestroy(cudaEvent_t event); 
# 256
__attribute__((unused)) extern cudaError_t __cudaCDP2FuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 257
__attribute__((unused)) extern cudaError_t __cudaCDP2Free(void * devPtr); 
# 258
__attribute__((unused)) extern cudaError_t __cudaCDP2Malloc(void ** devPtr, size_t size); 
# 259
__attribute__((unused)) extern cudaError_t __cudaCDP2MemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 260
__attribute__((unused)) extern cudaError_t __cudaCDP2MemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 261
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 262
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 263
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 264
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 265
__attribute__((unused)) extern cudaError_t __cudaCDP2MemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 266
__attribute__((unused)) extern cudaError_t __cudaCDP2MemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 267
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 268
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 269
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 270
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 271
__attribute__((unused)) extern cudaError_t __cudaCDP2RuntimeGetVersion(int * runtimeVersion); 
# 272
__attribute__((unused)) extern void *__cudaCDP2GetParameterBuffer(size_t alignment, size_t size); 
# 273
__attribute__((unused)) extern void *__cudaCDP2GetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 274
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 275
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 276
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 277
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 278
__attribute__((unused)) extern cudaError_t __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 279
__attribute__((unused)) extern cudaError_t __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 282
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 301 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline cudaGraphExec_t cudaGetCurrentGraphExec() 
# 302
{int volatile ___ = 1;
# 306
::exit(___);}
#if 0
# 302
{ 
# 303
unsigned long long current_graph_exec; 
# 304
__asm__("mov.u64 %0, %%current_graph_exec;" : "=l" (current_graph_exec) :); 
# 305
return (cudaGraphExec_t)current_graph_exec; 
# 306
} 
#endif
# 336 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const void * value, size_t size); 
# 364
__attribute__((unused)) extern cudaError_t cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t node, bool enable); 
# 391
__attribute__((unused)) extern cudaError_t cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t node, dim3 gridDim); 
# 420
__attribute__((unused)) extern cudaError_t cudaGraphKernelNodeUpdatesApply(const cudaGraphKernelNodeUpdate * updates, size_t updateCount); 
# 438
__attribute__((unused)) static inline void cudaTriggerProgrammaticLaunchCompletion() 
# 439
{int volatile ___ = 1;
# 441
::exit(___);}
#if 0
# 439
{ 
# 440
__asm__ volatile("griddepcontrol.launch_dependents;" : :); 
# 441
} 
#endif
# 454 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline void cudaGridDependencySynchronize() 
# 455
{int volatile ___ = 1;
# 457
::exit(___);}
#if 0
# 455
{ 
# 456
__asm__ volatile("griddepcontrol.wait;" : : : "memory"); 
# 457
} 
#endif
# 466 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned value); 
# 469
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 470
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 471
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 472
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 473
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 695 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline void *cudaGetParameterBuffer(size_t alignment, size_t size) 
# 696
{int volatile ___ = 1;(void)alignment;(void)size;
# 698
::exit(___);}
#if 0
# 696
{ 
# 697
return __cudaCDP2GetParameterBuffer(alignment, size); 
# 698
} 
#endif
# 705 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline void *cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize) 
# 706
{int volatile ___ = 1;(void)func;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;
# 708
::exit(___);}
#if 0
# 706
{ 
# 707
return __cudaCDP2GetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize); 
# 708
} 
#endif
# 715 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline cudaError_t cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
# 716
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;
# 718
::exit(___);}
#if 0
# 716
{ 
# 717
return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
# 718
} 
#endif
# 720 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline cudaError_t cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream) 
# 721
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;
# 723
::exit(___);}
#if 0
# 721
{ 
# 722
return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream); 
# 723
} 
#endif
# 781 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
# 782
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;
# 784
::exit(___);}
#if 0
# 782
{ 
# 783
return __cudaCDP2LaunchDevice(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
# 784
} 
#endif
# 786 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) static inline cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream) 
# 787
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;
# 789
::exit(___);}
#if 0
# 787
{ 
# 788
return __cudaCDP2LaunchDeviceV2(parameterBuffer, stream); 
# 789
} 
#endif
# 843 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
}
# 845
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 846
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 847
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 848
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 876
template< class T> __attribute__((unused)) static inline cudaError_t 
# 877
cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const T &value) 
# 878
{int volatile ___ = 1;(void)node;(void)offset;(void)value;
# 880
::exit(___);}
#if 0
# 878
{ 
# 879
return cudaGraphKernelNodeSetParam(node, offset, &value, sizeof(T)); 
# 880
} 
#endif
# 283 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern "C" {
# 323
extern cudaError_t cudaDeviceReset(); 
# 345
extern cudaError_t cudaDeviceSynchronize(); 
# 431
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 467
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 490
extern cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
# 524
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 561
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 605
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 632
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 662
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 712
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 755
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 799
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 865
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 903
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 935
extern cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
# 973
extern cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void * userData, cudaAsyncCallbackHandle_t * callback); 
# 996
extern cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback); 
# 1043
__attribute((deprecated)) extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 1089
__attribute((deprecated)) extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 1130
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 1156
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1205
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1238
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1274
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1321
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1386
extern cudaError_t cudaGetLastError(); 
# 1437
extern cudaError_t cudaPeekAtLastError(); 
# 1453
extern const char *cudaGetErrorName(cudaError_t error); 
# 1469
extern const char *cudaGetErrorString(cudaError_t error); 
# 1498
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1803
extern cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp * prop, int device); 
# 2005
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 2023
extern cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
# 2047
extern cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
# 2067
extern cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
# 2129
extern cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
# 2169
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 2191
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 2220
extern cudaError_t cudaInitDevice(int device, unsigned deviceFlags, unsigned flags); 
# 2266
extern cudaError_t cudaSetDevice(int device); 
# 2288
extern cudaError_t cudaGetDevice(int * device); 
# 2319
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 2389
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2434
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2474
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2506
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2554
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2581
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2606
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2643
extern cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long * streamId); 
# 2658
extern cudaError_t cudaCtxResetPersistingL2Cache(); 
# 2678
extern cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
# 2699
extern cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 2723
extern cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 2757
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2788
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
# 2796
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2863
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2887
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2912
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2996
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 3035
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 3076
extern cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaStreamCaptureMode mode); 
# 3127
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 3156
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 3194
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 3243
extern cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
# 3302
extern cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, const cudaGraphEdgeData ** edgeData_out = 0, size_t * numDependencies_out = 0); 
# 3342
extern cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
# 3377
extern cudaError_t cudaStreamUpdateCaptureDependencies_v2(cudaStream_t stream, cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, unsigned flags = 0); 
# 3414
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 3451
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 3492
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 3540
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
# 3573
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 3604
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 3634
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 3679
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3860
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3915
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3975
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3999
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 4153
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 4236
extern cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 4312
extern cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 4335
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 4402
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4464
extern cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
# 4521
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4622
__attribute((deprecated)) extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 4667
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 4700
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 4737
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 4761
extern cudaError_t cudaFuncGetName(const char ** name, const void * func); 
# 4783
extern cudaError_t cudaFuncGetParamInfo(const void * func, size_t paramIndex, size_t * paramOffset, size_t * paramSize); 
# 4807
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 4831
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 4897
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 4971
__attribute((deprecated)) extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 5027
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 5056
extern cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
# 5101
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 5136
extern cudaError_t cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 5175
extern cudaError_t cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 5295
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 5328
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 5365
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 5408
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 5460
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 5498
extern cudaError_t cudaFree(void * devPtr); 
# 5521
extern cudaError_t cudaFreeHost(void * ptr); 
# 5544
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 5567
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 5633
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 5730
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 5753
extern cudaError_t cudaHostUnregister(void * ptr); 
# 5798
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 5820
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 5859
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 6004
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 6149
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 6182
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 6287
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 6319
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 6437
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 6464
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 6498
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 6524
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 6553
extern cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
# 6576
extern cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
# 6600
extern cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
# 6628
extern cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
# 6658
extern cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
# 6703
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 6738
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 6787
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6837
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6887
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 6934
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6977
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 7021
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 7078
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7113
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 7176
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7234
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7291
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7342
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7393
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7422
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 7456
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 7502
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 7538
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 7579
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 7632
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 7660
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 7687
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 7757
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 7759
extern cudaError_t cudaMemPrefetchAsync_v2(const void * devPtr, size_t count, cudaMemLocation location, unsigned flags, cudaStream_t stream = 0); 
# 7873
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 7996
extern cudaError_t cudaMemAdvise_v2(const void * devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location); 
# 8078
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 8121
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 8181
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 8223
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 8266
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 8317
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 8367
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 8436
extern cudaError_t cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
# 8462
extern cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
# 8487
extern cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
# 8531
extern cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 8579
extern cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 8594
extern cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
# 8607
extern cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
# 8645
extern cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
# 8667
extern cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool); 
# 8703
extern cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
# 8728
extern cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8755
extern cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8778
extern cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
# 8807
extern cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
# 8960
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 9001
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 9043
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 9065
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 9129
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 9164
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 9203
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 9238
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 9270
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 9308
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 9337
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 9372
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 9402
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 9626
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9646
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 9666
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 9686
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 9707
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 9752
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 9772
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 9791
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 9825
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 9854
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 9901
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 9999
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 10032
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 10058
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 10078
extern cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
# 10101
extern cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 10125
extern cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 10176
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 10235
extern cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10304
extern cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10372
extern cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10404
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 10431
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 10470
extern cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10516
extern cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10562
extern cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10610
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 10633
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 10657
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 10699
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 10722
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 10746
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 10787
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 10814
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 10852
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 10896
extern cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10923
extern cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10951
extern cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10998
extern cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 11025
extern cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 11053
extern cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 11103
extern cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 11136
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
# 11164
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 11214
extern cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 11247
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
# 11275
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 11353
extern cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
# 11380
extern cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
# 11441
extern cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
# 11465
extern cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
# 11493
extern cudaError_t cudaDeviceGraphMemTrim(int device); 
# 11530
extern cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11564
extern cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11592
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 11620
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 11651
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 11682
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 11713
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 11747
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 11787
extern cudaError_t cudaGraphGetEdges_v2(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, cudaGraphEdgeData * edgeData, size_t * numEdges); 
# 11818
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 11855
extern cudaError_t cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, cudaGraphEdgeData * edgeData, size_t * pNumDependencies); 
# 11887
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 11925
extern cudaError_t cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, cudaGraphEdgeData * edgeData, size_t * pNumDependentNodes); 
# 11956
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11988
extern cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
# 12019
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 12054
extern cudaError_t cudaGraphRemoveDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
# 12084
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 12155
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0); 
# 12228
extern cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0); 
# 12335
extern cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams * instantiateParams); 
# 12360
extern cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long * flags); 
# 12419
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 12470
extern cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 12525
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 12588
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 12649
extern cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 12704
extern cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 12744
extern cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 12791
extern cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
# 12836
extern cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12881
extern cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12929
extern cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 12977
extern cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 13017
extern cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
# 13051
extern cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
# 13143
extern cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo * resultInfo); 
# 13168
extern cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 13199
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 13222
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 13243
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 13262
extern cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
# 13298
extern cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
# 13322
extern cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
# 13350
extern cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
# 13378
extern cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
# 13403
extern cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
# 13445
extern cudaError_t cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraphNodeParams * nodeParams); 
# 13489
extern cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaGraphNodeParams * nodeParams); 
# 13518
extern cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
# 13567
extern cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
# 13593
extern cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle * pHandle_out, cudaGraph_t graph, unsigned defaultLaunchValue = 0, unsigned flags = 0); 
# 13671
extern cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
# 13679
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 13858
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
# 13874
extern cudaError_t cudaGetKernel(cudaKernel_t * kernelPtr, const void * entryFuncAddr); 
# 14044 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
}
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 118
{ 
# 119
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 120
} 
# 122
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 123
{ 
# 124
int e = (((int)sizeof(unsigned short)) * 8); 
# 126
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 127
} 
# 129
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 130
{ 
# 131
int e = (((int)sizeof(unsigned short)) * 8); 
# 133
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 134
} 
# 136
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 137
{ 
# 138
int e = (((int)sizeof(unsigned short)) * 8); 
# 140
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 141
} 
# 143
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 144
{ 
# 145
int e = (((int)sizeof(unsigned short)) * 8); 
# 147
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 148
} 
# 150
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 151
{ 
# 152
int e = (((int)sizeof(char)) * 8); 
# 157
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 159
} 
# 161
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 162
{ 
# 163
int e = (((int)sizeof(signed char)) * 8); 
# 165
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 166
} 
# 168
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 169
{ 
# 170
int e = (((int)sizeof(unsigned char)) * 8); 
# 172
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 173
} 
# 175
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 176
{ 
# 177
int e = (((int)sizeof(signed char)) * 8); 
# 179
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 180
} 
# 182
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 183
{ 
# 184
int e = (((int)sizeof(unsigned char)) * 8); 
# 186
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 187
} 
# 189
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 190
{ 
# 191
int e = (((int)sizeof(signed char)) * 8); 
# 193
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 194
} 
# 196
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 197
{ 
# 198
int e = (((int)sizeof(unsigned char)) * 8); 
# 200
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 201
} 
# 203
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 204
{ 
# 205
int e = (((int)sizeof(signed char)) * 8); 
# 207
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 208
} 
# 210
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 211
{ 
# 212
int e = (((int)sizeof(unsigned char)) * 8); 
# 214
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 215
} 
# 217
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 218
{ 
# 219
int e = (((int)sizeof(short)) * 8); 
# 221
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 222
} 
# 224
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 225
{ 
# 226
int e = (((int)sizeof(unsigned short)) * 8); 
# 228
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 229
} 
# 231
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 232
{ 
# 233
int e = (((int)sizeof(short)) * 8); 
# 235
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 236
} 
# 238
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 239
{ 
# 240
int e = (((int)sizeof(unsigned short)) * 8); 
# 242
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 243
} 
# 245
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 246
{ 
# 247
int e = (((int)sizeof(short)) * 8); 
# 249
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 250
} 
# 252
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 253
{ 
# 254
int e = (((int)sizeof(unsigned short)) * 8); 
# 256
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 257
} 
# 259
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 260
{ 
# 261
int e = (((int)sizeof(short)) * 8); 
# 263
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 264
} 
# 266
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 267
{ 
# 268
int e = (((int)sizeof(unsigned short)) * 8); 
# 270
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 271
} 
# 273
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 274
{ 
# 275
int e = (((int)sizeof(int)) * 8); 
# 277
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 278
} 
# 280
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 281
{ 
# 282
int e = (((int)sizeof(unsigned)) * 8); 
# 284
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 285
} 
# 287
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 288
{ 
# 289
int e = (((int)sizeof(int)) * 8); 
# 291
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 292
} 
# 294
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 295
{ 
# 296
int e = (((int)sizeof(unsigned)) * 8); 
# 298
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 299
} 
# 301
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 302
{ 
# 303
int e = (((int)sizeof(int)) * 8); 
# 305
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 306
} 
# 308
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 309
{ 
# 310
int e = (((int)sizeof(unsigned)) * 8); 
# 312
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 313
} 
# 315
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 316
{ 
# 317
int e = (((int)sizeof(int)) * 8); 
# 319
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 320
} 
# 322
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 323
{ 
# 324
int e = (((int)sizeof(unsigned)) * 8); 
# 326
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 327
} 
# 389 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 390
{ 
# 391
int e = (((int)sizeof(float)) * 8); 
# 393
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 394
} 
# 396
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 397
{ 
# 398
int e = (((int)sizeof(float)) * 8); 
# 400
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 401
} 
# 403
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 404
{ 
# 405
int e = (((int)sizeof(float)) * 8); 
# 407
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 408
} 
# 410
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 411
{ 
# 412
int e = (((int)sizeof(float)) * 8); 
# 414
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 415
} 
# 417
static inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
# 418
{ 
# 419
int e = (((int)sizeof(char)) * 8); 
# 421
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
# 422
} 
# 424
template< cudaChannelFormatKind > inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 425
{ 
# 426
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 427
} 
# 430
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
# 431
{ 
# 432
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
# 433
} 
# 435
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
# 436
{ 
# 437
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
# 438
} 
# 440
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
# 441
{ 
# 442
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
# 443
} 
# 446
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
# 447
{ 
# 448
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
# 449
} 
# 451
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
# 452
{ 
# 453
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
# 454
} 
# 456
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
# 457
{ 
# 458
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
# 459
} 
# 462
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
# 463
{ 
# 464
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
# 465
} 
# 467
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
# 468
{ 
# 469
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
# 470
} 
# 472
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
# 473
{ 
# 474
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
# 475
} 
# 478
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
# 479
{ 
# 480
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
# 481
} 
# 483
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
# 484
{ 
# 485
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
# 486
} 
# 488
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
# 489
{ 
# 490
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
# 491
} 
# 494
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
# 495
{ 
# 496
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
# 497
} 
# 500
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
# 501
{ 
# 502
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
# 503
} 
# 506
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
# 507
{ 
# 508
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
# 509
} 
# 512
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
# 513
{ 
# 514
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
# 515
} 
# 518
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
# 519
{ 
# 520
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
# 521
} 
# 524
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
# 525
{ 
# 526
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
# 527
} 
# 530
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
# 531
{ 
# 532
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
# 533
} 
# 536
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
# 537
{ 
# 538
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
# 539
} 
# 542
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
# 543
{ 
# 544
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
# 545
} 
# 548
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
# 549
{ 
# 550
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
# 551
} 
# 554
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
# 555
{ 
# 556
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
# 557
} 
# 560
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
# 561
{ 
# 562
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
# 563
} 
# 566
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
# 567
{ 
# 568
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
# 569
} 
# 572
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
# 573
{ 
# 574
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
# 575
} 
# 578
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
# 579
{ 
# 580
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
# 581
} 
# 79 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 77 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 79
static inline uchar1 make_uchar1(unsigned char x); 
# 81
static inline char2 make_char2(signed char x, signed char y); 
# 83
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 85
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 87
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 89
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 91
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 93
static inline short1 make_short1(short x); 
# 95
static inline ushort1 make_ushort1(unsigned short x); 
# 97
static inline short2 make_short2(short x, short y); 
# 99
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 101
static inline short3 make_short3(short x, short y, short z); 
# 103
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 105
static inline short4 make_short4(short x, short y, short z, short w); 
# 107
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 109
static inline int1 make_int1(int x); 
# 111
static inline uint1 make_uint1(unsigned x); 
# 113
static inline int2 make_int2(int x, int y); 
# 115
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 117
static inline int3 make_int3(int x, int y, int z); 
# 119
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 121
static inline int4 make_int4(int x, int y, int z, int w); 
# 123
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 125
static inline long1 make_long1(long x); 
# 127
static inline ulong1 make_ulong1(unsigned long x); 
# 129
static inline long2 make_long2(long x, long y); 
# 131
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 133
static inline long3 make_long3(long x, long y, long z); 
# 135
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 137
static inline long4 make_long4(long x, long y, long z, long w); 
# 139
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 141
static inline float1 make_float1(float x); 
# 143
static inline float2 make_float2(float x, float y); 
# 145
static inline float3 make_float3(float x, float y, float z); 
# 147
static inline float4 make_float4(float x, float y, float z, float w); 
# 149
static inline longlong1 make_longlong1(long long x); 
# 151
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 153
static inline longlong2 make_longlong2(long long x, long long y); 
# 155
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 157
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 159
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 161
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 163
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 165
static inline double1 make_double1(double x); 
# 167
static inline double2 make_double2(double x, double y); 
# 169
static inline double3 make_double3(double x, double y, double z); 
# 171
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 28 "/usr/include/string.h" 3
extern "C" {
# 43 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 44
 __attribute((__nonnull__(1, 2))); 
# 47
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 48
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 61
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 64
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 65
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 69
extern "C++" {
# 71
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 72
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 73
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 74
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 89 "/usr/include/string.h" 3
}
# 99
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 100
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 101
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 110
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 111
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 122
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 123
 __attribute((__nonnull__(1, 2))); 
# 125
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 127
 __attribute((__nonnull__(1, 2))); 
# 130
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 131
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 137
extern int strcmp(const char * __s1, const char * __s2) throw()
# 138
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 140
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 144
extern int strcoll(const char * __s1, const char * __s2) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 149
 __attribute((__nonnull__(2))); 
# 156
extern int strcoll_l(const char * __s1, const char * __s2, locale_t __l) throw()
# 157
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 160
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, locale_t __l) throw()
# 161
 __attribute((__nonnull__(2, 4))); 
# 167
extern char *strdup(const char * __s) throw()
# 168
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 175
extern char *strndup(const char * __string, size_t __n) throw()
# 176
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 204 "/usr/include/string.h" 3
extern "C++" {
# 206
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 207
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 208
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 209
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 224 "/usr/include/string.h" 3
}
# 231
extern "C++" {
# 233
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 234
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 235
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 236
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 251 "/usr/include/string.h" 3
}
# 261
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 262
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 263
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 264
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 273
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 274
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 277
extern size_t strspn(const char * __s, const char * __accept) throw()
# 278
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 281
extern "C++" {
# 283
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 284
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 285
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 286
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 301 "/usr/include/string.h" 3
}
# 308
extern "C++" {
# 310
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 311
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 312
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 313
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 328 "/usr/include/string.h" 3
}
# 336
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 337
 __attribute((__nonnull__(2))); 
# 341
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 344
 __attribute((__nonnull__(2, 3))); 
# 346
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 348
 __attribute((__nonnull__(2, 3))); 
# 354
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 355
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 356
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 358
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 369
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 371
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 375
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 377
 __attribute((__nonnull__(1, 2))); 
# 378
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 380
 __attribute((__nonnull__(1, 2))); 
# 385
extern size_t strlen(const char * __s) throw()
# 386
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 391
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 392
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 397
extern char *strerror(int __errnum) throw(); 
# 421 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 422
 __attribute((__nonnull__(2))); 
# 428
extern char *strerror_l(int __errnum, locale_t __l) throw(); 
# 30 "/usr/include/strings.h" 3
extern "C" {
# 34
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 35
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 38
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 39
 __attribute((__nonnull__(1, 2))); 
# 42
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 46
extern "C++" {
# 48
extern char *index(char * __s, int __c) throw() __asm__("index")
# 49
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 50
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 51
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 66 "/usr/include/strings.h" 3
}
# 74
extern "C++" {
# 76
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 77
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 78
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 79
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 94 "/usr/include/strings.h" 3
}
# 104
extern int ffs(int __i) throw() __attribute((const)); 
# 110
extern int ffsl(long __l) throw() __attribute((const)); 
# 111
extern int ffsll(long long __ll) throw()
# 112
 __attribute((const)); 
# 116
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 117
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 120
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 121
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 128
extern int strcasecmp_l(const char * __s1, const char * __s2, locale_t __loc) throw()
# 129
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 133
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, locale_t __loc) throw()
# 135
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 138
}
# 436 "/usr/include/string.h" 3
extern void explicit_bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 440
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 442
 __attribute((__nonnull__(1, 2))); 
# 447
extern char *strsignal(int __sig) throw(); 
# 450
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 452
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 453
 __attribute((__nonnull__(1, 2))); 
# 457
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 459
 __attribute((__nonnull__(1, 2))); 
# 460
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 462
 __attribute((__nonnull__(1, 2))); 
# 467
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 468
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 471
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 474
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 482
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 483
 __attribute((__nonnull__(1))); 
# 484
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 485
 __attribute((__nonnull__(1))); 
# 499 "/usr/include/string.h" 3
}
# 26 "/usr/include/bits/timex.h" 3
struct timex { 
# 28
unsigned modes; 
# 29
__syscall_slong_t offset; 
# 30
__syscall_slong_t freq; 
# 31
__syscall_slong_t maxerror; 
# 32
__syscall_slong_t esterror; 
# 33
int status; 
# 34
__syscall_slong_t constant; 
# 35
__syscall_slong_t precision; 
# 36
__syscall_slong_t tolerance; 
# 37
timeval time; 
# 38
__syscall_slong_t tick; 
# 39
__syscall_slong_t ppsfreq; 
# 40
__syscall_slong_t jitter; 
# 41
int shift; 
# 42
__syscall_slong_t stabil; 
# 43
__syscall_slong_t jitcnt; 
# 44
__syscall_slong_t calcnt; 
# 45
__syscall_slong_t errcnt; 
# 46
__syscall_slong_t stbcnt; 
# 48
int tai; 
# 51
int: 32; int: 32; int: 32; int: 32; 
# 52
int: 32; int: 32; int: 32; int: 32; 
# 53
int: 32; int: 32; int: 32; 
# 54
}; 
# 75 "/usr/include/bits/time.h" 3
extern "C" {
# 78
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 80
}
# 7 "/usr/include/bits/types/struct_tm.h" 3
struct tm { 
# 9
int tm_sec; 
# 10
int tm_min; 
# 11
int tm_hour; 
# 12
int tm_mday; 
# 13
int tm_mon; 
# 14
int tm_year; 
# 15
int tm_wday; 
# 16
int tm_yday; 
# 17
int tm_isdst; 
# 20
long tm_gmtoff; 
# 21
const char *tm_zone; 
# 26
}; 
# 8 "/usr/include/bits/types/struct_itimerspec.h" 3
struct itimerspec { 
# 10
timespec it_interval; 
# 11
timespec it_value; 
# 12
}; 
# 49 "/usr/include/time.h" 3
struct sigevent; 
# 68 "/usr/include/time.h" 3
extern "C" {
# 72
extern clock_t clock() throw(); 
# 75
extern time_t time(time_t * __timer) throw(); 
# 78
extern double difftime(time_t __time1, time_t __time0) throw()
# 79
 __attribute((const)); 
# 82
extern time_t mktime(tm * __tp) throw(); 
# 88
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 95
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 104
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, locale_t __loc) throw(); 
# 111
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, locale_t __loc) throw(); 
# 119
extern tm *gmtime(const time_t * __timer) throw(); 
# 123
extern tm *localtime(const time_t * __timer) throw(); 
# 128
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 133
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 139
extern char *asctime(const tm * __tp) throw(); 
# 142
extern char *ctime(const time_t * __timer) throw(); 
# 149
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 153
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 159
extern char *__tzname[2]; 
# 160
extern int __daylight; 
# 161
extern long __timezone; 
# 166
extern char *tzname[2]; 
# 170
extern void tzset() throw(); 
# 174
extern int daylight; 
# 175
extern long timezone; 
# 190
extern time_t timegm(tm * __tp) throw(); 
# 193
extern time_t timelocal(tm * __tp) throw(); 
# 196
extern int dysize(int __year) throw() __attribute((const)); 
# 205
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 210
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 213
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 216
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 224
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 229
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 234
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 239
extern int timer_delete(timer_t __timerid) throw(); 
# 242
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 247
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 251
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 257
extern int timespec_get(timespec * __ts, int __base) throw()
# 258
 __attribute((__nonnull__(1))); 
# 274
extern int getdate_err; 
# 283
extern tm *getdate(const char * __string); 
# 297
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 301
}
# 88 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C" {
# 91
extern clock_t clock() throw(); 
# 96
extern void *memset(void *, int, size_t) throw(); 
# 97
extern void *memcpy(void *, const void *, size_t) throw(); 
# 99
}
# 124 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" {
# 222 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int abs(int a) throw(); 
# 230
extern long labs(long a) throw(); 
# 238
extern long long llabs(long long a) throw(); 
# 288
extern double fabs(double x) throw(); 
# 331
extern float fabsf(float x) throw(); 
# 341
extern inline int min(const int a, const int b); 
# 348
extern inline unsigned umin(const unsigned a, const unsigned b); 
# 355
extern inline long long llmin(const long long a, const long long b); 
# 362
extern inline unsigned long long ullmin(const unsigned long long a, const unsigned long long b); 
# 383
extern float fminf(float x, float y) throw(); 
# 403
extern double fmin(double x, double y) throw(); 
# 416 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline int max(const int a, const int b); 
# 424
extern inline unsigned umax(const unsigned a, const unsigned b); 
# 431
extern inline long long llmax(const long long a, const long long b); 
# 438
extern inline unsigned long long ullmax(const unsigned long long a, const unsigned long long b); 
# 459
extern float fmaxf(float x, float y) throw(); 
# 479
extern double fmax(double, double) throw(); 
# 523
extern double sin(double x) throw(); 
# 556
extern double cos(double x) throw(); 
# 575
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 591
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 636
extern double tan(double x) throw(); 
# 705
extern double sqrt(double x) throw(); 
# 777
extern double rsqrt(double x); 
# 847
extern float rsqrtf(float x); 
# 903
extern double log2(double x) throw(); 
# 968
extern double exp2(double x) throw(); 
# 1033
extern float exp2f(float x) throw(); 
# 1100 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 1163
extern float exp10f(float x) throw(); 
# 1256
extern double expm1(double x) throw(); 
# 1348
extern float expm1f(float x) throw(); 
# 1404
extern float log2f(float x) throw(); 
# 1458
extern double log10(double x) throw(); 
# 1528
extern double log(double x) throw(); 
# 1624
extern double log1p(double x) throw(); 
# 1723
extern float log1pf(float x) throw(); 
# 1787
extern double floor(double x) throw(); 
# 1866
extern double exp(double x) throw(); 
# 1907
extern double cosh(double x) throw(); 
# 1957
extern double sinh(double x) throw(); 
# 2007
extern double tanh(double x) throw(); 
# 2062
extern double acosh(double x) throw(); 
# 2120
extern float acoshf(float x) throw(); 
# 2173
extern double asinh(double x) throw(); 
# 2226
extern float asinhf(float x) throw(); 
# 2280
extern double atanh(double x) throw(); 
# 2334
extern float atanhf(float x) throw(); 
# 2383
extern double ldexp(double x, int exp) throw(); 
# 2429
extern float ldexpf(float x, int exp) throw(); 
# 2481
extern double logb(double x) throw(); 
# 2536
extern float logbf(float x) throw(); 
# 2576
extern int ilogb(double x) throw(); 
# 2616
extern int ilogbf(float x) throw(); 
# 2692
extern double scalbn(double x, int n) throw(); 
# 2768
extern float scalbnf(float x, int n) throw(); 
# 2844
extern double scalbln(double x, long n) throw(); 
# 2920
extern float scalblnf(float x, long n) throw(); 
# 2997
extern double frexp(double x, int * nptr) throw(); 
# 3071
extern float frexpf(float x, int * nptr) throw(); 
# 3123
extern double round(double x) throw(); 
# 3178
extern float roundf(float x) throw(); 
# 3196
extern long lround(double x) throw(); 
# 3214
extern long lroundf(float x) throw(); 
# 3232
extern long long llround(double x) throw(); 
# 3250
extern long long llroundf(float x) throw(); 
# 3378 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 3395
extern long lrint(double x) throw(); 
# 3412
extern long lrintf(float x) throw(); 
# 3429
extern long long llrint(double x) throw(); 
# 3446
extern long long llrintf(float x) throw(); 
# 3499
extern double nearbyint(double x) throw(); 
# 3552
extern float nearbyintf(float x) throw(); 
# 3614
extern double ceil(double x) throw(); 
# 3664
extern double trunc(double x) throw(); 
# 3717
extern float truncf(float x) throw(); 
# 3743
extern double fdim(double x, double y) throw(); 
# 3769
extern float fdimf(float x, float y) throw(); 
# 4069
extern double atan2(double y, double x) throw(); 
# 4140
extern double atan(double x) throw(); 
# 4163
extern double acos(double x) throw(); 
# 4214
extern double asin(double x) throw(); 
# 4282 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 4405
extern float hypotf(float x, float y) throw(); 
# 5191
extern double cbrt(double x) throw(); 
# 5277
extern float cbrtf(float x) throw(); 
# 5332 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rcbrt(double x); 
# 5382
extern float rcbrtf(float x); 
# 5442
extern double sinpi(double x); 
# 5502
extern float sinpif(float x); 
# 5554
extern double cospi(double x); 
# 5606
extern float cospif(float x); 
# 5636
extern void sincospi(double x, double * sptr, double * cptr); 
# 5666
extern void sincospif(float x, float * sptr, float * cptr); 
# 5999
extern double pow(double x, double y) throw(); 
# 6055
extern double modf(double x, double * iptr) throw(); 
# 6114
extern double fmod(double x, double y) throw(); 
# 6210
extern double remainder(double x, double y) throw(); 
# 6309
extern float remainderf(float x, float y) throw(); 
# 6381
extern double remquo(double x, double y, int * quo) throw(); 
# 6453
extern float remquof(float x, float y, int * quo) throw(); 
# 6494
extern double j0(double x) throw(); 
# 6536
extern float j0f(float x) throw(); 
# 6605
extern double j1(double x) throw(); 
# 6674
extern float j1f(float x) throw(); 
# 6717
extern double jn(int n, double x) throw(); 
# 6760
extern float jnf(int n, float x) throw(); 
# 6821
extern double y0(double x) throw(); 
# 6882
extern float y0f(float x) throw(); 
# 6943
extern double y1(double x) throw(); 
# 7004
extern float y1f(float x) throw(); 
# 7067
extern double yn(int n, double x) throw(); 
# 7130
extern float ynf(int n, float x) throw(); 
# 7319
extern double erf(double x) throw(); 
# 7401
extern float erff(float x) throw(); 
# 7473 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfinv(double x); 
# 7538
extern float erfinvf(float x); 
# 7577
extern double erfc(double x) throw(); 
# 7615
extern float erfcf(float x) throw(); 
# 7732
extern double lgamma(double x) throw(); 
# 7794 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcinv(double x); 
# 7849
extern float erfcinvf(float x); 
# 7917
extern double normcdfinv(double x); 
# 7985
extern float normcdfinvf(float x); 
# 8028
extern double normcdf(double x); 
# 8071
extern float normcdff(float x); 
# 8135
extern double erfcx(double x); 
# 8199
extern float erfcxf(float x); 
# 8318
extern float lgammaf(float x) throw(); 
# 8416
extern double tgamma(double x) throw(); 
# 8514
extern float tgammaf(float x) throw(); 
# 8527
extern double copysign(double x, double y) throw(); 
# 8540
extern float copysignf(float x, float y) throw(); 
# 8559
extern double nextafter(double x, double y) throw(); 
# 8578
extern float nextafterf(float x, float y) throw(); 
# 8594
extern double nan(const char * tagp) throw(); 
# 8610
extern float nanf(const char * tagp) throw(); 
# 8617 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinff(float) throw(); 
# 8618
extern int __isnanf(float) throw(); 
# 8628 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 8629
extern int __finitef(float) throw(); 
# 8630
extern int __signbit(double) throw(); 
# 8631
extern int __isnan(double) throw(); 
# 8632
extern int __isinf(double) throw(); 
# 8635
extern int __signbitf(float) throw(); 
# 8794
extern double fma(double x, double y, double z) throw(); 
# 8952
extern float fmaf(float x, float y, float z) throw(); 
# 8963 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbitl(long double) throw(); 
# 8969
extern int __finitel(long double) throw(); 
# 8970
extern int __isinfl(long double) throw(); 
# 8971
extern int __isnanl(long double) throw(); 
# 9021 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 9080
extern float asinf(float x) throw(); 
# 9160
extern float atanf(float x) throw(); 
# 9457
extern float atan2f(float y, float x) throw(); 
# 9491
extern float cosf(float x) throw(); 
# 9533
extern float sinf(float x) throw(); 
# 9575
extern float tanf(float x) throw(); 
# 9616
extern float coshf(float x) throw(); 
# 9666
extern float sinhf(float x) throw(); 
# 9716
extern float tanhf(float x) throw(); 
# 9768
extern float logf(float x) throw(); 
# 9848
extern float expf(float x) throw(); 
# 9900
extern float log10f(float x) throw(); 
# 9955
extern float modff(float x, float * iptr) throw(); 
# 10285
extern float powf(float x, float y) throw(); 
# 10354
extern float sqrtf(float x) throw(); 
# 10413
extern float ceilf(float x) throw(); 
# 10474
extern float floorf(float x) throw(); 
# 10532
extern float fmodf(float x, float y) throw(); 
# 10547 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 67 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
extern "C++" {
# 69
namespace std __attribute((__visibility__("default"))) { 
# 73
struct __true_type { }; 
# 74
struct __false_type { }; 
# 76
template< bool > 
# 77
struct __truth_type { 
# 78
typedef __false_type __type; }; 
# 81
template<> struct __truth_type< true>  { 
# 82
typedef __true_type __type; }; 
# 86
template< class _Sp, class _Tp> 
# 87
struct __traitor { 
# 89
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 90
typedef typename __truth_type< __value> ::__type __type; 
# 91
}; 
# 94
template< class , class > 
# 95
struct __are_same { 
# 97
enum { __value}; 
# 98
typedef __false_type __type; 
# 99
}; 
# 101
template< class _Tp> 
# 102
struct __are_same< _Tp, _Tp>  { 
# 104
enum { __value = 1}; 
# 105
typedef __true_type __type; 
# 106
}; 
# 109
template< class _Tp> 
# 110
struct __is_void { 
# 112
enum { __value}; 
# 113
typedef __false_type __type; 
# 114
}; 
# 117
template<> struct __is_void< void>  { 
# 119
enum { __value = 1}; 
# 120
typedef __true_type __type; 
# 121
}; 
# 126
template< class _Tp> 
# 127
struct __is_integer { 
# 129
enum { __value}; 
# 130
typedef __false_type __type; 
# 131
}; 
# 138
template<> struct __is_integer< bool>  { 
# 140
enum { __value = 1}; 
# 141
typedef __true_type __type; 
# 142
}; 
# 145
template<> struct __is_integer< char>  { 
# 147
enum { __value = 1}; 
# 148
typedef __true_type __type; 
# 149
}; 
# 152
template<> struct __is_integer< signed char>  { 
# 154
enum { __value = 1}; 
# 155
typedef __true_type __type; 
# 156
}; 
# 159
template<> struct __is_integer< unsigned char>  { 
# 161
enum { __value = 1}; 
# 162
typedef __true_type __type; 
# 163
}; 
# 167
template<> struct __is_integer< wchar_t>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 185 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char16_t>  { 
# 187
enum { __value = 1}; 
# 188
typedef __true_type __type; 
# 189
}; 
# 192
template<> struct __is_integer< char32_t>  { 
# 194
enum { __value = 1}; 
# 195
typedef __true_type __type; 
# 196
}; 
# 200
template<> struct __is_integer< short>  { 
# 202
enum { __value = 1}; 
# 203
typedef __true_type __type; 
# 204
}; 
# 207
template<> struct __is_integer< unsigned short>  { 
# 209
enum { __value = 1}; 
# 210
typedef __true_type __type; 
# 211
}; 
# 214
template<> struct __is_integer< int>  { 
# 216
enum { __value = 1}; 
# 217
typedef __true_type __type; 
# 218
}; 
# 221
template<> struct __is_integer< unsigned>  { 
# 223
enum { __value = 1}; 
# 224
typedef __true_type __type; 
# 225
}; 
# 228
template<> struct __is_integer< long>  { 
# 230
enum { __value = 1}; 
# 231
typedef __true_type __type; 
# 232
}; 
# 235
template<> struct __is_integer< unsigned long>  { 
# 237
enum { __value = 1}; 
# 238
typedef __true_type __type; 
# 239
}; 
# 242
template<> struct __is_integer< long long>  { 
# 244
enum { __value = 1}; 
# 245
typedef __true_type __type; 
# 246
}; 
# 249
template<> struct __is_integer< unsigned long long>  { 
# 251
enum { __value = 1}; 
# 252
typedef __true_type __type; 
# 253
}; 
# 272 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< unsigned __int128>  { enum { __value = 1}; typedef __true_type __type; }; 
# 289 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 290
struct __is_floating { 
# 292
enum { __value}; 
# 293
typedef __false_type __type; 
# 294
}; 
# 298
template<> struct __is_floating< float>  { 
# 300
enum { __value = 1}; 
# 301
typedef __true_type __type; 
# 302
}; 
# 305
template<> struct __is_floating< double>  { 
# 307
enum { __value = 1}; 
# 308
typedef __true_type __type; 
# 309
}; 
# 312
template<> struct __is_floating< long double>  { 
# 314
enum { __value = 1}; 
# 315
typedef __true_type __type; 
# 316
}; 
# 366 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 367
struct __is_pointer { 
# 369
enum { __value}; 
# 370
typedef __false_type __type; 
# 371
}; 
# 373
template< class _Tp> 
# 374
struct __is_pointer< _Tp *>  { 
# 376
enum { __value = 1}; 
# 377
typedef __true_type __type; 
# 378
}; 
# 383
template< class _Tp> 
# 384
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 386
}; 
# 391
template< class _Tp> 
# 392
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 394
}; 
# 399
template< class _Tp> 
# 400
struct __is_char { 
# 402
enum { __value}; 
# 403
typedef __false_type __type; 
# 404
}; 
# 407
template<> struct __is_char< char>  { 
# 409
enum { __value = 1}; 
# 410
typedef __true_type __type; 
# 411
}; 
# 415
template<> struct __is_char< wchar_t>  { 
# 417
enum { __value = 1}; 
# 418
typedef __true_type __type; 
# 419
}; 
# 422
template< class _Tp> 
# 423
struct __is_byte { 
# 425
enum { __value}; 
# 426
typedef __false_type __type; 
# 427
}; 
# 430
template<> struct __is_byte< char>  { 
# 432
enum { __value = 1}; 
# 433
typedef __true_type __type; 
# 434
}; 
# 437
template<> struct __is_byte< signed char>  { 
# 439
enum { __value = 1}; 
# 440
typedef __true_type __type; 
# 441
}; 
# 444
template<> struct __is_byte< unsigned char>  { 
# 446
enum { __value = 1}; 
# 447
typedef __true_type __type; 
# 448
}; 
# 451
enum class byte: unsigned char; 
# 454
template<> struct __is_byte< byte>  { 
# 456
enum { __value = 1}; 
# 457
typedef __true_type __type; 
# 458
}; 
# 470 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template< class > struct iterator_traits; 
# 473
template< class _Tp> 
# 474
struct __is_nonvolatile_trivially_copyable { 
# 476
enum { __value = __is_trivially_copyable(_Tp)}; 
# 477
}; 
# 482
template< class _Tp> 
# 483
struct __is_nonvolatile_trivially_copyable< volatile _Tp>  { 
# 485
enum { __value}; 
# 486
}; 
# 489
template< class _OutputIter, class _InputIter> 
# 490
struct __memcpyable { 
# 492
enum { __value}; 
# 493
}; 
# 495
template< class _Tp> 
# 496
struct __memcpyable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 498
}; 
# 500
template< class _Tp> 
# 501
struct __memcpyable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 503
}; 
# 510
template< class _Iter1, class _Iter2> 
# 511
struct __memcmpable { 
# 513
enum { __value}; 
# 514
}; 
# 517
template< class _Tp> 
# 518
struct __memcmpable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 520
}; 
# 522
template< class _Tp> 
# 523
struct __memcmpable< const _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 525
}; 
# 527
template< class _Tp> 
# 528
struct __memcmpable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 530
}; 
# 538
template< class _Tp, bool _TreatAsBytes = __is_byte< _Tp> ::__value> 
# 545
struct __is_memcmp_ordered { 
# 547
static const bool __value = (((_Tp)(-1)) > ((_Tp)1)); 
# 548
}; 
# 550
template< class _Tp> 
# 551
struct __is_memcmp_ordered< _Tp, false>  { 
# 553
static const bool __value = false; 
# 554
}; 
# 557
template< class _Tp, class _Up, bool  = sizeof(_Tp) == sizeof(_Up)> 
# 558
struct __is_memcmp_ordered_with { 
# 560
static const bool __value = (__is_memcmp_ordered< _Tp> ::__value && __is_memcmp_ordered< _Up> ::__value); 
# 562
}; 
# 564
template< class _Tp, class _Up> 
# 565
struct __is_memcmp_ordered_with< _Tp, _Up, false>  { 
# 567
static const bool __value = false; 
# 568
}; 
# 580 "/usr/include/c++/13/bits/cpp_type_traits.h" 3
template<> struct __is_memcmp_ordered_with< byte, byte, true>  { 
# 581
static constexpr inline bool __value = true; }; 
# 583
template< class _Tp, bool _SameSize> 
# 584
struct __is_memcmp_ordered_with< _Tp, byte, _SameSize>  { 
# 585
static constexpr inline bool __value = false; }; 
# 587
template< class _Up, bool _SameSize> 
# 588
struct __is_memcmp_ordered_with< byte, _Up, _SameSize>  { 
# 589
static constexpr inline bool __value = false; }; 
# 595
template< class _Tp> 
# 596
struct __is_move_iterator { 
# 598
enum { __value}; 
# 599
typedef __false_type __type; 
# 600
}; 
# 604
template< class _Iterator> inline _Iterator 
# 607
__miter_base(_Iterator __it) 
# 608
{ return __it; } 
# 611
}
# 612
}
# 37 "/usr/include/c++/13/ext/type_traits.h" 3
extern "C++" {
# 39
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
template< bool , class > 
# 45
struct __enable_if { 
# 46
}; 
# 48
template< class _Tp> 
# 49
struct __enable_if< true, _Tp>  { 
# 50
typedef _Tp __type; }; 
# 54
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 55
struct __conditional_type { 
# 56
typedef _Iftrue __type; }; 
# 58
template< class _Iftrue, class _Iffalse> 
# 59
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 60
typedef _Iffalse __type; }; 
# 64
template< class _Tp> 
# 65
struct __add_unsigned { 
# 68
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 71
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 72
}; 
# 75
template<> struct __add_unsigned< char>  { 
# 76
typedef unsigned char __type; }; 
# 79
template<> struct __add_unsigned< signed char>  { 
# 80
typedef unsigned char __type; }; 
# 83
template<> struct __add_unsigned< short>  { 
# 84
typedef unsigned short __type; }; 
# 87
template<> struct __add_unsigned< int>  { 
# 88
typedef unsigned __type; }; 
# 91
template<> struct __add_unsigned< long>  { 
# 92
typedef unsigned long __type; }; 
# 95
template<> struct __add_unsigned< long long>  { 
# 96
typedef unsigned long long __type; }; 
# 100
template<> struct __add_unsigned< bool> ; 
# 103
template<> struct __add_unsigned< wchar_t> ; 
# 107
template< class _Tp> 
# 108
struct __remove_unsigned { 
# 111
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 114
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 115
}; 
# 118
template<> struct __remove_unsigned< char>  { 
# 119
typedef signed char __type; }; 
# 122
template<> struct __remove_unsigned< unsigned char>  { 
# 123
typedef signed char __type; }; 
# 126
template<> struct __remove_unsigned< unsigned short>  { 
# 127
typedef short __type; }; 
# 130
template<> struct __remove_unsigned< unsigned>  { 
# 131
typedef int __type; }; 
# 134
template<> struct __remove_unsigned< unsigned long>  { 
# 135
typedef long __type; }; 
# 138
template<> struct __remove_unsigned< unsigned long long>  { 
# 139
typedef long long __type; }; 
# 143
template<> struct __remove_unsigned< bool> ; 
# 146
template<> struct __remove_unsigned< wchar_t> ; 
# 150
template< class _Type> constexpr bool 
# 153
__is_null_pointer(_Type *__ptr) 
# 154
{ return __ptr == 0; } 
# 156
template< class _Type> constexpr bool 
# 159
__is_null_pointer(_Type) 
# 160
{ return false; } 
# 164
constexpr bool __is_null_pointer(std::nullptr_t) 
# 165
{ return true; } 
# 170
template< class _Tp, bool  = std::template __is_integer< _Tp> ::__value> 
# 171
struct __promote { 
# 172
typedef double __type; }; 
# 177
template< class _Tp> 
# 178
struct __promote< _Tp, false>  { 
# 179
}; 
# 182
template<> struct __promote< long double>  { 
# 183
typedef long double __type; }; 
# 186
template<> struct __promote< double>  { 
# 187
typedef double __type; }; 
# 190
template<> struct __promote< float>  { 
# 191
typedef float __type; }; 
# 225
template< class ..._Tp> using __promoted_t = __decltype(((((typename __promote< _Tp> ::__type)0) + ... ))); 
# 230
template< class _Tp, class _Up> using __promote_2 = __promote< __promoted_t< _Tp, _Up> > ; 
# 233
template< class _Tp, class _Up, class _Vp> using __promote_3 = __promote< __promoted_t< _Tp, _Up, _Vp> > ; 
# 236
template< class _Tp, class _Up, class _Vp, class _Wp> using __promote_4 = __promote< __promoted_t< _Tp, _Up, _Vp, _Wp> > ; 
# 270 "/usr/include/c++/13/ext/type_traits.h" 3
}
# 271
}
# 34 "/usr/include/math.h" 3
extern "C" {
# 149 "/usr/include/math.h" 3
typedef float float_t; 
# 150
typedef double double_t; 
# 238 "/usr/include/math.h" 3
enum { 
# 239
FP_INT_UPWARD, 
# 242
FP_INT_DOWNWARD, 
# 245
FP_INT_TOWARDZERO, 
# 248
FP_INT_TONEARESTFROMZERO, 
# 251
FP_INT_TONEAREST
# 254
}; 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassify(double __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbit(double __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinf(double __value) throw() __attribute((const)); 
# 33
extern int __finite(double __value) throw() __attribute((const)); 
# 36
extern int __isnan(double __value) throw() __attribute((const)); 
# 39
extern int __iseqsig(double __x, double __y) throw(); 
# 42
extern int __issignaling(double __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 55
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 57
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 59
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 62
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 64
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 66
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 71
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 73
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 75
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 80
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 85
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 87
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 89
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 95
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 98
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 101
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 104
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 107
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 110
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 119
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 122
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 125
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 130
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 133
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 140
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 143
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 147
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 152
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 159
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 162
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 165
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 168
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 182 "/usr/include/bits/mathcalls.h" 3
extern int finite(double __value) throw() __attribute((const)); 
# 185
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 189
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 196
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 201
extern double nan(const char * __tagb) throw(); extern double __nan(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 218
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 219
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 220
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 221
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 222
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 228
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 229
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 230
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 235
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 241
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 249
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 256
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 259
extern double nextafter(double __x, double __y) throw(); extern double __nextafter(double __x, double __y) throw(); 
# 261
extern double nexttoward(double __x, long double __y) throw(); extern double __nexttoward(double __x, long double __y) throw(); 
# 266
extern double nextdown(double __x) throw(); extern double __nextdown(double __x) throw(); 
# 268
extern double nextup(double __x) throw(); extern double __nextup(double __x) throw(); 
# 272
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 276
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 280
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 285
extern long llogb(double __x) throw(); extern long __llogb(double __x) throw(); 
# 290
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 294
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 298
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 302
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 307
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 314
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 316
extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 320
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 322
extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 326
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 329
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 332
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 335
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 340
extern double roundeven(double __x) throw() __attribute((const)); extern double __roundeven(double __x) throw() __attribute((const)); 
# 345
extern __intmax_t fromfp(double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfp(double __x, int __round, unsigned __width) throw(); 
# 350
extern __uintmax_t ufromfp(double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfp(double __x, int __round, unsigned __width) throw(); 
# 356
extern __intmax_t fromfpx(double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpx(double __x, int __round, unsigned __width) throw(); 
# 362
extern __uintmax_t ufromfpx(double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpx(double __x, int __round, unsigned __width) throw(); 
# 365
extern double fmaxmag(double __x, double __y) throw() __attribute((const)); extern double __fmaxmag(double __x, double __y) throw() __attribute((const)); 
# 368
extern double fminmag(double __x, double __y) throw() __attribute((const)); extern double __fminmag(double __x, double __y) throw() __attribute((const)); 
# 371
extern int canonicalize(double * __cx, const double * __x) throw(); 
# 377
extern int totalorder(const double * __x, const double * __y) throw()
# 378
 __attribute((__pure__)); 
# 382
extern int totalordermag(const double * __x, const double * __y) throw()
# 383
 __attribute((__pure__)); 
# 386
extern double getpayload(const double * __x) throw(); extern double __getpayload(const double * __x) throw(); 
# 389
extern int setpayload(double * __x, double __payload) throw(); 
# 392
extern int setpayloadsig(double * __x, double __payload) throw(); 
# 400
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyf(float __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbitf(float __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinff(float __value) throw() __attribute((const)); 
# 33
extern int __finitef(float __value) throw() __attribute((const)); 
# 36
extern int __isnanf(float __value) throw() __attribute((const)); 
# 39
extern int __iseqsigf(float __x, float __y) throw(); 
# 42
extern int __issignalingf(float __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 55
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 57
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 59
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 62
extern float cosf(float __x) throw(); 
# 64
extern float sinf(float __x) throw(); 
# 66
extern float tanf(float __x) throw(); 
# 71
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 73
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 75
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 80
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 85
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 87
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 89
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 95
extern float expf(float __x) throw(); 
# 98
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 101
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 104
extern float logf(float __x) throw(); 
# 107
extern float log10f(float __x) throw(); 
# 110
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern float exp10f(float __x) throw(); 
# 119
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 122
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 125
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 130
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 133
extern float log2f(float __x) throw(); 
# 140
extern float powf(float __x, float __y) throw(); 
# 143
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 147
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 152
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 159
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 162
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 165
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 168
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 177
extern int isinff(float __value) throw() __attribute((const)); 
# 182
extern int finitef(float __value) throw() __attribute((const)); 
# 185
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 189
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 196
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 201
extern float nanf(const char * __tagb) throw(); extern float __nanf(const char * __tagb) throw(); 
# 211
extern int isnanf(float __value) throw() __attribute((const)); 
# 217
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 218
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 219
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 220
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 221
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 222
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 228
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 229
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 230
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 235
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 241
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 249
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 256
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 259
extern float nextafterf(float __x, float __y) throw(); extern float __nextafterf(float __x, float __y) throw(); 
# 261
extern float nexttowardf(float __x, long double __y) throw(); extern float __nexttowardf(float __x, long double __y) throw(); 
# 266
extern float nextdownf(float __x) throw(); extern float __nextdownf(float __x) throw(); 
# 268
extern float nextupf(float __x) throw(); extern float __nextupf(float __x) throw(); 
# 272
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 276
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 280
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 285
extern long llogbf(float __x) throw(); extern long __llogbf(float __x) throw(); 
# 290
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 294
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 298
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 302
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 307
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 314
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 316
extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 320
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 322
extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 326
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 329
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 332
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 335
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 340
extern float roundevenf(float __x) throw() __attribute((const)); extern float __roundevenf(float __x) throw() __attribute((const)); 
# 345
extern __intmax_t fromfpf(float __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf(float __x, int __round, unsigned __width) throw(); 
# 350
extern __uintmax_t ufromfpf(float __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf(float __x, int __round, unsigned __width) throw(); 
# 356
extern __intmax_t fromfpxf(float __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf(float __x, int __round, unsigned __width) throw(); 
# 362
extern __uintmax_t ufromfpxf(float __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf(float __x, int __round, unsigned __width) throw(); 
# 365
extern float fmaxmagf(float __x, float __y) throw() __attribute((const)); extern float __fmaxmagf(float __x, float __y) throw() __attribute((const)); 
# 368
extern float fminmagf(float __x, float __y) throw() __attribute((const)); extern float __fminmagf(float __x, float __y) throw() __attribute((const)); 
# 371
extern int canonicalizef(float * __cx, const float * __x) throw(); 
# 377
extern int totalorderf(const float * __x, const float * __y) throw()
# 378
 __attribute((__pure__)); 
# 382
extern int totalordermagf(const float * __x, const float * __y) throw()
# 383
 __attribute((__pure__)); 
# 386
extern float getpayloadf(const float * __x) throw(); extern float __getpayloadf(const float * __x) throw(); 
# 389
extern int setpayloadf(float * __x, float __payload) throw(); 
# 392
extern int setpayloadsigf(float * __x, float __payload) throw(); 
# 400
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyl(long double __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbitl(long double __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 33
extern int __finitel(long double __value) throw() __attribute((const)); 
# 36
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 39
extern int __iseqsigl(long double __x, long double __y) throw(); 
# 42
extern int __issignalingl(long double __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 55
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 57
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 59
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 62
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 64
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 66
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 71
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 73
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 75
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 80
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 85
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 87
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 89
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 95
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 98
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 101
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 104
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 107
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 110
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 119
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 122
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 125
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 130
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 133
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 140
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 143
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 147
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 152
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 159
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 162
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 165
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 168
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 177
extern int isinfl(long double __value) throw() __attribute((const)); 
# 182
extern int finitel(long double __value) throw() __attribute((const)); 
# 185
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 189
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 196
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 201
extern long double nanl(const char * __tagb) throw(); extern long double __nanl(const char * __tagb) throw(); 
# 211
extern int isnanl(long double __value) throw() __attribute((const)); 
# 217
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 218
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 219
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 220
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 221
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 222
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 228
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 229
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 230
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 235
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 241
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 249
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 256
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 259
extern long double nextafterl(long double __x, long double __y) throw(); extern long double __nextafterl(long double __x, long double __y) throw(); 
# 261
extern long double nexttowardl(long double __x, long double __y) throw(); extern long double __nexttowardl(long double __x, long double __y) throw(); 
# 266
extern long double nextdownl(long double __x) throw(); extern long double __nextdownl(long double __x) throw(); 
# 268
extern long double nextupl(long double __x) throw(); extern long double __nextupl(long double __x) throw(); 
# 272
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 276
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 280
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 285
extern long llogbl(long double __x) throw(); extern long __llogbl(long double __x) throw(); 
# 290
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 294
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 298
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 302
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 307
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 314
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 316
extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 320
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 322
extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 326
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 329
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 332
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 335
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 340
extern long double roundevenl(long double __x) throw() __attribute((const)); extern long double __roundevenl(long double __x) throw() __attribute((const)); 
# 345
extern __intmax_t fromfpl(long double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpl(long double __x, int __round, unsigned __width) throw(); 
# 350
extern __uintmax_t ufromfpl(long double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpl(long double __x, int __round, unsigned __width) throw(); 
# 356
extern __intmax_t fromfpxl(long double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxl(long double __x, int __round, unsigned __width) throw(); 
# 362
extern __uintmax_t ufromfpxl(long double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxl(long double __x, int __round, unsigned __width) throw(); 
# 365
extern long double fmaxmagl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxmagl(long double __x, long double __y) throw() __attribute((const)); 
# 368
extern long double fminmagl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminmagl(long double __x, long double __y) throw() __attribute((const)); 
# 371
extern int canonicalizel(long double * __cx, const long double * __x) throw(); 
# 377
extern int totalorderl(const long double * __x, const long double * __y) throw()
# 378
 __attribute((__pure__)); 
# 382
extern int totalordermagl(const long double * __x, const long double * __y) throw()
# 383
 __attribute((__pure__)); 
# 386
extern long double getpayloadl(const long double * __x) throw(); extern long double __getpayloadl(const long double * __x) throw(); 
# 389
extern int setpayloadl(long double * __x, long double __payload) throw(); 
# 392
extern int setpayloadsigl(long double * __x, long double __payload) throw(); 
# 400
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern float fadd(double __x, double __y) throw(); 
# 27
extern float fdiv(double __x, double __y) throw(); 
# 30
extern float fmul(double __x, double __y) throw(); 
# 33
extern float fsub(double __x, double __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern float faddl(long double __x, long double __y) throw(); 
# 27
extern float fdivl(long double __x, long double __y) throw(); 
# 30
extern float fmull(long double __x, long double __y) throw(); 
# 33
extern float fsubl(long double __x, long double __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern double daddl(long double __x, long double __y) throw(); 
# 27
extern double ddivl(long double __x, long double __y) throw(); 
# 30
extern double dmull(long double __x, long double __y) throw(); 
# 33
extern double dsubl(long double __x, long double __y) throw(); 
# 773 "/usr/include/math.h" 3
extern int signgam; 
# 854 "/usr/include/math.h" 3
enum { 
# 855
FP_NAN, 
# 858
FP_INFINITE, 
# 861
FP_ZERO, 
# 864
FP_SUBNORMAL, 
# 867
FP_NORMAL
# 870
}; 
# 23 "/usr/include/bits/iscanonical.h" 3
extern int __iscanonicall(long double __x) throw()
# 24
 __attribute((const)); 
# 46
extern "C++" {
# 47
inline int iscanonical(float __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 48
inline int iscanonical(double __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 49
inline int iscanonical(long double __val) { return __iscanonicall(__val); } 
# 53
}
# 985 "/usr/include/math.h" 3
extern "C++" {
# 986
inline int issignaling(float __val) { return __issignalingf(__val); } 
# 987
inline int issignaling(double __val) { return __issignaling(__val); } 
# 989
inline int issignaling(long double __val) 
# 990
{ 
# 994
return __issignalingl(__val); 
# 996
} 
# 1002
}
# 1016 "/usr/include/math.h" 3
extern "C++" {
# 1047 "/usr/include/math.h" 3
template< class __T> inline bool 
# 1048
iszero(__T __val) 
# 1049
{ 
# 1050
return __val == 0; 
# 1051
} 
# 1053
}
# 1278 "/usr/include/math.h" 3
extern "C++" {
# 1279
template< class > struct __iseqsig_type; 
# 1281
template<> struct __iseqsig_type< float>  { 
# 1283
static int __call(float __x, float __y) throw() 
# 1284
{ 
# 1285
return __iseqsigf(__x, __y); 
# 1286
} 
# 1287
}; 
# 1289
template<> struct __iseqsig_type< double>  { 
# 1291
static int __call(double __x, double __y) throw() 
# 1292
{ 
# 1293
return __iseqsig(__x, __y); 
# 1294
} 
# 1295
}; 
# 1297
template<> struct __iseqsig_type< long double>  { 
# 1299
static int __call(long double __x, long double __y) throw() 
# 1300
{ 
# 1302
return __iseqsigl(__x, __y); 
# 1306
} 
# 1307
}; 
# 1321 "/usr/include/math.h" 3
template< class _T1, class _T2> inline int 
# 1323
iseqsig(_T1 __x, _T2 __y) throw() 
# 1324
{ 
# 1326
typedef __decltype(((__x + __y) + (0.0F))) _T3; 
# 1330
return __iseqsig_type< __decltype(((__x + __y) + (0.0F)))> ::__call(__x, __y); 
# 1331
} 
# 1333
}
# 1338
}
# 79 "/usr/include/c++/13/cmath" 3
extern "C++" {
# 81
namespace std __attribute((__visibility__("default"))) { 
# 85
using ::acos;
# 89
constexpr float acos(float __x) 
# 90
{ return __builtin_acosf(__x); } 
# 93
constexpr long double acos(long double __x) 
# 94
{ return __builtin_acosl(__x); } 
# 97
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 101
acos(_Tp __x) 
# 102
{ return __builtin_acos(__x); } 
# 104
using ::asin;
# 108
constexpr float asin(float __x) 
# 109
{ return __builtin_asinf(__x); } 
# 112
constexpr long double asin(long double __x) 
# 113
{ return __builtin_asinl(__x); } 
# 116
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 120
asin(_Tp __x) 
# 121
{ return __builtin_asin(__x); } 
# 123
using ::atan;
# 127
constexpr float atan(float __x) 
# 128
{ return __builtin_atanf(__x); } 
# 131
constexpr long double atan(long double __x) 
# 132
{ return __builtin_atanl(__x); } 
# 135
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 139
atan(_Tp __x) 
# 140
{ return __builtin_atan(__x); } 
# 142
using ::atan2;
# 146
constexpr float atan2(float __y, float __x) 
# 147
{ return __builtin_atan2f(__y, __x); } 
# 150
constexpr long double atan2(long double __y, long double __x) 
# 151
{ return __builtin_atan2l(__y, __x); } 
# 154
using ::ceil;
# 158
constexpr float ceil(float __x) 
# 159
{ return __builtin_ceilf(__x); } 
# 162
constexpr long double ceil(long double __x) 
# 163
{ return __builtin_ceill(__x); } 
# 166
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 170
ceil(_Tp __x) 
# 171
{ return __builtin_ceil(__x); } 
# 173
using ::cos;
# 177
constexpr float cos(float __x) 
# 178
{ return __builtin_cosf(__x); } 
# 181
constexpr long double cos(long double __x) 
# 182
{ return __builtin_cosl(__x); } 
# 185
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 189
cos(_Tp __x) 
# 190
{ return __builtin_cos(__x); } 
# 192
using ::cosh;
# 196
constexpr float cosh(float __x) 
# 197
{ return __builtin_coshf(__x); } 
# 200
constexpr long double cosh(long double __x) 
# 201
{ return __builtin_coshl(__x); } 
# 204
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 208
cosh(_Tp __x) 
# 209
{ return __builtin_cosh(__x); } 
# 211
using ::exp;
# 215
constexpr float exp(float __x) 
# 216
{ return __builtin_expf(__x); } 
# 219
constexpr long double exp(long double __x) 
# 220
{ return __builtin_expl(__x); } 
# 223
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 227
exp(_Tp __x) 
# 228
{ return __builtin_exp(__x); } 
# 230
using ::fabs;
# 234
constexpr float fabs(float __x) 
# 235
{ return __builtin_fabsf(__x); } 
# 238
constexpr long double fabs(long double __x) 
# 239
{ return __builtin_fabsl(__x); } 
# 242
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 246
fabs(_Tp __x) 
# 247
{ return __builtin_fabs(__x); } 
# 249
using ::floor;
# 253
constexpr float floor(float __x) 
# 254
{ return __builtin_floorf(__x); } 
# 257
constexpr long double floor(long double __x) 
# 258
{ return __builtin_floorl(__x); } 
# 261
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 265
floor(_Tp __x) 
# 266
{ return __builtin_floor(__x); } 
# 268
using ::fmod;
# 272
constexpr float fmod(float __x, float __y) 
# 273
{ return __builtin_fmodf(__x, __y); } 
# 276
constexpr long double fmod(long double __x, long double __y) 
# 277
{ return __builtin_fmodl(__x, __y); } 
# 280
using ::frexp;
# 284
inline float frexp(float __x, int *__exp) 
# 285
{ return __builtin_frexpf(__x, __exp); } 
# 288
inline long double frexp(long double __x, int *__exp) 
# 289
{ return __builtin_frexpl(__x, __exp); } 
# 292
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 296
frexp(_Tp __x, int *__exp) 
# 297
{ return __builtin_frexp(__x, __exp); } 
# 299
using ::ldexp;
# 303
constexpr float ldexp(float __x, int __exp) 
# 304
{ return __builtin_ldexpf(__x, __exp); } 
# 307
constexpr long double ldexp(long double __x, int __exp) 
# 308
{ return __builtin_ldexpl(__x, __exp); } 
# 311
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 315
ldexp(_Tp __x, int __exp) 
# 316
{ return __builtin_ldexp(__x, __exp); } 
# 318
using ::log;
# 322
constexpr float log(float __x) 
# 323
{ return __builtin_logf(__x); } 
# 326
constexpr long double log(long double __x) 
# 327
{ return __builtin_logl(__x); } 
# 330
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 334
log(_Tp __x) 
# 335
{ return __builtin_log(__x); } 
# 337
using ::log10;
# 341
constexpr float log10(float __x) 
# 342
{ return __builtin_log10f(__x); } 
# 345
constexpr long double log10(long double __x) 
# 346
{ return __builtin_log10l(__x); } 
# 349
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 353
log10(_Tp __x) 
# 354
{ return __builtin_log10(__x); } 
# 356
using ::modf;
# 360
inline float modf(float __x, float *__iptr) 
# 361
{ return __builtin_modff(__x, __iptr); } 
# 364
inline long double modf(long double __x, long double *__iptr) 
# 365
{ return __builtin_modfl(__x, __iptr); } 
# 368
using ::pow;
# 372
constexpr float pow(float __x, float __y) 
# 373
{ return __builtin_powf(__x, __y); } 
# 376
constexpr long double pow(long double __x, long double __y) 
# 377
{ return __builtin_powl(__x, __y); } 
# 396 "/usr/include/c++/13/cmath" 3
using ::sin;
# 400
constexpr float sin(float __x) 
# 401
{ return __builtin_sinf(__x); } 
# 404
constexpr long double sin(long double __x) 
# 405
{ return __builtin_sinl(__x); } 
# 408
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 412
sin(_Tp __x) 
# 413
{ return __builtin_sin(__x); } 
# 415
using ::sinh;
# 419
constexpr float sinh(float __x) 
# 420
{ return __builtin_sinhf(__x); } 
# 423
constexpr long double sinh(long double __x) 
# 424
{ return __builtin_sinhl(__x); } 
# 427
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 431
sinh(_Tp __x) 
# 432
{ return __builtin_sinh(__x); } 
# 434
using ::sqrt;
# 438
constexpr float sqrt(float __x) 
# 439
{ return __builtin_sqrtf(__x); } 
# 442
constexpr long double sqrt(long double __x) 
# 443
{ return __builtin_sqrtl(__x); } 
# 446
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 450
sqrt(_Tp __x) 
# 451
{ return __builtin_sqrt(__x); } 
# 453
using ::tan;
# 457
constexpr float tan(float __x) 
# 458
{ return __builtin_tanf(__x); } 
# 461
constexpr long double tan(long double __x) 
# 462
{ return __builtin_tanl(__x); } 
# 465
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 469
tan(_Tp __x) 
# 470
{ return __builtin_tan(__x); } 
# 472
using ::tanh;
# 476
constexpr float tanh(float __x) 
# 477
{ return __builtin_tanhf(__x); } 
# 480
constexpr long double tanh(long double __x) 
# 481
{ return __builtin_tanhl(__x); } 
# 484
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 488
tanh(_Tp __x) 
# 489
{ return __builtin_tanh(__x); } 
# 1049 "/usr/include/c++/13/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1052
atan2(_Tp __y, _Up __x) 
# 1053
{ 
# 1054
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1055
return atan2((__type)__y, (__type)__x); 
# 1056
} 
# 1058
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1061
fmod(_Tp __x, _Up __y) 
# 1062
{ 
# 1063
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1064
return fmod((__type)__x, (__type)__y); 
# 1065
} 
# 1067
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1070
pow(_Tp __x, _Up __y) 
# 1071
{ 
# 1072
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1073
return pow((__type)__x, (__type)__y); 
# 1074
} 
# 1097 "/usr/include/c++/13/cmath" 3
constexpr int fpclassify(float __x) 
# 1098
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 1099
} 
# 1102
constexpr int fpclassify(double __x) 
# 1103
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 1104
} 
# 1107
constexpr int fpclassify(long double __x) 
# 1108
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 1109
} 
# 1113
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1116
fpclassify(_Tp __x) 
# 1117
{ return (__x != 0) ? 4 : 2; } 
# 1122
constexpr bool isfinite(float __x) 
# 1123
{ return __builtin_isfinite(__x); } 
# 1126
constexpr bool isfinite(double __x) 
# 1127
{ return __builtin_isfinite(__x); } 
# 1130
constexpr bool isfinite(long double __x) 
# 1131
{ return __builtin_isfinite(__x); } 
# 1135
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 1138
isfinite(_Tp) 
# 1139
{ return true; } 
# 1144
constexpr bool isinf(float __x) 
# 1145
{ return __builtin_isinf(__x); } 
# 1152
constexpr bool isinf(double __x) 
# 1153
{ return __builtin_isinf(__x); } 
# 1157
constexpr bool isinf(long double __x) 
# 1158
{ return __builtin_isinf(__x); } 
# 1162
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 1165
isinf(_Tp) 
# 1166
{ return false; } 
# 1171
constexpr bool isnan(float __x) 
# 1172
{ return __builtin_isnan(__x); } 
# 1179
constexpr bool isnan(double __x) 
# 1180
{ return __builtin_isnan(__x); } 
# 1184
constexpr bool isnan(long double __x) 
# 1185
{ return __builtin_isnan(__x); } 
# 1189
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 1192
isnan(_Tp) 
# 1193
{ return false; } 
# 1198
constexpr bool isnormal(float __x) 
# 1199
{ return __builtin_isnormal(__x); } 
# 1202
constexpr bool isnormal(double __x) 
# 1203
{ return __builtin_isnormal(__x); } 
# 1206
constexpr bool isnormal(long double __x) 
# 1207
{ return __builtin_isnormal(__x); } 
# 1211
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 1214
isnormal(_Tp __x) 
# 1215
{ return (__x != 0) ? true : false; } 
# 1221
constexpr bool signbit(float __x) 
# 1222
{ return __builtin_signbit(__x); } 
# 1225
constexpr bool signbit(double __x) 
# 1226
{ return __builtin_signbit(__x); } 
# 1229
constexpr bool signbit(long double __x) 
# 1230
{ return __builtin_signbit(__x); } 
# 1234
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 1237
signbit(_Tp __x) 
# 1238
{ return (__x < 0) ? true : false; } 
# 1243
constexpr bool isgreater(float __x, float __y) 
# 1244
{ return __builtin_isgreater(__x, __y); } 
# 1247
constexpr bool isgreater(double __x, double __y) 
# 1248
{ return __builtin_isgreater(__x, __y); } 
# 1251
constexpr bool isgreater(long double __x, long double __y) 
# 1252
{ return __builtin_isgreater(__x, __y); } 
# 1256
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1260
isgreater(_Tp __x, _Up __y) 
# 1261
{ 
# 1262
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1263
return __builtin_isgreater((__type)__x, (__type)__y); 
# 1264
} 
# 1269
constexpr bool isgreaterequal(float __x, float __y) 
# 1270
{ return __builtin_isgreaterequal(__x, __y); } 
# 1273
constexpr bool isgreaterequal(double __x, double __y) 
# 1274
{ return __builtin_isgreaterequal(__x, __y); } 
# 1277
constexpr bool isgreaterequal(long double __x, long double __y) 
# 1278
{ return __builtin_isgreaterequal(__x, __y); } 
# 1282
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1286
isgreaterequal(_Tp __x, _Up __y) 
# 1287
{ 
# 1288
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1289
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 1290
} 
# 1295
constexpr bool isless(float __x, float __y) 
# 1296
{ return __builtin_isless(__x, __y); } 
# 1299
constexpr bool isless(double __x, double __y) 
# 1300
{ return __builtin_isless(__x, __y); } 
# 1303
constexpr bool isless(long double __x, long double __y) 
# 1304
{ return __builtin_isless(__x, __y); } 
# 1308
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1312
isless(_Tp __x, _Up __y) 
# 1313
{ 
# 1314
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1315
return __builtin_isless((__type)__x, (__type)__y); 
# 1316
} 
# 1321
constexpr bool islessequal(float __x, float __y) 
# 1322
{ return __builtin_islessequal(__x, __y); } 
# 1325
constexpr bool islessequal(double __x, double __y) 
# 1326
{ return __builtin_islessequal(__x, __y); } 
# 1329
constexpr bool islessequal(long double __x, long double __y) 
# 1330
{ return __builtin_islessequal(__x, __y); } 
# 1334
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1338
islessequal(_Tp __x, _Up __y) 
# 1339
{ 
# 1340
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1341
return __builtin_islessequal((__type)__x, (__type)__y); 
# 1342
} 
# 1347
constexpr bool islessgreater(float __x, float __y) 
# 1348
{ return __builtin_islessgreater(__x, __y); } 
# 1351
constexpr bool islessgreater(double __x, double __y) 
# 1352
{ return __builtin_islessgreater(__x, __y); } 
# 1355
constexpr bool islessgreater(long double __x, long double __y) 
# 1356
{ return __builtin_islessgreater(__x, __y); } 
# 1360
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1364
islessgreater(_Tp __x, _Up __y) 
# 1365
{ 
# 1366
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1367
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 1368
} 
# 1373
constexpr bool isunordered(float __x, float __y) 
# 1374
{ return __builtin_isunordered(__x, __y); } 
# 1377
constexpr bool isunordered(double __x, double __y) 
# 1378
{ return __builtin_isunordered(__x, __y); } 
# 1381
constexpr bool isunordered(long double __x, long double __y) 
# 1382
{ return __builtin_isunordered(__x, __y); } 
# 1386
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 1390
isunordered(_Tp __x, _Up __y) 
# 1391
{ 
# 1392
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1393
return __builtin_isunordered((__type)__x, (__type)__y); 
# 1394
} 
# 1881 "/usr/include/c++/13/cmath" 3
using ::double_t;
# 1882
using ::float_t;
# 1885
using ::acosh;
# 1886
using ::acoshf;
# 1887
using ::acoshl;
# 1889
using ::asinh;
# 1890
using ::asinhf;
# 1891
using ::asinhl;
# 1893
using ::atanh;
# 1894
using ::atanhf;
# 1895
using ::atanhl;
# 1897
using ::cbrt;
# 1898
using ::cbrtf;
# 1899
using ::cbrtl;
# 1901
using ::copysign;
# 1902
using ::copysignf;
# 1903
using ::copysignl;
# 1905
using ::erf;
# 1906
using ::erff;
# 1907
using ::erfl;
# 1909
using ::erfc;
# 1910
using ::erfcf;
# 1911
using ::erfcl;
# 1913
using ::exp2;
# 1914
using ::exp2f;
# 1915
using ::exp2l;
# 1917
using ::expm1;
# 1918
using ::expm1f;
# 1919
using ::expm1l;
# 1921
using ::fdim;
# 1922
using ::fdimf;
# 1923
using ::fdiml;
# 1925
using ::fma;
# 1926
using ::fmaf;
# 1927
using ::fmal;
# 1929
using ::fmax;
# 1930
using ::fmaxf;
# 1931
using ::fmaxl;
# 1933
using ::fmin;
# 1934
using ::fminf;
# 1935
using ::fminl;
# 1937
using ::hypot;
# 1938
using ::hypotf;
# 1939
using ::hypotl;
# 1941
using ::ilogb;
# 1942
using ::ilogbf;
# 1943
using ::ilogbl;
# 1945
using ::lgamma;
# 1946
using ::lgammaf;
# 1947
using ::lgammal;
# 1950
using ::llrint;
# 1951
using ::llrintf;
# 1952
using ::llrintl;
# 1954
using ::llround;
# 1955
using ::llroundf;
# 1956
using ::llroundl;
# 1959
using ::log1p;
# 1960
using ::log1pf;
# 1961
using ::log1pl;
# 1963
using ::log2;
# 1964
using ::log2f;
# 1965
using ::log2l;
# 1967
using ::logb;
# 1968
using ::logbf;
# 1969
using ::logbl;
# 1971
using ::lrint;
# 1972
using ::lrintf;
# 1973
using ::lrintl;
# 1975
using ::lround;
# 1976
using ::lroundf;
# 1977
using ::lroundl;
# 1979
using ::nan;
# 1980
using ::nanf;
# 1981
using ::nanl;
# 1983
using ::nearbyint;
# 1984
using ::nearbyintf;
# 1985
using ::nearbyintl;
# 1987
using ::nextafter;
# 1988
using ::nextafterf;
# 1989
using ::nextafterl;
# 1991
using ::nexttoward;
# 1992
using ::nexttowardf;
# 1993
using ::nexttowardl;
# 1995
using ::remainder;
# 1996
using ::remainderf;
# 1997
using ::remainderl;
# 1999
using ::remquo;
# 2000
using ::remquof;
# 2001
using ::remquol;
# 2003
using ::rint;
# 2004
using ::rintf;
# 2005
using ::rintl;
# 2007
using ::round;
# 2008
using ::roundf;
# 2009
using ::roundl;
# 2011
using ::scalbln;
# 2012
using ::scalblnf;
# 2013
using ::scalblnl;
# 2015
using ::scalbn;
# 2016
using ::scalbnf;
# 2017
using ::scalbnl;
# 2019
using ::tgamma;
# 2020
using ::tgammaf;
# 2021
using ::tgammal;
# 2023
using ::trunc;
# 2024
using ::truncf;
# 2025
using ::truncl;
# 2030
constexpr float acosh(float __x) 
# 2031
{ return __builtin_acoshf(__x); } 
# 2034
constexpr long double acosh(long double __x) 
# 2035
{ return __builtin_acoshl(__x); } 
# 2039
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2042
acosh(_Tp __x) 
# 2043
{ return __builtin_acosh(__x); } 
# 2048
constexpr float asinh(float __x) 
# 2049
{ return __builtin_asinhf(__x); } 
# 2052
constexpr long double asinh(long double __x) 
# 2053
{ return __builtin_asinhl(__x); } 
# 2057
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2060
asinh(_Tp __x) 
# 2061
{ return __builtin_asinh(__x); } 
# 2066
constexpr float atanh(float __x) 
# 2067
{ return __builtin_atanhf(__x); } 
# 2070
constexpr long double atanh(long double __x) 
# 2071
{ return __builtin_atanhl(__x); } 
# 2075
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2078
atanh(_Tp __x) 
# 2079
{ return __builtin_atanh(__x); } 
# 2084
constexpr float cbrt(float __x) 
# 2085
{ return __builtin_cbrtf(__x); } 
# 2088
constexpr long double cbrt(long double __x) 
# 2089
{ return __builtin_cbrtl(__x); } 
# 2093
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2096
cbrt(_Tp __x) 
# 2097
{ return __builtin_cbrt(__x); } 
# 2102
constexpr float copysign(float __x, float __y) 
# 2103
{ return __builtin_copysignf(__x, __y); } 
# 2106
constexpr long double copysign(long double __x, long double __y) 
# 2107
{ return __builtin_copysignl(__x, __y); } 
# 2112
constexpr float erf(float __x) 
# 2113
{ return __builtin_erff(__x); } 
# 2116
constexpr long double erf(long double __x) 
# 2117
{ return __builtin_erfl(__x); } 
# 2121
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2124
erf(_Tp __x) 
# 2125
{ return __builtin_erf(__x); } 
# 2130
constexpr float erfc(float __x) 
# 2131
{ return __builtin_erfcf(__x); } 
# 2134
constexpr long double erfc(long double __x) 
# 2135
{ return __builtin_erfcl(__x); } 
# 2139
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2142
erfc(_Tp __x) 
# 2143
{ return __builtin_erfc(__x); } 
# 2148
constexpr float exp2(float __x) 
# 2149
{ return __builtin_exp2f(__x); } 
# 2152
constexpr long double exp2(long double __x) 
# 2153
{ return __builtin_exp2l(__x); } 
# 2157
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2160
exp2(_Tp __x) 
# 2161
{ return __builtin_exp2(__x); } 
# 2166
constexpr float expm1(float __x) 
# 2167
{ return __builtin_expm1f(__x); } 
# 2170
constexpr long double expm1(long double __x) 
# 2171
{ return __builtin_expm1l(__x); } 
# 2175
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2178
expm1(_Tp __x) 
# 2179
{ return __builtin_expm1(__x); } 
# 2184
constexpr float fdim(float __x, float __y) 
# 2185
{ return __builtin_fdimf(__x, __y); } 
# 2188
constexpr long double fdim(long double __x, long double __y) 
# 2189
{ return __builtin_fdiml(__x, __y); } 
# 2194
constexpr float fma(float __x, float __y, float __z) 
# 2195
{ return __builtin_fmaf(__x, __y, __z); } 
# 2198
constexpr long double fma(long double __x, long double __y, long double __z) 
# 2199
{ return __builtin_fmal(__x, __y, __z); } 
# 2204
constexpr float fmax(float __x, float __y) 
# 2205
{ return __builtin_fmaxf(__x, __y); } 
# 2208
constexpr long double fmax(long double __x, long double __y) 
# 2209
{ return __builtin_fmaxl(__x, __y); } 
# 2214
constexpr float fmin(float __x, float __y) 
# 2215
{ return __builtin_fminf(__x, __y); } 
# 2218
constexpr long double fmin(long double __x, long double __y) 
# 2219
{ return __builtin_fminl(__x, __y); } 
# 2224
constexpr float hypot(float __x, float __y) 
# 2225
{ return __builtin_hypotf(__x, __y); } 
# 2228
constexpr long double hypot(long double __x, long double __y) 
# 2229
{ return __builtin_hypotl(__x, __y); } 
# 2234
constexpr int ilogb(float __x) 
# 2235
{ return __builtin_ilogbf(__x); } 
# 2238
constexpr int ilogb(long double __x) 
# 2239
{ return __builtin_ilogbl(__x); } 
# 2243
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 2247
ilogb(_Tp __x) 
# 2248
{ return __builtin_ilogb(__x); } 
# 2253
constexpr float lgamma(float __x) 
# 2254
{ return __builtin_lgammaf(__x); } 
# 2257
constexpr long double lgamma(long double __x) 
# 2258
{ return __builtin_lgammal(__x); } 
# 2262
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2265
lgamma(_Tp __x) 
# 2266
{ return __builtin_lgamma(__x); } 
# 2271
constexpr long long llrint(float __x) 
# 2272
{ return __builtin_llrintf(__x); } 
# 2275
constexpr long long llrint(long double __x) 
# 2276
{ return __builtin_llrintl(__x); } 
# 2280
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 2283
llrint(_Tp __x) 
# 2284
{ return __builtin_llrint(__x); } 
# 2289
constexpr long long llround(float __x) 
# 2290
{ return __builtin_llroundf(__x); } 
# 2293
constexpr long long llround(long double __x) 
# 2294
{ return __builtin_llroundl(__x); } 
# 2298
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 2301
llround(_Tp __x) 
# 2302
{ return __builtin_llround(__x); } 
# 2307
constexpr float log1p(float __x) 
# 2308
{ return __builtin_log1pf(__x); } 
# 2311
constexpr long double log1p(long double __x) 
# 2312
{ return __builtin_log1pl(__x); } 
# 2316
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2319
log1p(_Tp __x) 
# 2320
{ return __builtin_log1p(__x); } 
# 2326
constexpr float log2(float __x) 
# 2327
{ return __builtin_log2f(__x); } 
# 2330
constexpr long double log2(long double __x) 
# 2331
{ return __builtin_log2l(__x); } 
# 2335
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2338
log2(_Tp __x) 
# 2339
{ return __builtin_log2(__x); } 
# 2344
constexpr float logb(float __x) 
# 2345
{ return __builtin_logbf(__x); } 
# 2348
constexpr long double logb(long double __x) 
# 2349
{ return __builtin_logbl(__x); } 
# 2353
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2356
logb(_Tp __x) 
# 2357
{ return __builtin_logb(__x); } 
# 2362
constexpr long lrint(float __x) 
# 2363
{ return __builtin_lrintf(__x); } 
# 2366
constexpr long lrint(long double __x) 
# 2367
{ return __builtin_lrintl(__x); } 
# 2371
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 2374
lrint(_Tp __x) 
# 2375
{ return __builtin_lrint(__x); } 
# 2380
constexpr long lround(float __x) 
# 2381
{ return __builtin_lroundf(__x); } 
# 2384
constexpr long lround(long double __x) 
# 2385
{ return __builtin_lroundl(__x); } 
# 2389
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 2392
lround(_Tp __x) 
# 2393
{ return __builtin_lround(__x); } 
# 2398
constexpr float nearbyint(float __x) 
# 2399
{ return __builtin_nearbyintf(__x); } 
# 2402
constexpr long double nearbyint(long double __x) 
# 2403
{ return __builtin_nearbyintl(__x); } 
# 2407
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2410
nearbyint(_Tp __x) 
# 2411
{ return __builtin_nearbyint(__x); } 
# 2416
constexpr float nextafter(float __x, float __y) 
# 2417
{ return __builtin_nextafterf(__x, __y); } 
# 2420
constexpr long double nextafter(long double __x, long double __y) 
# 2421
{ return __builtin_nextafterl(__x, __y); } 
# 2426
constexpr float nexttoward(float __x, long double __y) 
# 2427
{ return __builtin_nexttowardf(__x, __y); } 
# 2430
constexpr long double nexttoward(long double __x, long double __y) 
# 2431
{ return __builtin_nexttowardl(__x, __y); } 
# 2435
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2438
nexttoward(_Tp __x, long double __y) 
# 2439
{ return __builtin_nexttoward(__x, __y); } 
# 2444
constexpr float remainder(float __x, float __y) 
# 2445
{ return __builtin_remainderf(__x, __y); } 
# 2448
constexpr long double remainder(long double __x, long double __y) 
# 2449
{ return __builtin_remainderl(__x, __y); } 
# 2454
inline float remquo(float __x, float __y, int *__pquo) 
# 2455
{ return __builtin_remquof(__x, __y, __pquo); } 
# 2458
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 2459
{ return __builtin_remquol(__x, __y, __pquo); } 
# 2464
constexpr float rint(float __x) 
# 2465
{ return __builtin_rintf(__x); } 
# 2468
constexpr long double rint(long double __x) 
# 2469
{ return __builtin_rintl(__x); } 
# 2473
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2476
rint(_Tp __x) 
# 2477
{ return __builtin_rint(__x); } 
# 2482
constexpr float round(float __x) 
# 2483
{ return __builtin_roundf(__x); } 
# 2486
constexpr long double round(long double __x) 
# 2487
{ return __builtin_roundl(__x); } 
# 2491
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2494
round(_Tp __x) 
# 2495
{ return __builtin_round(__x); } 
# 2500
constexpr float scalbln(float __x, long __ex) 
# 2501
{ return __builtin_scalblnf(__x, __ex); } 
# 2504
constexpr long double scalbln(long double __x, long __ex) 
# 2505
{ return __builtin_scalblnl(__x, __ex); } 
# 2509
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2512
scalbln(_Tp __x, long __ex) 
# 2513
{ return __builtin_scalbln(__x, __ex); } 
# 2518
constexpr float scalbn(float __x, int __ex) 
# 2519
{ return __builtin_scalbnf(__x, __ex); } 
# 2522
constexpr long double scalbn(long double __x, int __ex) 
# 2523
{ return __builtin_scalbnl(__x, __ex); } 
# 2527
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2530
scalbn(_Tp __x, int __ex) 
# 2531
{ return __builtin_scalbn(__x, __ex); } 
# 2536
constexpr float tgamma(float __x) 
# 2537
{ return __builtin_tgammaf(__x); } 
# 2540
constexpr long double tgamma(long double __x) 
# 2541
{ return __builtin_tgammal(__x); } 
# 2545
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2548
tgamma(_Tp __x) 
# 2549
{ return __builtin_tgamma(__x); } 
# 2554
constexpr float trunc(float __x) 
# 2555
{ return __builtin_truncf(__x); } 
# 2558
constexpr long double trunc(long double __x) 
# 2559
{ return __builtin_truncl(__x); } 
# 2563
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 2566
trunc(_Tp __x) 
# 2567
{ return __builtin_trunc(__x); } 
# 3469 "/usr/include/c++/13/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3471
copysign(_Tp __x, _Up __y) 
# 3472
{ 
# 3473
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3474
return copysign((__type)__x, (__type)__y); 
# 3475
} 
# 3477
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3479
fdim(_Tp __x, _Up __y) 
# 3480
{ 
# 3481
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3482
return fdim((__type)__x, (__type)__y); 
# 3483
} 
# 3485
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3487
fmax(_Tp __x, _Up __y) 
# 3488
{ 
# 3489
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3490
return fmax((__type)__x, (__type)__y); 
# 3491
} 
# 3493
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3495
fmin(_Tp __x, _Up __y) 
# 3496
{ 
# 3497
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3498
return fmin((__type)__x, (__type)__y); 
# 3499
} 
# 3501
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3503
hypot(_Tp __x, _Up __y) 
# 3504
{ 
# 3505
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3506
return hypot((__type)__x, (__type)__y); 
# 3507
} 
# 3509
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3511
nextafter(_Tp __x, _Up __y) 
# 3512
{ 
# 3513
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3514
return nextafter((__type)__x, (__type)__y); 
# 3515
} 
# 3517
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3519
remainder(_Tp __x, _Up __y) 
# 3520
{ 
# 3521
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3522
return remainder((__type)__x, (__type)__y); 
# 3523
} 
# 3525
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 3527
remquo(_Tp __x, _Up __y, int *__pquo) 
# 3528
{ 
# 3529
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 3530
return remquo((__type)__x, (__type)__y, __pquo); 
# 3531
} 
# 3533
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 3535
fma(_Tp __x, _Up __y, _Vp __z) 
# 3536
{ 
# 3537
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 3538
return fma((__type)__x, (__type)__y, (__type)__z); 
# 3539
} 
# 3550
template< class _Tp> inline _Tp 
# 3552
__hypot3(_Tp __x, _Tp __y, _Tp __z) 
# 3553
{ 
# 3554
__x = std::abs(__x); 
# 3555
__y = std::abs(__y); 
# 3556
__z = std::abs(__z); 
# 3557
if (_Tp __a = (__x < __y) ? (__y < __z) ? __z : __y : ((__x < __z) ? __z : __x)) { 
# 3558
return __a * std::sqrt((((__x / __a) * (__x / __a)) + ((__y / __a) * (__y / __a))) + ((__z / __a) * (__z / __a))); } else { 
# 3562
return {}; }  
# 3563
} 
# 3566
inline float hypot(float __x, float __y, float __z) 
# 3567
{ return std::__hypot3< float> (__x, __y, __z); } 
# 3570
inline double hypot(double __x, double __y, double __z) 
# 3571
{ return std::__hypot3< double> (__x, __y, __z); } 
# 3574
inline long double hypot(long double __x, long double __y, long double __z) 
# 3575
{ return std::__hypot3< long double> (__x, __y, __z); } 
# 3577
template< class _Tp, class _Up, class _Vp> __gnu_cxx::__promoted_t< _Tp, _Up, _Vp>  
# 3579
hypot(_Tp __x, _Up __y, _Vp __z) 
# 3580
{ 
# 3581
using __type = __gnu_cxx::__promoted_t< _Tp, _Up, _Vp> ; 
# 3582
return std::__hypot3< __gnu_cxx::__promoted_t< _Tp, _Up, _Vp> > (__x, __y, __z); 
# 3583
} 
# 3696 "/usr/include/c++/13/cmath" 3
}
# 42 "/usr/include/c++/13/bits/functexcept.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 49
void __throw_bad_exception() __attribute((__noreturn__)); 
# 53
void __throw_bad_alloc() __attribute((__noreturn__)); 
# 56
void __throw_bad_array_new_length() __attribute((__noreturn__)); 
# 60
void __throw_bad_cast() __attribute((__noreturn__)); 
# 63
void __throw_bad_typeid() __attribute((__noreturn__)); 
# 67
void __throw_logic_error(const char *) __attribute((__noreturn__)); 
# 70
void __throw_domain_error(const char *) __attribute((__noreturn__)); 
# 73
void __throw_invalid_argument(const char *) __attribute((__noreturn__)); 
# 76
void __throw_length_error(const char *) __attribute((__noreturn__)); 
# 79
void __throw_out_of_range(const char *) __attribute((__noreturn__)); 
# 82
void __throw_out_of_range_fmt(const char *, ...) __attribute((__noreturn__))
# 83
 __attribute((__format__(__gnu_printf__, 1, 2))); 
# 86
void __throw_runtime_error(const char *) __attribute((__noreturn__)); 
# 89
void __throw_range_error(const char *) __attribute((__noreturn__)); 
# 92
void __throw_overflow_error(const char *) __attribute((__noreturn__)); 
# 95
void __throw_underflow_error(const char *) __attribute((__noreturn__)); 
# 99
void __throw_ios_failure(const char *) __attribute((__noreturn__)); 
# 102
void __throw_ios_failure(const char *, int) __attribute((__noreturn__)); 
# 106
void __throw_system_error(int) __attribute((__noreturn__)); 
# 110
void __throw_future_error(int) __attribute((__noreturn__)); 
# 114
void __throw_bad_function_call() __attribute((__noreturn__)); 
# 141 "/usr/include/c++/13/bits/functexcept.h" 3
}
# 37 "/usr/include/c++/13/ext/numeric_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 50
template< class _Tp> 
# 51
struct __is_integer_nonstrict : public std::__is_integer< _Tp>  { 
# 54
using std::__is_integer< _Tp> ::__value;
# 57
enum { __width = (__value) ? sizeof(_Tp) * (8) : (0)}; 
# 58
}; 
# 60
template< class _Value> 
# 61
struct __numeric_traits_integer { 
# 64
static_assert((__is_integer_nonstrict< _Value> ::__value), "invalid specialization");
# 70
static const bool __is_signed = (((_Value)(-1)) < 0); 
# 71
static const int __digits = (__is_integer_nonstrict< _Value> ::__width - __is_signed); 
# 75
static const _Value __max = (__is_signed ? (((((_Value)1) << (__digits - 1)) - 1) << 1) + 1 : (~((_Value)0))); 
# 78
static const _Value __min = (__is_signed ? (-__max) - 1 : ((_Value)0)); 
# 79
}; 
# 81
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__min; 
# 84
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__max; 
# 87
template< class _Value> const bool __numeric_traits_integer< _Value> ::__is_signed; 
# 90
template< class _Value> const int __numeric_traits_integer< _Value> ::__digits; 
# 137 "/usr/include/c++/13/ext/numeric_traits.h" 3
template< class _Tp> using __int_traits = __numeric_traits_integer< _Tp> ; 
# 157
template< class _Value> 
# 158
struct __numeric_traits_floating { 
# 161
static const int __max_digits10 = ((2) + ((((std::template __are_same< _Value, float> ::__value) ? 24 : ((std::template __are_same< _Value, double> ::__value) ? 53 : 64)) * 643L) / (2136))); 
# 164
static const bool __is_signed = true; 
# 165
static const int __digits10 = ((std::template __are_same< _Value, float> ::__value) ? 6 : ((std::template __are_same< _Value, double> ::__value) ? 15 : 18)); 
# 166
static const int __max_exponent10 = ((std::template __are_same< _Value, float> ::__value) ? 38 : ((std::template __are_same< _Value, double> ::__value) ? 308 : 4932)); 
# 167
}; 
# 169
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_digits10; 
# 172
template< class _Value> const bool __numeric_traits_floating< _Value> ::__is_signed; 
# 175
template< class _Value> const int __numeric_traits_floating< _Value> ::__digits10; 
# 178
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_exponent10; 
# 186
template< class _Value> 
# 187
struct __numeric_traits : public __numeric_traits_integer< _Value>  { 
# 189
}; 
# 192
template<> struct __numeric_traits< float>  : public __numeric_traits_floating< float>  { 
# 194
}; 
# 197
template<> struct __numeric_traits< double>  : public __numeric_traits_floating< double>  { 
# 199
}; 
# 202
template<> struct __numeric_traits< long double>  : public __numeric_traits_floating< long double>  { 
# 204
}; 
# 239 "/usr/include/c++/13/ext/numeric_traits.h" 3
}
# 40 "/usr/include/c++/13/type_traits" 3
namespace std __attribute((__visibility__("default"))) { 
# 44
template< class _Tp> class reference_wrapper; 
# 61
template< class _Tp, _Tp __v> 
# 62
struct integral_constant { 
# 64
static constexpr inline _Tp value = (__v); 
# 65
typedef _Tp value_type; 
# 66
typedef integral_constant type; 
# 67
constexpr operator value_type() const noexcept { return value; } 
# 72
constexpr value_type operator()() const noexcept { return value; } 
# 74
}; 
# 82
using true_type = integral_constant< bool, true> ; 
# 85
using false_type = integral_constant< bool, false> ; 
# 89
template< bool __v> using __bool_constant = integral_constant< bool, __v> ; 
# 97
template< bool __v> using bool_constant = integral_constant< bool, __v> ; 
# 105
template< bool , class _Tp = void> 
# 106
struct enable_if { 
# 107
}; 
# 110
template< class _Tp> 
# 111
struct enable_if< true, _Tp>  { 
# 112
typedef _Tp type; }; 
# 115
template< bool _Cond, class _Tp = void> using __enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 118
template< bool > 
# 119
struct __conditional { 
# 121
template< class _Tp, class > using type = _Tp; 
# 123
}; 
# 126
template<> struct __conditional< false>  { 
# 128
template< class , class _Up> using type = _Up; 
# 130
}; 
# 133
template< bool _Cond, class _If, class _Else> using __conditional_t = typename __conditional< _Cond> ::template type< _If, _Else> ; 
# 138
template< class _Type> 
# 139
struct __type_identity { 
# 140
using type = _Type; }; 
# 142
template< class _Tp> using __type_identity_t = typename __type_identity< _Tp> ::type; 
# 145
namespace __detail { 
# 148
template< class _Tp, class ...> using __first_t = _Tp; 
# 152
template< class ..._Bn> auto __or_fn(int)->__first_t< integral_constant< bool, false> , __enable_if_t< !((bool)_Bn::value)> ...> ; 
# 156
template< class ..._Bn> auto __or_fn(...)->true_type; 
# 159
template< class ..._Bn> auto __and_fn(int)->__first_t< integral_constant< bool, true> , __enable_if_t< (bool)_Bn::value> ...> ; 
# 163
template< class ..._Bn> auto __and_fn(...)->false_type; 
# 165
}
# 170
template< class ..._Bn> 
# 171
struct __or_ : public __decltype((__detail::__or_fn< _Bn...> (0))) { 
# 173
}; 
# 175
template< class ..._Bn> 
# 176
struct __and_ : public __decltype((__detail::__and_fn< _Bn...> (0))) { 
# 178
}; 
# 180
template< class _Pp> 
# 181
struct __not_ : public __bool_constant< !((bool)_Pp::value)>  { 
# 183
}; 
# 189
template< class ..._Bn> constexpr inline bool 
# 190
__or_v = (__or_< _Bn...> ::value); 
# 191
template< class ..._Bn> constexpr inline bool 
# 192
__and_v = (__and_< _Bn...> ::value); 
# 194
namespace __detail { 
# 196
template< class , class _B1, class ..._Bn> 
# 197
struct __disjunction_impl { 
# 198
using type = _B1; }; 
# 200
template< class _B1, class _B2, class ..._Bn> 
# 201
struct __disjunction_impl< __enable_if_t< !((bool)_B1::value)> , _B1, _B2, _Bn...>  { 
# 202
using type = typename __detail::__disjunction_impl< void, _B2, _Bn...> ::type; }; 
# 204
template< class , class _B1, class ..._Bn> 
# 205
struct __conjunction_impl { 
# 206
using type = _B1; }; 
# 208
template< class _B1, class _B2, class ..._Bn> 
# 209
struct __conjunction_impl< __enable_if_t< (bool)_B1::value> , _B1, _B2, _Bn...>  { 
# 210
using type = typename __detail::__conjunction_impl< void, _B2, _Bn...> ::type; }; 
# 211
}
# 216
template< class ..._Bn> 
# 217
struct conjunction : public __detail::__conjunction_impl< void, _Bn...> ::type { 
# 219
}; 
# 222
template<> struct conjunction< >  : public true_type { 
# 224
}; 
# 226
template< class ..._Bn> 
# 227
struct disjunction : public __detail::__disjunction_impl< void, _Bn...> ::type { 
# 229
}; 
# 232
template<> struct disjunction< >  : public false_type { 
# 234
}; 
# 236
template< class _Pp> 
# 237
struct negation : public __not_< _Pp> ::type { 
# 239
}; 
# 244
template< class ..._Bn> constexpr inline bool 
# 245
conjunction_v = (conjunction< _Bn...> ::value); 
# 247
template< class ..._Bn> constexpr inline bool 
# 248
disjunction_v = (disjunction< _Bn...> ::value); 
# 250
template< class _Pp> constexpr inline bool 
# 251
negation_v = (negation< _Pp> ::value); 
# 257
template< class > struct is_reference; 
# 259
template< class > struct is_function; 
# 261
template< class > struct is_void; 
# 263
template< class > struct remove_cv; 
# 265
template< class > struct is_const; 
# 269
template< class > struct __is_array_unknown_bounds; 
# 275
template< class _Tp, size_t  = sizeof(_Tp)> constexpr true_type 
# 276
__is_complete_or_unbounded(__type_identity< _Tp> ) 
# 277
{ return {}; } 
# 279
template< class _TypeIdentity, class 
# 280
_NestedType = typename _TypeIdentity::type> constexpr typename __or_< is_reference< _NestedType> , is_function< _NestedType> , is_void< _NestedType> , __is_array_unknown_bounds< _NestedType> > ::type 
# 286
__is_complete_or_unbounded(_TypeIdentity) 
# 287
{ return {}; } 
# 290
template< class _Tp> using __remove_cv_t = typename remove_cv< _Tp> ::type; 
# 297
template< class _Tp> 
# 298
struct is_void : public false_type { 
# 299
}; 
# 302
template<> struct is_void< void>  : public true_type { 
# 303
}; 
# 306
template<> struct is_void< const void>  : public true_type { 
# 307
}; 
# 310
template<> struct is_void< volatile void>  : public true_type { 
# 311
}; 
# 314
template<> struct is_void< const volatile void>  : public true_type { 
# 315
}; 
# 318
template< class > 
# 319
struct __is_integral_helper : public false_type { 
# 320
}; 
# 323
template<> struct __is_integral_helper< bool>  : public true_type { 
# 324
}; 
# 327
template<> struct __is_integral_helper< char>  : public true_type { 
# 328
}; 
# 331
template<> struct __is_integral_helper< signed char>  : public true_type { 
# 332
}; 
# 335
template<> struct __is_integral_helper< unsigned char>  : public true_type { 
# 336
}; 
# 342
template<> struct __is_integral_helper< wchar_t>  : public true_type { 
# 343
}; 
# 352
template<> struct __is_integral_helper< char16_t>  : public true_type { 
# 353
}; 
# 356
template<> struct __is_integral_helper< char32_t>  : public true_type { 
# 357
}; 
# 360
template<> struct __is_integral_helper< short>  : public true_type { 
# 361
}; 
# 364
template<> struct __is_integral_helper< unsigned short>  : public true_type { 
# 365
}; 
# 368
template<> struct __is_integral_helper< int>  : public true_type { 
# 369
}; 
# 372
template<> struct __is_integral_helper< unsigned>  : public true_type { 
# 373
}; 
# 376
template<> struct __is_integral_helper< long>  : public true_type { 
# 377
}; 
# 380
template<> struct __is_integral_helper< unsigned long>  : public true_type { 
# 381
}; 
# 384
template<> struct __is_integral_helper< long long>  : public true_type { 
# 385
}; 
# 388
template<> struct __is_integral_helper< unsigned long long>  : public true_type { 
# 389
}; 
# 396
template<> struct __is_integral_helper< __int128>  : public true_type { 
# 397
}; 
# 401
template<> struct __is_integral_helper< unsigned __int128>  : public true_type { 
# 402
}; 
# 440 "/usr/include/c++/13/type_traits" 3
template< class _Tp> 
# 441
struct is_integral : public __is_integral_helper< __remove_cv_t< _Tp> > ::type { 
# 443
}; 
# 446
template< class > 
# 447
struct __is_floating_point_helper : public false_type { 
# 448
}; 
# 451
template<> struct __is_floating_point_helper< float>  : public true_type { 
# 452
}; 
# 455
template<> struct __is_floating_point_helper< double>  : public true_type { 
# 456
}; 
# 459
template<> struct __is_floating_point_helper< long double>  : public true_type { 
# 460
}; 
# 500
template< class _Tp> 
# 501
struct is_floating_point : public __is_floating_point_helper< __remove_cv_t< _Tp> > ::type { 
# 503
}; 
# 506
template< class > 
# 507
struct is_array : public false_type { 
# 508
}; 
# 510
template< class _Tp, size_t _Size> 
# 511
struct is_array< _Tp [_Size]>  : public true_type { 
# 512
}; 
# 514
template< class _Tp> 
# 515
struct is_array< _Tp []>  : public true_type { 
# 516
}; 
# 518
template< class > 
# 519
struct __is_pointer_helper : public false_type { 
# 520
}; 
# 522
template< class _Tp> 
# 523
struct __is_pointer_helper< _Tp *>  : public true_type { 
# 524
}; 
# 527
template< class _Tp> 
# 528
struct is_pointer : public __is_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 530
}; 
# 533
template< class > 
# 534
struct is_lvalue_reference : public false_type { 
# 535
}; 
# 537
template< class _Tp> 
# 538
struct is_lvalue_reference< _Tp &>  : public true_type { 
# 539
}; 
# 542
template< class > 
# 543
struct is_rvalue_reference : public false_type { 
# 544
}; 
# 546
template< class _Tp> 
# 547
struct is_rvalue_reference< _Tp &&>  : public true_type { 
# 548
}; 
# 550
template< class > 
# 551
struct __is_member_object_pointer_helper : public false_type { 
# 552
}; 
# 554
template< class _Tp, class _Cp> 
# 555
struct __is_member_object_pointer_helper< _Tp (_Cp::*)>  : public __not_< is_function< _Tp> > ::type { 
# 556
}; 
# 559
template< class _Tp> 
# 560
struct is_member_object_pointer : public __is_member_object_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 562
}; 
# 564
template< class > 
# 565
struct __is_member_function_pointer_helper : public false_type { 
# 566
}; 
# 568
template< class _Tp, class _Cp> 
# 569
struct __is_member_function_pointer_helper< _Tp (_Cp::*)>  : public is_function< _Tp> ::type { 
# 570
}; 
# 573
template< class _Tp> 
# 574
struct is_member_function_pointer : public __is_member_function_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 576
}; 
# 579
template< class _Tp> 
# 580
struct is_enum : public integral_constant< bool, __is_enum(_Tp)>  { 
# 582
}; 
# 585
template< class _Tp> 
# 586
struct is_union : public integral_constant< bool, __is_union(_Tp)>  { 
# 588
}; 
# 591
template< class _Tp> 
# 592
struct is_class : public integral_constant< bool, __is_class(_Tp)>  { 
# 594
}; 
# 597
template< class _Tp> 
# 598
struct is_function : public __bool_constant< !is_const< const _Tp> ::value>  { 
# 599
}; 
# 601
template< class _Tp> 
# 602
struct is_function< _Tp &>  : public false_type { 
# 603
}; 
# 605
template< class _Tp> 
# 606
struct is_function< _Tp &&>  : public false_type { 
# 607
}; 
# 612
template< class _Tp> 
# 613
struct is_null_pointer : public false_type { 
# 614
}; 
# 617
template<> struct is_null_pointer< __decltype((nullptr))>  : public true_type { 
# 618
}; 
# 621
template<> struct is_null_pointer< const __decltype((nullptr))>  : public true_type { 
# 622
}; 
# 625
template<> struct is_null_pointer< volatile __decltype((nullptr))>  : public true_type { 
# 626
}; 
# 629
template<> struct is_null_pointer< const volatile __decltype((nullptr))>  : public true_type { 
# 630
}; 
# 634
template< class _Tp> 
# 635
struct __is_nullptr_t : public is_null_pointer< _Tp>  { 
# 637
}; 
# 642
template< class _Tp> 
# 643
struct is_reference : public false_type { 
# 645
}; 
# 647
template< class _Tp> 
# 648
struct is_reference< _Tp &>  : public true_type { 
# 650
}; 
# 652
template< class _Tp> 
# 653
struct is_reference< _Tp &&>  : public true_type { 
# 655
}; 
# 658
template< class _Tp> 
# 659
struct is_arithmetic : public __or_< is_integral< _Tp> , is_floating_point< _Tp> > ::type { 
# 661
}; 
# 664
template< class _Tp> 
# 665
struct is_fundamental : public __or_< is_arithmetic< _Tp> , is_void< _Tp> , is_null_pointer< _Tp> > ::type { 
# 668
}; 
# 671
template< class _Tp> 
# 672
struct is_object : public __not_< __or_< is_function< _Tp> , is_reference< _Tp> , is_void< _Tp> > > ::type { 
# 675
}; 
# 677
template< class > struct is_member_pointer; 
# 681
template< class _Tp> 
# 682
struct is_scalar : public __or_< is_arithmetic< _Tp> , is_enum< _Tp> , is_pointer< _Tp> , is_member_pointer< _Tp> , is_null_pointer< _Tp> > ::type { 
# 685
}; 
# 688
template< class _Tp> 
# 689
struct is_compound : public __not_< is_fundamental< _Tp> > ::type { 
# 690
}; 
# 693
template< class _Tp> 
# 694
struct __is_member_pointer_helper : public false_type { 
# 695
}; 
# 697
template< class _Tp, class _Cp> 
# 698
struct __is_member_pointer_helper< _Tp (_Cp::*)>  : public true_type { 
# 699
}; 
# 703
template< class _Tp> 
# 704
struct is_member_pointer : public __is_member_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 706
}; 
# 708
template< class , class > struct is_same; 
# 712
template< class _Tp, class ..._Types> using __is_one_of = __or_< is_same< _Tp, _Types> ...> ; 
# 717
template< class _Tp> using __is_signed_integer = __is_one_of< __remove_cv_t< _Tp> , signed char, signed short, signed int, signed long, signed long long, signed __int128> ; 
# 737 "/usr/include/c++/13/type_traits" 3
template< class _Tp> using __is_unsigned_integer = __is_one_of< __remove_cv_t< _Tp> , unsigned char, unsigned short, unsigned, unsigned long, unsigned long long, unsigned __int128> ; 
# 756 "/usr/include/c++/13/type_traits" 3
template< class _Tp> using __is_standard_integer = __or_< __is_signed_integer< _Tp> , __is_unsigned_integer< _Tp> > ; 
# 761
template< class ...> using __void_t = void; 
# 767
template< class > 
# 768
struct is_const : public false_type { 
# 769
}; 
# 771
template< class _Tp> 
# 772
struct is_const< const _Tp>  : public true_type { 
# 773
}; 
# 776
template< class > 
# 777
struct is_volatile : public false_type { 
# 778
}; 
# 780
template< class _Tp> 
# 781
struct is_volatile< volatile _Tp>  : public true_type { 
# 782
}; 
# 785
template< class _Tp> 
# 786
struct is_trivial : public integral_constant< bool, __is_trivial(_Tp)>  { 
# 789
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 791
}; 
# 794
template< class _Tp> 
# 795
struct is_trivially_copyable : public integral_constant< bool, __is_trivially_copyable(_Tp)>  { 
# 798
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 800
}; 
# 803
template< class _Tp> 
# 804
struct is_standard_layout : public integral_constant< bool, __is_standard_layout(_Tp)>  { 
# 807
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 809
}; 
# 816
template< class _Tp> 
# 819
struct is_pod : public integral_constant< bool, __is_pod(_Tp)>  { 
# 822
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 824
}; 
# 830
template< class _Tp> 
# 833
struct is_literal_type : public integral_constant< bool, __is_literal_type(_Tp)>  { 
# 836
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 838
}; 
# 841
template< class _Tp> 
# 842
struct is_empty : public integral_constant< bool, __is_empty(_Tp)>  { 
# 844
}; 
# 847
template< class _Tp> 
# 848
struct is_polymorphic : public integral_constant< bool, __is_polymorphic(_Tp)>  { 
# 850
}; 
# 856
template< class _Tp> 
# 857
struct is_final : public integral_constant< bool, __is_final(_Tp)>  { 
# 859
}; 
# 863
template< class _Tp> 
# 864
struct is_abstract : public integral_constant< bool, __is_abstract(_Tp)>  { 
# 866
}; 
# 869
template< class _Tp, bool 
# 870
 = is_arithmetic< _Tp> ::value> 
# 871
struct __is_signed_helper : public false_type { 
# 872
}; 
# 874
template< class _Tp> 
# 875
struct __is_signed_helper< _Tp, true>  : public integral_constant< bool, ((_Tp)(-1)) < ((_Tp)0)>  { 
# 877
}; 
# 881
template< class _Tp> 
# 882
struct is_signed : public __is_signed_helper< _Tp> ::type { 
# 884
}; 
# 887
template< class _Tp> 
# 888
struct is_unsigned : public __and_< is_arithmetic< _Tp> , __not_< is_signed< _Tp> > > ::type { 
# 890
}; 
# 893
template< class _Tp, class _Up = _Tp &&> _Up __declval(int); 
# 897
template< class _Tp> _Tp __declval(long); 
# 902
template< class _Tp> auto declval() noexcept->__decltype((__declval< _Tp> (0))); 
# 905
template< class > struct remove_all_extents; 
# 909
template< class _Tp> 
# 910
struct __is_array_known_bounds : public false_type { 
# 912
}; 
# 914
template< class _Tp, size_t _Size> 
# 915
struct __is_array_known_bounds< _Tp [_Size]>  : public true_type { 
# 917
}; 
# 919
template< class _Tp> 
# 920
struct __is_array_unknown_bounds : public false_type { 
# 922
}; 
# 924
template< class _Tp> 
# 925
struct __is_array_unknown_bounds< _Tp []>  : public true_type { 
# 927
}; 
# 936
struct __do_is_destructible_impl { 
# 938
template< class _Tp, class  = __decltype((declval< _Tp &> ().~_Tp()))> static true_type __test(int); 
# 941
template< class > static false_type __test(...); 
# 943
}; 
# 945
template< class _Tp> 
# 946
struct __is_destructible_impl : public __do_is_destructible_impl { 
# 949
typedef __decltype((__test< _Tp> (0))) type; 
# 950
}; 
# 952
template< class _Tp, bool 
# 953
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 956
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_destructible_safe; 
# 959
template< class _Tp> 
# 960
struct __is_destructible_safe< _Tp, false, false>  : public __is_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 963
}; 
# 965
template< class _Tp> 
# 966
struct __is_destructible_safe< _Tp, true, false>  : public false_type { 
# 967
}; 
# 969
template< class _Tp> 
# 970
struct __is_destructible_safe< _Tp, false, true>  : public true_type { 
# 971
}; 
# 975
template< class _Tp> 
# 976
struct is_destructible : public __is_destructible_safe< _Tp> ::type { 
# 979
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 981
}; 
# 989
struct __do_is_nt_destructible_impl { 
# 991
template< class _Tp> static __bool_constant< noexcept(declval< _Tp &> ().~_Tp())>  __test(int); 
# 995
template< class > static false_type __test(...); 
# 997
}; 
# 999
template< class _Tp> 
# 1000
struct __is_nt_destructible_impl : public __do_is_nt_destructible_impl { 
# 1003
typedef __decltype((__test< _Tp> (0))) type; 
# 1004
}; 
# 1006
template< class _Tp, bool 
# 1007
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 1010
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_nt_destructible_safe; 
# 1013
template< class _Tp> 
# 1014
struct __is_nt_destructible_safe< _Tp, false, false>  : public __is_nt_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 1017
}; 
# 1019
template< class _Tp> 
# 1020
struct __is_nt_destructible_safe< _Tp, true, false>  : public false_type { 
# 1021
}; 
# 1023
template< class _Tp> 
# 1024
struct __is_nt_destructible_safe< _Tp, false, true>  : public true_type { 
# 1025
}; 
# 1029
template< class _Tp> 
# 1030
struct is_nothrow_destructible : public __is_nt_destructible_safe< _Tp> ::type { 
# 1033
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1035
}; 
# 1038
template< class _Tp, class ..._Args> using __is_constructible_impl = __bool_constant< __is_constructible(_Tp, _Args...)> ; 
# 1044
template< class _Tp, class ..._Args> 
# 1045
struct is_constructible : public __is_constructible_impl< _Tp, _Args...>  { 
# 1048
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1050
}; 
# 1053
template< class _Tp> 
# 1054
struct is_default_constructible : public __is_constructible_impl< _Tp>  { 
# 1057
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1059
}; 
# 1062
template< class _Tp, class  = void> 
# 1063
struct __add_lvalue_reference_helper { 
# 1064
using type = _Tp; }; 
# 1066
template< class _Tp> 
# 1067
struct __add_lvalue_reference_helper< _Tp, __void_t< _Tp &> >  { 
# 1068
using type = _Tp &; }; 
# 1070
template< class _Tp> using __add_lval_ref_t = typename __add_lvalue_reference_helper< _Tp> ::type; 
# 1075
template< class _Tp> 
# 1076
struct is_copy_constructible : public __is_constructible_impl< _Tp, __add_lval_ref_t< const _Tp> >  { 
# 1079
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1081
}; 
# 1084
template< class _Tp, class  = void> 
# 1085
struct __add_rvalue_reference_helper { 
# 1086
using type = _Tp; }; 
# 1088
template< class _Tp> 
# 1089
struct __add_rvalue_reference_helper< _Tp, __void_t< _Tp &&> >  { 
# 1090
using type = _Tp &&; }; 
# 1092
template< class _Tp> using __add_rval_ref_t = typename __add_rvalue_reference_helper< _Tp> ::type; 
# 1097
template< class _Tp> 
# 1098
struct is_move_constructible : public __is_constructible_impl< _Tp, __add_rval_ref_t< _Tp> >  { 
# 1101
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1103
}; 
# 1106
template< class _Tp, class ..._Args> using __is_nothrow_constructible_impl = __bool_constant< __is_nothrow_constructible(_Tp, _Args...)> ; 
# 1112
template< class _Tp, class ..._Args> 
# 1113
struct is_nothrow_constructible : public __is_nothrow_constructible_impl< _Tp, _Args...>  { 
# 1116
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1118
}; 
# 1121
template< class _Tp> 
# 1122
struct is_nothrow_default_constructible : public __is_nothrow_constructible_impl< _Tp>  { 
# 1125
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1127
}; 
# 1130
template< class _Tp> 
# 1131
struct is_nothrow_copy_constructible : public __is_nothrow_constructible_impl< _Tp, __add_lval_ref_t< const _Tp> >  { 
# 1134
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1136
}; 
# 1139
template< class _Tp> 
# 1140
struct is_nothrow_move_constructible : public __is_nothrow_constructible_impl< _Tp, __add_rval_ref_t< _Tp> >  { 
# 1143
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1145
}; 
# 1148
template< class _Tp, class _Up> using __is_assignable_impl = __bool_constant< __is_assignable(_Tp, _Up)> ; 
# 1153
template< class _Tp, class _Up> 
# 1154
struct is_assignable : public __is_assignable_impl< _Tp, _Up>  { 
# 1157
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1159
}; 
# 1162
template< class _Tp> 
# 1163
struct is_copy_assignable : public __is_assignable_impl< __add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> >  { 
# 1167
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1169
}; 
# 1172
template< class _Tp> 
# 1173
struct is_move_assignable : public __is_assignable_impl< __add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> >  { 
# 1176
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1178
}; 
# 1181
template< class _Tp, class _Up> using __is_nothrow_assignable_impl = __bool_constant< __is_nothrow_assignable(_Tp, _Up)> ; 
# 1187
template< class _Tp, class _Up> 
# 1188
struct is_nothrow_assignable : public __is_nothrow_assignable_impl< _Tp, _Up>  { 
# 1191
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1193
}; 
# 1196
template< class _Tp> 
# 1197
struct is_nothrow_copy_assignable : public __is_nothrow_assignable_impl< __add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> >  { 
# 1201
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1203
}; 
# 1206
template< class _Tp> 
# 1207
struct is_nothrow_move_assignable : public __is_nothrow_assignable_impl< __add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> >  { 
# 1211
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1213
}; 
# 1216
template< class _Tp, class ..._Args> using __is_trivially_constructible_impl = __bool_constant< __is_trivially_constructible(_Tp, _Args...)> ; 
# 1222
template< class _Tp, class ..._Args> 
# 1223
struct is_trivially_constructible : public __is_trivially_constructible_impl< _Tp, _Args...>  { 
# 1226
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1228
}; 
# 1231
template< class _Tp> 
# 1232
struct is_trivially_default_constructible : public __is_trivially_constructible_impl< _Tp>  { 
# 1235
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1237
}; 
# 1239
struct __do_is_implicitly_default_constructible_impl { 
# 1241
template< class _Tp> static void __helper(const _Tp &); 
# 1244
template< class _Tp> static true_type __test(const _Tp &, __decltype((__helper< const _Tp &> ({}))) * = 0); 
# 1248
static false_type __test(...); 
# 1249
}; 
# 1251
template< class _Tp> 
# 1252
struct __is_implicitly_default_constructible_impl : public __do_is_implicitly_default_constructible_impl { 
# 1255
typedef __decltype((__test(declval< _Tp> ()))) type; 
# 1256
}; 
# 1258
template< class _Tp> 
# 1259
struct __is_implicitly_default_constructible_safe : public __is_implicitly_default_constructible_impl< _Tp> ::type { 
# 1261
}; 
# 1263
template< class _Tp> 
# 1264
struct __is_implicitly_default_constructible : public __and_< __is_constructible_impl< _Tp> , __is_implicitly_default_constructible_safe< _Tp> > ::type { 
# 1267
}; 
# 1270
template< class _Tp> 
# 1271
struct is_trivially_copy_constructible : public __is_trivially_constructible_impl< _Tp, __add_lval_ref_t< const _Tp> >  { 
# 1274
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1276
}; 
# 1279
template< class _Tp> 
# 1280
struct is_trivially_move_constructible : public __is_trivially_constructible_impl< _Tp, __add_rval_ref_t< _Tp> >  { 
# 1283
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1285
}; 
# 1288
template< class _Tp, class _Up> using __is_trivially_assignable_impl = __bool_constant< __is_trivially_assignable(_Tp, _Up)> ; 
# 1294
template< class _Tp, class _Up> 
# 1295
struct is_trivially_assignable : public __is_trivially_assignable_impl< _Tp, _Up>  { 
# 1298
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1300
}; 
# 1303
template< class _Tp> 
# 1304
struct is_trivially_copy_assignable : public __is_trivially_assignable_impl< __add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> >  { 
# 1308
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1310
}; 
# 1313
template< class _Tp> 
# 1314
struct is_trivially_move_assignable : public __is_trivially_assignable_impl< __add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> >  { 
# 1318
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1320
}; 
# 1323
template< class _Tp> 
# 1324
struct is_trivially_destructible : public __and_< __is_destructible_safe< _Tp> , __bool_constant< __has_trivial_destructor(_Tp)> > ::type { 
# 1328
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1330
}; 
# 1334
template< class _Tp> 
# 1335
struct has_virtual_destructor : public integral_constant< bool, __has_virtual_destructor(_Tp)>  { 
# 1338
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1340
}; 
# 1346
template< class _Tp> 
# 1347
struct alignment_of : public integral_constant< unsigned long, __alignof__(_Tp)>  { 
# 1350
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1352
}; 
# 1355
template< class > 
# 1356
struct rank : public integral_constant< unsigned long, 0UL>  { 
# 1357
}; 
# 1359
template< class _Tp, size_t _Size> 
# 1360
struct rank< _Tp [_Size]>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1361
}; 
# 1363
template< class _Tp> 
# 1364
struct rank< _Tp []>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1365
}; 
# 1368
template< class , unsigned _Uint = 0U> 
# 1369
struct extent : public integral_constant< unsigned long, 0UL>  { 
# 1370
}; 
# 1372
template< class _Tp, size_t _Size> 
# 1373
struct extent< _Tp [_Size], 0>  : public integral_constant< unsigned long, _Size>  { 
# 1374
}; 
# 1376
template< class _Tp, unsigned _Uint, size_t _Size> 
# 1377
struct extent< _Tp [_Size], _Uint>  : public std::extent< _Tp, _Uint - (1)> ::type { 
# 1378
}; 
# 1380
template< class _Tp> 
# 1381
struct extent< _Tp [], 0>  : public integral_constant< unsigned long, 0UL>  { 
# 1382
}; 
# 1384
template< class _Tp, unsigned _Uint> 
# 1385
struct extent< _Tp [], _Uint>  : public std::extent< _Tp, _Uint - (1)> ::type { 
# 1386
}; 
# 1392
template< class _Tp, class _Up> 
# 1393
struct is_same : public integral_constant< bool, __is_same(_Tp, _Up)>  { 
# 1399
}; 
# 1409 "/usr/include/c++/13/type_traits" 3
template< class _Base, class _Derived> 
# 1410
struct is_base_of : public integral_constant< bool, __is_base_of(_Base, _Derived)>  { 
# 1412
}; 
# 1415
template< class _From, class _To> 
# 1416
struct is_convertible : public __bool_constant< __is_convertible(_From, _To)>  { 
# 1418
}; 
# 1458 "/usr/include/c++/13/type_traits" 3
template< class _ToElementType, class _FromElementType> using __is_array_convertible = is_convertible< _FromElementType (*)[], _ToElementType (*)[]> ; 
# 1522 "/usr/include/c++/13/type_traits" 3
template< class _Tp> 
# 1523
struct remove_const { 
# 1524
typedef _Tp type; }; 
# 1526
template< class _Tp> 
# 1527
struct remove_const< const _Tp>  { 
# 1528
typedef _Tp type; }; 
# 1531
template< class _Tp> 
# 1532
struct remove_volatile { 
# 1533
typedef _Tp type; }; 
# 1535
template< class _Tp> 
# 1536
struct remove_volatile< volatile _Tp>  { 
# 1537
typedef _Tp type; }; 
# 1545
template< class _Tp> 
# 1546
struct remove_cv { 
# 1547
using type = _Tp; }; 
# 1549
template< class _Tp> 
# 1550
struct remove_cv< const _Tp>  { 
# 1551
using type = _Tp; }; 
# 1553
template< class _Tp> 
# 1554
struct remove_cv< volatile _Tp>  { 
# 1555
using type = _Tp; }; 
# 1557
template< class _Tp> 
# 1558
struct remove_cv< const volatile _Tp>  { 
# 1559
using type = _Tp; }; 
# 1563
template< class _Tp> 
# 1564
struct add_const { 
# 1565
using type = const _Tp; }; 
# 1568
template< class _Tp> 
# 1569
struct add_volatile { 
# 1570
using type = volatile _Tp; }; 
# 1573
template< class _Tp> 
# 1574
struct add_cv { 
# 1575
using type = const volatile _Tp; }; 
# 1582
template< class _Tp> using remove_const_t = typename remove_const< _Tp> ::type; 
# 1586
template< class _Tp> using remove_volatile_t = typename remove_volatile< _Tp> ::type; 
# 1590
template< class _Tp> using remove_cv_t = typename remove_cv< _Tp> ::type; 
# 1594
template< class _Tp> using add_const_t = typename add_const< _Tp> ::type; 
# 1598
template< class _Tp> using add_volatile_t = typename add_volatile< _Tp> ::type; 
# 1602
template< class _Tp> using add_cv_t = typename add_cv< _Tp> ::type; 
# 1614
template< class _Tp> 
# 1615
struct remove_reference { 
# 1616
using type = _Tp; }; 
# 1618
template< class _Tp> 
# 1619
struct remove_reference< _Tp &>  { 
# 1620
using type = _Tp; }; 
# 1622
template< class _Tp> 
# 1623
struct remove_reference< _Tp &&>  { 
# 1624
using type = _Tp; }; 
# 1628
template< class _Tp> 
# 1629
struct add_lvalue_reference { 
# 1630
using type = __add_lval_ref_t< _Tp> ; }; 
# 1633
template< class _Tp> 
# 1634
struct add_rvalue_reference { 
# 1635
using type = __add_rval_ref_t< _Tp> ; }; 
# 1639
template< class _Tp> using remove_reference_t = typename remove_reference< _Tp> ::type; 
# 1643
template< class _Tp> using add_lvalue_reference_t = typename add_lvalue_reference< _Tp> ::type; 
# 1647
template< class _Tp> using add_rvalue_reference_t = typename add_rvalue_reference< _Tp> ::type; 
# 1656
template< class _Unqualified, bool _IsConst, bool _IsVol> struct __cv_selector; 
# 1659
template< class _Unqualified> 
# 1660
struct __cv_selector< _Unqualified, false, false>  { 
# 1661
typedef _Unqualified __type; }; 
# 1663
template< class _Unqualified> 
# 1664
struct __cv_selector< _Unqualified, false, true>  { 
# 1665
typedef volatile _Unqualified __type; }; 
# 1667
template< class _Unqualified> 
# 1668
struct __cv_selector< _Unqualified, true, false>  { 
# 1669
typedef const _Unqualified __type; }; 
# 1671
template< class _Unqualified> 
# 1672
struct __cv_selector< _Unqualified, true, true>  { 
# 1673
typedef const volatile _Unqualified __type; }; 
# 1675
template< class _Qualified, class _Unqualified, bool 
# 1676
_IsConst = is_const< _Qualified> ::value, bool 
# 1677
_IsVol = is_volatile< _Qualified> ::value> 
# 1678
class __match_cv_qualifiers { 
# 1680
typedef __cv_selector< _Unqualified, _IsConst, _IsVol>  __match; 
# 1683
public: typedef typename __cv_selector< _Unqualified, _IsConst, _IsVol> ::__type __type; 
# 1684
}; 
# 1687
template< class _Tp> 
# 1688
struct __make_unsigned { 
# 1689
typedef _Tp __type; }; 
# 1692
template<> struct __make_unsigned< char>  { 
# 1693
typedef unsigned char __type; }; 
# 1696
template<> struct __make_unsigned< signed char>  { 
# 1697
typedef unsigned char __type; }; 
# 1700
template<> struct __make_unsigned< short>  { 
# 1701
typedef unsigned short __type; }; 
# 1704
template<> struct __make_unsigned< int>  { 
# 1705
typedef unsigned __type; }; 
# 1708
template<> struct __make_unsigned< long>  { 
# 1709
typedef unsigned long __type; }; 
# 1712
template<> struct __make_unsigned< long long>  { 
# 1713
typedef unsigned long long __type; }; 
# 1718
template<> struct __make_unsigned< __int128>  { 
# 1719
typedef unsigned __int128 __type; }; 
# 1741 "/usr/include/c++/13/type_traits" 3
template< class _Tp, bool 
# 1742
_IsInt = is_integral< _Tp> ::value, bool 
# 1743
_IsEnum = is_enum< _Tp> ::value> class __make_unsigned_selector; 
# 1746
template< class _Tp> 
# 1747
class __make_unsigned_selector< _Tp, true, false>  { 
# 1749
using __unsigned_type = typename __make_unsigned< __remove_cv_t< _Tp> > ::__type; 
# 1753
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1755
}; 
# 1757
class __make_unsigned_selector_base { 
# 1760
protected: template< class ...> struct _List { }; 
# 1762
template< class _Tp, class ..._Up> 
# 1763
struct _List< _Tp, _Up...>  : public __make_unsigned_selector_base::_List< _Up...>  { 
# 1764
static constexpr inline std::size_t __size = sizeof(_Tp); }; 
# 1766
template< size_t _Sz, class _Tp, bool  = _Sz <= _Tp::__size> struct __select; 
# 1769
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1770
struct __select< _Sz, _List< _Uint, _UInts...> , true>  { 
# 1771
using __type = _Uint; }; 
# 1773
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1774
struct __select< _Sz, _List< _Uint, _UInts...> , false>  : public __make_unsigned_selector_base::__select< _Sz, _List< _UInts...> >  { 
# 1776
}; 
# 1777
}; 
# 1780
template< class _Tp> 
# 1781
class __make_unsigned_selector< _Tp, false, true>  : private __make_unsigned_selector_base { 
# 1785
using _UInts = _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> ; 
# 1788
using __unsigned_type = typename __select< sizeof(_Tp), _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> > ::__type; 
# 1791
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1793
}; 
# 1800
template<> struct __make_unsigned< wchar_t>  { 
# 1802
using __type = __make_unsigned_selector< wchar_t, false, true> ::__type; 
# 1804
}; 
# 1816 "/usr/include/c++/13/type_traits" 3
template<> struct __make_unsigned< char16_t>  { 
# 1818
using __type = __make_unsigned_selector< char16_t, false, true> ::__type; 
# 1820
}; 
# 1823
template<> struct __make_unsigned< char32_t>  { 
# 1825
using __type = __make_unsigned_selector< char32_t, false, true> ::__type; 
# 1827
}; 
# 1834
template< class _Tp> 
# 1835
struct make_unsigned { 
# 1836
typedef typename __make_unsigned_selector< _Tp> ::__type type; }; 
# 1839
template<> struct make_unsigned< bool> ; 
# 1840
template<> struct make_unsigned< const bool> ; 
# 1841
template<> struct make_unsigned< volatile bool> ; 
# 1842
template<> struct make_unsigned< const volatile bool> ; 
# 1847
template< class _Tp> 
# 1848
struct __make_signed { 
# 1849
typedef _Tp __type; }; 
# 1852
template<> struct __make_signed< char>  { 
# 1853
typedef signed char __type; }; 
# 1856
template<> struct __make_signed< unsigned char>  { 
# 1857
typedef signed char __type; }; 
# 1860
template<> struct __make_signed< unsigned short>  { 
# 1861
typedef signed short __type; }; 
# 1864
template<> struct __make_signed< unsigned>  { 
# 1865
typedef signed int __type; }; 
# 1868
template<> struct __make_signed< unsigned long>  { 
# 1869
typedef signed long __type; }; 
# 1872
template<> struct __make_signed< unsigned long long>  { 
# 1873
typedef signed long long __type; }; 
# 1878
template<> struct __make_signed< unsigned __int128>  { 
# 1879
typedef __int128 __type; }; 
# 1901 "/usr/include/c++/13/type_traits" 3
template< class _Tp, bool 
# 1902
_IsInt = is_integral< _Tp> ::value, bool 
# 1903
_IsEnum = is_enum< _Tp> ::value> class __make_signed_selector; 
# 1906
template< class _Tp> 
# 1907
class __make_signed_selector< _Tp, true, false>  { 
# 1909
using __signed_type = typename __make_signed< __remove_cv_t< _Tp> > ::__type; 
# 1913
public: using __type = typename __match_cv_qualifiers< _Tp, __signed_type> ::__type; 
# 1915
}; 
# 1918
template< class _Tp> 
# 1919
class __make_signed_selector< _Tp, false, true>  { 
# 1921
typedef typename __make_unsigned_selector< _Tp> ::__type __unsigned_type; 
# 1924
public: typedef typename std::__make_signed_selector< __unsigned_type> ::__type __type; 
# 1925
}; 
# 1932
template<> struct __make_signed< wchar_t>  { 
# 1934
using __type = __make_signed_selector< wchar_t, false, true> ::__type; 
# 1936
}; 
# 1948 "/usr/include/c++/13/type_traits" 3
template<> struct __make_signed< char16_t>  { 
# 1950
using __type = __make_signed_selector< char16_t, false, true> ::__type; 
# 1952
}; 
# 1955
template<> struct __make_signed< char32_t>  { 
# 1957
using __type = __make_signed_selector< char32_t, false, true> ::__type; 
# 1959
}; 
# 1966
template< class _Tp> 
# 1967
struct make_signed { 
# 1968
typedef typename __make_signed_selector< _Tp> ::__type type; }; 
# 1971
template<> struct make_signed< bool> ; 
# 1972
template<> struct make_signed< const bool> ; 
# 1973
template<> struct make_signed< volatile bool> ; 
# 1974
template<> struct make_signed< const volatile bool> ; 
# 1978
template< class _Tp> using make_signed_t = typename make_signed< _Tp> ::type; 
# 1982
template< class _Tp> using make_unsigned_t = typename make_unsigned< _Tp> ::type; 
# 1989
template< class _Tp> 
# 1990
struct remove_extent { 
# 1991
typedef _Tp type; }; 
# 1993
template< class _Tp, size_t _Size> 
# 1994
struct remove_extent< _Tp [_Size]>  { 
# 1995
typedef _Tp type; }; 
# 1997
template< class _Tp> 
# 1998
struct remove_extent< _Tp []>  { 
# 1999
typedef _Tp type; }; 
# 2002
template< class _Tp> 
# 2003
struct remove_all_extents { 
# 2004
typedef _Tp type; }; 
# 2006
template< class _Tp, size_t _Size> 
# 2007
struct remove_all_extents< _Tp [_Size]>  { 
# 2008
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 2010
template< class _Tp> 
# 2011
struct remove_all_extents< _Tp []>  { 
# 2012
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 2016
template< class _Tp> using remove_extent_t = typename remove_extent< _Tp> ::type; 
# 2020
template< class _Tp> using remove_all_extents_t = typename remove_all_extents< _Tp> ::type; 
# 2026
template< class _Tp, class > 
# 2027
struct __remove_pointer_helper { 
# 2028
typedef _Tp type; }; 
# 2030
template< class _Tp, class _Up> 
# 2031
struct __remove_pointer_helper< _Tp, _Up *>  { 
# 2032
typedef _Up type; }; 
# 2035
template< class _Tp> 
# 2036
struct remove_pointer : public __remove_pointer_helper< _Tp, __remove_cv_t< _Tp> >  { 
# 2038
}; 
# 2040
template< class _Tp, class  = void> 
# 2041
struct __add_pointer_helper { 
# 2042
using type = _Tp; }; 
# 2044
template< class _Tp> 
# 2045
struct __add_pointer_helper< _Tp, __void_t< _Tp *> >  { 
# 2046
using type = _Tp *; }; 
# 2049
template< class _Tp> 
# 2050
struct add_pointer : public __add_pointer_helper< _Tp>  { 
# 2052
}; 
# 2054
template< class _Tp> 
# 2055
struct add_pointer< _Tp &>  { 
# 2056
using type = _Tp *; }; 
# 2058
template< class _Tp> 
# 2059
struct add_pointer< _Tp &&>  { 
# 2060
using type = _Tp *; }; 
# 2064
template< class _Tp> using remove_pointer_t = typename remove_pointer< _Tp> ::type; 
# 2068
template< class _Tp> using add_pointer_t = typename add_pointer< _Tp> ::type; 
# 2072
template< size_t _Len> 
# 2073
struct __aligned_storage_msa { 
# 2075
union __type { 
# 2077
unsigned char __data[_Len]; 
# 2078
struct __attribute((__aligned__)) { } __align; 
# 2079
}; 
# 2080
}; 
# 2095
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> 
# 2099
struct aligned_storage { 
# 2101
union type { 
# 2103
unsigned char __data[_Len]; 
# 2104
struct __attribute((__aligned__(_Align))) { } __align; 
# 2105
}; 
# 2106
}; 
# 2108
template< class ..._Types> 
# 2109
struct __strictest_alignment { 
# 2111
static const size_t _S_alignment = (0); 
# 2112
static const size_t _S_size = (0); 
# 2113
}; 
# 2115
template< class _Tp, class ..._Types> 
# 2116
struct __strictest_alignment< _Tp, _Types...>  { 
# 2118
static const size_t _S_alignment = ((__alignof__(_Tp) > __strictest_alignment< _Types...> ::_S_alignment) ? __alignof__(_Tp) : __strictest_alignment< _Types...> ::_S_alignment); 
# 2121
static const size_t _S_size = ((sizeof(_Tp) > __strictest_alignment< _Types...> ::_S_size) ? sizeof(_Tp) : __strictest_alignment< _Types...> ::_S_size); 
# 2124
}; 
# 2126
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
# 2141
template< size_t _Len, class ..._Types> 
# 2144
struct aligned_union { 
# 2147
static_assert((sizeof...(_Types) != (0)), "At least one type is required");
# 2149
private: using __strictest = __strictest_alignment< _Types...> ; 
# 2150
static const size_t _S_len = ((_Len > __strictest::_S_size) ? _Len : __strictest::_S_size); 
# 2154
public: static const size_t alignment_value = (__strictest::_S_alignment); 
# 2156
typedef typename aligned_storage< _S_len, alignment_value> ::type type; 
# 2157
}; 
# 2159
template< size_t _Len, class ..._Types> const size_t aligned_union< _Len, _Types...> ::alignment_value; 
# 2161
#pragma GCC diagnostic pop
# 2167
template< class _Up> 
# 2168
struct __decay_selector : public __conditional_t< is_const< const _Up> ::value, remove_cv< _Up> , add_pointer< _Up> >  { 
# 2172
}; 
# 2174
template< class _Up, size_t _Nm> 
# 2175
struct __decay_selector< _Up [_Nm]>  { 
# 2176
using type = _Up *; }; 
# 2178
template< class _Up> 
# 2179
struct __decay_selector< _Up []>  { 
# 2180
using type = _Up *; }; 
# 2185
template< class _Tp> 
# 2186
struct decay { 
# 2187
using type = typename __decay_selector< _Tp> ::type; }; 
# 2189
template< class _Tp> 
# 2190
struct decay< _Tp &>  { 
# 2191
using type = typename __decay_selector< _Tp> ::type; }; 
# 2193
template< class _Tp> 
# 2194
struct decay< _Tp &&>  { 
# 2195
using type = typename __decay_selector< _Tp> ::type; }; 
# 2200
template< class _Tp> 
# 2201
struct __strip_reference_wrapper { 
# 2203
typedef _Tp __type; 
# 2204
}; 
# 2206
template< class _Tp> 
# 2207
struct __strip_reference_wrapper< reference_wrapper< _Tp> >  { 
# 2209
typedef _Tp &__type; 
# 2210
}; 
# 2213
template< class _Tp> using __decay_t = typename decay< _Tp> ::type; 
# 2216
template< class _Tp> using __decay_and_strip = __strip_reference_wrapper< __decay_t< _Tp> > ; 
# 2223
template< class ..._Cond> using _Require = __enable_if_t< __and_< _Cond...> ::value> ; 
# 2227
template< class _Tp> using __remove_cvref_t = typename remove_cv< typename remove_reference< _Tp> ::type> ::type; 
# 2234
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 2235
struct conditional { 
# 2236
typedef _Iftrue type; }; 
# 2239
template< class _Iftrue, class _Iffalse> 
# 2240
struct conditional< false, _Iftrue, _Iffalse>  { 
# 2241
typedef _Iffalse type; }; 
# 2244
template< class ..._Tp> struct common_type; 
# 2256
template< class _Tp> 
# 2257
struct __success_type { 
# 2258
typedef _Tp type; }; 
# 2260
struct __failure_type { 
# 2261
}; 
# 2263
struct __do_common_type_impl { 
# 2265
template< class _Tp, class _Up> using __cond_t = __decltype((true ? std::declval< _Tp> () : std::declval< _Up> ())); 
# 2271
template< class _Tp, class _Up> static __success_type< __decay_t< __cond_t< _Tp, _Up> > >  _S_test(int); 
# 2283 "/usr/include/c++/13/type_traits" 3
template< class , class > static __failure_type _S_test_2(...); 
# 2287
template< class _Tp, class _Up> static __decltype((_S_test_2< _Tp, _Up> (0))) _S_test(...); 
# 2290
}; 
# 2294
template<> struct common_type< >  { 
# 2295
}; 
# 2298
template< class _Tp0> 
# 2299
struct common_type< _Tp0>  : public std::common_type< _Tp0, _Tp0>  { 
# 2301
}; 
# 2304
template< class _Tp1, class _Tp2, class 
# 2305
_Dp1 = __decay_t< _Tp1> , class _Dp2 = __decay_t< _Tp2> > 
# 2306
struct __common_type_impl { 
# 2310
using type = common_type< _Dp1, _Dp2> ; 
# 2311
}; 
# 2313
template< class _Tp1, class _Tp2> 
# 2314
struct __common_type_impl< _Tp1, _Tp2, _Tp1, _Tp2>  : private __do_common_type_impl { 
# 2319
using type = __decltype((_S_test< _Tp1, _Tp2> (0))); 
# 2320
}; 
# 2323
template< class _Tp1, class _Tp2> 
# 2324
struct common_type< _Tp1, _Tp2>  : public __common_type_impl< _Tp1, _Tp2> ::type { 
# 2326
}; 
# 2328
template< class ...> 
# 2329
struct __common_type_pack { 
# 2330
}; 
# 2332
template< class , class , class  = void> struct __common_type_fold; 
# 2336
template< class _Tp1, class _Tp2, class ..._Rp> 
# 2337
struct common_type< _Tp1, _Tp2, _Rp...>  : public __common_type_fold< std::common_type< _Tp1, _Tp2> , __common_type_pack< _Rp...> >  { 
# 2340
}; 
# 2345
template< class _CTp, class ..._Rp> 
# 2346
struct __common_type_fold< _CTp, __common_type_pack< _Rp...> , __void_t< typename _CTp::type> >  : public common_type< typename _CTp::type, _Rp...>  { 
# 2349
}; 
# 2352
template< class _CTp, class _Rp> 
# 2353
struct __common_type_fold< _CTp, _Rp, void>  { 
# 2354
}; 
# 2356
template< class _Tp, bool  = is_enum< _Tp> ::value> 
# 2357
struct __underlying_type_impl { 
# 2359
using type = __underlying_type(_Tp); 
# 2360
}; 
# 2362
template< class _Tp> 
# 2363
struct __underlying_type_impl< _Tp, false>  { 
# 2364
}; 
# 2368
template< class _Tp> 
# 2369
struct underlying_type : public __underlying_type_impl< _Tp>  { 
# 2371
}; 
# 2374
template< class _Tp> 
# 2375
struct __declval_protector { 
# 2377
static const bool __stop = false; 
# 2378
}; 
# 2385
template< class _Tp> auto 
# 2386
declval() noexcept->__decltype((__declval< _Tp> (0))) 
# 2387
{ 
# 2388
static_assert((__declval_protector< _Tp> ::__stop), "declval() must not be used!");
# 2390
return __declval< _Tp> (0); 
# 2391
} 
# 2394
template< class _Signature> struct result_of; 
# 2402
struct __invoke_memfun_ref { }; 
# 2403
struct __invoke_memfun_deref { }; 
# 2404
struct __invoke_memobj_ref { }; 
# 2405
struct __invoke_memobj_deref { }; 
# 2406
struct __invoke_other { }; 
# 2409
template< class _Tp, class _Tag> 
# 2410
struct __result_of_success : public __success_type< _Tp>  { 
# 2411
using __invoke_type = _Tag; }; 
# 2414
struct __result_of_memfun_ref_impl { 
# 2416
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype(((std::declval< _Tp1> ().*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_ref>  _S_test(int); 
# 2421
template< class ...> static __failure_type _S_test(...); 
# 2423
}; 
# 2425
template< class _MemPtr, class _Arg, class ..._Args> 
# 2426
struct __result_of_memfun_ref : private __result_of_memfun_ref_impl { 
# 2429
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2430
}; 
# 2433
struct __result_of_memfun_deref_impl { 
# 2435
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype((((*std::declval< _Tp1> ()).*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_deref>  _S_test(int); 
# 2440
template< class ...> static __failure_type _S_test(...); 
# 2442
}; 
# 2444
template< class _MemPtr, class _Arg, class ..._Args> 
# 2445
struct __result_of_memfun_deref : private __result_of_memfun_deref_impl { 
# 2448
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2449
}; 
# 2452
struct __result_of_memobj_ref_impl { 
# 2454
template< class _Fp, class _Tp1> static __result_of_success< __decltype((std::declval< _Tp1> ().*std::declval< _Fp> ())), __invoke_memobj_ref>  _S_test(int); 
# 2459
template< class , class > static __failure_type _S_test(...); 
# 2461
}; 
# 2463
template< class _MemPtr, class _Arg> 
# 2464
struct __result_of_memobj_ref : private __result_of_memobj_ref_impl { 
# 2467
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2468
}; 
# 2471
struct __result_of_memobj_deref_impl { 
# 2473
template< class _Fp, class _Tp1> static __result_of_success< __decltype(((*std::declval< _Tp1> ()).*std::declval< _Fp> ())), __invoke_memobj_deref>  _S_test(int); 
# 2478
template< class , class > static __failure_type _S_test(...); 
# 2480
}; 
# 2482
template< class _MemPtr, class _Arg> 
# 2483
struct __result_of_memobj_deref : private __result_of_memobj_deref_impl { 
# 2486
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2487
}; 
# 2489
template< class _MemPtr, class _Arg> struct __result_of_memobj; 
# 2492
template< class _Res, class _Class, class _Arg> 
# 2493
struct __result_of_memobj< _Res (_Class::*), _Arg>  { 
# 2495
typedef __remove_cvref_t< _Arg>  _Argval; 
# 2496
typedef _Res (_Class::*_MemPtr); 
# 2501
typedef typename __conditional_t< __or_< is_same< _Argval, _Class> , is_base_of< _Class, _Argval> > ::value, __result_of_memobj_ref< _MemPtr, _Arg> , __result_of_memobj_deref< _MemPtr, _Arg> > ::type type; 
# 2502
}; 
# 2504
template< class _MemPtr, class _Arg, class ..._Args> struct __result_of_memfun; 
# 2507
template< class _Res, class _Class, class _Arg, class ..._Args> 
# 2508
struct __result_of_memfun< _Res (_Class::*), _Arg, _Args...>  { 
# 2510
typedef typename remove_reference< _Arg> ::type _Argval; 
# 2511
typedef _Res (_Class::*_MemPtr); 
# 2515
typedef typename __conditional_t< is_base_of< _Class, _Argval> ::value, __result_of_memfun_ref< _MemPtr, _Arg, _Args...> , __result_of_memfun_deref< _MemPtr, _Arg, _Args...> > ::type type; 
# 2516
}; 
# 2523
template< class _Tp, class _Up = __remove_cvref_t< _Tp> > 
# 2524
struct __inv_unwrap { 
# 2526
using type = _Tp; 
# 2527
}; 
# 2529
template< class _Tp, class _Up> 
# 2530
struct __inv_unwrap< _Tp, reference_wrapper< _Up> >  { 
# 2532
using type = _Up &; 
# 2533
}; 
# 2535
template< bool , bool , class _Functor, class ..._ArgTypes> 
# 2536
struct __result_of_impl { 
# 2538
typedef __failure_type type; 
# 2539
}; 
# 2541
template< class _MemPtr, class _Arg> 
# 2542
struct __result_of_impl< true, false, _MemPtr, _Arg>  : public __result_of_memobj< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type>  { 
# 2545
}; 
# 2547
template< class _MemPtr, class _Arg, class ..._Args> 
# 2548
struct __result_of_impl< false, true, _MemPtr, _Arg, _Args...>  : public __result_of_memfun< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type, _Args...>  { 
# 2551
}; 
# 2554
struct __result_of_other_impl { 
# 2556
template< class _Fn, class ..._Args> static __result_of_success< __decltype((std::declval< _Fn> ()(std::declval< _Args> ()...))), __invoke_other>  _S_test(int); 
# 2561
template< class ...> static __failure_type _S_test(...); 
# 2563
}; 
# 2565
template< class _Functor, class ..._ArgTypes> 
# 2566
struct __result_of_impl< false, false, _Functor, _ArgTypes...>  : private __result_of_other_impl { 
# 2569
typedef __decltype((_S_test< _Functor, _ArgTypes...> (0))) type; 
# 2570
}; 
# 2573
template< class _Functor, class ..._ArgTypes> 
# 2574
struct __invoke_result : public __result_of_impl< is_member_object_pointer< typename remove_reference< _Functor> ::type> ::value, is_member_function_pointer< typename remove_reference< _Functor> ::type> ::value, _Functor, _ArgTypes...> ::type { 
# 2584
}; 
# 2587
template< class _Functor, class ..._ArgTypes> 
# 2588
struct result_of< _Functor (_ArgTypes ...)>  : public __invoke_result< _Functor, _ArgTypes...>  { 
# 2590
}; 
# 2593
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
# 2596
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> using aligned_storage_t = typename aligned_storage< _Len, _Align> ::type; 
# 2600
template< size_t _Len, class ..._Types> using aligned_union_t = typename aligned_union< _Len, _Types...> ::type; 
# 2602
#pragma GCC diagnostic pop
# 2605
template< class _Tp> using decay_t = typename decay< _Tp> ::type; 
# 2609
template< bool _Cond, class _Tp = void> using enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2613
template< bool _Cond, class _Iftrue, class _Iffalse> using conditional_t = typename conditional< _Cond, _Iftrue, _Iffalse> ::type; 
# 2617
template< class ..._Tp> using common_type_t = typename common_type< _Tp...> ::type; 
# 2621
template< class _Tp> using underlying_type_t = typename underlying_type< _Tp> ::type; 
# 2625
template< class _Tp> using result_of_t = typename result_of< _Tp> ::type; 
# 2632
template< class ...> using void_t = void; 
# 2659 "/usr/include/c++/13/type_traits" 3
template< class _Default, class _AlwaysVoid, 
# 2660
template< class ...>  class _Op, class ..._Args> 
# 2661
struct __detector { 
# 2663
using type = _Default; 
# 2664
using __is_detected = false_type; 
# 2665
}; 
# 2668
template< class _Default, template< class ...>  class _Op, class ...
# 2669
_Args> 
# 2670
struct __detector< _Default, __void_t< _Op< _Args...> > , _Op, _Args...>  { 
# 2672
using type = _Op< _Args...> ; 
# 2673
using __is_detected = true_type; 
# 2674
}; 
# 2676
template< class _Default, template< class ...>  class _Op, class ...
# 2677
_Args> using __detected_or = __detector< _Default, void, _Op, _Args...> ; 
# 2682
template< class _Default, template< class ...>  class _Op, class ...
# 2683
_Args> using __detected_or_t = typename __detected_or< _Default, _Op, _Args...> ::type; 
# 2701 "/usr/include/c++/13/type_traits" 3
template< class _Tp> struct __is_swappable; 
# 2704
template< class _Tp> struct __is_nothrow_swappable; 
# 2707
template< class > 
# 2708
struct __is_tuple_like_impl : public false_type { 
# 2709
}; 
# 2712
template< class _Tp> 
# 2713
struct __is_tuple_like : public __is_tuple_like_impl< __remove_cvref_t< _Tp> > ::type { 
# 2715
}; 
# 2718
template< class _Tp> inline _Require< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> >  swap(_Tp &, _Tp &) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value); 
# 2728
template< class _Tp, size_t _Nm> inline __enable_if_t< __is_swappable< _Tp> ::value>  swap(_Tp (& __a)[_Nm], _Tp (& __b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value); 
# 2736
namespace __swappable_details { 
# 2737
using std::swap;
# 2739
struct __do_is_swappable_impl { 
# 2741
template< class _Tp, class 
# 2742
 = __decltype((swap(std::declval< _Tp &> (), std::declval< _Tp &> ())))> static true_type 
# 2741
__test(int); 
# 2745
template< class > static false_type __test(...); 
# 2747
}; 
# 2749
struct __do_is_nothrow_swappable_impl { 
# 2751
template< class _Tp> static __bool_constant< noexcept(swap(std::declval< _Tp &> (), std::declval< _Tp &> ()))>  __test(int); 
# 2756
template< class > static false_type __test(...); 
# 2758
}; 
# 2760
}
# 2762
template< class _Tp> 
# 2763
struct __is_swappable_impl : public __swappable_details::__do_is_swappable_impl { 
# 2766
typedef __decltype((__test< _Tp> (0))) type; 
# 2767
}; 
# 2769
template< class _Tp> 
# 2770
struct __is_nothrow_swappable_impl : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2773
typedef __decltype((__test< _Tp> (0))) type; 
# 2774
}; 
# 2776
template< class _Tp> 
# 2777
struct __is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2779
}; 
# 2781
template< class _Tp> 
# 2782
struct __is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2784
}; 
# 2792
template< class _Tp> 
# 2793
struct is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2796
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2798
}; 
# 2801
template< class _Tp> 
# 2802
struct is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2805
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2807
}; 
# 2811
template< class _Tp> constexpr inline bool 
# 2812
is_swappable_v = (is_swappable< _Tp> ::value); 
# 2816
template< class _Tp> constexpr inline bool 
# 2817
is_nothrow_swappable_v = (is_nothrow_swappable< _Tp> ::value); 
# 2822
namespace __swappable_with_details { 
# 2823
using std::swap;
# 2825
struct __do_is_swappable_with_impl { 
# 2827
template< class _Tp, class _Up, class 
# 2828
 = __decltype((swap(std::declval< _Tp> (), std::declval< _Up> ()))), class 
# 2830
 = __decltype((swap(std::declval< _Up> (), std::declval< _Tp> ())))> static true_type 
# 2827
__test(int); 
# 2833
template< class , class > static false_type __test(...); 
# 2835
}; 
# 2837
struct __do_is_nothrow_swappable_with_impl { 
# 2839
template< class _Tp, class _Up> static __bool_constant< noexcept(swap(std::declval< _Tp> (), std::declval< _Up> ())) && noexcept(swap(std::declval< _Up> (), std::declval< _Tp> ()))>  __test(int); 
# 2846
template< class , class > static false_type __test(...); 
# 2848
}; 
# 2850
}
# 2852
template< class _Tp, class _Up> 
# 2853
struct __is_swappable_with_impl : public __swappable_with_details::__do_is_swappable_with_impl { 
# 2856
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2857
}; 
# 2860
template< class _Tp> 
# 2861
struct __is_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_swappable_impl { 
# 2864
typedef __decltype((__test< _Tp &> (0))) type; 
# 2865
}; 
# 2867
template< class _Tp, class _Up> 
# 2868
struct __is_nothrow_swappable_with_impl : public __swappable_with_details::__do_is_nothrow_swappable_with_impl { 
# 2871
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2872
}; 
# 2875
template< class _Tp> 
# 2876
struct __is_nothrow_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2879
typedef __decltype((__test< _Tp &> (0))) type; 
# 2880
}; 
# 2884
template< class _Tp, class _Up> 
# 2885
struct is_swappable_with : public __is_swappable_with_impl< _Tp, _Up> ::type { 
# 2888
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2890
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2892
}; 
# 2895
template< class _Tp, class _Up> 
# 2896
struct is_nothrow_swappable_with : public __is_nothrow_swappable_with_impl< _Tp, _Up> ::type { 
# 2899
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2901
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2903
}; 
# 2907
template< class _Tp, class _Up> constexpr inline bool 
# 2908
is_swappable_with_v = (is_swappable_with< _Tp, _Up> ::value); 
# 2912
template< class _Tp, class _Up> constexpr inline bool 
# 2913
is_nothrow_swappable_with_v = (is_nothrow_swappable_with< _Tp, _Up> ::value); 
# 2924
template< class _Result, class _Ret, bool 
# 2925
 = is_void< _Ret> ::value, class  = void> 
# 2926
struct __is_invocable_impl : public false_type { 
# 2929
using __nothrow_conv = false_type; 
# 2930
}; 
# 2933
template< class _Result, class _Ret> 
# 2934
struct __is_invocable_impl< _Result, _Ret, true, __void_t< typename _Result::type> >  : public true_type { 
# 2939
using __nothrow_conv = true_type; 
# 2940
}; 
# 2942
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
# 2945
template< class _Result, class _Ret> 
# 2946
struct __is_invocable_impl< _Result, _Ret, false, __void_t< typename _Result::type> >  { 
# 2952
private: using _Res_t = typename _Result::type; 
# 2956
static _Res_t _S_get() noexcept; 
# 2959
template< class _Tp> static void _S_conv(__type_identity_t< _Tp> ) noexcept; 
# 2963
template< class _Tp, bool 
# 2964
_Nothrow = noexcept(_S_conv< _Tp> ((_S_get)())), class 
# 2965
 = __decltype((_S_conv< _Tp> ((_S_get)()))), bool 
# 2967
_Dangle = __reference_converts_from_temporary(_Tp, _Res_t)> static __bool_constant< _Nothrow && (!_Dangle)>  
# 2963
_S_test(int); 
# 2975
template< class _Tp, bool  = false> static false_type _S_test(...); 
# 2981
public: using type = __decltype((_S_test< _Ret, true> (1))); 
# 2984
using __nothrow_conv = __decltype((_S_test< _Ret> (1))); 
# 2985
}; 
#pragma GCC diagnostic pop
# 2988
template< class _Fn, class ..._ArgTypes> 
# 2989
struct __is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 2991
}; 
# 2993
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2994
__call_is_nt(__invoke_memfun_ref) 
# 2995
{ 
# 2996
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2997
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2999
} 
# 3001
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 3002
__call_is_nt(__invoke_memfun_deref) 
# 3003
{ 
# 3004
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 3006
} 
# 3008
template< class _Fn, class _Tp> constexpr bool 
# 3009
__call_is_nt(__invoke_memobj_ref) 
# 3010
{ 
# 3011
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 3012
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())); 
# 3013
} 
# 3015
template< class _Fn, class _Tp> constexpr bool 
# 3016
__call_is_nt(__invoke_memobj_deref) 
# 3017
{ 
# 3018
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())); 
# 3019
} 
# 3021
template< class _Fn, class ..._Args> constexpr bool 
# 3022
__call_is_nt(__invoke_other) 
# 3023
{ 
# 3024
return noexcept(std::declval< _Fn> ()(std::declval< _Args> ()...)); 
# 3025
} 
# 3027
template< class _Result, class _Fn, class ..._Args> 
# 3028
struct __call_is_nothrow : public __bool_constant< std::__call_is_nt< _Fn, _Args...> (typename _Result::__invoke_type{})>  { 
# 3032
}; 
# 3034
template< class _Fn, class ..._Args> using __call_is_nothrow_ = __call_is_nothrow< __invoke_result< _Fn, _Args...> , _Fn, _Args...> ; 
# 3039
template< class _Fn, class ..._Args> 
# 3040
struct __is_nothrow_invocable : public __and_< __is_invocable< _Fn, _Args...> , __call_is_nothrow_< _Fn, _Args...> > ::type { 
# 3043
}; 
# 3045
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
struct __nonesuchbase { }; 
# 3048
struct __nonesuch : private __nonesuchbase { 
# 3049
~__nonesuch() = delete;
# 3050
__nonesuch(const __nonesuch &) = delete;
# 3051
void operator=(const __nonesuch &) = delete;
# 3052
}; 
#pragma GCC diagnostic pop
# 3060
template< class _Functor, class ..._ArgTypes> 
# 3061
struct invoke_result : public __invoke_result< _Functor, _ArgTypes...>  { 
# 3064
static_assert((std::__is_complete_or_unbounded(__type_identity< _Functor> {})), "_Functor must be a complete class or an unbounded array");
# 3066
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3069
}; 
# 3072
template< class _Fn, class ..._Args> using invoke_result_t = typename invoke_result< _Fn, _Args...> ::type; 
# 3076
template< class _Fn, class ..._ArgTypes> 
# 3077
struct is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 3080
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3082
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3085
}; 
# 3088
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 3089
struct is_invocable_r : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> ::type { 
# 3092
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3094
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3097
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 3099
}; 
# 3102
template< class _Fn, class ..._ArgTypes> 
# 3103
struct is_nothrow_invocable : public __and_< __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 3107
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3109
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3112
}; 
# 3118
template< class _Result, class _Ret> using __is_nt_invocable_impl = typename __is_invocable_impl< _Result, _Ret> ::__nothrow_conv; 
# 3124
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 3125
struct is_nothrow_invocable_r : public __and_< __is_nt_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 3129
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3131
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3134
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 3136
}; 
# 3155
template< class _Tp> constexpr inline bool 
# 3156
is_void_v = (is_void< _Tp> ::value); 
# 3157
template< class _Tp> constexpr inline bool 
# 3158
is_null_pointer_v = (is_null_pointer< _Tp> ::value); 
# 3159
template< class _Tp> constexpr inline bool 
# 3160
is_integral_v = (is_integral< _Tp> ::value); 
# 3161
template< class _Tp> constexpr inline bool 
# 3162
is_floating_point_v = (is_floating_point< _Tp> ::value); 
# 3164
template< class _Tp> constexpr inline bool 
# 3165
is_array_v = false; 
# 3166
template< class _Tp> constexpr inline bool 
# 3167
is_array_v< _Tp []>  = true; 
# 3168
template< class _Tp, size_t _Num> constexpr inline bool 
# 3169
is_array_v< _Tp [_Num]>  = true; 
# 3171
template< class _Tp> constexpr inline bool 
# 3172
is_pointer_v = (is_pointer< _Tp> ::value); 
# 3173
template< class _Tp> constexpr inline bool 
# 3174
is_lvalue_reference_v = false; 
# 3175
template< class _Tp> constexpr inline bool 
# 3176
is_lvalue_reference_v< _Tp &>  = true; 
# 3177
template< class _Tp> constexpr inline bool 
# 3178
is_rvalue_reference_v = false; 
# 3179
template< class _Tp> constexpr inline bool 
# 3180
is_rvalue_reference_v< _Tp &&>  = true; 
# 3181
template< class _Tp> constexpr inline bool 
# 3182
is_member_object_pointer_v = (is_member_object_pointer< _Tp> ::value); 
# 3184
template< class _Tp> constexpr inline bool 
# 3185
is_member_function_pointer_v = (is_member_function_pointer< _Tp> ::value); 
# 3187
template< class _Tp> constexpr inline bool 
# 3188
is_enum_v = __is_enum(_Tp); 
# 3189
template< class _Tp> constexpr inline bool 
# 3190
is_union_v = __is_union(_Tp); 
# 3191
template< class _Tp> constexpr inline bool 
# 3192
is_class_v = __is_class(_Tp); 
# 3193
template< class _Tp> constexpr inline bool 
# 3194
is_function_v = (is_function< _Tp> ::value); 
# 3195
template< class _Tp> constexpr inline bool 
# 3196
is_reference_v = false; 
# 3197
template< class _Tp> constexpr inline bool 
# 3198
is_reference_v< _Tp &>  = true; 
# 3199
template< class _Tp> constexpr inline bool 
# 3200
is_reference_v< _Tp &&>  = true; 
# 3201
template< class _Tp> constexpr inline bool 
# 3202
is_arithmetic_v = (is_arithmetic< _Tp> ::value); 
# 3203
template< class _Tp> constexpr inline bool 
# 3204
is_fundamental_v = (is_fundamental< _Tp> ::value); 
# 3205
template< class _Tp> constexpr inline bool 
# 3206
is_object_v = (is_object< _Tp> ::value); 
# 3207
template< class _Tp> constexpr inline bool 
# 3208
is_scalar_v = (is_scalar< _Tp> ::value); 
# 3209
template< class _Tp> constexpr inline bool 
# 3210
is_compound_v = (is_compound< _Tp> ::value); 
# 3211
template< class _Tp> constexpr inline bool 
# 3212
is_member_pointer_v = (is_member_pointer< _Tp> ::value); 
# 3213
template< class _Tp> constexpr inline bool 
# 3214
is_const_v = false; 
# 3215
template< class _Tp> constexpr inline bool 
# 3216
is_const_v< const _Tp>  = true; 
# 3217
template< class _Tp> constexpr inline bool 
# 3218
is_volatile_v = false; 
# 3219
template< class _Tp> constexpr inline bool 
# 3220
is_volatile_v< volatile _Tp>  = true; 
# 3222
template< class _Tp> constexpr inline bool 
# 3223
is_trivial_v = __is_trivial(_Tp); 
# 3224
template< class _Tp> constexpr inline bool 
# 3225
is_trivially_copyable_v = __is_trivially_copyable(_Tp); 
# 3226
template< class _Tp> constexpr inline bool 
# 3227
is_standard_layout_v = __is_standard_layout(_Tp); 
# 3228
template< class _Tp> constexpr inline bool 
# 3230
is_pod_v = __is_pod(_Tp); 
# 3231
template< class _Tp> constexpr inline bool 
# 3233
is_literal_type_v = __is_literal_type(_Tp); 
# 3234
template< class _Tp> constexpr inline bool 
# 3235
is_empty_v = __is_empty(_Tp); 
# 3236
template< class _Tp> constexpr inline bool 
# 3237
is_polymorphic_v = __is_polymorphic(_Tp); 
# 3238
template< class _Tp> constexpr inline bool 
# 3239
is_abstract_v = __is_abstract(_Tp); 
# 3240
template< class _Tp> constexpr inline bool 
# 3241
is_final_v = __is_final(_Tp); 
# 3243
template< class _Tp> constexpr inline bool 
# 3244
is_signed_v = (is_signed< _Tp> ::value); 
# 3245
template< class _Tp> constexpr inline bool 
# 3246
is_unsigned_v = (is_unsigned< _Tp> ::value); 
# 3248
template< class _Tp, class ..._Args> constexpr inline bool 
# 3249
is_constructible_v = __is_constructible(_Tp, _Args...); 
# 3250
template< class _Tp> constexpr inline bool 
# 3251
is_default_constructible_v = __is_constructible(_Tp); 
# 3252
template< class _Tp> constexpr inline bool 
# 3253
is_copy_constructible_v = __is_constructible(_Tp, __add_lval_ref_t< const _Tp> ); 
# 3255
template< class _Tp> constexpr inline bool 
# 3256
is_move_constructible_v = __is_constructible(_Tp, __add_rval_ref_t< _Tp> ); 
# 3259
template< class _Tp, class _Up> constexpr inline bool 
# 3260
is_assignable_v = __is_assignable(_Tp, _Up); 
# 3261
template< class _Tp> constexpr inline bool 
# 3262
is_copy_assignable_v = __is_assignable(__add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> ); 
# 3264
template< class _Tp> constexpr inline bool 
# 3265
is_move_assignable_v = __is_assignable(__add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> ); 
# 3268
template< class _Tp> constexpr inline bool 
# 3269
is_destructible_v = (is_destructible< _Tp> ::value); 
# 3271
template< class _Tp, class ..._Args> constexpr inline bool 
# 3272
is_trivially_constructible_v = __is_trivially_constructible(_Tp, _Args...); 
# 3274
template< class _Tp> constexpr inline bool 
# 3275
is_trivially_default_constructible_v = __is_trivially_constructible(_Tp); 
# 3277
template< class _Tp> constexpr inline bool 
# 3278
is_trivially_copy_constructible_v = __is_trivially_constructible(_Tp, __add_lval_ref_t< const _Tp> ); 
# 3280
template< class _Tp> constexpr inline bool 
# 3281
is_trivially_move_constructible_v = __is_trivially_constructible(_Tp, __add_rval_ref_t< _Tp> ); 
# 3284
template< class _Tp, class _Up> constexpr inline bool 
# 3285
is_trivially_assignable_v = __is_trivially_assignable(_Tp, _Up); 
# 3287
template< class _Tp> constexpr inline bool 
# 3288
is_trivially_copy_assignable_v = __is_trivially_assignable(__add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> ); 
# 3291
template< class _Tp> constexpr inline bool 
# 3292
is_trivially_move_assignable_v = __is_trivially_assignable(__add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> ); 
# 3295
template< class _Tp> constexpr inline bool 
# 3296
is_trivially_destructible_v = (is_trivially_destructible< _Tp> ::value); 
# 3298
template< class _Tp, class ..._Args> constexpr inline bool 
# 3299
is_nothrow_constructible_v = __is_nothrow_constructible(_Tp, _Args...); 
# 3301
template< class _Tp> constexpr inline bool 
# 3302
is_nothrow_default_constructible_v = __is_nothrow_constructible(_Tp); 
# 3304
template< class _Tp> constexpr inline bool 
# 3305
is_nothrow_copy_constructible_v = __is_nothrow_constructible(_Tp, __add_lval_ref_t< const _Tp> ); 
# 3307
template< class _Tp> constexpr inline bool 
# 3308
is_nothrow_move_constructible_v = __is_nothrow_constructible(_Tp, __add_rval_ref_t< _Tp> ); 
# 3311
template< class _Tp, class _Up> constexpr inline bool 
# 3312
is_nothrow_assignable_v = __is_nothrow_assignable(_Tp, _Up); 
# 3314
template< class _Tp> constexpr inline bool 
# 3315
is_nothrow_copy_assignable_v = __is_nothrow_assignable(__add_lval_ref_t< _Tp> , __add_lval_ref_t< const _Tp> ); 
# 3318
template< class _Tp> constexpr inline bool 
# 3319
is_nothrow_move_assignable_v = __is_nothrow_assignable(__add_lval_ref_t< _Tp> , __add_rval_ref_t< _Tp> ); 
# 3322
template< class _Tp> constexpr inline bool 
# 3323
is_nothrow_destructible_v = (is_nothrow_destructible< _Tp> ::value); 
# 3326
template< class _Tp> constexpr inline bool 
# 3327
has_virtual_destructor_v = __has_virtual_destructor(_Tp); 
# 3330
template< class _Tp> constexpr inline size_t 
# 3331
alignment_of_v = (alignment_of< _Tp> ::value); 
# 3333
template< class _Tp> constexpr inline size_t 
# 3334
rank_v = (0); 
# 3335
template< class _Tp, size_t _Size> constexpr inline size_t 
# 3336
rank_v< _Tp [_Size]>  = 1 + rank_v< _Tp> ; 
# 3337
template< class _Tp> constexpr inline size_t 
# 3338
rank_v< _Tp []>  = 1 + rank_v< _Tp> ; 
# 3340
template< class _Tp, unsigned _Idx = 0U> constexpr inline size_t 
# 3341
extent_v = (0); 
# 3342
template< class _Tp, size_t _Size> constexpr inline size_t 
# 3343
extent_v< _Tp [_Size], 0>  = _Size; 
# 3344
template< class _Tp, unsigned _Idx, size_t _Size> constexpr inline size_t 
# 3345
extent_v< _Tp [_Size], _Idx>  = extent_v< _Tp, _Idx - (1)> ; 
# 3346
template< class _Tp> constexpr inline size_t 
# 3347
extent_v< _Tp [], 0>  = (0); 
# 3348
template< class _Tp, unsigned _Idx> constexpr inline size_t 
# 3349
extent_v< _Tp [], _Idx>  = extent_v< _Tp, _Idx - (1)> ; 
# 3352
template< class _Tp, class _Up> constexpr inline bool 
# 3353
is_same_v = __is_same(_Tp, _Up); 
# 3360 "/usr/include/c++/13/type_traits" 3
template< class _Base, class _Derived> constexpr inline bool 
# 3361
is_base_of_v = __is_base_of(_Base, _Derived); 
# 3363
template< class _From, class _To> constexpr inline bool 
# 3364
is_convertible_v = __is_convertible(_From, _To); 
# 3369
template< class _Fn, class ..._Args> constexpr inline bool 
# 3370
is_invocable_v = (is_invocable< _Fn, _Args...> ::value); 
# 3371
template< class _Fn, class ..._Args> constexpr inline bool 
# 3372
is_nothrow_invocable_v = (is_nothrow_invocable< _Fn, _Args...> ::value); 
# 3374
template< class _Ret, class _Fn, class ..._Args> constexpr inline bool 
# 3375
is_invocable_r_v = (is_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3377
template< class _Ret, class _Fn, class ..._Args> constexpr inline bool 
# 3378
is_nothrow_invocable_r_v = (is_nothrow_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3386
template< class _Tp> 
# 3387
struct has_unique_object_representations : public bool_constant< __has_unique_object_representations(remove_cv_t< remove_all_extents_t< _Tp> > )>  { 
# 3392
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 3394
}; 
# 3397
template< class _Tp> constexpr inline bool 
# 3398
has_unique_object_representations_v = (has_unique_object_representations< _Tp> ::value); 
# 3406
template< class _Tp> 
# 3407
struct is_aggregate : public bool_constant< __is_aggregate(remove_cv_t< _Tp> )>  { 
# 3409
}; 
# 3415
template< class _Tp> constexpr inline bool 
# 3416
is_aggregate_v = __is_aggregate(remove_cv_t< _Tp> ); 
# 3834 "/usr/include/c++/13/type_traits" 3
}
# 40 "/usr/include/c++/13/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 49
template< class _Tp> constexpr _Tp *
# 51
__addressof(_Tp &__r) noexcept 
# 52
{ return __builtin_addressof(__r); } 
# 67
template< class _Tp> 
# 68
[[__nodiscard__]] constexpr _Tp &&
# 70
forward(typename remove_reference< _Tp> ::type &__t) noexcept 
# 71
{ return static_cast< _Tp &&>(__t); } 
# 79
template< class _Tp> 
# 80
[[__nodiscard__]] constexpr _Tp &&
# 82
forward(typename remove_reference< _Tp> ::type &&__t) noexcept 
# 83
{ 
# 84
static_assert((!std::template is_lvalue_reference< _Tp> ::value), "std::forward must not be used to convert an rvalue to an lvalue");
# 86
return static_cast< _Tp &&>(__t); 
# 87
} 
# 94
template< class _Tp> 
# 95
[[__nodiscard__]] constexpr typename remove_reference< _Tp> ::type &&
# 97
move(_Tp &&__t) noexcept 
# 98
{ return static_cast< typename remove_reference< _Tp> ::type &&>(__t); } 
# 101
template< class _Tp> 
# 102
struct __move_if_noexcept_cond : public __and_< __not_< is_nothrow_move_constructible< _Tp> > , is_copy_constructible< _Tp> > ::type { 
# 104
}; 
# 114
template< class _Tp> 
# 115
[[__nodiscard__]] constexpr __conditional_t< __move_if_noexcept_cond< _Tp> ::value, const _Tp &, _Tp &&>  
# 118
move_if_noexcept(_Tp &__x) noexcept 
# 119
{ return std::move(__x); } 
# 135
template< class _Tp> 
# 136
[[__nodiscard__]] constexpr _Tp *
# 138
addressof(_Tp &__r) noexcept 
# 139
{ return std::__addressof(__r); } 
# 143
template < typename _Tp >
    const _Tp * addressof ( const _Tp && ) = delete;
# 147
template< class _Tp, class _Up = _Tp> inline _Tp 
# 150
__exchange(_Tp &__obj, _Up &&__new_val) 
# 151
{ 
# 152
_Tp __old_val = std::move(__obj); 
# 153
__obj = std::forward< _Up> (__new_val); 
# 154
return __old_val; 
# 155
} 
# 179 "/usr/include/c++/13/bits/move.h" 3
template< class _Tp> inline typename enable_if< __and_< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> > ::value> ::type 
# 189
swap(_Tp &__a, _Tp &__b) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value) 
# 192
{ 
# 197
_Tp __tmp = std::move(__a); 
# 198
__a = std::move(__b); 
# 199
__b = std::move(__tmp); 
# 200
} 
# 205
template< class _Tp, size_t _Nm> inline typename enable_if< __is_swappable< _Tp> ::value> ::type 
# 213
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value) 
# 215
{ 
# 216
for (size_t __n = (0); __n < _Nm; ++__n) { 
# 217
swap(__a[__n], __b[__n]); }  
# 218
} 
# 222
}
# 43 "/usr/include/c++/13/bits/utility.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
template< class _Tp> struct tuple_size; 
# 55
template< class _Tp, class 
# 56
_Up = typename remove_cv< _Tp> ::type, class 
# 57
 = typename enable_if< is_same< _Tp, _Up> ::value> ::type, size_t 
# 58
 = tuple_size< _Tp> ::value> using __enable_if_has_tuple_size = _Tp; 
# 61
template< class _Tp> 
# 62
struct tuple_size< const __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 63
}; 
# 65
template< class _Tp> 
# 66
struct tuple_size< volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 67
}; 
# 69
template< class _Tp> 
# 70
struct tuple_size< const volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 71
}; 
# 74
template< class _Tp> constexpr inline size_t 
# 75
tuple_size_v = (tuple_size< _Tp> ::value); 
# 79
template< size_t __i, class _Tp> struct tuple_element; 
# 83
template< size_t __i, class _Tp> using __tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 86
template< size_t __i, class _Tp> 
# 87
struct tuple_element< __i, const _Tp>  { 
# 89
using type = const __tuple_element_t< __i, _Tp> ; 
# 90
}; 
# 92
template< size_t __i, class _Tp> 
# 93
struct tuple_element< __i, volatile _Tp>  { 
# 95
using type = volatile __tuple_element_t< __i, _Tp> ; 
# 96
}; 
# 98
template< size_t __i, class _Tp> 
# 99
struct tuple_element< __i, const volatile _Tp>  { 
# 101
using type = const volatile __tuple_element_t< __i, _Tp> ; 
# 102
}; 
# 108
template< class _Tp, class ..._Types> constexpr size_t 
# 110
__find_uniq_type_in_pack() 
# 111
{ 
# 112
constexpr size_t __sz = sizeof...(_Types); 
# 113
constexpr bool __found[__sz] = {__is_same(_Tp, _Types)...}; 
# 114
size_t __n = __sz; 
# 115
for (size_t __i = (0); __i < __sz; ++__i) 
# 116
{ 
# 117
if (__found[__i]) 
# 118
{ 
# 119
if (__n < __sz) { 
# 120
return __sz; }  
# 121
__n = __i; 
# 122
}  
# 123
}  
# 124
return __n; 
# 125
} 
# 134
template< size_t __i, class _Tp> using tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 140
template< size_t ..._Indexes> struct _Index_tuple { }; 
# 143
template< size_t _Num> 
# 144
struct _Build_index_tuple { 
# 154 "/usr/include/c++/13/bits/utility.h" 3
using __type = _Index_tuple< __integer_pack(_Num)...> ; 
# 156
}; 
# 163
template< class _Tp, _Tp ..._Idx> 
# 164
struct integer_sequence { 
# 169
typedef _Tp value_type; 
# 170
static constexpr size_t size() noexcept { return sizeof...(_Idx); } 
# 171
}; 
# 174
template< class _Tp, _Tp _Num> using make_integer_sequence = integer_sequence< _Tp, __integer_pack((_Tp)_Num)...> ; 
# 183
template< size_t ..._Idx> using index_sequence = integer_sequence< unsigned long, _Idx...> ; 
# 187
template< size_t _Num> using make_index_sequence = make_integer_sequence< unsigned long, _Num> ; 
# 191
template< class ..._Types> using index_sequence_for = make_index_sequence< sizeof...(_Types)> ; 
# 196
struct in_place_t { 
# 197
explicit in_place_t() = default;
# 198
}; 
# 200
constexpr inline in_place_t in_place{}; 
# 202
template< class _Tp> struct in_place_type_t { 
# 204
explicit in_place_type_t() = default;
# 205
}; 
# 207
template< class _Tp> constexpr inline in_place_type_t< _Tp>  
# 208
in_place_type{}; 
# 210
template< size_t _Idx> struct in_place_index_t { 
# 212
explicit in_place_index_t() = default;
# 213
}; 
# 215
template< size_t _Idx> constexpr inline in_place_index_t< _Idx>  
# 216
in_place_index{}; 
# 218
template< class > constexpr inline bool 
# 219
__is_in_place_type_v = false; 
# 221
template< class _Tp> constexpr inline bool 
# 222
__is_in_place_type_v< in_place_type_t< _Tp> >  = true; 
# 224
template< class _Tp> using __is_in_place_type = bool_constant< __is_in_place_type_v< _Tp> > ; 
# 230
template< size_t _Np, class ..._Types> 
# 231
struct _Nth_type { 
# 232
}; 
# 234
template< class _Tp0, class ..._Rest> 
# 235
struct _Nth_type< 0, _Tp0, _Rest...>  { 
# 236
using type = _Tp0; }; 
# 238
template< class _Tp0, class _Tp1, class ..._Rest> 
# 239
struct _Nth_type< 1, _Tp0, _Tp1, _Rest...>  { 
# 240
using type = _Tp1; }; 
# 242
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 243
struct _Nth_type< 2, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 244
using type = _Tp2; }; 
# 246
template< size_t _Np, class _Tp0, class _Tp1, class _Tp2, class ...
# 247
_Rest> 
# 251
struct _Nth_type< _Np, _Tp0, _Tp1, _Tp2, _Rest...>  : public std::_Nth_type< _Np - (3), _Rest...>  { 
# 253
}; 
# 256
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 257
struct _Nth_type< 0, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 258
using type = _Tp0; }; 
# 260
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 261
struct _Nth_type< 1, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 262
using type = _Tp1; }; 
# 270
}
# 69 "/usr/include/c++/13/bits/stl_pair.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 80
struct piecewise_construct_t { explicit piecewise_construct_t() = default;}; 
# 83
constexpr inline piecewise_construct_t piecewise_construct = piecewise_construct_t(); 
# 89
template< class ...> class tuple; 
# 92
template< size_t ...> struct _Index_tuple; 
# 101
template< bool , class _T1, class _T2> 
# 102
struct _PCC { 
# 104
template< class _U1, class _U2> static constexpr bool 
# 105
_ConstructiblePair() 
# 106
{ 
# 107
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, const _U2 &> > ::value; 
# 109
} 
# 111
template< class _U1, class _U2> static constexpr bool 
# 112
_ImplicitlyConvertiblePair() 
# 113
{ 
# 114
return __and_< is_convertible< const _U1 &, _T1> , is_convertible< const _U2 &, _T2> > ::value; 
# 116
} 
# 118
template< class _U1, class _U2> static constexpr bool 
# 119
_MoveConstructiblePair() 
# 120
{ 
# 121
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, _U2 &&> > ::value; 
# 123
} 
# 125
template< class _U1, class _U2> static constexpr bool 
# 126
_ImplicitlyMoveConvertiblePair() 
# 127
{ 
# 128
return __and_< is_convertible< _U1 &&, _T1> , is_convertible< _U2 &&, _T2> > ::value; 
# 130
} 
# 131
}; 
# 133
template< class _T1, class _T2> 
# 134
struct _PCC< false, _T1, _T2>  { 
# 136
template< class _U1, class _U2> static constexpr bool 
# 137
_ConstructiblePair() 
# 138
{ 
# 139
return false; 
# 140
} 
# 142
template< class _U1, class _U2> static constexpr bool 
# 143
_ImplicitlyConvertiblePair() 
# 144
{ 
# 145
return false; 
# 146
} 
# 148
template< class _U1, class _U2> static constexpr bool 
# 149
_MoveConstructiblePair() 
# 150
{ 
# 151
return false; 
# 152
} 
# 154
template< class _U1, class _U2> static constexpr bool 
# 155
_ImplicitlyMoveConvertiblePair() 
# 156
{ 
# 157
return false; 
# 158
} 
# 159
}; 
# 163
template< class _U1, class _U2> class __pair_base { 
# 166
template< class _T1, class _T2> friend struct pair; 
# 167
__pair_base() = default;
# 168
~__pair_base() = default;
# 169
__pair_base(const __pair_base &) = default;
# 170
__pair_base &operator=(const __pair_base &) = delete;
# 172
}; 
# 186
template< class _T1, class _T2> 
# 187
struct pair : public __pair_base< _T1, _T2>  { 
# 190
typedef _T1 first_type; 
# 191
typedef _T2 second_type; 
# 193
_T1 first; 
# 194
_T2 second; 
# 197
constexpr pair(const pair &) = default;
# 198
constexpr pair(pair &&) = default;
# 200
template< class ..._Args1, class ..._Args2> pair(std::piecewise_construct_t, tuple< _Args1...> , tuple< _Args2...> ); 
# 206
void swap(pair &__p) noexcept(__and_< __is_nothrow_swappable< _T1> , __is_nothrow_swappable< _T2> > ::value) 
# 209
{ 
# 210
using std::swap;
# 211
swap(first, __p.first); 
# 212
swap(second, __p.second); 
# 213
} 
# 235 "/usr/include/c++/13/bits/stl_pair.h" 3
private: template< class ..._Args1, std::size_t ..._Indexes1, class ...
# 236
_Args2, std::size_t ..._Indexes2> 
# 235
pair(tuple< _Args1...>  &, tuple< _Args2...>  &, _Index_tuple< _Indexes1...> , _Index_tuple< _Indexes2...> ); 
# 531 "/usr/include/c++/13/bits/stl_pair.h" 3
public: 
# 525
template< class _U1 = _T1, class 
# 526
_U2 = _T2, typename enable_if< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > ::value, bool> ::type 
# 530
 = true> constexpr 
# 531
pair() : first(), second() 
# 532
{ } 
# 534
template< class _U1 = _T1, class 
# 535
_U2 = _T2, typename enable_if< __and_< is_default_constructible< _U1> , is_default_constructible< _U2> , __not_< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > > > ::value, bool> ::type 
# 542
 = false> constexpr explicit 
# 543
pair() : first(), second() 
# 544
{ } 
# 548
using _PCCP = _PCC< true, _T1, _T2> ; 
# 552
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 557
 = true> constexpr 
# 558
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 559
{ } 
# 562
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 567
 = false> constexpr explicit 
# 568
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 569
{ } 
# 573
template< class _U1, class _U2> using _PCCFP = _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ; 
# 579
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 584
 = true> constexpr 
# 585
pair(const pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 587
{ ; } 
# 589
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 594
 = false> constexpr explicit 
# 595
pair(const pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 597
{ ; } 
# 613 "/usr/include/c++/13/bits/stl_pair.h" 3
private: struct __zero_as_null_pointer_constant { 
# 615
__zero_as_null_pointer_constant(int (__zero_as_null_pointer_constant::*)) 
# 616
{ } 
# 617
template < typename _Tp,
   typename = __enable_if_t < is_null_pointer < _Tp > :: value > >
 __zero_as_null_pointer_constant ( _Tp ) = delete;
# 620
}; 
# 636
public: 
# 627
template< class _U1, std::__enable_if_t< __and_< __not_< is_reference< _U1> > , is_pointer< _T2> , is_constructible< _T1, _U1> , __not_< is_constructible< _T1, const _U1 &> > , is_convertible< _U1, _T1> > ::value, bool>  
# 633
 = true> constexpr 
# 636
pair(_U1 &&__x, __zero_as_null_pointer_constant, ...) : first(std::forward< _U1> (__x)), second(nullptr) 
# 638
{ ; } 
# 640
template< class _U1, std::__enable_if_t< __and_< __not_< is_reference< _U1> > , is_pointer< _T2> , is_constructible< _T1, _U1> , __not_< is_constructible< _T1, const _U1 &> > , __not_< is_convertible< _U1, _T1> > > ::value, bool>  
# 646
 = false> constexpr explicit 
# 649
pair(_U1 &&__x, __zero_as_null_pointer_constant, ...) : first(std::forward< _U1> (__x)), second(nullptr) 
# 651
{ ; } 
# 653
template< class _U2, std::__enable_if_t< __and_< is_pointer< _T1> , __not_< is_reference< _U2> > , is_constructible< _T2, _U2> , __not_< is_constructible< _T2, const _U2 &> > , is_convertible< _U2, _T2> > ::value, bool>  
# 659
 = true> constexpr 
# 662
pair(__zero_as_null_pointer_constant, _U2 &&__y, ...) : first(nullptr), second(std::forward< _U2> (__y)) 
# 664
{ ; } 
# 666
template< class _U2, std::__enable_if_t< __and_< is_pointer< _T1> , __not_< is_reference< _U2> > , is_constructible< _T2, _U2> , __not_< is_constructible< _T2, const _U2 &> > , __not_< is_convertible< _U2, _T2> > > ::value, bool>  
# 672
 = false> constexpr explicit 
# 675
pair(__zero_as_null_pointer_constant, _U2 &&__y, ...) : first(nullptr), second(std::forward< _U2> (__y)) 
# 677
{ ; } 
# 681
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 686
 = true> constexpr 
# 687
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 689
{ ; } 
# 691
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 696
 = false> constexpr explicit 
# 697
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 699
{ ; } 
# 702
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 707
 = true> constexpr 
# 708
pair(pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 711
{ ; } 
# 713
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 718
 = false> constexpr explicit 
# 719
pair(pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 722
{ ; } 
# 727
pair &operator=(std::__conditional_t< __and_< is_copy_assignable< _T1> , is_copy_assignable< _T2> > ::value, const pair &, const std::__nonesuch &>  
# 729
__p) 
# 730
{ 
# 731
(first) = (__p.first); 
# 732
(second) = (__p.second); 
# 733
return *this; 
# 734
} 
# 737
pair &operator=(std::__conditional_t< __and_< is_move_assignable< _T1> , is_move_assignable< _T2> > ::value, pair &&, std::__nonesuch &&>  
# 739
__p) noexcept(__and_< is_nothrow_move_assignable< _T1> , is_nothrow_move_assignable< _T2> > ::value) 
# 742
{ 
# 743
(first) = std::forward< first_type> ((__p.first)); 
# 744
(second) = std::forward< second_type> ((__p.second)); 
# 745
return *this; 
# 746
} 
# 748
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, const _U1 &> , is_assignable< _T2 &, const _U2 &> > ::value, pair &> ::type 
# 752
operator=(const pair< _U1, _U2>  &__p) 
# 753
{ 
# 754
(first) = (__p.first); 
# 755
(second) = (__p.second); 
# 756
return *this; 
# 757
} 
# 759
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, _U1 &&> , is_assignable< _T2 &, _U2 &&> > ::value, pair &> ::type 
# 763
operator=(pair< _U1, _U2>  &&__p) 
# 764
{ 
# 765
(first) = std::forward< _U1> ((__p.first)); 
# 766
(second) = std::forward< _U2> ((__p.second)); 
# 767
return *this; 
# 768
} 
# 801 "/usr/include/c++/13/bits/stl_pair.h" 3
}; 
# 806
template< class _T1, class _T2> pair(_T1, _T2)->pair< _T1, _T2> ; 
# 810
template< class _T1, class _T2> constexpr bool 
# 812
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 813
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 833 "/usr/include/c++/13/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr bool 
# 835
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 836
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 837
} 
# 840
template< class _T1, class _T2> constexpr bool 
# 842
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 843
{ return !(__x == __y); } 
# 846
template< class _T1, class _T2> constexpr bool 
# 848
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 849
{ return __y < __x; } 
# 852
template< class _T1, class _T2> constexpr bool 
# 854
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 855
{ return !(__y < __x); } 
# 858
template< class _T1, class _T2> constexpr bool 
# 860
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 861
{ return !(__x < __y); } 
# 870
template< class _T1, class _T2> inline typename enable_if< __and_< __is_swappable< _T1> , __is_swappable< _T2> > ::value> ::type 
# 879
swap(pair< _T1, _T2>  &__x, pair< _T1, _T2>  &__y) noexcept(noexcept(__x.swap(__y))) 
# 881
{ __x.swap(__y); } 
# 893 "/usr/include/c++/13/bits/stl_pair.h" 3
template < typename _T1, typename _T2 >
    typename enable_if < ! __and_ < __is_swappable < _T1 >,
          __is_swappable < _T2 > > :: value > :: type
    swap ( pair < _T1, _T2 > &, pair < _T1, _T2 > & ) = delete;
# 919
template< class _T1, class _T2> constexpr pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  
# 922
make_pair(_T1 &&__x, _T2 &&__y) 
# 923
{ 
# 924
typedef typename __decay_and_strip< _T1> ::__type __ds_type1; 
# 925
typedef typename __decay_and_strip< _T2> ::__type __ds_type2; 
# 926
typedef pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  __pair_type; 
# 927
return __pair_type(std::forward< _T1> (__x), std::forward< _T2> (__y)); 
# 928
} 
# 942 "/usr/include/c++/13/bits/stl_pair.h" 3
template< class _T1, class _T2> 
# 943
struct __is_tuple_like_impl< pair< _T1, _T2> >  : public true_type { 
# 944
}; 
# 948
template< class _Tp1, class _Tp2> 
# 949
struct tuple_size< pair< _Tp1, _Tp2> >  : public integral_constant< unsigned long, 2UL>  { 
# 950
}; 
# 953
template< class _Tp1, class _Tp2> 
# 954
struct tuple_element< 0, pair< _Tp1, _Tp2> >  { 
# 955
typedef _Tp1 type; }; 
# 958
template< class _Tp1, class _Tp2> 
# 959
struct tuple_element< 1, pair< _Tp1, _Tp2> >  { 
# 960
typedef _Tp2 type; }; 
# 963
template< class _Tp1, class _Tp2> constexpr inline size_t 
# 964
tuple_size_v< pair< _Tp1, _Tp2> >  = (2); 
# 966
template< class _Tp1, class _Tp2> constexpr inline size_t 
# 967
tuple_size_v< const pair< _Tp1, _Tp2> >  = (2); 
# 969
template< class _Tp> constexpr inline bool 
# 970
__is_pair = false; 
# 972
template< class _Tp, class _Up> constexpr inline bool 
# 973
__is_pair< pair< _Tp, _Up> >  = true; 
# 977
template< size_t _Int> struct __pair_get; 
# 981
template<> struct __pair_get< 0UL>  { 
# 983
template< class _Tp1, class _Tp2> static constexpr _Tp1 &
# 985
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 986
{ return __pair.first; } 
# 988
template< class _Tp1, class _Tp2> static constexpr _Tp1 &&
# 990
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 991
{ return std::forward< _Tp1> ((__pair.first)); } 
# 993
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &
# 995
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 996
{ return __pair.first; } 
# 998
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &&
# 1000
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 1001
{ return std::forward< const _Tp1> ((__pair.first)); } 
# 1002
}; 
# 1005
template<> struct __pair_get< 1UL>  { 
# 1007
template< class _Tp1, class _Tp2> static constexpr _Tp2 &
# 1009
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 1010
{ return __pair.second; } 
# 1012
template< class _Tp1, class _Tp2> static constexpr _Tp2 &&
# 1014
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 1015
{ return std::forward< _Tp2> ((__pair.second)); } 
# 1017
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &
# 1019
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 1020
{ return __pair.second; } 
# 1022
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &&
# 1024
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 1025
{ return std::forward< const _Tp2> ((__pair.second)); } 
# 1026
}; 
# 1033
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 1035
get(pair< _Tp1, _Tp2>  &__in) noexcept 
# 1036
{ return __pair_get< _Int> ::__get(__in); } 
# 1038
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 1040
get(pair< _Tp1, _Tp2>  &&__in) noexcept 
# 1041
{ return __pair_get< _Int> ::__move_get(std::move(__in)); } 
# 1043
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 1045
get(const pair< _Tp1, _Tp2>  &__in) noexcept 
# 1046
{ return __pair_get< _Int> ::__const_get(__in); } 
# 1048
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 1050
get(const pair< _Tp1, _Tp2>  &&__in) noexcept 
# 1051
{ return __pair_get< _Int> ::__const_move_get(std::move(__in)); } 
# 1057
template< class _Tp, class _Up> constexpr _Tp &
# 1059
get(pair< _Tp, _Up>  &__p) noexcept 
# 1060
{ return __p.first; } 
# 1062
template< class _Tp, class _Up> constexpr const _Tp &
# 1064
get(const pair< _Tp, _Up>  &__p) noexcept 
# 1065
{ return __p.first; } 
# 1067
template< class _Tp, class _Up> constexpr _Tp &&
# 1069
get(pair< _Tp, _Up>  &&__p) noexcept 
# 1070
{ return std::move((__p.first)); } 
# 1072
template< class _Tp, class _Up> constexpr const _Tp &&
# 1074
get(const pair< _Tp, _Up>  &&__p) noexcept 
# 1075
{ return std::move((__p.first)); } 
# 1077
template< class _Tp, class _Up> constexpr _Tp &
# 1079
get(pair< _Up, _Tp>  &__p) noexcept 
# 1080
{ return __p.second; } 
# 1082
template< class _Tp, class _Up> constexpr const _Tp &
# 1084
get(const pair< _Up, _Tp>  &__p) noexcept 
# 1085
{ return __p.second; } 
# 1087
template< class _Tp, class _Up> constexpr _Tp &&
# 1089
get(pair< _Up, _Tp>  &&__p) noexcept 
# 1090
{ return std::move((__p.second)); } 
# 1092
template< class _Tp, class _Up> constexpr const _Tp &&
# 1094
get(const pair< _Up, _Tp>  &&__p) noexcept 
# 1095
{ return std::move((__p.second)); } 
# 1119 "/usr/include/c++/13/bits/stl_pair.h" 3
}
# 74 "/usr/include/c++/13/bits/stl_iterator_base_types.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 93
struct input_iterator_tag { }; 
# 96
struct output_iterator_tag { }; 
# 99
struct forward_iterator_tag : public input_iterator_tag { }; 
# 103
struct bidirectional_iterator_tag : public forward_iterator_tag { }; 
# 107
struct random_access_iterator_tag : public bidirectional_iterator_tag { }; 
# 125
template< class _Category, class _Tp, class _Distance = ptrdiff_t, class 
# 126
_Pointer = _Tp *, class _Reference = _Tp &> 
# 127
struct iterator { 
# 130
typedef _Category iterator_category; 
# 132
typedef _Tp value_type; 
# 134
typedef _Distance difference_type; 
# 136
typedef _Pointer pointer; 
# 138
typedef _Reference reference; 
# 139
}; 
# 149
template< class _Iterator> struct iterator_traits; 
# 155
template< class _Iterator, class  = __void_t< > > 
# 156
struct __iterator_traits { }; 
# 160
template< class _Iterator> 
# 161
struct __iterator_traits< _Iterator, __void_t< typename _Iterator::iterator_category, typename _Iterator::value_type, typename _Iterator::difference_type, typename _Iterator::pointer, typename _Iterator::reference> >  { 
# 168
typedef typename _Iterator::iterator_category iterator_category; 
# 169
typedef typename _Iterator::value_type value_type; 
# 170
typedef typename _Iterator::difference_type difference_type; 
# 171
typedef typename _Iterator::pointer pointer; 
# 172
typedef typename _Iterator::reference reference; 
# 173
}; 
# 176
template< class _Iterator> 
# 177
struct iterator_traits : public __iterator_traits< _Iterator>  { 
# 178
}; 
# 209 "/usr/include/c++/13/bits/stl_iterator_base_types.h" 3
template< class _Tp> 
# 210
struct iterator_traits< _Tp *>  { 
# 212
typedef random_access_iterator_tag iterator_category; 
# 213
typedef _Tp value_type; 
# 214
typedef ptrdiff_t difference_type; 
# 215
typedef _Tp *pointer; 
# 216
typedef _Tp &reference; 
# 217
}; 
# 220
template< class _Tp> 
# 221
struct iterator_traits< const _Tp *>  { 
# 223
typedef random_access_iterator_tag iterator_category; 
# 224
typedef _Tp value_type; 
# 225
typedef ptrdiff_t difference_type; 
# 226
typedef const _Tp *pointer; 
# 227
typedef const _Tp &reference; 
# 228
}; 
# 235
template< class _Iter> 
# 236
__attribute((__always_inline__)) constexpr typename iterator_traits< _Iter> ::iterator_category 
# 239
__iterator_category(const _Iter &) 
# 240
{ return typename iterator_traits< _Iter> ::iterator_category(); } 
# 245
template< class _Iter> using __iter_category_t = typename iterator_traits< _Iter> ::iterator_category; 
# 249
template< class _InIter> using _RequireInputIter = __enable_if_t< is_convertible< __iter_category_t< _InIter> , input_iterator_tag> ::value> ; 
# 254
template< class _It, class 
# 255
_Cat = __iter_category_t< _It> > 
# 256
struct __is_random_access_iter : public is_base_of< random_access_iterator_tag, _Cat>  { 
# 259
typedef is_base_of< std::random_access_iterator_tag, _Cat>  _Base; 
# 260
enum { __value = is_base_of< std::random_access_iterator_tag, _Cat> ::value}; 
# 261
}; 
# 270 "/usr/include/c++/13/bits/stl_iterator_base_types.h" 3
}
# 68 "/usr/include/c++/13/bits/stl_iterator_base_funcs.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 74
template< class > struct _List_iterator; 
# 75
template< class > struct _List_const_iterator; 
# 78
template< class _InputIterator> constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 81
__distance(_InputIterator __first, _InputIterator __last, input_iterator_tag) 
# 83
{ 
# 87
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 88
while (__first != __last) 
# 89
{ 
# 90
++__first; 
# 91
++__n; 
# 92
}  
# 93
return __n; 
# 94
} 
# 96
template< class _RandomAccessIterator> 
# 97
__attribute((__always_inline__)) constexpr typename iterator_traits< _RandomAccessIterator> ::difference_type 
# 100
__distance(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag) 
# 102
{ 
# 106
return __last - __first; 
# 107
} 
# 111
template< class _Tp> ptrdiff_t __distance(_List_iterator< _Tp> , _List_iterator< _Tp> , input_iterator_tag); 
# 117
template< class _Tp> ptrdiff_t __distance(_List_const_iterator< _Tp> , _List_const_iterator< _Tp> , input_iterator_tag); 
# 126
template < typename _OutputIterator >
    void
    __distance ( _OutputIterator, _OutputIterator, output_iterator_tag ) = delete;
# 144
template< class _InputIterator> 
# 145
[[__nodiscard__]] __attribute((__always_inline__)) constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 148
distance(_InputIterator __first, _InputIterator __last) 
# 149
{ 
# 151
return std::__distance(__first, __last, std::__iterator_category(__first)); 
# 153
} 
# 155
template< class _InputIterator, class _Distance> constexpr void 
# 157
__advance(_InputIterator &__i, _Distance __n, input_iterator_tag) 
# 158
{ 
# 161
do { if (std::__is_constant_evaluated() && (!((bool)(__n >= 0)))) { __builtin_unreachable(); }  } while (false); 
# 162
while (__n--) { 
# 163
++__i; }  
# 164
} 
# 166
template< class _BidirectionalIterator, class _Distance> constexpr void 
# 168
__advance(_BidirectionalIterator &__i, _Distance __n, bidirectional_iterator_tag) 
# 170
{ 
# 174
if (__n > 0) { 
# 175
while (__n--) { 
# 176
++__i; }  } else { 
# 178
while (__n++) { 
# 179
--__i; }  }  
# 180
} 
# 182
template< class _RandomAccessIterator, class _Distance> constexpr void 
# 184
__advance(_RandomAccessIterator &__i, _Distance __n, random_access_iterator_tag) 
# 186
{ 
# 190
if (__builtin_constant_p(__n) && (__n == 1)) { 
# 191
++__i; } else { 
# 192
if (__builtin_constant_p(__n) && (__n == (-1))) { 
# 193
--__i; } else { 
# 195
__i += __n; }  }  
# 196
} 
# 200
template < typename _OutputIterator, typename _Distance >
    void
    __advance ( _OutputIterator &, _Distance, output_iterator_tag ) = delete;
# 217
template< class _InputIterator, class _Distance> 
# 218
__attribute((__always_inline__)) constexpr void 
# 220
advance(_InputIterator &__i, _Distance __n) 
# 221
{ 
# 223
typename iterator_traits< _InputIterator> ::difference_type __d = __n; 
# 224
std::__advance(__i, __d, std::__iterator_category(__i)); 
# 225
} 
# 229
template< class _InputIterator> 
# 230
[[__nodiscard__]] [[__gnu__::__always_inline__]] constexpr _InputIterator 
# 232
next(_InputIterator __x, typename iterator_traits< _InputIterator> ::difference_type 
# 233
__n = 1) 
# 234
{ 
# 237
std::advance(__x, __n); 
# 238
return __x; 
# 239
} 
# 241
template< class _BidirectionalIterator> 
# 242
[[__nodiscard__]] [[__gnu__::__always_inline__]] constexpr _BidirectionalIterator 
# 244
prev(_BidirectionalIterator __x, typename iterator_traits< _BidirectionalIterator> ::difference_type 
# 245
__n = 1) 
# 246
{ 
# 250
std::advance(__x, -__n); 
# 251
return __x; 
# 252
} 
# 257
}
# 49 "/usr/include/c++/13/bits/ptr_traits.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 55
class __undefined; 
# 59
template< class _Tp> 
# 60
struct __get_first_arg { 
# 61
using type = __undefined; }; 
# 63
template< template< class , class ...>  class _SomeTemplate, class _Tp, class ...
# 64
_Types> 
# 65
struct __get_first_arg< _SomeTemplate< _Tp, _Types...> >  { 
# 66
using type = _Tp; }; 
# 70
template< class _Tp, class _Up> 
# 71
struct __replace_first_arg { 
# 72
}; 
# 74
template< template< class , class ...>  class _SomeTemplate, class _Up, class 
# 75
_Tp, class ..._Types> 
# 76
struct __replace_first_arg< _SomeTemplate< _Tp, _Types...> , _Up>  { 
# 77
using type = _SomeTemplate< _Up, _Types...> ; }; 
# 80
template< class _Ptr, class  = void> 
# 81
struct __ptr_traits_elem : public __get_first_arg< _Ptr>  { 
# 82
}; 
# 90
template< class _Ptr> 
# 91
struct __ptr_traits_elem< _Ptr, __void_t< typename _Ptr::element_type> >  { 
# 92
using type = typename _Ptr::element_type; }; 
# 95
template< class _Ptr> using __ptr_traits_elem_t = typename __ptr_traits_elem< _Ptr> ::type; 
# 101
template< class _Ptr, class _Elt, bool  = is_void< _Elt> ::value> 
# 102
struct __ptr_traits_ptr_to { 
# 104
using pointer = _Ptr; 
# 105
using element_type = _Elt; 
# 114
static pointer pointer_to(element_type &__r) 
# 120
{ return pointer::pointer_to(__r); } 
# 121
}; 
# 124
template< class _Ptr, class _Elt> 
# 125
struct __ptr_traits_ptr_to< _Ptr, _Elt, true>  { 
# 126
}; 
# 129
template< class _Tp> 
# 130
struct __ptr_traits_ptr_to< _Tp *, _Tp, false>  { 
# 132
using pointer = _Tp *; 
# 133
using element_type = _Tp; 
# 141
static pointer pointer_to(element_type &__r) noexcept 
# 142
{ return std::addressof(__r); } 
# 143
}; 
# 145
template< class _Ptr, class _Elt> 
# 146
struct __ptr_traits_impl : public __ptr_traits_ptr_to< _Ptr, _Elt>  { 
# 150
private: 
# 149
template< class _Tp> using __diff_t = typename _Tp::difference_type; 
# 152
template< class _Tp, class _Up> using __rebind = __type_identity< typename _Tp::template rebind< _Up> > ; 
# 157
public: using pointer = _Ptr; 
# 160
using element_type = _Elt; 
# 163
using difference_type = std::__detected_or_t< std::ptrdiff_t, __diff_t, _Ptr> ; 
# 166
template< class _Up> using rebind = typename std::__detected_or_t< __replace_first_arg< _Ptr, _Up> , __rebind, _Ptr, _Up> ::type; 
# 169
}; 
# 173
template< class _Ptr> 
# 174
struct __ptr_traits_impl< _Ptr, __undefined>  { 
# 175
}; 
# 183
template< class _Ptr> 
# 184
struct pointer_traits : public __ptr_traits_impl< _Ptr, __ptr_traits_elem_t< _Ptr> >  { 
# 185
}; 
# 193
template< class _Tp> 
# 194
struct pointer_traits< _Tp *>  : public __ptr_traits_ptr_to< _Tp *, _Tp>  { 
# 197
typedef _Tp *pointer; 
# 199
typedef _Tp element_type; 
# 201
typedef std::ptrdiff_t difference_type; 
# 203
template< class _Up> using rebind = _Up *; 
# 204
}; 
# 207
template< class _Ptr, class _Tp> using __ptr_rebind = typename pointer_traits< _Ptr> ::template rebind< _Tp> ; 
# 210
template< class _Tp> constexpr _Tp *
# 212
__to_address(_Tp *__ptr) noexcept 
# 213
{ 
# 214
static_assert((!std::template is_function< _Tp> ::value), "not a function pointer");
# 215
return __ptr; 
# 216
} 
# 219
template< class _Ptr> constexpr typename pointer_traits< _Ptr> ::element_type *
# 221
__to_address(const _Ptr &__ptr) 
# 222
{ return std::__to_address(__ptr.operator->()); } 
# 267 "/usr/include/c++/13/bits/ptr_traits.h" 3
}
# 88 "/usr/include/c++/13/bits/stl_iterator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 113 "/usr/include/c++/13/bits/stl_iterator.h" 3
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
# 135
template< class _Iterator> 
# 136
class reverse_iterator : public iterator< typename iterator_traits< _Iterator> ::iterator_category, typename iterator_traits< _Iterator> ::value_type, typename iterator_traits< _Iterator> ::difference_type, typename iterator_traits< _Iterator> ::pointer, typename iterator_traits< _Iterator> ::reference>  { 
# 143
template< class _Iter> friend class reverse_iterator; 
# 155 "/usr/include/c++/13/bits/stl_iterator.h" 3
protected: _Iterator current; 
# 157
typedef iterator_traits< _Iterator>  __traits_type; 
# 160
public: typedef _Iterator iterator_type; 
# 161
typedef typename iterator_traits< _Iterator> ::pointer pointer; 
# 163
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 164
typedef typename iterator_traits< _Iterator> ::reference reference; 
# 186 "/usr/include/c++/13/bits/stl_iterator.h" 3
constexpr reverse_iterator() noexcept(noexcept((_Iterator()))) : current() 
# 189
{ } 
# 195
constexpr explicit reverse_iterator(iterator_type __x) noexcept(noexcept(((_Iterator)__x))) : current(__x) 
# 198
{ } 
# 204
constexpr reverse_iterator(const reverse_iterator &__x) noexcept(noexcept(((_Iterator)(__x.current)))) : current(__x.current) 
# 207
{ } 
# 210
reverse_iterator &operator=(const reverse_iterator &) = default;
# 217
template< class _Iter> constexpr 
# 222
reverse_iterator(const reverse_iterator< _Iter>  &__x) noexcept(noexcept(((_Iterator)(__x.current)))) : current((__x.current)) 
# 225
{ } 
# 228
template< class _Iter> constexpr reverse_iterator &
# 235
operator=(const reverse_iterator< _Iter>  &__x) noexcept(noexcept(((current) = (__x.current)))) 
# 237
{ 
# 238
(current) = (__x.current); 
# 239
return *this; 
# 240
} 
# 246
[[__nodiscard__]] constexpr iterator_type 
# 248
base() const noexcept(noexcept(((_Iterator)(current)))) 
# 250
{ return current; } 
# 262
[[__nodiscard__]] constexpr reference 
# 264
operator*() const 
# 265
{ 
# 266
_Iterator __tmp = current; 
# 267
return *(--__tmp); 
# 268
} 
# 275
[[__nodiscard__]] constexpr pointer 
# 277
operator->() const 
# 282
{ 
# 285
_Iterator __tmp = current; 
# 286
--__tmp; 
# 287
return _S_to_pointer(__tmp); 
# 288
} 
# 296
constexpr reverse_iterator &operator++() 
# 297
{ 
# 298
--(current); 
# 299
return *this; 
# 300
} 
# 308
constexpr reverse_iterator operator++(int) 
# 309
{ 
# 310
reverse_iterator __tmp = *this; 
# 311
--(current); 
# 312
return __tmp; 
# 313
} 
# 321
constexpr reverse_iterator &operator--() 
# 322
{ 
# 323
++(current); 
# 324
return *this; 
# 325
} 
# 333
constexpr reverse_iterator operator--(int) 
# 334
{ 
# 335
reverse_iterator __tmp = *this; 
# 336
++(current); 
# 337
return __tmp; 
# 338
} 
# 345
[[__nodiscard__]] constexpr reverse_iterator 
# 347
operator+(difference_type __n) const 
# 348
{ return ((reverse_iterator)((current) - __n)); } 
# 357
constexpr reverse_iterator &operator+=(difference_type __n) 
# 358
{ 
# 359
(current) -= __n; 
# 360
return *this; 
# 361
} 
# 368
[[__nodiscard__]] constexpr reverse_iterator 
# 370
operator-(difference_type __n) const 
# 371
{ return ((reverse_iterator)((current) + __n)); } 
# 380
constexpr reverse_iterator &operator-=(difference_type __n) 
# 381
{ 
# 382
(current) += __n; 
# 383
return *this; 
# 384
} 
# 391
[[__nodiscard__]] constexpr reference 
# 393
operator[](difference_type __n) const 
# 394
{ return *((*this) + __n); } 
# 425 "/usr/include/c++/13/bits/stl_iterator.h" 3
private: 
# 423
template< class _Tp> static constexpr _Tp *
# 425
_S_to_pointer(_Tp *__p) 
# 426
{ return __p; } 
# 428
template< class _Tp> static constexpr pointer 
# 430
_S_to_pointer(_Tp __t) 
# 431
{ return __t.operator->(); } 
# 432
}; 
# 445
template< class _Iterator> 
# 446
[[__nodiscard__]] constexpr bool 
# 448
operator==(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 449
__y) 
# 450
{ return __x.base() == __y.base(); } 
# 452
template< class _Iterator> 
# 453
[[__nodiscard__]] constexpr bool 
# 455
operator<(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 456
__y) 
# 457
{ return __y.base() < __x.base(); } 
# 459
template< class _Iterator> 
# 460
[[__nodiscard__]] constexpr bool 
# 462
operator!=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 463
__y) 
# 464
{ return !(__x == __y); } 
# 466
template< class _Iterator> 
# 467
[[__nodiscard__]] constexpr bool 
# 469
operator>(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 470
__y) 
# 471
{ return __y < __x; } 
# 473
template< class _Iterator> 
# 474
[[__nodiscard__]] constexpr bool 
# 476
operator<=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 477
__y) 
# 478
{ return !(__y < __x); } 
# 480
template< class _Iterator> 
# 481
[[__nodiscard__]] constexpr bool 
# 483
operator>=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 484
__y) 
# 485
{ return !(__x < __y); } 
# 490
template< class _IteratorL, class _IteratorR> 
# 491
[[__nodiscard__]] constexpr bool 
# 493
operator==(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 494
__y) 
# 495
{ return __x.base() == __y.base(); } 
# 497
template< class _IteratorL, class _IteratorR> 
# 498
[[__nodiscard__]] constexpr bool 
# 500
operator<(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 501
__y) 
# 502
{ return __x.base() > __y.base(); } 
# 504
template< class _IteratorL, class _IteratorR> 
# 505
[[__nodiscard__]] constexpr bool 
# 507
operator!=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 508
__y) 
# 509
{ return __x.base() != __y.base(); } 
# 511
template< class _IteratorL, class _IteratorR> 
# 512
[[__nodiscard__]] constexpr bool 
# 514
operator>(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 515
__y) 
# 516
{ return __x.base() < __y.base(); } 
# 518
template< class _IteratorL, class _IteratorR> constexpr bool 
# 520
operator<=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 521
__y) 
# 522
{ return __x.base() >= __y.base(); } 
# 524
template< class _IteratorL, class _IteratorR> 
# 525
[[__nodiscard__]] constexpr bool 
# 527
operator>=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 528
__y) 
# 529
{ return __x.base() <= __y.base(); } 
# 622 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> 
# 623
[[__nodiscard__]] constexpr auto 
# 625
operator-(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 626
__y)->__decltype((__y.base() - __x.base())) 
# 628
{ return __y.base() - __x.base(); } 
# 631
template< class _Iterator> 
# 632
[[__nodiscard__]] constexpr reverse_iterator< _Iterator>  
# 634
operator+(typename reverse_iterator< _Iterator> ::difference_type __n, const reverse_iterator< _Iterator>  &
# 635
__x) 
# 636
{ return ((reverse_iterator< _Iterator> )(__x.base() - __n)); } 
# 640
template< class _Iterator> constexpr reverse_iterator< _Iterator>  
# 642
__make_reverse_iterator(_Iterator __i) 
# 643
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 651
template< class _Iterator> 
# 652
[[__nodiscard__]] constexpr reverse_iterator< _Iterator>  
# 654
make_reverse_iterator(_Iterator __i) 
# 655
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 666 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 669
__niter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__niter_base(__it.base())))) 
# 671
{ return __make_reverse_iterator(__niter_base(__it.base())); } 
# 673
template< class _Iterator> 
# 674
struct __is_move_iterator< reverse_iterator< _Iterator> >  : public std::__is_move_iterator< _Iterator>  { 
# 676
}; 
# 678
template< class _Iterator> auto 
# 681
__miter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__miter_base(__it.base())))) 
# 683
{ return __make_reverse_iterator(__miter_base(__it.base())); } 
# 697
template< class _Container> 
# 698
class back_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 702
protected: _Container *container; 
# 706
public: typedef _Container container_type; 
# 713
explicit back_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 714
{ } 
# 737 "/usr/include/c++/13/bits/stl_iterator.h" 3
back_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 738
{ 
# 739
(container)->push_back(__value); 
# 740
return *this; 
# 741
} 
# 745
back_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 746
{ 
# 747
(container)->push_back(std::move(__value)); 
# 748
return *this; 
# 749
} 
# 753
[[__nodiscard__]] back_insert_iterator &
# 755
operator*() 
# 756
{ return *this; } 
# 761
back_insert_iterator &operator++() 
# 762
{ return *this; } 
# 767
back_insert_iterator operator++(int) 
# 768
{ return *this; } 
# 769
}; 
# 782
template< class _Container> 
# 783
[[__nodiscard__]] inline back_insert_iterator< _Container>  
# 785
back_inserter(_Container &__x) 
# 786
{ return ((back_insert_iterator< _Container> )(__x)); } 
# 798
template< class _Container> 
# 799
class front_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 803
protected: _Container *container; 
# 807
public: typedef _Container container_type; 
# 814
explicit front_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 815
{ } 
# 838 "/usr/include/c++/13/bits/stl_iterator.h" 3
front_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 839
{ 
# 840
(container)->push_front(__value); 
# 841
return *this; 
# 842
} 
# 846
front_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 847
{ 
# 848
(container)->push_front(std::move(__value)); 
# 849
return *this; 
# 850
} 
# 854
[[__nodiscard__]] front_insert_iterator &
# 856
operator*() 
# 857
{ return *this; } 
# 862
front_insert_iterator &operator++() 
# 863
{ return *this; } 
# 868
front_insert_iterator operator++(int) 
# 869
{ return *this; } 
# 870
}; 
# 883
template< class _Container> 
# 884
[[__nodiscard__]] inline front_insert_iterator< _Container>  
# 886
front_inserter(_Container &__x) 
# 887
{ return ((front_insert_iterator< _Container> )(__x)); } 
# 903
template< class _Container> 
# 904
class insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 910
typedef typename _Container::iterator _Iter; 
# 913
protected: _Container *container; 
# 914
_Iter iter; 
# 918
public: typedef _Container container_type; 
# 929
insert_iterator(_Container &__x, _Iter __i) : container(std::__addressof(__x)), iter(__i) 
# 930
{ } 
# 966 "/usr/include/c++/13/bits/stl_iterator.h" 3
insert_iterator &operator=(const typename _Container::value_type &__value) 
# 967
{ 
# 968
(iter) = (container)->insert(iter, __value); 
# 969
++(iter); 
# 970
return *this; 
# 971
} 
# 975
insert_iterator &operator=(typename _Container::value_type &&__value) 
# 976
{ 
# 977
(iter) = (container)->insert(iter, std::move(__value)); 
# 978
++(iter); 
# 979
return *this; 
# 980
} 
# 984
[[__nodiscard__]] insert_iterator &
# 986
operator*() 
# 987
{ return *this; } 
# 992
insert_iterator &operator++() 
# 993
{ return *this; } 
# 998
insert_iterator &operator++(int) 
# 999
{ return *this; } 
# 1000
}; 
# 1002
#pragma GCC diagnostic pop
# 1023 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _Container> 
# 1024
[[__nodiscard__]] inline insert_iterator< _Container>  
# 1026
inserter(_Container &__x, typename _Container::iterator __i) 
# 1027
{ return insert_iterator< _Container> (__x, __i); } 
# 1033
}
# 1035
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 1046
template< class _Iterator, class _Container> 
# 1047
class __normal_iterator { 
# 1050
protected: _Iterator _M_current; 
# 1052
typedef std::iterator_traits< _Iterator>  __traits_type; 
# 1055
template< class _Iter> using __convertible_from = std::__enable_if_t< std::is_convertible< _Iter, _Iterator> ::value> ; 
# 1061
public: typedef _Iterator iterator_type; 
# 1062
typedef typename std::iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 1063
typedef typename std::iterator_traits< _Iterator> ::value_type value_type; 
# 1064
typedef typename std::iterator_traits< _Iterator> ::difference_type difference_type; 
# 1065
typedef typename std::iterator_traits< _Iterator> ::reference reference; 
# 1066
typedef typename std::iterator_traits< _Iterator> ::pointer pointer; 
# 1072
constexpr __normal_iterator() noexcept : _M_current(_Iterator()) 
# 1073
{ } 
# 1076
explicit __normal_iterator(const _Iterator &__i) noexcept : _M_current(__i) 
# 1077
{ } 
# 1081
template< class _Iter, class  = __convertible_from< _Iter> > 
# 1083
__normal_iterator(const __normal_iterator< _Iter, _Container>  &__i) noexcept : _M_current(__i.base()) 
# 1094 "/usr/include/c++/13/bits/stl_iterator.h" 3
{ } 
# 1099
reference operator*() const noexcept 
# 1100
{ return *(_M_current); } 
# 1104
pointer operator->() const noexcept 
# 1105
{ return _M_current; } 
# 1109
__normal_iterator &operator++() noexcept 
# 1110
{ 
# 1111
++(_M_current); 
# 1112
return *this; 
# 1113
} 
# 1117
__normal_iterator operator++(int) noexcept 
# 1118
{ return ((__normal_iterator)((_M_current)++)); } 
# 1123
__normal_iterator &operator--() noexcept 
# 1124
{ 
# 1125
--(_M_current); 
# 1126
return *this; 
# 1127
} 
# 1131
__normal_iterator operator--(int) noexcept 
# 1132
{ return ((__normal_iterator)((_M_current)--)); } 
# 1137
reference operator[](difference_type __n) const noexcept 
# 1138
{ return (_M_current)[__n]; } 
# 1142
__normal_iterator &operator+=(difference_type __n) noexcept 
# 1143
{ (_M_current) += __n; return *this; } 
# 1147
__normal_iterator operator+(difference_type __n) const noexcept 
# 1148
{ return ((__normal_iterator)((_M_current) + __n)); } 
# 1152
__normal_iterator &operator-=(difference_type __n) noexcept 
# 1153
{ (_M_current) -= __n; return *this; } 
# 1157
__normal_iterator operator-(difference_type __n) const noexcept 
# 1158
{ return ((__normal_iterator)((_M_current) - __n)); } 
# 1162
const _Iterator &base() const noexcept 
# 1163
{ return _M_current; } 
# 1164
}; 
# 1214 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR, class _Container> 
# 1215
[[__nodiscard__]] inline bool 
# 1217
operator==(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1218
__rhs) noexcept 
# 1220
{ return __lhs.base() == __rhs.base(); } 
# 1222
template< class _Iterator, class _Container> 
# 1223
[[__nodiscard__]] inline bool 
# 1225
operator==(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1226
__rhs) noexcept 
# 1228
{ return __lhs.base() == __rhs.base(); } 
# 1230
template< class _IteratorL, class _IteratorR, class _Container> 
# 1231
[[__nodiscard__]] inline bool 
# 1233
operator!=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1234
__rhs) noexcept 
# 1236
{ return __lhs.base() != __rhs.base(); } 
# 1238
template< class _Iterator, class _Container> 
# 1239
[[__nodiscard__]] inline bool 
# 1241
operator!=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1242
__rhs) noexcept 
# 1244
{ return __lhs.base() != __rhs.base(); } 
# 1247
template< class _IteratorL, class _IteratorR, class _Container> 
# 1248
[[__nodiscard__]] inline bool 
# 1250
operator<(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1251
__rhs) noexcept 
# 1253
{ return __lhs.base() < __rhs.base(); } 
# 1255
template< class _Iterator, class _Container> 
# 1256
[[__nodiscard__]] inline bool 
# 1258
operator<(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1259
__rhs) noexcept 
# 1261
{ return __lhs.base() < __rhs.base(); } 
# 1263
template< class _IteratorL, class _IteratorR, class _Container> 
# 1264
[[__nodiscard__]] inline bool 
# 1266
operator>(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1267
__rhs) noexcept 
# 1269
{ return __lhs.base() > __rhs.base(); } 
# 1271
template< class _Iterator, class _Container> 
# 1272
[[__nodiscard__]] inline bool 
# 1274
operator>(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1275
__rhs) noexcept 
# 1277
{ return __lhs.base() > __rhs.base(); } 
# 1279
template< class _IteratorL, class _IteratorR, class _Container> 
# 1280
[[__nodiscard__]] inline bool 
# 1282
operator<=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1283
__rhs) noexcept 
# 1285
{ return __lhs.base() <= __rhs.base(); } 
# 1287
template< class _Iterator, class _Container> 
# 1288
[[__nodiscard__]] inline bool 
# 1290
operator<=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1291
__rhs) noexcept 
# 1293
{ return __lhs.base() <= __rhs.base(); } 
# 1295
template< class _IteratorL, class _IteratorR, class _Container> 
# 1296
[[__nodiscard__]] inline bool 
# 1298
operator>=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1299
__rhs) noexcept 
# 1301
{ return __lhs.base() >= __rhs.base(); } 
# 1303
template< class _Iterator, class _Container> 
# 1304
[[__nodiscard__]] inline bool 
# 1306
operator>=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1307
__rhs) noexcept 
# 1309
{ return __lhs.base() >= __rhs.base(); } 
# 1316
template< class _IteratorL, class _IteratorR, class _Container> 
# 1319
[[__nodiscard__]] inline auto 
# 1321
operator-(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1322
__rhs) noexcept->__decltype((__lhs.base() - __rhs.base())) 
# 1329
{ return __lhs.base() - __rhs.base(); } 
# 1331
template< class _Iterator, class _Container> 
# 1332
[[__nodiscard__]] inline typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1334
operator-(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1335
__rhs) noexcept 
# 1337
{ return __lhs.base() - __rhs.base(); } 
# 1339
template< class _Iterator, class _Container> 
# 1340
[[__nodiscard__]] inline __normal_iterator< _Iterator, _Container>  
# 1342
operator+(typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1343
__n, const __normal_iterator< _Iterator, _Container>  &__i) noexcept 
# 1345
{ return ((__normal_iterator< _Iterator, _Container> )(__i.base() + __n)); } 
# 1348
}
# 1350
namespace std __attribute((__visibility__("default"))) { 
# 1354
template< class _Iterator, class _Container> _Iterator 
# 1357
__niter_base(__gnu_cxx::__normal_iterator< _Iterator, _Container>  __it) noexcept(std::template is_nothrow_copy_constructible< _Iterator> ::value) 
# 1359
{ return __it.base(); } 
# 1366
template< class _Iterator, class _Container> constexpr auto 
# 1368
__to_address(const __gnu_cxx::__normal_iterator< _Iterator, _Container>  &
# 1369
__it) noexcept->__decltype((std::__to_address(__it.base()))) 
# 1371
{ return std::__to_address(__it.base()); } 
# 1421 "/usr/include/c++/13/bits/stl_iterator.h" 3
namespace __detail { 
# 1437 "/usr/include/c++/13/bits/stl_iterator.h" 3
}
# 1448
template< class _Iterator> 
# 1449
class move_iterator { 
# 1454
_Iterator _M_current; 
# 1456
using __traits_type = iterator_traits< _Iterator> ; 
# 1458
using __base_ref = typename iterator_traits< _Iterator> ::reference; 
# 1461
template< class _Iter2> friend class move_iterator; 
# 1488 "/usr/include/c++/13/bits/stl_iterator.h" 3
public: using iterator_type = _Iterator; 
# 1501 "/usr/include/c++/13/bits/stl_iterator.h" 3
typedef typename iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 1502
typedef typename iterator_traits< _Iterator> ::value_type value_type; 
# 1503
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 1505
typedef _Iterator pointer; 
# 1508
using reference = __conditional_t< is_reference< __base_ref> ::value, typename remove_reference< __base_ref> ::type &&, __base_ref> ; 
# 1515
constexpr move_iterator() : _M_current() 
# 1516
{ } 
# 1519
constexpr explicit move_iterator(iterator_type __i) : _M_current(std::move(__i)) 
# 1520
{ } 
# 1522
template< class _Iter> constexpr 
# 1527
move_iterator(const move_iterator< _Iter>  &__i) : _M_current((__i._M_current)) 
# 1528
{ } 
# 1530
template< class _Iter> constexpr move_iterator &
# 1536
operator=(const move_iterator< _Iter>  &__i) 
# 1537
{ 
# 1538
(_M_current) = (__i._M_current); 
# 1539
return *this; 
# 1540
} 
# 1543
[[__nodiscard__]] constexpr iterator_type 
# 1545
base() const 
# 1546
{ return _M_current; } 
# 1559 "/usr/include/c++/13/bits/stl_iterator.h" 3
[[__nodiscard__]] constexpr reference 
# 1561
operator*() const 
# 1565
{ return static_cast< reference>(*(_M_current)); } 
# 1568
[[__nodiscard__]] constexpr pointer 
# 1570
operator->() const 
# 1571
{ return _M_current; } 
# 1574
constexpr move_iterator &operator++() 
# 1575
{ 
# 1576
++(_M_current); 
# 1577
return *this; 
# 1578
} 
# 1581
constexpr move_iterator operator++(int) 
# 1582
{ 
# 1583
move_iterator __tmp = *this; 
# 1584
++(_M_current); 
# 1585
return __tmp; 
# 1586
} 
# 1595
constexpr move_iterator &operator--() 
# 1596
{ 
# 1597
--(_M_current); 
# 1598
return *this; 
# 1599
} 
# 1602
constexpr move_iterator operator--(int) 
# 1603
{ 
# 1604
move_iterator __tmp = *this; 
# 1605
--(_M_current); 
# 1606
return __tmp; 
# 1607
} 
# 1609
[[__nodiscard__]] constexpr move_iterator 
# 1611
operator+(difference_type __n) const 
# 1612
{ return ((move_iterator)((_M_current) + __n)); } 
# 1615
constexpr move_iterator &operator+=(difference_type __n) 
# 1616
{ 
# 1617
(_M_current) += __n; 
# 1618
return *this; 
# 1619
} 
# 1621
[[__nodiscard__]] constexpr move_iterator 
# 1623
operator-(difference_type __n) const 
# 1624
{ return ((move_iterator)((_M_current) - __n)); } 
# 1627
constexpr move_iterator &operator-=(difference_type __n) 
# 1628
{ 
# 1629
(_M_current) -= __n; 
# 1630
return *this; 
# 1631
} 
# 1633
[[__nodiscard__]] constexpr reference 
# 1635
operator[](difference_type __n) const 
# 1639
{ return std::move((_M_current)[__n]); } 
# 1673 "/usr/include/c++/13/bits/stl_iterator.h" 3
}; 
# 1675
template< class _IteratorL, class _IteratorR> 
# 1676
[[__nodiscard__]] constexpr bool 
# 1678
operator==(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1679
__y) 
# 1683
{ return __x.base() == __y.base(); } 
# 1694 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> 
# 1695
[[__nodiscard__]] constexpr bool 
# 1697
operator!=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1698
__y) 
# 1699
{ return !(__x == __y); } 
# 1702
template< class _IteratorL, class _IteratorR> 
# 1703
[[__nodiscard__]] constexpr bool 
# 1705
operator<(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1706
__y) 
# 1710
{ return __x.base() < __y.base(); } 
# 1712
template< class _IteratorL, class _IteratorR> 
# 1713
[[__nodiscard__]] constexpr bool 
# 1715
operator<=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1716
__y) 
# 1720
{ return !(__y < __x); } 
# 1722
template< class _IteratorL, class _IteratorR> 
# 1723
[[__nodiscard__]] constexpr bool 
# 1725
operator>(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1726
__y) 
# 1730
{ return __y < __x; } 
# 1732
template< class _IteratorL, class _IteratorR> 
# 1733
[[__nodiscard__]] constexpr bool 
# 1735
operator>=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1736
__y) 
# 1740
{ return !(__x < __y); } 
# 1745
template< class _Iterator> 
# 1746
[[__nodiscard__]] constexpr bool 
# 1748
operator==(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1749
__y) 
# 1750
{ return __x.base() == __y.base(); } 
# 1760 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _Iterator> 
# 1761
[[__nodiscard__]] constexpr bool 
# 1763
operator!=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1764
__y) 
# 1765
{ return !(__x == __y); } 
# 1767
template< class _Iterator> 
# 1768
[[__nodiscard__]] constexpr bool 
# 1770
operator<(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1771
__y) 
# 1772
{ return __x.base() < __y.base(); } 
# 1774
template< class _Iterator> 
# 1775
[[__nodiscard__]] constexpr bool 
# 1777
operator<=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1778
__y) 
# 1779
{ return !(__y < __x); } 
# 1781
template< class _Iterator> 
# 1782
[[__nodiscard__]] constexpr bool 
# 1784
operator>(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1785
__y) 
# 1786
{ return __y < __x; } 
# 1788
template< class _Iterator> 
# 1789
[[__nodiscard__]] constexpr bool 
# 1791
operator>=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1792
__y) 
# 1793
{ return !(__x < __y); } 
# 1797
template< class _IteratorL, class _IteratorR> 
# 1798
[[__nodiscard__]] constexpr auto 
# 1800
operator-(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1801
__y)->__decltype((__x.base() - __y.base())) 
# 1803
{ return __x.base() - __y.base(); } 
# 1805
template< class _Iterator> 
# 1806
[[__nodiscard__]] constexpr move_iterator< _Iterator>  
# 1808
operator+(typename move_iterator< _Iterator> ::difference_type __n, const move_iterator< _Iterator>  &
# 1809
__x) 
# 1810
{ return __x + __n; } 
# 1812
template< class _Iterator> 
# 1813
[[__nodiscard__]] constexpr move_iterator< _Iterator>  
# 1815
make_move_iterator(_Iterator __i) 
# 1816
{ return ((move_iterator< _Iterator> )(std::move(__i))); } 
# 1818
template< class _Iterator, class _ReturnType = __conditional_t< __move_if_noexcept_cond< typename iterator_traits< _Iterator> ::value_type> ::value, _Iterator, move_iterator< _Iterator> > > constexpr _ReturnType 
# 1823
__make_move_if_noexcept_iterator(_Iterator __i) 
# 1824
{ return (_ReturnType)__i; } 
# 1828
template< class _Tp, class _ReturnType = __conditional_t< __move_if_noexcept_cond< _Tp> ::value, const _Tp *, move_iterator< _Tp *> > > constexpr _ReturnType 
# 1832
__make_move_if_noexcept_iterator(_Tp *__i) 
# 1833
{ return (_ReturnType)__i; } 
# 2951 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 2954
__niter_base(move_iterator< _Iterator>  __it)->__decltype((make_move_iterator(__niter_base(__it.base())))) 
# 2956
{ return make_move_iterator(__niter_base(__it.base())); } 
# 2958
template< class _Iterator> 
# 2959
struct __is_move_iterator< move_iterator< _Iterator> >  { 
# 2961
enum { __value = 1}; 
# 2962
typedef __true_type __type; 
# 2963
}; 
# 2965
template< class _Iterator> auto 
# 2968
__miter_base(move_iterator< _Iterator>  __it)->__decltype((__miter_base(__it.base()))) 
# 2970
{ return __miter_base(__it.base()); } 
# 2983 "/usr/include/c++/13/bits/stl_iterator.h" 3
template< class _InputIterator> using __iter_key_t = remove_const_t< typename iterator_traits< _InputIterator> ::value_type::first_type> ; 
# 2987
template< class _InputIterator> using __iter_val_t = typename iterator_traits< _InputIterator> ::value_type::second_type; 
# 2991
template< class _T1, class _T2> struct pair; 
# 2994
template< class _InputIterator> using __iter_to_alloc_t = pair< const __iter_key_t< _InputIterator> , __iter_val_t< _InputIterator> > ; 
# 3000
}
# 48 "/usr/include/c++/13/debug/debug.h" 3
namespace std { 
# 50
namespace __debug { }
# 51
}
# 56
namespace __gnu_debug { 
# 58
using namespace std::__debug;
# 60
template< class _Ite, class _Seq, class _Cat> struct _Safe_iterator; 
# 62
}
# 35 "/usr/include/c++/13/bits/predefined_ops.h" 3
namespace __gnu_cxx { 
# 37
namespace __ops { 
# 39
struct _Iter_less_iter { 
# 41
template< class _Iterator1, class _Iterator2> constexpr bool 
# 44
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 45
{ return (*__it1) < (*__it2); } 
# 46
}; 
# 50
constexpr _Iter_less_iter __iter_less_iter() 
# 51
{ return _Iter_less_iter(); } 
# 53
struct _Iter_less_val { 
# 56
constexpr _Iter_less_val() = default;
# 63
explicit _Iter_less_val(_Iter_less_iter) { } 
# 65
template< class _Iterator, class _Value> bool 
# 68
operator()(_Iterator __it, _Value &__val) const 
# 69
{ return (*__it) < __val; } 
# 70
}; 
# 74
inline _Iter_less_val __iter_less_val() 
# 75
{ return _Iter_less_val(); } 
# 79
inline _Iter_less_val __iter_comp_val(_Iter_less_iter) 
# 80
{ return _Iter_less_val(); } 
# 82
struct _Val_less_iter { 
# 85
constexpr _Val_less_iter() = default;
# 92
explicit _Val_less_iter(_Iter_less_iter) { } 
# 94
template< class _Value, class _Iterator> bool 
# 97
operator()(_Value &__val, _Iterator __it) const 
# 98
{ return __val < (*__it); } 
# 99
}; 
# 103
inline _Val_less_iter __val_less_iter() 
# 104
{ return _Val_less_iter(); } 
# 108
inline _Val_less_iter __val_comp_iter(_Iter_less_iter) 
# 109
{ return _Val_less_iter(); } 
# 111
struct _Iter_equal_to_iter { 
# 113
template< class _Iterator1, class _Iterator2> bool 
# 116
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 117
{ return (*__it1) == (*__it2); } 
# 118
}; 
# 122
inline _Iter_equal_to_iter __iter_equal_to_iter() 
# 123
{ return _Iter_equal_to_iter(); } 
# 125
struct _Iter_equal_to_val { 
# 127
template< class _Iterator, class _Value> bool 
# 130
operator()(_Iterator __it, _Value &__val) const 
# 131
{ return (*__it) == __val; } 
# 132
}; 
# 136
inline _Iter_equal_to_val __iter_equal_to_val() 
# 137
{ return _Iter_equal_to_val(); } 
# 141
inline _Iter_equal_to_val __iter_comp_val(_Iter_equal_to_iter) 
# 142
{ return _Iter_equal_to_val(); } 
# 144
template< class _Compare> 
# 145
struct _Iter_comp_iter { 
# 147
_Compare _M_comp; 
# 150
constexpr explicit _Iter_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 152
{ } 
# 154
template< class _Iterator1, class _Iterator2> constexpr bool 
# 157
operator()(_Iterator1 __it1, _Iterator2 __it2) 
# 158
{ return (bool)(_M_comp)(*__it1, *__it2); } 
# 159
}; 
# 161
template< class _Compare> constexpr _Iter_comp_iter< _Compare>  
# 164
__iter_comp_iter(_Compare __comp) 
# 165
{ return ((_Iter_comp_iter< _Compare> )(std::move(__comp))); } 
# 167
template< class _Compare> 
# 168
struct _Iter_comp_val { 
# 170
_Compare _M_comp; 
# 174
explicit _Iter_comp_val(_Compare __comp) : _M_comp(std::move(__comp)) 
# 176
{ } 
# 180
explicit _Iter_comp_val(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 182
{ } 
# 187
explicit _Iter_comp_val(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 189
{ } 
# 192
template< class _Iterator, class _Value> bool 
# 195
operator()(_Iterator __it, _Value &__val) 
# 196
{ return (bool)(_M_comp)(*__it, __val); } 
# 197
}; 
# 199
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 202
__iter_comp_val(_Compare __comp) 
# 203
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 205
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 208
__iter_comp_val(_Iter_comp_iter< _Compare>  __comp) 
# 209
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 211
template< class _Compare> 
# 212
struct _Val_comp_iter { 
# 214
_Compare _M_comp; 
# 218
explicit _Val_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 220
{ } 
# 224
explicit _Val_comp_iter(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 226
{ } 
# 231
explicit _Val_comp_iter(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 233
{ } 
# 236
template< class _Value, class _Iterator> bool 
# 239
operator()(_Value &__val, _Iterator __it) 
# 240
{ return (bool)(_M_comp)(__val, *__it); } 
# 241
}; 
# 243
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 246
__val_comp_iter(_Compare __comp) 
# 247
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 249
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 252
__val_comp_iter(_Iter_comp_iter< _Compare>  __comp) 
# 253
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 255
template< class _Value> 
# 256
struct _Iter_equals_val { 
# 258
_Value &_M_value; 
# 262
explicit _Iter_equals_val(_Value &__value) : _M_value(__value) 
# 264
{ } 
# 266
template< class _Iterator> bool 
# 269
operator()(_Iterator __it) 
# 270
{ return (*__it) == (_M_value); } 
# 271
}; 
# 273
template< class _Value> inline _Iter_equals_val< _Value>  
# 276
__iter_equals_val(_Value &__val) 
# 277
{ return ((_Iter_equals_val< _Value> )(__val)); } 
# 279
template< class _Iterator1> 
# 280
struct _Iter_equals_iter { 
# 282
_Iterator1 _M_it1; 
# 286
explicit _Iter_equals_iter(_Iterator1 __it1) : _M_it1(__it1) 
# 288
{ } 
# 290
template< class _Iterator2> bool 
# 293
operator()(_Iterator2 __it2) 
# 294
{ return (*__it2) == (*(_M_it1)); } 
# 295
}; 
# 297
template< class _Iterator> inline _Iter_equals_iter< _Iterator>  
# 300
__iter_comp_iter(_Iter_equal_to_iter, _Iterator __it) 
# 301
{ return ((_Iter_equals_iter< _Iterator> )(__it)); } 
# 303
template< class _Predicate> 
# 304
struct _Iter_pred { 
# 306
_Predicate _M_pred; 
# 310
explicit _Iter_pred(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 312
{ } 
# 314
template< class _Iterator> bool 
# 317
operator()(_Iterator __it) 
# 318
{ return (bool)(_M_pred)(*__it); } 
# 319
}; 
# 321
template< class _Predicate> inline _Iter_pred< _Predicate>  
# 324
__pred_iter(_Predicate __pred) 
# 325
{ return ((_Iter_pred< _Predicate> )(std::move(__pred))); } 
# 327
template< class _Compare, class _Value> 
# 328
struct _Iter_comp_to_val { 
# 330
_Compare _M_comp; 
# 331
_Value &_M_value; 
# 334
_Iter_comp_to_val(_Compare __comp, _Value &__value) : _M_comp(std::move(__comp)), _M_value(__value) 
# 336
{ } 
# 338
template< class _Iterator> bool 
# 341
operator()(_Iterator __it) 
# 342
{ return (bool)(_M_comp)(*__it, _M_value); } 
# 343
}; 
# 345
template< class _Compare, class _Value> _Iter_comp_to_val< _Compare, _Value>  
# 348
__iter_comp_val(_Compare __comp, _Value &__val) 
# 349
{ 
# 350
return _Iter_comp_to_val< _Compare, _Value> (std::move(__comp), __val); 
# 351
} 
# 353
template< class _Compare, class _Iterator1> 
# 354
struct _Iter_comp_to_iter { 
# 356
_Compare _M_comp; 
# 357
_Iterator1 _M_it1; 
# 360
_Iter_comp_to_iter(_Compare __comp, _Iterator1 __it1) : _M_comp(std::move(__comp)), _M_it1(__it1) 
# 362
{ } 
# 364
template< class _Iterator2> bool 
# 367
operator()(_Iterator2 __it2) 
# 368
{ return (bool)(_M_comp)(*__it2, *(_M_it1)); } 
# 369
}; 
# 371
template< class _Compare, class _Iterator> inline _Iter_comp_to_iter< _Compare, _Iterator>  
# 374
__iter_comp_iter(_Iter_comp_iter< _Compare>  __comp, _Iterator __it) 
# 375
{ 
# 376
return _Iter_comp_to_iter< _Compare, _Iterator> (std::move((__comp._M_comp)), __it); 
# 378
} 
# 380
template< class _Predicate> 
# 381
struct _Iter_negate { 
# 383
_Predicate _M_pred; 
# 387
explicit _Iter_negate(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 389
{ } 
# 391
template< class _Iterator> bool 
# 394
operator()(_Iterator __it) 
# 395
{ return !((bool)(_M_pred)(*__it)); } 
# 396
}; 
# 398
template< class _Predicate> inline _Iter_negate< _Predicate>  
# 401
__negate(_Iter_pred< _Predicate>  __pred) 
# 402
{ return ((_Iter_negate< _Predicate> )(std::move((__pred._M_pred)))); } 
# 404
}
# 405
}
# 55 "/usr/include/c++/13/bit" 3
namespace std __attribute((__visibility__("default"))) { 
# 149 "/usr/include/c++/13/bit" 3
template< class _Tp> constexpr _Tp 
# 151
__rotl(_Tp __x, int __s) noexcept 
# 152
{ 
# 153
constexpr auto _Nd = (__gnu_cxx::__int_traits< _Tp> ::__digits); 
# 154
if constexpr ((_Nd & (_Nd - 1)) == 0) 
# 155
{ 
# 158
constexpr unsigned __uNd = (_Nd); 
# 159
const unsigned __r = __s; 
# 160
return (__x << (__r % __uNd)) | (__x >> ((-__r) % __uNd)); 
# 161
}  
# 162
const int __r = __s % _Nd; 
# 163
if (__r == 0) { 
# 164
return __x; } else { 
# 165
if (__r > 0) { 
# 166
return (__x << __r) | (__x >> ((_Nd - __r) % _Nd)); } else { 
# 168
return (__x >> (-__r)) | (__x << ((_Nd + __r) % _Nd)); }  }  
# 169
} 
# 171
template< class _Tp> constexpr _Tp 
# 173
__rotr(_Tp __x, int __s) noexcept 
# 174
{ 
# 175
constexpr auto _Nd = (__gnu_cxx::__int_traits< _Tp> ::__digits); 
# 176
if constexpr ((_Nd & (_Nd - 1)) == 0) 
# 177
{ 
# 180
constexpr unsigned __uNd = (_Nd); 
# 181
const unsigned __r = __s; 
# 182
return (__x >> (__r % __uNd)) | (__x << ((-__r) % __uNd)); 
# 183
}  
# 184
const int __r = __s % _Nd; 
# 185
if (__r == 0) { 
# 186
return __x; } else { 
# 187
if (__r > 0) { 
# 188
return (__x >> __r) | (__x << ((_Nd - __r) % _Nd)); } else { 
# 190
return (__x << (-__r)) | (__x >> ((_Nd + __r) % _Nd)); }  }  
# 191
} 
# 193
template< class _Tp> constexpr int 
# 195
__countl_zero(_Tp __x) noexcept 
# 196
{ 
# 197
using __gnu_cxx::__int_traits;
# 198
constexpr auto _Nd = (__int_traits< _Tp> ::__digits); 
# 200
if (__x == 0) { 
# 201
return _Nd; }  
# 203
constexpr auto _Nd_ull = __int_traits< unsigned long long> ::__digits; 
# 204
constexpr auto _Nd_ul = __int_traits< unsigned long> ::__digits; 
# 205
constexpr auto _Nd_u = __int_traits< unsigned> ::__digits; 
# 207
if constexpr (_Nd <= _Nd_u) 
# 208
{ 
# 209
constexpr int __diff = (_Nd_u - _Nd); 
# 210
return __builtin_clz(__x) - __diff; 
# 211
} else { 
# 212
if constexpr (_Nd <= _Nd_ul) 
# 213
{ 
# 214
constexpr int __diff = (_Nd_ul - _Nd); 
# 215
return __builtin_clzl(__x) - __diff; 
# 216
} else { 
# 217
if constexpr (_Nd <= _Nd_ull) 
# 218
{ 
# 219
constexpr int __diff = (_Nd_ull - _Nd); 
# 220
return __builtin_clzll(__x) - __diff; 
# 221
} else 
# 223
{ 
# 224
static_assert((_Nd <= (2 * _Nd_ull)), "Maximum supported integer size is 128-bit");
# 227
unsigned long long __high = __x >> _Nd_ull; 
# 228
if (__high != (0)) 
# 229
{ 
# 230
constexpr int __diff = ((2 * _Nd_ull) - _Nd); 
# 231
return __builtin_clzll(__high) - __diff; 
# 232
}  
# 233
constexpr auto __max_ull = __int_traits< unsigned long long> ::__max; 
# 234
unsigned long long __low = __x & __max_ull; 
# 235
return (_Nd - _Nd_ull) + __builtin_clzll(__low); 
# 236
}  }  }  
# 237
} 
# 239
template< class _Tp> constexpr int 
# 241
__countl_one(_Tp __x) noexcept 
# 242
{ 
# 243
return std::__countl_zero< _Tp> ((_Tp)(~__x)); 
# 244
} 
# 246
template< class _Tp> constexpr int 
# 248
__countr_zero(_Tp __x) noexcept 
# 249
{ 
# 250
using __gnu_cxx::__int_traits;
# 251
constexpr auto _Nd = (__int_traits< _Tp> ::__digits); 
# 253
if (__x == 0) { 
# 254
return _Nd; }  
# 256
constexpr auto _Nd_ull = __int_traits< unsigned long long> ::__digits; 
# 257
constexpr auto _Nd_ul = __int_traits< unsigned long> ::__digits; 
# 258
constexpr auto _Nd_u = __int_traits< unsigned> ::__digits; 
# 260
if constexpr (_Nd <= _Nd_u) { 
# 261
return __builtin_ctz(__x); } else { 
# 262
if constexpr (_Nd <= _Nd_ul) { 
# 263
return __builtin_ctzl(__x); } else { 
# 264
if constexpr (_Nd <= _Nd_ull) { 
# 265
return __builtin_ctzll(__x); } else 
# 267
{ 
# 268
static_assert((_Nd <= (2 * _Nd_ull)), "Maximum supported integer size is 128-bit");
# 271
constexpr auto __max_ull = __int_traits< unsigned long long> ::__max; 
# 272
unsigned long long __low = __x & __max_ull; 
# 273
if (__low != (0)) { 
# 274
return __builtin_ctzll(__low); }  
# 275
unsigned long long __high = __x >> _Nd_ull; 
# 276
return __builtin_ctzll(__high) + _Nd_ull; 
# 277
}  }  }  
# 278
} 
# 280
template< class _Tp> constexpr int 
# 282
__countr_one(_Tp __x) noexcept 
# 283
{ 
# 284
return std::__countr_zero((_Tp)(~__x)); 
# 285
} 
# 287
template< class _Tp> constexpr int 
# 289
__popcount(_Tp __x) noexcept 
# 290
{ 
# 291
using __gnu_cxx::__int_traits;
# 292
constexpr auto _Nd = (__int_traits< _Tp> ::__digits); 
# 294
constexpr auto _Nd_ull = __int_traits< unsigned long long> ::__digits; 
# 295
constexpr auto _Nd_ul = __int_traits< unsigned long> ::__digits; 
# 296
constexpr auto _Nd_u = __int_traits< unsigned> ::__digits; 
# 298
if constexpr (_Nd <= _Nd_u) { 
# 299
return __builtin_popcount(__x); } else { 
# 300
if constexpr (_Nd <= _Nd_ul) { 
# 301
return __builtin_popcountl(__x); } else { 
# 302
if constexpr (_Nd <= _Nd_ull) { 
# 303
return __builtin_popcountll(__x); } else 
# 305
{ 
# 306
static_assert((_Nd <= (2 * _Nd_ull)), "Maximum supported integer size is 128-bit");
# 309
constexpr auto __max_ull = __int_traits< unsigned long long> ::__max; 
# 310
unsigned long long __low = __x & __max_ull; 
# 311
unsigned long long __high = __x >> _Nd_ull; 
# 312
return __builtin_popcountll(__low) + __builtin_popcountll(__high); 
# 313
}  }  }  
# 314
} 
# 316
template< class _Tp> constexpr bool 
# 318
__has_single_bit(_Tp __x) noexcept 
# 319
{ return std::__popcount(__x) == 1; } 
# 321
template< class _Tp> constexpr _Tp 
# 323
__bit_ceil(_Tp __x) noexcept 
# 324
{ 
# 325
using __gnu_cxx::__int_traits;
# 326
constexpr auto _Nd = (__int_traits< _Tp> ::__digits); 
# 327
if ((__x == 0) || (__x == 1)) { 
# 328
return 1; }  
# 329
auto __shift_exponent = _Nd - std::__countl_zero((_Tp)(__x - 1U)); 
# 334
if (!std::__is_constant_evaluated()) 
# 335
{ 
# 336
do { if (std::__is_constant_evaluated() && (!((bool)(__shift_exponent != __int_traits< _Tp> ::__digits)))) { __builtin_unreachable(); }  } while (false); 
# 337
}  
# 339
using __promoted_type = __decltype((__x << 1)); 
# 340
if constexpr (!is_same< __decltype((__x << 1)), _Tp> ::value) 
# 341
{ 
# 347
const int __extra_exp = ((sizeof(__promoted_type) / sizeof(_Tp)) / (2)); 
# 348
__shift_exponent |= ((__shift_exponent & _Nd) << __extra_exp); 
# 349
}  
# 350
return ((_Tp)1U) << __shift_exponent; 
# 351
} 
# 353
template< class _Tp> constexpr _Tp 
# 355
__bit_floor(_Tp __x) noexcept 
# 356
{ 
# 357
constexpr auto _Nd = (__gnu_cxx::__int_traits< _Tp> ::__digits); 
# 358
if (__x == 0) { 
# 359
return 0; }  
# 360
return ((_Tp)1U) << (_Nd - std::__countl_zero((_Tp)(__x >> 1))); 
# 361
} 
# 363
template< class _Tp> constexpr int 
# 365
__bit_width(_Tp __x) noexcept 
# 366
{ 
# 367
constexpr auto _Nd = (__gnu_cxx::__int_traits< _Tp> ::__digits); 
# 368
return _Nd - std::__countl_zero(__x); 
# 369
} 
# 479 "/usr/include/c++/13/bit" 3
}
# 82 "/usr/include/c++/13/bits/stl_algobase.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 90
template< class _Tp, class _Up> constexpr int 
# 93
__memcmp(const _Tp *__first1, const _Up *__first2, size_t __num) 
# 94
{ 
# 96
static_assert((sizeof(_Tp) == sizeof(_Up)), "can be compared with memcmp");
# 108 "/usr/include/c++/13/bits/stl_algobase.h" 3
return __builtin_memcmp(__first1, __first2, sizeof(_Tp) * __num); 
# 109
} 
# 152 "/usr/include/c++/13/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> inline void 
# 155
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) 
# 156
{ 
# 185 "/usr/include/c++/13/bits/stl_algobase.h" 3
swap(*__a, *__b); 
# 187
} 
# 201
template< class _ForwardIterator1, class _ForwardIterator2> _ForwardIterator2 
# 204
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 205
__first2) 
# 206
{ 
# 212
; 
# 214
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 215
std::iter_swap(__first1, __first2); }  
# 216
return __first2; 
# 217
} 
# 230
template< class _Tp> constexpr const _Tp &
# 233
min(const _Tp &__a, const _Tp &__b) 
# 234
{ 
# 238
if (__b < __a) { 
# 239
return __b; }  
# 240
return __a; 
# 241
} 
# 254
template< class _Tp> constexpr const _Tp &
# 257
max(const _Tp &__a, const _Tp &__b) 
# 258
{ 
# 262
if (__a < __b) { 
# 263
return __b; }  
# 264
return __a; 
# 265
} 
# 278
template< class _Tp, class _Compare> constexpr const _Tp &
# 281
min(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 282
{ 
# 284
if (__comp(__b, __a)) { 
# 285
return __b; }  
# 286
return __a; 
# 287
} 
# 300
template< class _Tp, class _Compare> constexpr const _Tp &
# 303
max(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 304
{ 
# 306
if (__comp(__a, __b)) { 
# 307
return __b; }  
# 308
return __a; 
# 309
} 
# 313
template< class _Iterator> inline _Iterator 
# 316
__niter_base(_Iterator __it) noexcept(std::template is_nothrow_copy_constructible< _Iterator> ::value) 
# 318
{ return __it; } 
# 320
template< class _Ite, class _Seq> _Ite __niter_base(const __gnu_debug::_Safe_iterator< _Ite, _Seq, random_access_iterator_tag>  &); 
# 328
template< class _From, class _To> inline _From 
# 331
__niter_wrap(_From __from, _To __res) 
# 332
{ return __from + (__res - std::__niter_base(__from)); } 
# 335
template< class _Iterator> inline _Iterator 
# 338
__niter_wrap(const _Iterator &, _Iterator __res) 
# 339
{ return __res; } 
# 347
template< bool _IsMove, bool _IsSimple, class _Category> 
# 348
struct __copy_move { 
# 350
template< class _II, class _OI> static _OI 
# 353
__copy_m(_II __first, _II __last, _OI __result) 
# 354
{ 
# 355
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 356
(*__result) = (*__first); }  
# 357
return __result; 
# 358
} 
# 359
}; 
# 362
template< class _Category> 
# 363
struct __copy_move< true, false, _Category>  { 
# 365
template< class _II, class _OI> static _OI 
# 368
__copy_m(_II __first, _II __last, _OI __result) 
# 369
{ 
# 370
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 371
(*__result) = std::move(*__first); }  
# 372
return __result; 
# 373
} 
# 374
}; 
# 378
template<> struct __copy_move< false, false, random_access_iterator_tag>  { 
# 380
template< class _II, class _OI> static _OI 
# 383
__copy_m(_II __first, _II __last, _OI __result) 
# 384
{ 
# 385
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 386
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 387
{ 
# 388
(*__result) = (*__first); 
# 389
++__first; 
# 390
++__result; 
# 391
}  
# 392
return __result; 
# 393
} 
# 395
template< class _Tp, class _Up> static void 
# 397
__assign_one(_Tp *__to, _Up *__from) 
# 398
{ (*__to) = (*__from); } 
# 399
}; 
# 403
template<> struct __copy_move< true, false, random_access_iterator_tag>  { 
# 405
template< class _II, class _OI> static _OI 
# 408
__copy_m(_II __first, _II __last, _OI __result) 
# 409
{ 
# 410
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 411
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 412
{ 
# 413
(*__result) = std::move(*__first); 
# 414
++__first; 
# 415
++__result; 
# 416
}  
# 417
return __result; 
# 418
} 
# 420
template< class _Tp, class _Up> static void 
# 422
__assign_one(_Tp *__to, _Up *__from) 
# 423
{ (*__to) = std::move(*__from); } 
# 424
}; 
# 427
template< bool _IsMove> 
# 428
struct __copy_move< _IsMove, true, random_access_iterator_tag>  { 
# 430
template< class _Tp, class _Up> static _Up *
# 433
__copy_m(_Tp *__first, _Tp *__last, _Up *__result) 
# 434
{ 
# 435
const ptrdiff_t _Num = __last - __first; 
# 436
if (__builtin_expect(_Num > (1), true)) { 
# 437
__builtin_memmove(__result, __first, sizeof(_Tp) * _Num); } else { 
# 438
if (_Num == (1)) { 
# 439
std::template __copy_move< _IsMove, false, random_access_iterator_tag> ::__assign_one(__result, __first); }  }  
# 441
return __result + _Num; 
# 442
} 
# 443
}; 
# 447
template< class _Tp, class _Ref, class _Ptr> struct _Deque_iterator; 
# 450
struct _Bit_iterator; 
# 457
template< class _CharT> struct char_traits; 
# 460
template< class _CharT, class _Traits> class istreambuf_iterator; 
# 463
template< class _CharT, class _Traits> class ostreambuf_iterator; 
# 466
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(_CharT *, _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 472
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(const _CharT *, const _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 478
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _CharT *); 
# 484
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _Deque_iterator< _CharT, _CharT &, _CharT *> ); 
# 494
template< bool _IsMove, class _II, class _OI> inline _OI 
# 497
__copy_move_a2(_II __first, _II __last, _OI __result) 
# 498
{ 
# 499
typedef typename iterator_traits< _II> ::iterator_category _Category; 
# 505
return std::template __copy_move< _IsMove, __memcpyable< _OI, _II> ::__value, typename iterator_traits< _II> ::iterator_category> ::__copy_m(__first, __last, __result); 
# 507
} 
# 509
template< bool _IsMove, class 
# 510
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 509
__copy_move_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 516
template< bool _IsMove, class 
# 517
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 516
__copy_move_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 523
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 529
template< bool _IsMove, class _II, class _OI> inline _OI 
# 532
__copy_move_a1(_II __first, _II __last, _OI __result) 
# 533
{ return std::__copy_move_a2< _IsMove> (__first, __last, __result); } 
# 535
template< bool _IsMove, class _II, class _OI> inline _OI 
# 538
__copy_move_a(_II __first, _II __last, _OI __result) 
# 539
{ 
# 540
return std::__niter_wrap(__result, std::__copy_move_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 544
} 
# 546
template< bool _IsMove, class 
# 547
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 546
__copy_move_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 553
template< bool _IsMove, class 
# 554
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 553
__copy_move_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 559
template< bool _IsMove, class 
# 560
_IIte, class _ISeq, class _ICat, class 
# 561
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 559
__copy_move_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 567
template< class _InputIterator, class _Size, class _OutputIterator> _OutputIterator 
# 570
__copy_n_a(_InputIterator __first, _Size __n, _OutputIterator __result, bool) 
# 572
{ 
# 573
if (__n > 0) 
# 574
{ 
# 575
while (true) 
# 576
{ 
# 577
(*__result) = (*__first); 
# 578
++__result; 
# 579
if ((--__n) > 0) { 
# 580
++__first; } else { 
# 582
break; }  
# 583
}  
# 584
}  
# 585
return __result; 
# 586
} 
# 589
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _CharT *, bool); 
# 595
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _Deque_iterator< _CharT, _CharT &, _CharT *> , bool); 
# 621
template< class _II, class _OI> inline _OI 
# 624
copy(_II __first, _II __last, _OI __result) 
# 625
{ 
# 630
; 
# 632
return std::__copy_move_a< __is_move_iterator< _II> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 634
} 
# 654
template< class _II, class _OI> inline _OI 
# 657
move(_II __first, _II __last, _OI __result) 
# 658
{ 
# 663
; 
# 665
return std::__copy_move_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 667
} 
# 674
template< bool _IsMove, bool _IsSimple, class _Category> 
# 675
struct __copy_move_backward { 
# 677
template< class _BI1, class _BI2> static _BI2 
# 680
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 681
{ 
# 682
while (__first != __last) { 
# 683
(*(--__result)) = (*(--__last)); }  
# 684
return __result; 
# 685
} 
# 686
}; 
# 689
template< class _Category> 
# 690
struct __copy_move_backward< true, false, _Category>  { 
# 692
template< class _BI1, class _BI2> static _BI2 
# 695
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 696
{ 
# 697
while (__first != __last) { 
# 698
(*(--__result)) = std::move(*(--__last)); }  
# 699
return __result; 
# 700
} 
# 701
}; 
# 705
template<> struct __copy_move_backward< false, false, random_access_iterator_tag>  { 
# 707
template< class _BI1, class _BI2> static _BI2 
# 710
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 711
{ 
# 713
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 714
for (; __n > 0; --__n) { 
# 715
(*(--__result)) = (*(--__last)); }  
# 716
return __result; 
# 717
} 
# 718
}; 
# 722
template<> struct __copy_move_backward< true, false, random_access_iterator_tag>  { 
# 724
template< class _BI1, class _BI2> static _BI2 
# 727
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 728
{ 
# 730
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 731
for (; __n > 0; --__n) { 
# 732
(*(--__result)) = std::move(*(--__last)); }  
# 733
return __result; 
# 734
} 
# 735
}; 
# 738
template< bool _IsMove> 
# 739
struct __copy_move_backward< _IsMove, true, random_access_iterator_tag>  { 
# 741
template< class _Tp, class _Up> static _Up *
# 744
__copy_move_b(_Tp *__first, _Tp *__last, _Up *__result) 
# 745
{ 
# 746
const ptrdiff_t _Num = __last - __first; 
# 747
if (__builtin_expect(_Num > (1), true)) { 
# 748
__builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num); } else { 
# 749
if (_Num == (1)) { 
# 750
std::template __copy_move< _IsMove, false, random_access_iterator_tag> ::__assign_one(__result - 1, __first); }  }  
# 752
return __result - _Num; 
# 753
} 
# 754
}; 
# 756
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 759
__copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result) 
# 760
{ 
# 761
typedef typename iterator_traits< _BI1> ::iterator_category _Category; 
# 767
return std::template __copy_move_backward< _IsMove, __memcpyable< _BI2, _BI1> ::__value, typename iterator_traits< _BI1> ::iterator_category> ::__copy_move_b(__first, __last, __result); 
# 772
} 
# 774
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 777
__copy_move_backward_a1(_BI1 __first, _BI1 __last, _BI2 __result) 
# 778
{ return std::__copy_move_backward_a2< _IsMove> (__first, __last, __result); } 
# 780
template< bool _IsMove, class 
# 781
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 780
__copy_move_backward_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 787
template< bool _IsMove, class 
# 788
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 787
__copy_move_backward_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 795
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_backward_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 802
template< bool _IsMove, class _II, class _OI> inline _OI 
# 805
__copy_move_backward_a(_II __first, _II __last, _OI __result) 
# 806
{ 
# 807
return std::__niter_wrap(__result, std::__copy_move_backward_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 811
} 
# 813
template< bool _IsMove, class 
# 814
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 813
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 821
template< bool _IsMove, class 
# 822
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 821
__copy_move_backward_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 827
template< bool _IsMove, class 
# 828
_IIte, class _ISeq, class _ICat, class 
# 829
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 827
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 854
template< class _BI1, class _BI2> inline _BI2 
# 857
copy_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 858
{ 
# 864
; 
# 866
return std::__copy_move_backward_a< __is_move_iterator< _BI1> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 868
} 
# 889
template< class _BI1, class _BI2> inline _BI2 
# 892
move_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 893
{ 
# 899
; 
# 901
return std::__copy_move_backward_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 904
} 
# 911
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, void> ::__type 
# 915
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 916
__value) 
# 917
{ 
# 918
for (; __first != __last; ++__first) { 
# 919
(*__first) = __value; }  
# 920
} 
# 922
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, void> ::__type 
# 926
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 927
__value) 
# 928
{ 
# 929
const _Tp __tmp = __value; 
# 930
for (; __first != __last; ++__first) { 
# 931
(*__first) = __tmp; }  
# 932
} 
# 935
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_byte< _Tp> ::__value, void> ::__type 
# 939
__fill_a1(_Tp *__first, _Tp *__last, const _Tp &__c) 
# 940
{ 
# 941
const _Tp __tmp = __c; 
# 950 "/usr/include/c++/13/bits/stl_algobase.h" 3
if (const size_t __len = __last - __first) { 
# 951
__builtin_memset(__first, static_cast< unsigned char>(__tmp), __len); }  
# 952
} 
# 954
template< class _Ite, class _Cont, class _Tp> inline void 
# 957
__fill_a1(__gnu_cxx::__normal_iterator< _Ite, _Cont>  __first, __gnu_cxx::__normal_iterator< _Ite, _Cont>  
# 958
__last, const _Tp &
# 959
__value) 
# 960
{ std::__fill_a1(__first.base(), __last.base(), __value); } 
# 962
template< class _Tp, class _VTp> void __fill_a1(const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _VTp &); 
# 970
void __fill_a1(_Bit_iterator, _Bit_iterator, const bool &); 
# 973
template< class _FIte, class _Tp> inline void 
# 976
__fill_a(_FIte __first, _FIte __last, const _Tp &__value) 
# 977
{ std::__fill_a1(__first, __last, __value); } 
# 979
template< class _Ite, class _Seq, class _Cat, class _Tp> void __fill_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const _Tp &); 
# 997
template< class _ForwardIterator, class _Tp> inline void 
# 1000
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value) 
# 1001
{ 
# 1005
; 
# 1007
std::__fill_a(__first, __last, __value); 
# 1008
} 
# 1012
constexpr int __size_to_integer(int __n) { return __n; } 
# 1014
constexpr unsigned __size_to_integer(unsigned __n) { return __n; } 
# 1016
constexpr long __size_to_integer(long __n) { return __n; } 
# 1018
constexpr unsigned long __size_to_integer(unsigned long __n) { return __n; } 
# 1020
constexpr long long __size_to_integer(long long __n) { return __n; } 
# 1022
constexpr unsigned long long __size_to_integer(unsigned long long __n) { return __n; } 
# 1026
constexpr __int128 __size_to_integer(__int128 __n) { return __n; } 
# 1028
constexpr unsigned __int128 __size_to_integer(unsigned __int128 __n) { return __n; } 
# 1050 "/usr/include/c++/13/bits/stl_algobase.h" 3
constexpr long long __size_to_integer(float __n) { return (long long)__n; } 
# 1052
constexpr long long __size_to_integer(double __n) { return (long long)__n; } 
# 1054
constexpr long long __size_to_integer(long double __n) { return (long long)__n; } 
# 1060
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1064
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1065
{ 
# 1066
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1067
(*__first) = __value; }  
# 1068
return __first; 
# 1069
} 
# 1071
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1075
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1076
{ 
# 1077
const _Tp __tmp = __value; 
# 1078
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1079
(*__first) = __tmp; }  
# 1080
return __first; 
# 1081
} 
# 1083
template< class _Ite, class _Seq, class _Cat, class _Size, class 
# 1084
_Tp> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 1083
__fill_n_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  & __first, _Size __n, const _Tp & __value, input_iterator_tag); 
# 1090
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1093
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, output_iterator_tag) 
# 1095
{ 
# 1097
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1099
return __fill_n_a1(__first, __n, __value); 
# 1100
} 
# 1102
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1105
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, input_iterator_tag) 
# 1107
{ 
# 1109
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1111
return __fill_n_a1(__first, __n, __value); 
# 1112
} 
# 1114
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1117
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, random_access_iterator_tag) 
# 1119
{ 
# 1121
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1123
if (__n <= 0) { 
# 1124
return __first; }  
# 1126
; 
# 1128
std::__fill_a(__first, __first + __n, __value); 
# 1129
return __first + __n; 
# 1130
} 
# 1149
template< class _OI, class _Size, class _Tp> inline _OI 
# 1152
fill_n(_OI __first, _Size __n, const _Tp &__value) 
# 1153
{ 
# 1157
return std::__fill_n_a(__first, std::__size_to_integer(__n), __value, std::__iterator_category(__first)); 
# 1159
} 
# 1161
template< bool _BoolType> 
# 1162
struct __equal { 
# 1164
template< class _II1, class _II2> static bool 
# 1167
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1168
{ 
# 1169
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1170
if (!((*__first1) == (*__first2))) { 
# 1171
return false; }  }  
# 1172
return true; 
# 1173
} 
# 1174
}; 
# 1177
template<> struct __equal< true>  { 
# 1179
template< class _Tp> static bool 
# 1182
equal(const _Tp *__first1, const _Tp *__last1, const _Tp *__first2) 
# 1183
{ 
# 1184
if (const size_t __len = __last1 - __first1) { 
# 1185
return !std::__memcmp(__first1, __first2, __len); }  
# 1186
return true; 
# 1187
} 
# 1188
}; 
# 1190
template< class _Tp, class _Ref, class _Ptr, class _II> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _II); 
# 1197
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1198
_Tp2, class _Ref2, class _Ptr2> bool 
# 1197
__equal_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1204
template< class _II, class _Tp, class _Ref, class _Ptr> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_II, _II, _Deque_iterator< _Tp, _Ref, _Ptr> ); 
# 1210
template< class _II1, class _II2> inline bool 
# 1213
__equal_aux1(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1214
{ 
# 1215
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1216
const bool __simple = ((__is_integer< typename iterator_traits< _II1> ::value_type> ::__value || __is_pointer< typename iterator_traits< _II1> ::value_type> ::__value) && __memcmpable< _II1, _II2> ::__value); 
# 1219
return std::template __equal< __simple> ::equal(__first1, __last1, __first2); 
# 1220
} 
# 1222
template< class _II1, class _II2> inline bool 
# 1225
__equal_aux(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1226
{ 
# 1227
return std::__equal_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2)); 
# 1230
} 
# 1232
template< class _II1, class _Seq1, class _Cat1, class _II2> bool __equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, _II2); 
# 1238
template< class _II1, class _II2, class _Seq2, class _Cat2> bool __equal_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1243
template< class _II1, class _Seq1, class _Cat1, class 
# 1244
_II2, class _Seq2, class _Cat2> bool 
# 1243
__equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1250
template< class , class > 
# 1251
struct __lc_rai { 
# 1253
template< class _II1, class _II2> static _II1 
# 1256
__newlast1(_II1, _II1 __last1, _II2, _II2) 
# 1257
{ return __last1; } 
# 1259
template< class _II> static bool 
# 1262
__cnd2(_II __first, _II __last) 
# 1263
{ return __first != __last; } 
# 1264
}; 
# 1267
template<> struct __lc_rai< random_access_iterator_tag, random_access_iterator_tag>  { 
# 1269
template< class _RAI1, class _RAI2> static _RAI1 
# 1272
__newlast1(_RAI1 __first1, _RAI1 __last1, _RAI2 
# 1273
__first2, _RAI2 __last2) 
# 1274
{ 
# 1276
const typename iterator_traits< _RAI1> ::difference_type __diff1 = __last1 - __first1; 
# 1278
const typename iterator_traits< _RAI2> ::difference_type __diff2 = __last2 - __first2; 
# 1279
return (__diff2 < __diff1) ? __first1 + __diff2 : __last1; 
# 1280
} 
# 1282
template< class _RAI> static bool 
# 1284
__cnd2(_RAI, _RAI) 
# 1285
{ return true; } 
# 1286
}; 
# 1288
template< class _II1, class _II2, class _Compare> bool 
# 1291
__lexicographical_compare_impl(_II1 __first1, _II1 __last1, _II2 
# 1292
__first2, _II2 __last2, _Compare 
# 1293
__comp) 
# 1294
{ 
# 1295
typedef typename iterator_traits< _II1> ::iterator_category _Category1; 
# 1296
typedef typename iterator_traits< _II2> ::iterator_category _Category2; 
# 1297
typedef __lc_rai< typename iterator_traits< _II1> ::iterator_category, typename iterator_traits< _II2> ::iterator_category>  __rai_type; 
# 1299
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2); 
# 1300
for (; (__first1 != __last1) && __rai_type::__cnd2(__first2, __last2); (++__first1), ((void)(++__first2))) 
# 1302
{ 
# 1303
if (__comp(__first1, __first2)) { 
# 1304
return true; }  
# 1305
if (__comp(__first2, __first1)) { 
# 1306
return false; }  
# 1307
}  
# 1308
return (__first1 == __last1) && (__first2 != __last2); 
# 1309
} 
# 1311
template< bool _BoolType> 
# 1312
struct __lexicographical_compare { 
# 1314
template< class _II1, class _II2> static bool 
# 1317
__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1318
{ 
# 1319
using __gnu_cxx::__ops::__iter_less_iter;
# 1320
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __iter_less_iter()); 
# 1323
} 
# 1325
template< class _II1, class _II2> static int 
# 1328
__3way(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1329
{ 
# 1330
while (__first1 != __last1) 
# 1331
{ 
# 1332
if (__first2 == __last2) { 
# 1333
return +1; }  
# 1334
if ((*__first1) < (*__first2)) { 
# 1335
return -1; }  
# 1336
if ((*__first2) < (*__first1)) { 
# 1337
return +1; }  
# 1338
++__first1; 
# 1339
++__first2; 
# 1340
}  
# 1341
return ((int)(__first2 == __last2)) - 1; 
# 1342
} 
# 1343
}; 
# 1346
template<> struct __lexicographical_compare< true>  { 
# 1348
template< class _Tp, class _Up> static bool 
# 1351
__lc(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1352
__first2, const _Up *__last2) 
# 1353
{ return __3way(__first1, __last1, __first2, __last2) < 0; } 
# 1355
template< class _Tp, class _Up> static ptrdiff_t 
# 1358
__3way(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1359
__first2, const _Up *__last2) 
# 1360
{ 
# 1361
const size_t __len1 = __last1 - __first1; 
# 1362
const size_t __len2 = __last2 - __first2; 
# 1363
if (const size_t __len = std::min(__len1, __len2)) { 
# 1364
if (int __result = std::__memcmp(__first1, __first2, __len)) { 
# 1365
return __result; }  }  
# 1366
return (ptrdiff_t)(__len1 - __len2); 
# 1367
} 
# 1368
}; 
# 1370
template< class _II1, class _II2> inline bool 
# 1373
__lexicographical_compare_aux1(_II1 __first1, _II1 __last1, _II2 
# 1374
__first2, _II2 __last2) 
# 1375
{ 
# 1376
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1377
typedef typename iterator_traits< _II2> ::value_type _ValueType2; 
# 1378
const bool __simple = (__is_memcmp_ordered_with< typename iterator_traits< _II1> ::value_type, typename iterator_traits< _II2> ::value_type> ::__value && __is_pointer< _II1> ::__value && __is_pointer< _II2> ::__value); 
# 1391 "/usr/include/c++/13/bits/stl_algobase.h" 3
return std::template __lexicographical_compare< __simple> ::__lc(__first1, __last1, __first2, __last2); 
# 1393
} 
# 1395
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1396
_Tp2> bool 
# 1395
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Tp2 *, _Tp2 *); 
# 1403
template< class _Tp1, class 
# 1404
_Tp2, class _Ref2, class _Ptr2> bool 
# 1403
__lexicographical_compare_aux1(_Tp1 *, _Tp1 *, _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1410
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1411
_Tp2, class _Ref2, class _Ptr2> bool 
# 1410
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1419
template< class _II1, class _II2> inline bool 
# 1422
__lexicographical_compare_aux(_II1 __first1, _II1 __last1, _II2 
# 1423
__first2, _II2 __last2) 
# 1424
{ 
# 1425
return std::__lexicographical_compare_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2), std::__niter_base(__last2)); 
# 1429
} 
# 1431
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1432
_II2> bool 
# 1431
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, _II2, _II2); 
# 1439
template< class _II1, class 
# 1440
_Iter2, class _Seq2, class _Cat2> bool 
# 1439
__lexicographical_compare_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1447
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1448
_Iter2, class _Seq2, class _Cat2> bool 
# 1447
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1456
template< class _ForwardIterator, class _Tp, class _Compare> _ForwardIterator 
# 1459
__lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1460
__val, _Compare __comp) 
# 1461
{ 
# 1463
typedef typename iterator_traits< _ForwardIterator> ::difference_type _DistanceType; 
# 1465
_DistanceType __len = std::distance(__first, __last); 
# 1467
while (__len > 0) 
# 1468
{ 
# 1469
_DistanceType __half = __len >> 1; 
# 1470
_ForwardIterator __middle = __first; 
# 1471
std::advance(__middle, __half); 
# 1472
if (__comp(__middle, __val)) 
# 1473
{ 
# 1474
__first = __middle; 
# 1475
++__first; 
# 1476
__len = ((__len - __half) - 1); 
# 1477
} else { 
# 1479
__len = __half; }  
# 1480
}  
# 1481
return __first; 
# 1482
} 
# 1495
template< class _ForwardIterator, class _Tp> inline _ForwardIterator 
# 1498
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1499
__val) 
# 1500
{ 
# 1505
; 
# 1507
return std::__lower_bound(__first, __last, __val, __gnu_cxx::__ops::__iter_less_val()); 
# 1509
} 
# 1513
template< class _Tp> constexpr _Tp 
# 1515
__lg(_Tp __n) 
# 1516
{ 
# 1518
return std::__bit_width((make_unsigned_t< _Tp> )__n) - 1; 
# 1528 "/usr/include/c++/13/bits/stl_algobase.h" 3
} 
# 1544
template< class _II1, class _II2> inline bool 
# 1547
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1548
{ 
# 1555
; 
# 1557
return std::__equal_aux(__first1, __last1, __first2); 
# 1558
} 
# 1575
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1578
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1579
__first2, _BinaryPredicate __binary_pred) 
# 1580
{ 
# 1584
; 
# 1586
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1587
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1588
return false; }  }  
# 1589
return true; 
# 1590
} 
# 1594
template< class _II1, class _II2> inline bool 
# 1597
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1598
{ 
# 1599
using _RATag = random_access_iterator_tag; 
# 1600
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1601
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1602
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1603
if (_RAIters()) 
# 1604
{ 
# 1605
auto __d1 = std::distance(__first1, __last1); 
# 1606
auto __d2 = std::distance(__first2, __last2); 
# 1607
if (__d1 != __d2) { 
# 1608
return false; }  
# 1609
return std::equal(__first1, __last1, __first2); 
# 1610
}  
# 1612
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1614
if (!((*__first1) == (*__first2))) { 
# 1615
return false; }  }  
# 1616
return (__first1 == __last1) && (__first2 == __last2); 
# 1617
} 
# 1620
template< class _II1, class _II2, class _BinaryPredicate> inline bool 
# 1623
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2, _BinaryPredicate 
# 1624
__binary_pred) 
# 1625
{ 
# 1626
using _RATag = random_access_iterator_tag; 
# 1627
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1628
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1629
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1630
if (_RAIters()) 
# 1631
{ 
# 1632
auto __d1 = std::distance(__first1, __last1); 
# 1633
auto __d2 = std::distance(__first2, __last2); 
# 1634
if (__d1 != __d2) { 
# 1635
return false; }  
# 1636
return std::equal(__first1, __last1, __first2, __binary_pred); 
# 1638
}  
# 1640
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1642
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1643
return false; }  }  
# 1644
return (__first1 == __last1) && (__first2 == __last2); 
# 1645
} 
# 1665
template< class _II1, class _II2> inline bool 
# 1668
equal(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1669
{ 
# 1676
; 
# 1677
; 
# 1679
return std::__equal4(__first1, __last1, __first2, __last2); 
# 1680
} 
# 1698
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1701
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1702
__first2, _IIter2 __last2, _BinaryPredicate __binary_pred) 
# 1703
{ 
# 1707
; 
# 1708
; 
# 1710
return std::__equal4(__first1, __last1, __first2, __last2, __binary_pred); 
# 1712
} 
# 1730
template< class _II1, class _II2> inline bool 
# 1733
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1734
__first2, _II2 __last2) 
# 1735
{ 
# 1745
; 
# 1746
; 
# 1748
return std::__lexicographical_compare_aux(__first1, __last1, __first2, __last2); 
# 1750
} 
# 1765
template< class _II1, class _II2, class _Compare> inline bool 
# 1768
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1769
__first2, _II2 __last2, _Compare __comp) 
# 1770
{ 
# 1774
; 
# 1775
; 
# 1777
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__comp)); 
# 1780
} 
# 1877 "/usr/include/c++/13/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 1878
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1881
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1882
__first2, _BinaryPredicate __binary_pred) 
# 1883
{ 
# 1884
while ((__first1 != __last1) && __binary_pred(__first1, __first2)) 
# 1885
{ 
# 1886
++__first1; 
# 1887
++__first2; 
# 1888
}  
# 1889
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1890
} 
# 1905
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1908
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1909
__first2) 
# 1910
{ 
# 1917
; 
# 1919
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 1921
} 
# 1939
template< class _InputIterator1, class _InputIterator2, class 
# 1940
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 1943
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1944
__first2, _BinaryPredicate __binary_pred) 
# 1945
{ 
# 1949
; 
# 1951
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 1953
} 
# 1957
template< class _InputIterator1, class _InputIterator2, class 
# 1958
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1961
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1962
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 1963
__binary_pred) 
# 1964
{ 
# 1965
while ((__first1 != __last1) && (__first2 != __last2) && __binary_pred(__first1, __first2)) 
# 1967
{ 
# 1968
++__first1; 
# 1969
++__first2; 
# 1970
}  
# 1971
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1972
} 
# 1988
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1991
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1992
__first2, _InputIterator2 __last2) 
# 1993
{ 
# 2000
; 
# 2001
; 
# 2003
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2005
} 
# 2024
template< class _InputIterator1, class _InputIterator2, class 
# 2025
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 2028
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 2029
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 2030
__binary_pred) 
# 2031
{ 
# 2035
; 
# 2036
; 
# 2038
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 2040
} 
# 2046
template< class _InputIterator, class _Predicate> inline _InputIterator 
# 2049
__find_if(_InputIterator __first, _InputIterator __last, _Predicate 
# 2050
__pred, input_iterator_tag) 
# 2051
{ 
# 2052
while ((__first != __last) && (!__pred(__first))) { 
# 2053
++__first; }  
# 2054
return __first; 
# 2055
} 
# 2058
template< class _RandomAccessIterator, class _Predicate> _RandomAccessIterator 
# 2061
__find_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate 
# 2062
__pred, random_access_iterator_tag) 
# 2063
{ 
# 2065
typename iterator_traits< _RandomAccessIterator> ::difference_type __trip_count = (__last - __first) >> 2; 
# 2067
for (; __trip_count > 0; --__trip_count) 
# 2068
{ 
# 2069
if (__pred(__first)) { 
# 2070
return __first; }  
# 2071
++__first; 
# 2073
if (__pred(__first)) { 
# 2074
return __first; }  
# 2075
++__first; 
# 2077
if (__pred(__first)) { 
# 2078
return __first; }  
# 2079
++__first; 
# 2081
if (__pred(__first)) { 
# 2082
return __first; }  
# 2083
++__first; 
# 2084
}  
# 2086
switch (__last - __first) 
# 2087
{ 
# 2088
case 3:  
# 2089
if (__pred(__first)) { 
# 2090
return __first; }  
# 2091
++__first; 
# 2093
case 2:  
# 2094
if (__pred(__first)) { 
# 2095
return __first; }  
# 2096
++__first; 
# 2098
case 1:  
# 2099
if (__pred(__first)) { 
# 2100
return __first; }  
# 2101
++__first; 
# 2103
case 0:  
# 2104
default:  
# 2105
return __last; 
# 2106
}  
# 2107
} 
# 2109
template< class _Iterator, class _Predicate> inline _Iterator 
# 2112
__find_if(_Iterator __first, _Iterator __last, _Predicate __pred) 
# 2113
{ 
# 2114
return __find_if(__first, __last, __pred, std::__iterator_category(__first)); 
# 2116
} 
# 2118
template< class _InputIterator, class _Predicate> typename iterator_traits< _InputIterator> ::difference_type 
# 2121
__count_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) 
# 2122
{ 
# 2123
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 2124
for (; __first != __last; ++__first) { 
# 2125
if (__pred(__first)) { 
# 2126
++__n; }  }  
# 2127
return __n; 
# 2128
} 
# 2130
template< class _ForwardIterator, class _Predicate> _ForwardIterator 
# 2133
__remove_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate 
# 2134
__pred) 
# 2135
{ 
# 2136
__first = std::__find_if(__first, __last, __pred); 
# 2137
if (__first == __last) { 
# 2138
return __first; }  
# 2139
_ForwardIterator __result = __first; 
# 2140
++__first; 
# 2141
for (; __first != __last; ++__first) { 
# 2142
if (!__pred(__first)) 
# 2143
{ 
# 2144
(*__result) = std::move(*__first); 
# 2145
++__result; 
# 2146
}  }  
# 2147
return __result; 
# 2148
} 
# 2151
template< class _ForwardIterator1, class _ForwardIterator2, class 
# 2152
_BinaryPredicate> bool 
# 2155
__is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2156
__first2, _BinaryPredicate __pred) 
# 2157
{ 
# 2160
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 2161
if (!__pred(__first1, __first2)) { 
# 2162
break; }  }  
# 2164
if (__first1 == __last1) { 
# 2165
return true; }  
# 2169
_ForwardIterator2 __last2 = __first2; 
# 2170
std::advance(__last2, std::distance(__first1, __last1)); 
# 2171
for (_ForwardIterator1 __scan = __first1; __scan != __last1; ++__scan) 
# 2172
{ 
# 2173
if (__scan != std::__find_if(__first1, __scan, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan))) { 
# 2175
continue; }  
# 2177
auto __matches = std::__count_if(__first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)); 
# 2180
if ((0 == __matches) || (std::__count_if(__scan, __last1, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)) != __matches)) { 
# 2184
return false; }  
# 2185
}   
# 2186
return true; 
# 2187
} 
# 2201
template< class _ForwardIterator1, class _ForwardIterator2> inline bool 
# 2204
is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2205
__first2) 
# 2206
{ 
# 2213
; 
# 2215
return std::__is_permutation(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2217
} 
# 2221
}
# 158 "/usr/include/c++/13/limits" 3
namespace std __attribute((__visibility__("default"))) { 
# 167
enum float_round_style { 
# 169
round_indeterminate = (-1), 
# 170
round_toward_zero = 0, 
# 171
round_to_nearest, 
# 172
round_toward_infinity, 
# 173
round_toward_neg_infinity
# 174
}; 
# 182
enum float_denorm_style { 
# 185
denorm_indeterminate = (-1), 
# 187
denorm_absent = 0, 
# 189
denorm_present
# 190
}; 
# 202
struct __numeric_limits_base { 
# 206
static constexpr inline bool is_specialized = false; 
# 211
static constexpr inline int digits = 0; 
# 214
static constexpr inline int digits10 = 0; 
# 219
static constexpr inline int max_digits10 = 0; 
# 223
static constexpr inline bool is_signed = false; 
# 226
static constexpr inline bool is_integer = false; 
# 231
static constexpr inline bool is_exact = false; 
# 235
static constexpr inline int radix = 0; 
# 239
static constexpr inline int min_exponent = 0; 
# 243
static constexpr inline int min_exponent10 = 0; 
# 248
static constexpr inline int max_exponent = 0; 
# 252
static constexpr inline int max_exponent10 = 0; 
# 255
static constexpr inline bool has_infinity = false; 
# 259
static constexpr inline bool has_quiet_NaN = false; 
# 263
static constexpr inline bool has_signaling_NaN = false; 
# 266
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 270
static constexpr inline bool has_denorm_loss = false; 
# 274
static constexpr inline bool is_iec559 = false; 
# 279
static constexpr inline bool is_bounded = false; 
# 288
static constexpr inline bool is_modulo = false; 
# 291
static constexpr inline bool traps = false; 
# 294
static constexpr inline bool tinyness_before = false; 
# 299
static constexpr inline float_round_style round_style = round_toward_zero; 
# 301
}; 
# 311
template< class _Tp> 
# 312
struct numeric_limits : public __numeric_limits_base { 
# 317
static constexpr _Tp min() noexcept { return _Tp(); } 
# 321
static constexpr _Tp max() noexcept { return _Tp(); } 
# 327
static constexpr _Tp lowest() noexcept { return _Tp(); } 
# 333
static constexpr _Tp epsilon() noexcept { return _Tp(); } 
# 337
static constexpr _Tp round_error() noexcept { return _Tp(); } 
# 341
static constexpr _Tp infinity() noexcept { return _Tp(); } 
# 346
static constexpr _Tp quiet_NaN() noexcept { return _Tp(); } 
# 351
static constexpr _Tp signaling_NaN() noexcept { return _Tp(); } 
# 357
static constexpr _Tp denorm_min() noexcept { return _Tp(); } 
# 358
}; 
# 363
template< class _Tp> 
# 364
struct numeric_limits< const _Tp>  : public std::numeric_limits< _Tp>  { 
# 365
}; 
# 367
template< class _Tp> 
# 368
struct numeric_limits< volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 369
}; 
# 371
template< class _Tp> 
# 372
struct numeric_limits< const volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 373
}; 
# 384
template<> struct numeric_limits< bool>  { 
# 386
static constexpr inline bool is_specialized = true; 
# 389
static constexpr bool min() noexcept { return false; } 
# 392
static constexpr bool max() noexcept { return true; } 
# 396
static constexpr bool lowest() noexcept { return min(); } 
# 398
static constexpr inline int digits = 1; 
# 399
static constexpr inline int digits10 = 0; 
# 401
static constexpr inline int max_digits10 = 0; 
# 403
static constexpr inline bool is_signed = false; 
# 404
static constexpr inline bool is_integer = true; 
# 405
static constexpr inline bool is_exact = true; 
# 406
static constexpr inline int radix = 2; 
# 409
static constexpr bool epsilon() noexcept { return false; } 
# 412
static constexpr bool round_error() noexcept { return false; } 
# 414
static constexpr inline int min_exponent = 0; 
# 415
static constexpr inline int min_exponent10 = 0; 
# 416
static constexpr inline int max_exponent = 0; 
# 417
static constexpr inline int max_exponent10 = 0; 
# 419
static constexpr inline bool has_infinity = false; 
# 420
static constexpr inline bool has_quiet_NaN = false; 
# 421
static constexpr inline bool has_signaling_NaN = false; 
# 422
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 424
static constexpr inline bool has_denorm_loss = false; 
# 427
static constexpr bool infinity() noexcept { return false; } 
# 430
static constexpr bool quiet_NaN() noexcept { return false; } 
# 433
static constexpr bool signaling_NaN() noexcept { return false; } 
# 436
static constexpr bool denorm_min() noexcept { return false; } 
# 438
static constexpr inline bool is_iec559 = false; 
# 439
static constexpr inline bool is_bounded = true; 
# 440
static constexpr inline bool is_modulo = false; 
# 445
static constexpr inline bool traps = true; 
# 446
static constexpr inline bool tinyness_before = false; 
# 447
static constexpr inline float_round_style round_style = round_toward_zero; 
# 449
}; 
# 453
template<> struct numeric_limits< char>  { 
# 455
static constexpr inline bool is_specialized = true; 
# 458
static constexpr char min() noexcept { return ((((char)(-1)) < 0) ? (-((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0)))) - 1 : ((char)0)); } 
# 461
static constexpr char max() noexcept { return ((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0))); } 
# 465
static constexpr char lowest() noexcept { return min(); } 
# 468
static constexpr inline int digits = ((sizeof(char) * (8)) - (((char)(-1)) < 0)); 
# 469
static constexpr inline int digits10 = ((((sizeof(char) * (8)) - (((char)(-1)) < 0)) * (643L)) / (2136)); 
# 471
static constexpr inline int max_digits10 = 0; 
# 473
static constexpr inline bool is_signed = (((char)(-1)) < 0); 
# 474
static constexpr inline bool is_integer = true; 
# 475
static constexpr inline bool is_exact = true; 
# 476
static constexpr inline int radix = 2; 
# 479
static constexpr char epsilon() noexcept { return 0; } 
# 482
static constexpr char round_error() noexcept { return 0; } 
# 484
static constexpr inline int min_exponent = 0; 
# 485
static constexpr inline int min_exponent10 = 0; 
# 486
static constexpr inline int max_exponent = 0; 
# 487
static constexpr inline int max_exponent10 = 0; 
# 489
static constexpr inline bool has_infinity = false; 
# 490
static constexpr inline bool has_quiet_NaN = false; 
# 491
static constexpr inline bool has_signaling_NaN = false; 
# 492
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 494
static constexpr inline bool has_denorm_loss = false; 
# 497
static constexpr char infinity() noexcept { return ((char)0); } 
# 500
static constexpr char quiet_NaN() noexcept { return ((char)0); } 
# 503
static constexpr char signaling_NaN() noexcept { return ((char)0); } 
# 506
static constexpr char denorm_min() noexcept { return static_cast< char>(0); } 
# 508
static constexpr inline bool is_iec559 = false; 
# 509
static constexpr inline bool is_bounded = true; 
# 510
static constexpr inline bool is_modulo = (!is_signed); 
# 512
static constexpr inline bool traps = true; 
# 513
static constexpr inline bool tinyness_before = false; 
# 514
static constexpr inline float_round_style round_style = round_toward_zero; 
# 516
}; 
# 520
template<> struct numeric_limits< signed char>  { 
# 522
static constexpr inline bool is_specialized = true; 
# 525
static constexpr signed char min() noexcept { return (-127) - 1; } 
# 528
static constexpr signed char max() noexcept { return 127; } 
# 532
static constexpr signed char lowest() noexcept { return min(); } 
# 535
static constexpr inline int digits = ((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)); 
# 536
static constexpr inline int digits10 = ((((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)) * (643L)) / (2136)); 
# 539
static constexpr inline int max_digits10 = 0; 
# 541
static constexpr inline bool is_signed = true; 
# 542
static constexpr inline bool is_integer = true; 
# 543
static constexpr inline bool is_exact = true; 
# 544
static constexpr inline int radix = 2; 
# 547
static constexpr signed char epsilon() noexcept { return 0; } 
# 550
static constexpr signed char round_error() noexcept { return 0; } 
# 552
static constexpr inline int min_exponent = 0; 
# 553
static constexpr inline int min_exponent10 = 0; 
# 554
static constexpr inline int max_exponent = 0; 
# 555
static constexpr inline int max_exponent10 = 0; 
# 557
static constexpr inline bool has_infinity = false; 
# 558
static constexpr inline bool has_quiet_NaN = false; 
# 559
static constexpr inline bool has_signaling_NaN = false; 
# 560
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 562
static constexpr inline bool has_denorm_loss = false; 
# 565
static constexpr signed char infinity() noexcept { return static_cast< signed char>(0); } 
# 568
static constexpr signed char quiet_NaN() noexcept { return static_cast< signed char>(0); } 
# 571
static constexpr signed char signaling_NaN() noexcept 
# 572
{ return static_cast< signed char>(0); } 
# 575
static constexpr signed char denorm_min() noexcept 
# 576
{ return static_cast< signed char>(0); } 
# 578
static constexpr inline bool is_iec559 = false; 
# 579
static constexpr inline bool is_bounded = true; 
# 580
static constexpr inline bool is_modulo = false; 
# 582
static constexpr inline bool traps = true; 
# 583
static constexpr inline bool tinyness_before = false; 
# 584
static constexpr inline float_round_style round_style = round_toward_zero; 
# 586
}; 
# 590
template<> struct numeric_limits< unsigned char>  { 
# 592
static constexpr inline bool is_specialized = true; 
# 595
static constexpr unsigned char min() noexcept { return 0; } 
# 598
static constexpr unsigned char max() noexcept { return ((127) * 2U) + (1); } 
# 602
static constexpr unsigned char lowest() noexcept { return min(); } 
# 605
static constexpr inline int digits = ((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)); 
# 607
static constexpr inline int digits10 = ((((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)) * (643L)) / (2136)); 
# 610
static constexpr inline int max_digits10 = 0; 
# 612
static constexpr inline bool is_signed = false; 
# 613
static constexpr inline bool is_integer = true; 
# 614
static constexpr inline bool is_exact = true; 
# 615
static constexpr inline int radix = 2; 
# 618
static constexpr unsigned char epsilon() noexcept { return 0; } 
# 621
static constexpr unsigned char round_error() noexcept { return 0; } 
# 623
static constexpr inline int min_exponent = 0; 
# 624
static constexpr inline int min_exponent10 = 0; 
# 625
static constexpr inline int max_exponent = 0; 
# 626
static constexpr inline int max_exponent10 = 0; 
# 628
static constexpr inline bool has_infinity = false; 
# 629
static constexpr inline bool has_quiet_NaN = false; 
# 630
static constexpr inline bool has_signaling_NaN = false; 
# 631
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 633
static constexpr inline bool has_denorm_loss = false; 
# 636
static constexpr unsigned char infinity() noexcept 
# 637
{ return static_cast< unsigned char>(0); } 
# 640
static constexpr unsigned char quiet_NaN() noexcept 
# 641
{ return static_cast< unsigned char>(0); } 
# 644
static constexpr unsigned char signaling_NaN() noexcept 
# 645
{ return static_cast< unsigned char>(0); } 
# 648
static constexpr unsigned char denorm_min() noexcept 
# 649
{ return static_cast< unsigned char>(0); } 
# 651
static constexpr inline bool is_iec559 = false; 
# 652
static constexpr inline bool is_bounded = true; 
# 653
static constexpr inline bool is_modulo = true; 
# 655
static constexpr inline bool traps = true; 
# 656
static constexpr inline bool tinyness_before = false; 
# 657
static constexpr inline float_round_style round_style = round_toward_zero; 
# 659
}; 
# 663
template<> struct numeric_limits< wchar_t>  { 
# 665
static constexpr inline bool is_specialized = true; 
# 668
static constexpr wchar_t min() noexcept { return ((((wchar_t)(-1)) < 0) ? (-((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0)))) - 1 : ((wchar_t)0)); } 
# 671
static constexpr wchar_t max() noexcept { return ((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0))); } 
# 675
static constexpr wchar_t lowest() noexcept { return min(); } 
# 678
static constexpr inline int digits = ((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)); 
# 679
static constexpr inline int digits10 = ((((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) * (643L)) / (2136)); 
# 682
static constexpr inline int max_digits10 = 0; 
# 684
static constexpr inline bool is_signed = (((wchar_t)(-1)) < 0); 
# 685
static constexpr inline bool is_integer = true; 
# 686
static constexpr inline bool is_exact = true; 
# 687
static constexpr inline int radix = 2; 
# 690
static constexpr wchar_t epsilon() noexcept { return 0; } 
# 693
static constexpr wchar_t round_error() noexcept { return 0; } 
# 695
static constexpr inline int min_exponent = 0; 
# 696
static constexpr inline int min_exponent10 = 0; 
# 697
static constexpr inline int max_exponent = 0; 
# 698
static constexpr inline int max_exponent10 = 0; 
# 700
static constexpr inline bool has_infinity = false; 
# 701
static constexpr inline bool has_quiet_NaN = false; 
# 702
static constexpr inline bool has_signaling_NaN = false; 
# 703
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 705
static constexpr inline bool has_denorm_loss = false; 
# 708
static constexpr wchar_t infinity() noexcept { return ((wchar_t)0); } 
# 711
static constexpr wchar_t quiet_NaN() noexcept { return ((wchar_t)0); } 
# 714
static constexpr wchar_t signaling_NaN() noexcept { return ((wchar_t)0); } 
# 717
static constexpr wchar_t denorm_min() noexcept { return ((wchar_t)0); } 
# 719
static constexpr inline bool is_iec559 = false; 
# 720
static constexpr inline bool is_bounded = true; 
# 721
static constexpr inline bool is_modulo = (!is_signed); 
# 723
static constexpr inline bool traps = true; 
# 724
static constexpr inline bool tinyness_before = false; 
# 725
static constexpr inline float_round_style round_style = round_toward_zero; 
# 727
}; 
# 797 "/usr/include/c++/13/limits" 3
template<> struct numeric_limits< char16_t>  { 
# 799
static constexpr inline bool is_specialized = true; 
# 802
static constexpr char16_t min() noexcept { return ((((char16_t)(-1)) < 0) ? (-((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0)))) - 1 : ((char16_t)0)); } 
# 805
static constexpr char16_t max() noexcept { return ((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0))); } 
# 808
static constexpr char16_t lowest() noexcept { return min(); } 
# 810
static constexpr inline int digits = ((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)); 
# 811
static constexpr inline int digits10 = ((((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) * (643L)) / (2136)); 
# 812
static constexpr inline int max_digits10 = 0; 
# 813
static constexpr inline bool is_signed = (((char16_t)(-1)) < 0); 
# 814
static constexpr inline bool is_integer = true; 
# 815
static constexpr inline bool is_exact = true; 
# 816
static constexpr inline int radix = 2; 
# 819
static constexpr char16_t epsilon() noexcept { return 0; } 
# 822
static constexpr char16_t round_error() noexcept { return 0; } 
# 824
static constexpr inline int min_exponent = 0; 
# 825
static constexpr inline int min_exponent10 = 0; 
# 826
static constexpr inline int max_exponent = 0; 
# 827
static constexpr inline int max_exponent10 = 0; 
# 829
static constexpr inline bool has_infinity = false; 
# 830
static constexpr inline bool has_quiet_NaN = false; 
# 831
static constexpr inline bool has_signaling_NaN = false; 
# 832
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 833
static constexpr inline bool has_denorm_loss = false; 
# 836
static constexpr char16_t infinity() noexcept { return ((char16_t)0); } 
# 839
static constexpr char16_t quiet_NaN() noexcept { return ((char16_t)0); } 
# 842
static constexpr char16_t signaling_NaN() noexcept { return ((char16_t)0); } 
# 845
static constexpr char16_t denorm_min() noexcept { return ((char16_t)0); } 
# 847
static constexpr inline bool is_iec559 = false; 
# 848
static constexpr inline bool is_bounded = true; 
# 849
static constexpr inline bool is_modulo = (!is_signed); 
# 851
static constexpr inline bool traps = true; 
# 852
static constexpr inline bool tinyness_before = false; 
# 853
static constexpr inline float_round_style round_style = round_toward_zero; 
# 854
}; 
# 858
template<> struct numeric_limits< char32_t>  { 
# 860
static constexpr inline bool is_specialized = true; 
# 863
static constexpr char32_t min() noexcept { return ((((char32_t)(-1)) < (0)) ? (-((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0)))) - (1) : ((char32_t)0)); } 
# 866
static constexpr char32_t max() noexcept { return ((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0))); } 
# 869
static constexpr char32_t lowest() noexcept { return min(); } 
# 871
static constexpr inline int digits = ((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))); 
# 872
static constexpr inline int digits10 = ((((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) * (643L)) / (2136)); 
# 873
static constexpr inline int max_digits10 = 0; 
# 874
static constexpr inline bool is_signed = (((char32_t)(-1)) < (0)); 
# 875
static constexpr inline bool is_integer = true; 
# 876
static constexpr inline bool is_exact = true; 
# 877
static constexpr inline int radix = 2; 
# 880
static constexpr char32_t epsilon() noexcept { return 0; } 
# 883
static constexpr char32_t round_error() noexcept { return 0; } 
# 885
static constexpr inline int min_exponent = 0; 
# 886
static constexpr inline int min_exponent10 = 0; 
# 887
static constexpr inline int max_exponent = 0; 
# 888
static constexpr inline int max_exponent10 = 0; 
# 890
static constexpr inline bool has_infinity = false; 
# 891
static constexpr inline bool has_quiet_NaN = false; 
# 892
static constexpr inline bool has_signaling_NaN = false; 
# 893
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 894
static constexpr inline bool has_denorm_loss = false; 
# 897
static constexpr char32_t infinity() noexcept { return ((char32_t)0); } 
# 900
static constexpr char32_t quiet_NaN() noexcept { return ((char32_t)0); } 
# 903
static constexpr char32_t signaling_NaN() noexcept { return ((char32_t)0); } 
# 906
static constexpr char32_t denorm_min() noexcept { return ((char32_t)0); } 
# 908
static constexpr inline bool is_iec559 = false; 
# 909
static constexpr inline bool is_bounded = true; 
# 910
static constexpr inline bool is_modulo = (!is_signed); 
# 912
static constexpr inline bool traps = true; 
# 913
static constexpr inline bool tinyness_before = false; 
# 914
static constexpr inline float_round_style round_style = round_toward_zero; 
# 915
}; 
# 920
template<> struct numeric_limits< short>  { 
# 922
static constexpr inline bool is_specialized = true; 
# 925
static constexpr short min() noexcept { return (-32767) - 1; } 
# 928
static constexpr short max() noexcept { return 32767; } 
# 932
static constexpr short lowest() noexcept { return min(); } 
# 935
static constexpr inline int digits = ((sizeof(short) * (8)) - (((short)(-1)) < 0)); 
# 936
static constexpr inline int digits10 = ((((sizeof(short) * (8)) - (((short)(-1)) < 0)) * (643L)) / (2136)); 
# 938
static constexpr inline int max_digits10 = 0; 
# 940
static constexpr inline bool is_signed = true; 
# 941
static constexpr inline bool is_integer = true; 
# 942
static constexpr inline bool is_exact = true; 
# 943
static constexpr inline int radix = 2; 
# 946
static constexpr short epsilon() noexcept { return 0; } 
# 949
static constexpr short round_error() noexcept { return 0; } 
# 951
static constexpr inline int min_exponent = 0; 
# 952
static constexpr inline int min_exponent10 = 0; 
# 953
static constexpr inline int max_exponent = 0; 
# 954
static constexpr inline int max_exponent10 = 0; 
# 956
static constexpr inline bool has_infinity = false; 
# 957
static constexpr inline bool has_quiet_NaN = false; 
# 958
static constexpr inline bool has_signaling_NaN = false; 
# 959
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 961
static constexpr inline bool has_denorm_loss = false; 
# 964
static constexpr short infinity() noexcept { return ((short)0); } 
# 967
static constexpr short quiet_NaN() noexcept { return ((short)0); } 
# 970
static constexpr short signaling_NaN() noexcept { return ((short)0); } 
# 973
static constexpr short denorm_min() noexcept { return ((short)0); } 
# 975
static constexpr inline bool is_iec559 = false; 
# 976
static constexpr inline bool is_bounded = true; 
# 977
static constexpr inline bool is_modulo = false; 
# 979
static constexpr inline bool traps = true; 
# 980
static constexpr inline bool tinyness_before = false; 
# 981
static constexpr inline float_round_style round_style = round_toward_zero; 
# 983
}; 
# 987
template<> struct numeric_limits< unsigned short>  { 
# 989
static constexpr inline bool is_specialized = true; 
# 992
static constexpr unsigned short min() noexcept { return 0; } 
# 995
static constexpr unsigned short max() noexcept { return ((32767) * 2U) + (1); } 
# 999
static constexpr unsigned short lowest() noexcept { return min(); } 
# 1002
static constexpr inline int digits = ((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)); 
# 1004
static constexpr inline int digits10 = ((((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)) * (643L)) / (2136)); 
# 1007
static constexpr inline int max_digits10 = 0; 
# 1009
static constexpr inline bool is_signed = false; 
# 1010
static constexpr inline bool is_integer = true; 
# 1011
static constexpr inline bool is_exact = true; 
# 1012
static constexpr inline int radix = 2; 
# 1015
static constexpr unsigned short epsilon() noexcept { return 0; } 
# 1018
static constexpr unsigned short round_error() noexcept { return 0; } 
# 1020
static constexpr inline int min_exponent = 0; 
# 1021
static constexpr inline int min_exponent10 = 0; 
# 1022
static constexpr inline int max_exponent = 0; 
# 1023
static constexpr inline int max_exponent10 = 0; 
# 1025
static constexpr inline bool has_infinity = false; 
# 1026
static constexpr inline bool has_quiet_NaN = false; 
# 1027
static constexpr inline bool has_signaling_NaN = false; 
# 1028
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1030
static constexpr inline bool has_denorm_loss = false; 
# 1033
static constexpr unsigned short infinity() noexcept 
# 1034
{ return static_cast< unsigned short>(0); } 
# 1037
static constexpr unsigned short quiet_NaN() noexcept 
# 1038
{ return static_cast< unsigned short>(0); } 
# 1041
static constexpr unsigned short signaling_NaN() noexcept 
# 1042
{ return static_cast< unsigned short>(0); } 
# 1045
static constexpr unsigned short denorm_min() noexcept 
# 1046
{ return static_cast< unsigned short>(0); } 
# 1048
static constexpr inline bool is_iec559 = false; 
# 1049
static constexpr inline bool is_bounded = true; 
# 1050
static constexpr inline bool is_modulo = true; 
# 1052
static constexpr inline bool traps = true; 
# 1053
static constexpr inline bool tinyness_before = false; 
# 1054
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1056
}; 
# 1060
template<> struct numeric_limits< int>  { 
# 1062
static constexpr inline bool is_specialized = true; 
# 1065
static constexpr int min() noexcept { return (-2147483647) - 1; } 
# 1068
static constexpr int max() noexcept { return 2147483647; } 
# 1072
static constexpr int lowest() noexcept { return min(); } 
# 1075
static constexpr inline int digits = ((sizeof(int) * (8)) - (((int)(-1)) < 0)); 
# 1076
static constexpr inline int digits10 = ((((sizeof(int) * (8)) - (((int)(-1)) < 0)) * (643L)) / (2136)); 
# 1078
static constexpr inline int max_digits10 = 0; 
# 1080
static constexpr inline bool is_signed = true; 
# 1081
static constexpr inline bool is_integer = true; 
# 1082
static constexpr inline bool is_exact = true; 
# 1083
static constexpr inline int radix = 2; 
# 1086
static constexpr int epsilon() noexcept { return 0; } 
# 1089
static constexpr int round_error() noexcept { return 0; } 
# 1091
static constexpr inline int min_exponent = 0; 
# 1092
static constexpr inline int min_exponent10 = 0; 
# 1093
static constexpr inline int max_exponent = 0; 
# 1094
static constexpr inline int max_exponent10 = 0; 
# 1096
static constexpr inline bool has_infinity = false; 
# 1097
static constexpr inline bool has_quiet_NaN = false; 
# 1098
static constexpr inline bool has_signaling_NaN = false; 
# 1099
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1101
static constexpr inline bool has_denorm_loss = false; 
# 1104
static constexpr int infinity() noexcept { return static_cast< int>(0); } 
# 1107
static constexpr int quiet_NaN() noexcept { return static_cast< int>(0); } 
# 1110
static constexpr int signaling_NaN() noexcept { return static_cast< int>(0); } 
# 1113
static constexpr int denorm_min() noexcept { return static_cast< int>(0); } 
# 1115
static constexpr inline bool is_iec559 = false; 
# 1116
static constexpr inline bool is_bounded = true; 
# 1117
static constexpr inline bool is_modulo = false; 
# 1119
static constexpr inline bool traps = true; 
# 1120
static constexpr inline bool tinyness_before = false; 
# 1121
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1123
}; 
# 1127
template<> struct numeric_limits< unsigned>  { 
# 1129
static constexpr inline bool is_specialized = true; 
# 1132
static constexpr unsigned min() noexcept { return 0; } 
# 1135
static constexpr unsigned max() noexcept { return ((2147483647) * 2U) + (1); } 
# 1139
static constexpr unsigned lowest() noexcept { return min(); } 
# 1142
static constexpr inline int digits = ((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))); 
# 1144
static constexpr inline int digits10 = ((((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))) * (643L)) / (2136)); 
# 1147
static constexpr inline int max_digits10 = 0; 
# 1149
static constexpr inline bool is_signed = false; 
# 1150
static constexpr inline bool is_integer = true; 
# 1151
static constexpr inline bool is_exact = true; 
# 1152
static constexpr inline int radix = 2; 
# 1155
static constexpr unsigned epsilon() noexcept { return 0; } 
# 1158
static constexpr unsigned round_error() noexcept { return 0; } 
# 1160
static constexpr inline int min_exponent = 0; 
# 1161
static constexpr inline int min_exponent10 = 0; 
# 1162
static constexpr inline int max_exponent = 0; 
# 1163
static constexpr inline int max_exponent10 = 0; 
# 1165
static constexpr inline bool has_infinity = false; 
# 1166
static constexpr inline bool has_quiet_NaN = false; 
# 1167
static constexpr inline bool has_signaling_NaN = false; 
# 1168
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1170
static constexpr inline bool has_denorm_loss = false; 
# 1173
static constexpr unsigned infinity() noexcept { return static_cast< unsigned>(0); } 
# 1176
static constexpr unsigned quiet_NaN() noexcept 
# 1177
{ return static_cast< unsigned>(0); } 
# 1180
static constexpr unsigned signaling_NaN() noexcept 
# 1181
{ return static_cast< unsigned>(0); } 
# 1184
static constexpr unsigned denorm_min() noexcept 
# 1185
{ return static_cast< unsigned>(0); } 
# 1187
static constexpr inline bool is_iec559 = false; 
# 1188
static constexpr inline bool is_bounded = true; 
# 1189
static constexpr inline bool is_modulo = true; 
# 1191
static constexpr inline bool traps = true; 
# 1192
static constexpr inline bool tinyness_before = false; 
# 1193
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1195
}; 
# 1199
template<> struct numeric_limits< long>  { 
# 1201
static constexpr inline bool is_specialized = true; 
# 1204
static constexpr long min() noexcept { return (-9223372036854775807L) - (1); } 
# 1207
static constexpr long max() noexcept { return 9223372036854775807L; } 
# 1211
static constexpr long lowest() noexcept { return min(); } 
# 1214
static constexpr inline int digits = ((sizeof(long) * (8)) - (((long)(-1)) < (0))); 
# 1215
static constexpr inline int digits10 = ((((sizeof(long) * (8)) - (((long)(-1)) < (0))) * (643L)) / (2136)); 
# 1217
static constexpr inline int max_digits10 = 0; 
# 1219
static constexpr inline bool is_signed = true; 
# 1220
static constexpr inline bool is_integer = true; 
# 1221
static constexpr inline bool is_exact = true; 
# 1222
static constexpr inline int radix = 2; 
# 1225
static constexpr long epsilon() noexcept { return 0; } 
# 1228
static constexpr long round_error() noexcept { return 0; } 
# 1230
static constexpr inline int min_exponent = 0; 
# 1231
static constexpr inline int min_exponent10 = 0; 
# 1232
static constexpr inline int max_exponent = 0; 
# 1233
static constexpr inline int max_exponent10 = 0; 
# 1235
static constexpr inline bool has_infinity = false; 
# 1236
static constexpr inline bool has_quiet_NaN = false; 
# 1237
static constexpr inline bool has_signaling_NaN = false; 
# 1238
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1240
static constexpr inline bool has_denorm_loss = false; 
# 1243
static constexpr long infinity() noexcept { return static_cast< long>(0); } 
# 1246
static constexpr long quiet_NaN() noexcept { return static_cast< long>(0); } 
# 1249
static constexpr long signaling_NaN() noexcept { return static_cast< long>(0); } 
# 1252
static constexpr long denorm_min() noexcept { return static_cast< long>(0); } 
# 1254
static constexpr inline bool is_iec559 = false; 
# 1255
static constexpr inline bool is_bounded = true; 
# 1256
static constexpr inline bool is_modulo = false; 
# 1258
static constexpr inline bool traps = true; 
# 1259
static constexpr inline bool tinyness_before = false; 
# 1260
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1262
}; 
# 1266
template<> struct numeric_limits< unsigned long>  { 
# 1268
static constexpr inline bool is_specialized = true; 
# 1271
static constexpr unsigned long min() noexcept { return 0; } 
# 1274
static constexpr unsigned long max() noexcept { return ((9223372036854775807L) * 2UL) + (1); } 
# 1278
static constexpr unsigned long lowest() noexcept { return min(); } 
# 1281
static constexpr inline int digits = ((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))); 
# 1283
static constexpr inline int digits10 = ((((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))) * (643L)) / (2136)); 
# 1286
static constexpr inline int max_digits10 = 0; 
# 1288
static constexpr inline bool is_signed = false; 
# 1289
static constexpr inline bool is_integer = true; 
# 1290
static constexpr inline bool is_exact = true; 
# 1291
static constexpr inline int radix = 2; 
# 1294
static constexpr unsigned long epsilon() noexcept { return 0; } 
# 1297
static constexpr unsigned long round_error() noexcept { return 0; } 
# 1299
static constexpr inline int min_exponent = 0; 
# 1300
static constexpr inline int min_exponent10 = 0; 
# 1301
static constexpr inline int max_exponent = 0; 
# 1302
static constexpr inline int max_exponent10 = 0; 
# 1304
static constexpr inline bool has_infinity = false; 
# 1305
static constexpr inline bool has_quiet_NaN = false; 
# 1306
static constexpr inline bool has_signaling_NaN = false; 
# 1307
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1309
static constexpr inline bool has_denorm_loss = false; 
# 1312
static constexpr unsigned long infinity() noexcept 
# 1313
{ return static_cast< unsigned long>(0); } 
# 1316
static constexpr unsigned long quiet_NaN() noexcept 
# 1317
{ return static_cast< unsigned long>(0); } 
# 1320
static constexpr unsigned long signaling_NaN() noexcept 
# 1321
{ return static_cast< unsigned long>(0); } 
# 1324
static constexpr unsigned long denorm_min() noexcept 
# 1325
{ return static_cast< unsigned long>(0); } 
# 1327
static constexpr inline bool is_iec559 = false; 
# 1328
static constexpr inline bool is_bounded = true; 
# 1329
static constexpr inline bool is_modulo = true; 
# 1331
static constexpr inline bool traps = true; 
# 1332
static constexpr inline bool tinyness_before = false; 
# 1333
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1335
}; 
# 1339
template<> struct numeric_limits< long long>  { 
# 1341
static constexpr inline bool is_specialized = true; 
# 1344
static constexpr long long min() noexcept { return (-9223372036854775807LL) - (1); } 
# 1347
static constexpr long long max() noexcept { return 9223372036854775807LL; } 
# 1351
static constexpr long long lowest() noexcept { return min(); } 
# 1354
static constexpr inline int digits = ((sizeof(long long) * (8)) - (((long long)(-1)) < (0))); 
# 1356
static constexpr inline int digits10 = ((((sizeof(long long) * (8)) - (((long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1359
static constexpr inline int max_digits10 = 0; 
# 1361
static constexpr inline bool is_signed = true; 
# 1362
static constexpr inline bool is_integer = true; 
# 1363
static constexpr inline bool is_exact = true; 
# 1364
static constexpr inline int radix = 2; 
# 1367
static constexpr long long epsilon() noexcept { return 0; } 
# 1370
static constexpr long long round_error() noexcept { return 0; } 
# 1372
static constexpr inline int min_exponent = 0; 
# 1373
static constexpr inline int min_exponent10 = 0; 
# 1374
static constexpr inline int max_exponent = 0; 
# 1375
static constexpr inline int max_exponent10 = 0; 
# 1377
static constexpr inline bool has_infinity = false; 
# 1378
static constexpr inline bool has_quiet_NaN = false; 
# 1379
static constexpr inline bool has_signaling_NaN = false; 
# 1380
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1382
static constexpr inline bool has_denorm_loss = false; 
# 1385
static constexpr long long infinity() noexcept { return static_cast< long long>(0); } 
# 1388
static constexpr long long quiet_NaN() noexcept { return static_cast< long long>(0); } 
# 1391
static constexpr long long signaling_NaN() noexcept 
# 1392
{ return static_cast< long long>(0); } 
# 1395
static constexpr long long denorm_min() noexcept { return static_cast< long long>(0); } 
# 1397
static constexpr inline bool is_iec559 = false; 
# 1398
static constexpr inline bool is_bounded = true; 
# 1399
static constexpr inline bool is_modulo = false; 
# 1401
static constexpr inline bool traps = true; 
# 1402
static constexpr inline bool tinyness_before = false; 
# 1403
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1405
}; 
# 1409
template<> struct numeric_limits< unsigned long long>  { 
# 1411
static constexpr inline bool is_specialized = true; 
# 1414
static constexpr unsigned long long min() noexcept { return 0; } 
# 1417
static constexpr unsigned long long max() noexcept { return ((9223372036854775807LL) * 2ULL) + (1); } 
# 1421
static constexpr unsigned long long lowest() noexcept { return min(); } 
# 1424
static constexpr inline int digits = ((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))); 
# 1426
static constexpr inline int digits10 = ((((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1429
static constexpr inline int max_digits10 = 0; 
# 1431
static constexpr inline bool is_signed = false; 
# 1432
static constexpr inline bool is_integer = true; 
# 1433
static constexpr inline bool is_exact = true; 
# 1434
static constexpr inline int radix = 2; 
# 1437
static constexpr unsigned long long epsilon() noexcept { return 0; } 
# 1440
static constexpr unsigned long long round_error() noexcept { return 0; } 
# 1442
static constexpr inline int min_exponent = 0; 
# 1443
static constexpr inline int min_exponent10 = 0; 
# 1444
static constexpr inline int max_exponent = 0; 
# 1445
static constexpr inline int max_exponent10 = 0; 
# 1447
static constexpr inline bool has_infinity = false; 
# 1448
static constexpr inline bool has_quiet_NaN = false; 
# 1449
static constexpr inline bool has_signaling_NaN = false; 
# 1450
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1452
static constexpr inline bool has_denorm_loss = false; 
# 1455
static constexpr unsigned long long infinity() noexcept 
# 1456
{ return static_cast< unsigned long long>(0); } 
# 1459
static constexpr unsigned long long quiet_NaN() noexcept 
# 1460
{ return static_cast< unsigned long long>(0); } 
# 1463
static constexpr unsigned long long signaling_NaN() noexcept 
# 1464
{ return static_cast< unsigned long long>(0); } 
# 1467
static constexpr unsigned long long denorm_min() noexcept 
# 1468
{ return static_cast< unsigned long long>(0); } 
# 1470
static constexpr inline bool is_iec559 = false; 
# 1471
static constexpr inline bool is_bounded = true; 
# 1472
static constexpr inline bool is_modulo = true; 
# 1474
static constexpr inline bool traps = true; 
# 1475
static constexpr inline bool tinyness_before = false; 
# 1476
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1478
}; 
# 1637 "/usr/include/c++/13/limits" 3
template<> struct numeric_limits< __int128>  { static constexpr inline bool is_specialized = true; static constexpr __int128 min() noexcept { return ((((__int128)(-1)) < (0)) ? (-((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0)))) - (1) : ((__int128)0)); } static constexpr __int128 max() noexcept { return ((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0))); } static constexpr inline int digits = (128 - 1); static constexpr inline int digits10 = (((128 - 1) * 643L) / (2136)); static constexpr inline bool is_signed = true; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr __int128 epsilon() noexcept { return 0; } static constexpr __int128 round_error() noexcept { return 0; } static constexpr __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr __int128 infinity() noexcept { return static_cast< __int128>(0); } static constexpr __int128 quiet_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 signaling_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 denorm_min() noexcept { return static_cast< __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = false; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; template<> struct numeric_limits< unsigned __int128>  { static constexpr inline bool is_specialized = true; static constexpr unsigned __int128 min() noexcept { return 0; } static constexpr unsigned __int128 max() noexcept { return ((((unsigned __int128)(-1)) < (0)) ? (((((unsigned __int128)1) << ((128 - (((unsigned __int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((unsigned __int128)0))); } static constexpr unsigned __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int digits = 128; static constexpr inline int digits10 = (((128) * 643L) / (2136)); static constexpr inline bool is_signed = false; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr unsigned __int128 epsilon() noexcept { return 0; } static constexpr unsigned __int128 round_error() noexcept { return 0; } static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr unsigned __int128 infinity() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 quiet_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 signaling_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 denorm_min() noexcept { return static_cast< unsigned __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = true; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; 
# 1670 "/usr/include/c++/13/limits" 3
template<> struct numeric_limits< float>  { 
# 1672
static constexpr inline bool is_specialized = true; 
# 1675
static constexpr float min() noexcept { return (1.1754944E-38F); } 
# 1678
static constexpr float max() noexcept { return (3.4028235E38F); } 
# 1682
static constexpr float lowest() noexcept { return -(3.4028235E38F); } 
# 1685
static constexpr inline int digits = 24; 
# 1686
static constexpr inline int digits10 = 6; 
# 1688
static constexpr inline int max_digits10 = ((2) + (((24) * 643L) / (2136))); 
# 1691
static constexpr inline bool is_signed = true; 
# 1692
static constexpr inline bool is_integer = false; 
# 1693
static constexpr inline bool is_exact = false; 
# 1694
static constexpr inline int radix = 2; 
# 1697
static constexpr float epsilon() noexcept { return (1.1920929E-7F); } 
# 1700
static constexpr float round_error() noexcept { return (0.5F); } 
# 1702
static constexpr inline int min_exponent = (-125); 
# 1703
static constexpr inline int min_exponent10 = (-37); 
# 1704
static constexpr inline int max_exponent = 128; 
# 1705
static constexpr inline int max_exponent10 = 38; 
# 1707
static constexpr inline bool has_infinity = (1); 
# 1708
static constexpr inline bool has_quiet_NaN = (1); 
# 1709
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1710
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1712
static constexpr inline bool has_denorm_loss = false; 
# 1716
static constexpr float infinity() noexcept { return __builtin_huge_valf(); } 
# 1719
static constexpr float quiet_NaN() noexcept { return __builtin_nanf(""); } 
# 1722
static constexpr float signaling_NaN() noexcept { return __builtin_nansf(""); } 
# 1725
static constexpr float denorm_min() noexcept { return (1.4E-45F); } 
# 1727
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1729
static constexpr inline bool is_bounded = true; 
# 1730
static constexpr inline bool is_modulo = false; 
# 1732
static constexpr inline bool traps = false; 
# 1733
static constexpr inline bool tinyness_before = false; 
# 1735
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1737
}; 
# 1745
template<> struct numeric_limits< double>  { 
# 1747
static constexpr inline bool is_specialized = true; 
# 1750
static constexpr double min() noexcept { return ((double)(2.2250738585072013831E-308L)); } 
# 1753
static constexpr double max() noexcept { return ((double)(1.7976931348623157081E308L)); } 
# 1757
static constexpr double lowest() noexcept { return -((double)(1.7976931348623157081E308L)); } 
# 1760
static constexpr inline int digits = 53; 
# 1761
static constexpr inline int digits10 = 15; 
# 1763
static constexpr inline int max_digits10 = ((2) + (((53) * 643L) / (2136))); 
# 1766
static constexpr inline bool is_signed = true; 
# 1767
static constexpr inline bool is_integer = false; 
# 1768
static constexpr inline bool is_exact = false; 
# 1769
static constexpr inline int radix = 2; 
# 1772
static constexpr double epsilon() noexcept { return ((double)(2.2204460492503130808E-16L)); } 
# 1775
static constexpr double round_error() noexcept { return (0.5); } 
# 1777
static constexpr inline int min_exponent = (-1021); 
# 1778
static constexpr inline int min_exponent10 = (-307); 
# 1779
static constexpr inline int max_exponent = 1024; 
# 1780
static constexpr inline int max_exponent10 = 308; 
# 1782
static constexpr inline bool has_infinity = (1); 
# 1783
static constexpr inline bool has_quiet_NaN = (1); 
# 1784
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1785
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1787
static constexpr inline bool has_denorm_loss = false; 
# 1791
static constexpr double infinity() noexcept { return __builtin_huge_val(); } 
# 1794
static constexpr double quiet_NaN() noexcept { return __builtin_nan(""); } 
# 1797
static constexpr double signaling_NaN() noexcept { return __builtin_nans(""); } 
# 1800
static constexpr double denorm_min() noexcept { return ((double)(4.940656458412465442E-324L)); } 
# 1802
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1804
static constexpr inline bool is_bounded = true; 
# 1805
static constexpr inline bool is_modulo = false; 
# 1807
static constexpr inline bool traps = false; 
# 1808
static constexpr inline bool tinyness_before = false; 
# 1810
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1812
}; 
# 1820
template<> struct numeric_limits< long double>  { 
# 1822
static constexpr inline bool is_specialized = true; 
# 1825
static constexpr long double min() noexcept { return (3.3621031431120935063E-4932L); } 
# 1828
static constexpr long double max() noexcept { return (1.189731495357231765E4932L); } 
# 1832
static constexpr long double lowest() noexcept { return -(1.189731495357231765E4932L); } 
# 1835
static constexpr inline int digits = 64; 
# 1836
static constexpr inline int digits10 = 18; 
# 1838
static constexpr inline int max_digits10 = ((2) + (((64) * 643L) / (2136))); 
# 1841
static constexpr inline bool is_signed = true; 
# 1842
static constexpr inline bool is_integer = false; 
# 1843
static constexpr inline bool is_exact = false; 
# 1844
static constexpr inline int radix = 2; 
# 1847
static constexpr long double epsilon() noexcept { return (1.084202172485504434E-19L); } 
# 1850
static constexpr long double round_error() noexcept { return (0.5L); } 
# 1852
static constexpr inline int min_exponent = (-16381); 
# 1853
static constexpr inline int min_exponent10 = (-4931); 
# 1854
static constexpr inline int max_exponent = 16384; 
# 1855
static constexpr inline int max_exponent10 = 4932; 
# 1857
static constexpr inline bool has_infinity = (1); 
# 1858
static constexpr inline bool has_quiet_NaN = (1); 
# 1859
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1860
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1862
static constexpr inline bool has_denorm_loss = false; 
# 1866
static constexpr long double infinity() noexcept { return __builtin_huge_vall(); } 
# 1869
static constexpr long double quiet_NaN() noexcept { return __builtin_nanl(""); } 
# 1872
static constexpr long double signaling_NaN() noexcept { return __builtin_nansl(""); } 
# 1875
static constexpr long double denorm_min() noexcept { return (3.6E-4951L); } 
# 1877
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1879
static constexpr inline bool is_bounded = true; 
# 1880
static constexpr inline bool is_modulo = false; 
# 1882
static constexpr inline bool traps = false; 
# 1883
static constexpr inline bool tinyness_before = false; 
# 1885
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1887
}; 
# 2077 "/usr/include/c++/13/limits" 3
}
# 39 "/usr/include/c++/13/tr1/special_function_util.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 50 "/usr/include/c++/13/tr1/special_function_util.h" 3
namespace __detail { 
# 55
template< class _Tp> 
# 56
struct __floating_point_constant { 
# 58
static const _Tp __value; 
# 59
}; 
# 63
template< class _Tp> 
# 64
struct __numeric_constants { 
# 67
static _Tp __pi() throw() 
# 68
{ return static_cast< _Tp>((3.1415926535897932385L)); } 
# 70
static _Tp __pi_2() throw() 
# 71
{ return static_cast< _Tp>((1.5707963267948966193L)); } 
# 73
static _Tp __pi_3() throw() 
# 74
{ return static_cast< _Tp>((1.0471975511965977461L)); } 
# 76
static _Tp __pi_4() throw() 
# 77
{ return static_cast< _Tp>((0.78539816339744830963L)); } 
# 79
static _Tp __1_pi() throw() 
# 80
{ return static_cast< _Tp>((0.31830988618379067154L)); } 
# 82
static _Tp __2_sqrtpi() throw() 
# 83
{ return static_cast< _Tp>((1.1283791670955125738L)); } 
# 85
static _Tp __sqrt2() throw() 
# 86
{ return static_cast< _Tp>((1.4142135623730950488L)); } 
# 88
static _Tp __sqrt3() throw() 
# 89
{ return static_cast< _Tp>((1.7320508075688772936L)); } 
# 91
static _Tp __sqrtpio2() throw() 
# 92
{ return static_cast< _Tp>((1.2533141373155002512L)); } 
# 94
static _Tp __sqrt1_2() throw() 
# 95
{ return static_cast< _Tp>((0.7071067811865475244L)); } 
# 97
static _Tp __lnpi() throw() 
# 98
{ return static_cast< _Tp>((1.1447298858494001742L)); } 
# 100
static _Tp __gamma_e() throw() 
# 101
{ return static_cast< _Tp>((0.5772156649015328606L)); } 
# 103
static _Tp __euler() throw() 
# 104
{ return static_cast< _Tp>((2.7182818284590452354L)); } 
# 105
}; 
# 120 "/usr/include/c++/13/tr1/special_function_util.h" 3
template< class _Tp> inline bool 
# 121
__isnan(const _Tp __x) 
# 122
{ return __builtin_isnan(__x); } 
# 125
template<> inline bool __isnan< float> (float __x) 
# 126
{ return __builtin_isnanf(__x); } 
# 129
template<> inline bool __isnan< long double> (long double __x) 
# 130
{ return __builtin_isnanl(__x); } 
# 133
}
# 139
}
# 51 "/usr/include/c++/13/tr1/gamma.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/13/tr1/gamma.tcc" 3
namespace __detail { 
# 76
template< class _Tp> _Tp 
# 78
__bernoulli_series(unsigned __n) 
# 79
{ 
# 81
static const _Tp __num[28] = {((_Tp)1UL), ((-((_Tp)1UL)) / ((_Tp)2UL)), (((_Tp)1UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)1UL) / ((_Tp)42UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)5UL) / ((_Tp)66UL)), ((_Tp)0UL), ((-((_Tp)691UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)7UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)3617UL)) / ((_Tp)510UL)), ((_Tp)0UL), (((_Tp)43867UL) / ((_Tp)798UL)), ((_Tp)0UL), ((-((_Tp)174611)) / ((_Tp)330UL)), ((_Tp)0UL), (((_Tp)854513UL) / ((_Tp)138UL)), ((_Tp)0UL), ((-((_Tp)236364091UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)8553103UL) / ((_Tp)6UL)), ((_Tp)0UL)}; 
# 98
if (__n == (0)) { 
# 99
return (_Tp)1; }  
# 101
if (__n == (1)) { 
# 102
return (-((_Tp)1)) / ((_Tp)2); }  
# 105
if ((__n % (2)) == (1)) { 
# 106
return (_Tp)0; }  
# 109
if (__n < (28)) { 
# 110
return __num[__n]; }  
# 113
_Tp __fact = ((_Tp)1); 
# 114
if (((__n / (2)) % (2)) == (0)) { 
# 115
__fact *= ((_Tp)(-1)); }  
# 116
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 117
__fact *= (__k / (((_Tp)2) * __numeric_constants< _Tp> ::__pi())); }  
# 118
__fact *= ((_Tp)2); 
# 120
_Tp __sum = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (1000); ++__i) 
# 122
{ 
# 123
_Tp __term = std::pow((_Tp)__i, -((_Tp)__n)); 
# 124
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__sum += __term; 
# 127
}  
# 129
return __fact * __sum; 
# 130
} 
# 139
template< class _Tp> inline _Tp 
# 141
__bernoulli(int __n) 
# 142
{ return __bernoulli_series< _Tp> (__n); } 
# 153
template< class _Tp> _Tp 
# 155
__log_gamma_bernoulli(_Tp __x) 
# 156
{ 
# 157
_Tp __lg = (((__x - ((_Tp)(0.5L))) * std::log(__x)) - __x) + (((_Tp)(0.5L)) * std::log(((_Tp)2) * __numeric_constants< _Tp> ::__pi())); 
# 161
const _Tp __xx = __x * __x; 
# 162
_Tp __help = ((_Tp)1) / __x; 
# 163
for (unsigned __i = (1); __i < (20); ++__i) 
# 164
{ 
# 165
const _Tp __2i = (_Tp)((2) * __i); 
# 166
__help /= ((__2i * (__2i - ((_Tp)1))) * __xx); 
# 167
__lg += (__bernoulli< _Tp> ((2) * __i) * __help); 
# 168
}  
# 170
return __lg; 
# 171
} 
# 181
template< class _Tp> _Tp 
# 183
__log_gamma_lanczos(_Tp __x) 
# 184
{ 
# 185
const _Tp __xm1 = __x - ((_Tp)1); 
# 187
static const _Tp __lanczos_cheb_7[9] = {((_Tp)(0.99999999999980993226L)), ((_Tp)(676.52036812188509857L)), ((_Tp)(-(1259.1392167224028704L))), ((_Tp)(771.32342877765307887L)), ((_Tp)(-(176.61502916214059906L))), ((_Tp)(12.507343278686904814L)), ((_Tp)(-(0.1385710952657201169L))), ((_Tp)(9.9843695780195708595E-6L)), ((_Tp)(1.5056327351493115584E-7L))}; 
# 199
static const _Tp __LOGROOT2PI = ((_Tp)(0.9189385332046727418L)); 
# 202
_Tp __sum = (__lanczos_cheb_7[0]); 
# 203
for (unsigned __k = (1); __k < (9); ++__k) { 
# 204
__sum += ((__lanczos_cheb_7[__k]) / (__xm1 + __k)); }  
# 206
const _Tp __term1 = (__xm1 + ((_Tp)(0.5L))) * std::log((__xm1 + ((_Tp)(7.5L))) / __numeric_constants< _Tp> ::__euler()); 
# 209
const _Tp __term2 = __LOGROOT2PI + std::log(__sum); 
# 210
const _Tp __result = __term1 + (__term2 - ((_Tp)7)); 
# 212
return __result; 
# 213
} 
# 225
template< class _Tp> _Tp 
# 227
__log_gamma(_Tp __x) 
# 228
{ 
# 229
if (__x > ((_Tp)(0.5L))) { 
# 230
return __log_gamma_lanczos(__x); } else 
# 232
{ 
# 233
const _Tp __sin_fact = std::abs(std::sin(__numeric_constants< _Tp> ::__pi() * __x)); 
# 235
if (__sin_fact == ((_Tp)0)) { 
# 236
std::__throw_domain_error("Argument is nonpositive integer in __log_gamma"); }  
# 238
return (__numeric_constants< _Tp> ::__lnpi() - std::log(__sin_fact)) - __log_gamma_lanczos(((_Tp)1) - __x); 
# 241
}  
# 242
} 
# 252
template< class _Tp> _Tp 
# 254
__log_gamma_sign(_Tp __x) 
# 255
{ 
# 256
if (__x > ((_Tp)0)) { 
# 257
return (_Tp)1; } else 
# 259
{ 
# 260
const _Tp __sin_fact = std::sin(__numeric_constants< _Tp> ::__pi() * __x); 
# 262
if (__sin_fact > ((_Tp)0)) { 
# 263
return 1; } else { 
# 264
if (__sin_fact < ((_Tp)0)) { 
# 265
return -((_Tp)1); } else { 
# 267
return (_Tp)0; }  }  
# 268
}  
# 269
} 
# 283
template< class _Tp> _Tp 
# 285
__log_bincoef(unsigned __n, unsigned __k) 
# 286
{ 
# 288
static const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 292
_Tp __coeff = (std::lgamma((_Tp)((1) + __n)) - std::lgamma((_Tp)((1) + __k))) - std::lgamma((_Tp)(((1) + __n) - __k)); 
# 300
} 
# 314
template< class _Tp> _Tp 
# 316
__bincoef(unsigned __n, unsigned __k) 
# 317
{ 
# 319
static const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 323
const _Tp __log_coeff = __log_bincoef< _Tp> (__n, __k); 
# 324
if (__log_coeff > __max_bincoeff) { 
# 325
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 327
return std::exp(__log_coeff); }  
# 328
} 
# 337
template< class _Tp> inline _Tp 
# 339
__gamma(_Tp __x) 
# 340
{ return std::exp(__log_gamma(__x)); } 
# 356
template< class _Tp> _Tp 
# 358
__psi_series(_Tp __x) 
# 359
{ 
# 360
_Tp __sum = (-__numeric_constants< _Tp> ::__gamma_e()) - (((_Tp)1) / __x); 
# 361
const unsigned __max_iter = (100000); 
# 362
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 363
{ 
# 364
const _Tp __term = __x / (__k * (__k + __x)); 
# 365
__sum += __term; 
# 366
if (std::abs(__term / __sum) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 367
break; }  
# 368
}  
# 369
return __sum; 
# 370
} 
# 386
template< class _Tp> _Tp 
# 388
__psi_asymp(_Tp __x) 
# 389
{ 
# 390
_Tp __sum = std::log(__x) - (((_Tp)(0.5L)) / __x); 
# 391
const _Tp __xx = __x * __x; 
# 392
_Tp __xp = __xx; 
# 393
const unsigned __max_iter = (100); 
# 394
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 395
{ 
# 396
const _Tp __term = __bernoulli< _Tp> ((2) * __k) / (((2) * __k) * __xp); 
# 397
__sum -= __term; 
# 398
if (std::abs(__term / __sum) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 399
break; }  
# 400
__xp *= __xx; 
# 401
}  
# 402
return __sum; 
# 403
} 
# 417
template< class _Tp> _Tp 
# 419
__psi(_Tp __x) 
# 420
{ 
# 421
const int __n = static_cast< int>(__x + (0.5L)); 
# 422
const _Tp __eps = ((_Tp)4) * std::template numeric_limits< _Tp> ::epsilon(); 
# 423
if ((__n <= 0) && (std::abs(__x - ((_Tp)__n)) < __eps)) { 
# 424
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 425
if (__x < ((_Tp)0)) 
# 426
{ 
# 427
const _Tp __pi = __numeric_constants< _Tp> ::__pi(); 
# 428
return __psi(((_Tp)1) - __x) - ((__pi * std::cos(__pi * __x)) / std::sin(__pi * __x)); 
# 430
} else { 
# 431
if (__x > ((_Tp)100)) { 
# 432
return __psi_asymp(__x); } else { 
# 434
return __psi_series(__x); }  }  }  
# 435
} 
# 446
template< class _Tp> _Tp 
# 448
__psi(unsigned __n, _Tp __x) 
# 449
{ 
# 450
if (__x <= ((_Tp)0)) { 
# 451
std::__throw_domain_error("Argument out of range in __psi"); } else { 
# 453
if (__n == (0)) { 
# 454
return __psi(__x); } else 
# 456
{ 
# 457
const _Tp __hzeta = __hurwitz_zeta((_Tp)(__n + (1)), __x); 
# 459
const _Tp __ln_nfact = std::lgamma((_Tp)(__n + (1))); 
# 463
_Tp __result = std::exp(__ln_nfact) * __hzeta; 
# 464
if ((__n % (2)) == (1)) { 
# 465
__result = (-__result); }  
# 466
return __result; 
# 467
}  }  
# 468
} 
# 469
}
# 476
}
# 55 "/usr/include/c++/13/tr1/bessel_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 71 "/usr/include/c++/13/tr1/bessel_function.tcc" 3
namespace __detail { 
# 98
template< class _Tp> void 
# 100
__gamma_temme(_Tp __mu, _Tp &
# 101
__gam1, _Tp &__gam2, _Tp &__gampl, _Tp &__gammi) 
# 102
{ 
# 104
__gampl = (((_Tp)1) / std::tgamma(((_Tp)1) + __mu)); 
# 105
__gammi = (((_Tp)1) / std::tgamma(((_Tp)1) - __mu)); 
# 111
if (std::abs(__mu) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 112
__gam1 = (-((_Tp)__numeric_constants< _Tp> ::__gamma_e())); } else { 
# 114
__gam1 = ((__gammi - __gampl) / (((_Tp)2) * __mu)); }  
# 116
__gam2 = ((__gammi + __gampl) / ((_Tp)2)); 
# 119
} 
# 136
template< class _Tp> void 
# 138
__bessel_jn(_Tp __nu, _Tp __x, _Tp &
# 139
__Jnu, _Tp &__Nnu, _Tp &__Jpnu, _Tp &__Npnu) 
# 140
{ 
# 141
if (__x == ((_Tp)0)) 
# 142
{ 
# 143
if (__nu == ((_Tp)0)) 
# 144
{ 
# 145
__Jnu = ((_Tp)1); 
# 146
__Jpnu = ((_Tp)0); 
# 147
} else { 
# 148
if (__nu == ((_Tp)1)) 
# 149
{ 
# 150
__Jnu = ((_Tp)0); 
# 151
__Jpnu = ((_Tp)(0.5L)); 
# 152
} else 
# 154
{ 
# 155
__Jnu = ((_Tp)0); 
# 156
__Jpnu = ((_Tp)0); 
# 157
}  }  
# 158
__Nnu = (-std::template numeric_limits< _Tp> ::infinity()); 
# 159
__Npnu = std::template numeric_limits< _Tp> ::infinity(); 
# 160
return; 
# 161
}  
# 163
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 168
const _Tp __fp_min = std::sqrt(std::template numeric_limits< _Tp> ::min()); 
# 169
const int __max_iter = 15000; 
# 170
const _Tp __x_min = ((_Tp)2); 
# 172
const int __nl = (__x < __x_min) ? static_cast< int>(__nu + ((_Tp)(0.5L))) : std::max(0, static_cast< int>((__nu - __x) + ((_Tp)(1.5L)))); 
# 176
const _Tp __mu = __nu - __nl; 
# 177
const _Tp __mu2 = __mu * __mu; 
# 178
const _Tp __xi = ((_Tp)1) / __x; 
# 179
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 180
_Tp __w = __xi2 / __numeric_constants< _Tp> ::__pi(); 
# 181
int __isign = 1; 
# 182
_Tp __h = __nu * __xi; 
# 183
if (__h < __fp_min) { 
# 184
__h = __fp_min; }  
# 185
_Tp __b = __xi2 * __nu; 
# 186
_Tp __d = ((_Tp)0); 
# 187
_Tp __c = __h; 
# 188
int __i; 
# 189
for (__i = 1; __i <= __max_iter; ++__i) 
# 190
{ 
# 191
__b += __xi2; 
# 192
__d = (__b - __d); 
# 193
if (std::abs(__d) < __fp_min) { 
# 194
__d = __fp_min; }  
# 195
__c = (__b - (((_Tp)1) / __c)); 
# 196
if (std::abs(__c) < __fp_min) { 
# 197
__c = __fp_min; }  
# 198
__d = (((_Tp)1) / __d); 
# 199
const _Tp __del = __c * __d; 
# 200
__h *= __del; 
# 201
if (__d < ((_Tp)0)) { 
# 202
__isign = (-__isign); }  
# 203
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 204
break; }  
# 205
}  
# 206
if (__i > __max_iter) { 
# 207
std::__throw_runtime_error("Argument x too large in __bessel_jn; try asymptotic expansion."); }  
# 209
_Tp __Jnul = __isign * __fp_min; 
# 210
_Tp __Jpnul = __h * __Jnul; 
# 211
_Tp __Jnul1 = __Jnul; 
# 212
_Tp __Jpnu1 = __Jpnul; 
# 213
_Tp __fact = __nu * __xi; 
# 214
for (int __l = __nl; __l >= 1; --__l) 
# 215
{ 
# 216
const _Tp __Jnutemp = (__fact * __Jnul) + __Jpnul; 
# 217
__fact -= __xi; 
# 218
__Jpnul = ((__fact * __Jnutemp) - __Jnul); 
# 219
__Jnul = __Jnutemp; 
# 220
}  
# 221
if (__Jnul == ((_Tp)0)) { 
# 222
__Jnul = __eps; }  
# 223
_Tp __f = __Jpnul / __Jnul; 
# 224
_Tp __Nmu, __Nnu1, __Npmu, __Jmu; 
# 225
if (__x < __x_min) 
# 226
{ 
# 227
const _Tp __x2 = __x / ((_Tp)2); 
# 228
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 229
_Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 231
_Tp __d = (-std::log(__x2)); 
# 232
_Tp __e = __mu * __d; 
# 233
_Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 235
_Tp __gam1, __gam2, __gampl, __gammi; 
# 236
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 237
_Tp __ff = ((((_Tp)2) / __numeric_constants< _Tp> ::__pi()) * __fact) * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 239
__e = std::exp(__e); 
# 240
_Tp __p = __e / (__numeric_constants< _Tp> ::__pi() * __gampl); 
# 241
_Tp __q = ((_Tp)1) / ((__e * __numeric_constants< _Tp> ::__pi()) * __gammi); 
# 242
const _Tp __pimu2 = __pimu / ((_Tp)2); 
# 243
_Tp __fact3 = (std::abs(__pimu2) < __eps) ? (_Tp)1 : (std::sin(__pimu2) / __pimu2); 
# 245
_Tp __r = ((__numeric_constants< _Tp> ::__pi() * __pimu2) * __fact3) * __fact3; 
# 246
_Tp __c = ((_Tp)1); 
# 247
__d = ((-__x2) * __x2); 
# 248
_Tp __sum = __ff + (__r * __q); 
# 249
_Tp __sum1 = __p; 
# 250
for (__i = 1; __i <= __max_iter; ++__i) 
# 251
{ 
# 252
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 253
__c *= (__d / ((_Tp)__i)); 
# 254
__p /= (((_Tp)__i) - __mu); 
# 255
__q /= (((_Tp)__i) + __mu); 
# 256
const _Tp __del = __c * (__ff + (__r * __q)); 
# 257
__sum += __del; 
# 258
const _Tp __del1 = (__c * __p) - (__i * __del); 
# 259
__sum1 += __del1; 
# 260
if (std::abs(__del) < (__eps * (((_Tp)1) + std::abs(__sum)))) { 
# 261
break; }  
# 262
}  
# 263
if (__i > __max_iter) { 
# 264
std::__throw_runtime_error("Bessel y series failed to converge in __bessel_jn."); }  
# 266
__Nmu = (-__sum); 
# 267
__Nnu1 = ((-__sum1) * __xi2); 
# 268
__Npmu = (((__mu * __xi) * __Nmu) - __Nnu1); 
# 269
__Jmu = (__w / (__Npmu - (__f * __Nmu))); 
# 270
} else 
# 272
{ 
# 273
_Tp __a = ((_Tp)(0.25L)) - __mu2; 
# 274
_Tp __q = ((_Tp)1); 
# 275
_Tp __p = ((-__xi) / ((_Tp)2)); 
# 276
_Tp __br = ((_Tp)2) * __x; 
# 277
_Tp __bi = ((_Tp)2); 
# 278
_Tp __fact = (__a * __xi) / ((__p * __p) + (__q * __q)); 
# 279
_Tp __cr = __br + (__q * __fact); 
# 280
_Tp __ci = __bi + (__p * __fact); 
# 281
_Tp __den = (__br * __br) + (__bi * __bi); 
# 282
_Tp __dr = __br / __den; 
# 283
_Tp __di = (-__bi) / __den; 
# 284
_Tp __dlr = (__cr * __dr) - (__ci * __di); 
# 285
_Tp __dli = (__cr * __di) + (__ci * __dr); 
# 286
_Tp __temp = (__p * __dlr) - (__q * __dli); 
# 287
__q = ((__p * __dli) + (__q * __dlr)); 
# 288
__p = __temp; 
# 289
int __i; 
# 290
for (__i = 2; __i <= __max_iter; ++__i) 
# 291
{ 
# 292
__a += ((_Tp)(2 * (__i - 1))); 
# 293
__bi += ((_Tp)2); 
# 294
__dr = ((__a * __dr) + __br); 
# 295
__di = ((__a * __di) + __bi); 
# 296
if ((std::abs(__dr) + std::abs(__di)) < __fp_min) { 
# 297
__dr = __fp_min; }  
# 298
__fact = (__a / ((__cr * __cr) + (__ci * __ci))); 
# 299
__cr = (__br + (__cr * __fact)); 
# 300
__ci = (__bi - (__ci * __fact)); 
# 301
if ((std::abs(__cr) + std::abs(__ci)) < __fp_min) { 
# 302
__cr = __fp_min; }  
# 303
__den = ((__dr * __dr) + (__di * __di)); 
# 304
__dr /= __den; 
# 305
__di /= (-__den); 
# 306
__dlr = ((__cr * __dr) - (__ci * __di)); 
# 307
__dli = ((__cr * __di) + (__ci * __dr)); 
# 308
__temp = ((__p * __dlr) - (__q * __dli)); 
# 309
__q = ((__p * __dli) + (__q * __dlr)); 
# 310
__p = __temp; 
# 311
if ((std::abs(__dlr - ((_Tp)1)) + std::abs(__dli)) < __eps) { 
# 312
break; }  
# 313
}  
# 314
if (__i > __max_iter) { 
# 315
std::__throw_runtime_error("Lentz\'s method failed in __bessel_jn."); }  
# 317
const _Tp __gam = (__p - __f) / __q; 
# 318
__Jmu = std::sqrt(__w / (((__p - __f) * __gam) + __q)); 
# 320
__Jmu = std::copysign(__Jmu, __Jnul); 
# 325
__Nmu = (__gam * __Jmu); 
# 326
__Npmu = ((__p + (__q / __gam)) * __Nmu); 
# 327
__Nnu1 = (((__mu * __xi) * __Nmu) - __Npmu); 
# 328
}  
# 329
__fact = (__Jmu / __Jnul); 
# 330
__Jnu = (__fact * __Jnul1); 
# 331
__Jpnu = (__fact * __Jpnu1); 
# 332
for (__i = 1; __i <= __nl; ++__i) 
# 333
{ 
# 334
const _Tp __Nnutemp = (((__mu + __i) * __xi2) * __Nnu1) - __Nmu; 
# 335
__Nmu = __Nnu1; 
# 336
__Nnu1 = __Nnutemp; 
# 337
}  
# 338
__Nnu = __Nmu; 
# 339
__Npnu = (((__nu * __xi) * __Nmu) - __Nnu1); 
# 342
} 
# 361
template< class _Tp> void 
# 363
__cyl_bessel_jn_asymp(_Tp __nu, _Tp __x, _Tp &__Jnu, _Tp &__Nnu) 
# 364
{ 
# 365
const _Tp __mu = (((_Tp)4) * __nu) * __nu; 
# 366
const _Tp __8x = ((_Tp)8) * __x; 
# 368
_Tp __P = ((_Tp)0); 
# 369
_Tp __Q = ((_Tp)0); 
# 371
_Tp __k = ((_Tp)0); 
# 372
_Tp __term = ((_Tp)1); 
# 374
int __epsP = 0; 
# 375
int __epsQ = 0; 
# 377
_Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 379
do 
# 380
{ 
# 381
__term *= ((__k == 0) ? (_Tp)1 : ((-(__mu - (((2 * __k) - 1) * ((2 * __k) - 1)))) / (__k * __8x))); 
# 385
__epsP = (std::abs(__term) < (__eps * std::abs(__P))); 
# 386
__P += __term; 
# 388
__k++; 
# 390
__term *= ((__mu - (((2 * __k) - 1) * ((2 * __k) - 1))) / (__k * __8x)); 
# 391
__epsQ = (std::abs(__term) < (__eps * std::abs(__Q))); 
# 392
__Q += __term; 
# 394
if (__epsP && __epsQ && (__k > (__nu / (2.0)))) { 
# 395
break; }  
# 397
__k++; 
# 398
} 
# 399
while (__k < 1000); 
# 401
const _Tp __chi = __x - ((__nu + ((_Tp)(0.5L))) * __numeric_constants< _Tp> ::__pi_2()); 
# 404
const _Tp __c = std::cos(__chi); 
# 405
const _Tp __s = std::sin(__chi); 
# 407
const _Tp __coef = std::sqrt(((_Tp)2) / (__numeric_constants< _Tp> ::__pi() * __x)); 
# 410
__Jnu = (__coef * ((__c * __P) - (__s * __Q))); 
# 411
__Nnu = (__coef * ((__s * __P) + (__c * __Q))); 
# 414
} 
# 444
template< class _Tp> _Tp 
# 446
__cyl_bessel_ij_series(_Tp __nu, _Tp __x, _Tp __sgn, unsigned 
# 447
__max_iter) 
# 448
{ 
# 449
if (__x == ((_Tp)0)) { 
# 450
return (__nu == ((_Tp)0)) ? (_Tp)1 : ((_Tp)0); }  
# 452
const _Tp __x2 = __x / ((_Tp)2); 
# 453
_Tp __fact = __nu * std::log(__x2); 
# 455
__fact -= std::lgamma(__nu + ((_Tp)1)); 
# 459
__fact = std::exp(__fact); 
# 460
const _Tp __xx4 = (__sgn * __x2) * __x2; 
# 461
_Tp __Jn = ((_Tp)1); 
# 462
_Tp __term = ((_Tp)1); 
# 464
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 465
{ 
# 466
__term *= (__xx4 / (((_Tp)__i) * (__nu + ((_Tp)__i)))); 
# 467
__Jn += __term; 
# 468
if (std::abs(__term / __Jn) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 469
break; }  
# 470
}  
# 472
return __fact * __Jn; 
# 473
} 
# 490
template< class _Tp> _Tp 
# 492
__cyl_bessel_j(_Tp __nu, _Tp __x) 
# 493
{ 
# 494
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 495
std::__throw_domain_error("Bad argument in __cyl_bessel_j."); } else { 
# 497
if (__isnan(__nu) || __isnan(__x)) { 
# 498
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 499
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 500
return __cyl_bessel_ij_series(__nu, __x, -((_Tp)1), 200); } else { 
# 501
if (__x > ((_Tp)1000)) 
# 502
{ 
# 503
_Tp __J_nu, __N_nu; 
# 504
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 505
return __J_nu; 
# 506
} else 
# 508
{ 
# 509
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 510
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 511
return __J_nu; 
# 512
}  }  }  }  
# 513
} 
# 532
template< class _Tp> _Tp 
# 534
__cyl_neumann_n(_Tp __nu, _Tp __x) 
# 535
{ 
# 536
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 537
std::__throw_domain_error("Bad argument in __cyl_neumann_n."); } else { 
# 539
if (__isnan(__nu) || __isnan(__x)) { 
# 540
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 541
if (__x > ((_Tp)1000)) 
# 542
{ 
# 543
_Tp __J_nu, __N_nu; 
# 544
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 545
return __N_nu; 
# 546
} else 
# 548
{ 
# 549
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 550
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 551
return __N_nu; 
# 552
}  }  }  
# 553
} 
# 569
template< class _Tp> void 
# 571
__sph_bessel_jn(unsigned __n, _Tp __x, _Tp &
# 572
__j_n, _Tp &__n_n, _Tp &__jp_n, _Tp &__np_n) 
# 573
{ 
# 574
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 576
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 577
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 579
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 582
__j_n = (__factor * __J_nu); 
# 583
__n_n = (__factor * __N_nu); 
# 584
__jp_n = ((__factor * __Jp_nu) - (__j_n / (((_Tp)2) * __x))); 
# 585
__np_n = ((__factor * __Np_nu) - (__n_n / (((_Tp)2) * __x))); 
# 588
} 
# 604
template< class _Tp> _Tp 
# 606
__sph_bessel(unsigned __n, _Tp __x) 
# 607
{ 
# 608
if (__x < ((_Tp)0)) { 
# 609
std::__throw_domain_error("Bad argument in __sph_bessel."); } else { 
# 611
if (__isnan(__x)) { 
# 612
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 613
if (__x == ((_Tp)0)) 
# 614
{ 
# 615
if (__n == (0)) { 
# 616
return (_Tp)1; } else { 
# 618
return (_Tp)0; }  
# 619
} else 
# 621
{ 
# 622
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 623
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 624
return __j_n; 
# 625
}  }  }  
# 626
} 
# 642
template< class _Tp> _Tp 
# 644
__sph_neumann(unsigned __n, _Tp __x) 
# 645
{ 
# 646
if (__x < ((_Tp)0)) { 
# 647
std::__throw_domain_error("Bad argument in __sph_neumann."); } else { 
# 649
if (__isnan(__x)) { 
# 650
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 651
if (__x == ((_Tp)0)) { 
# 652
return -std::template numeric_limits< _Tp> ::infinity(); } else 
# 654
{ 
# 655
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 656
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 657
return __n_n; 
# 658
}  }  }  
# 659
} 
# 660
}
# 667
}
# 49 "/usr/include/c++/13/tr1/beta_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/13/tr1/beta_function.tcc" 3
namespace __detail { 
# 79
template< class _Tp> _Tp 
# 81
__beta_gamma(_Tp __x, _Tp __y) 
# 82
{ 
# 84
_Tp __bet; 
# 86
if (__x > __y) 
# 87
{ 
# 88
__bet = (std::tgamma(__x) / std::tgamma(__x + __y)); 
# 90
__bet *= std::tgamma(__y); 
# 91
} else 
# 93
{ 
# 94
__bet = (std::tgamma(__y) / std::tgamma(__x + __y)); 
# 96
__bet *= std::tgamma(__x); 
# 97
}  
# 111 "/usr/include/c++/13/tr1/beta_function.tcc" 3
return __bet; 
# 112
} 
# 127
template< class _Tp> _Tp 
# 129
__beta_lgamma(_Tp __x, _Tp __y) 
# 130
{ 
# 132
_Tp __bet = (std::lgamma(__x) + std::lgamma(__y)) - std::lgamma(__x + __y); 
# 140
__bet = std::exp(__bet); 
# 141
return __bet; 
# 142
} 
# 158
template< class _Tp> _Tp 
# 160
__beta_product(_Tp __x, _Tp __y) 
# 161
{ 
# 163
_Tp __bet = (__x + __y) / (__x * __y); 
# 165
unsigned __max_iter = (1000000); 
# 166
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 167
{ 
# 168
_Tp __term = (((_Tp)1) + ((__x + __y) / __k)) / ((((_Tp)1) + (__x / __k)) * (((_Tp)1) + (__y / __k))); 
# 170
__bet *= __term; 
# 171
}  
# 173
return __bet; 
# 174
} 
# 189
template< class _Tp> inline _Tp 
# 191
__beta(_Tp __x, _Tp __y) 
# 192
{ 
# 193
if (__isnan(__x) || __isnan(__y)) { 
# 194
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 196
return __beta_lgamma(__x, __y); }  
# 197
} 
# 198
}
# 205
}
# 45 "/usr/include/c++/13/tr1/ell_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 59 "/usr/include/c++/13/tr1/ell_integral.tcc" 3
namespace __detail { 
# 76
template< class _Tp> _Tp 
# 78
__ellint_rf(_Tp __x, _Tp __y, _Tp __z) 
# 79
{ 
# 80
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 81
const _Tp __lolim = ((_Tp)5) * __min; 
# 83
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 84
std::__throw_domain_error("Argument less than zero in __ellint_rf."); } else { 
# 86
if ((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) { 
# 88
std::__throw_domain_error("Argument too small in __ellint_rf"); } else 
# 90
{ 
# 91
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 92
const _Tp __c1 = (((_Tp)1) / ((_Tp)24)); 
# 93
const _Tp __c2 = (((_Tp)1) / ((_Tp)10)); 
# 94
const _Tp __c3 = (((_Tp)3) / ((_Tp)44)); 
# 95
const _Tp __c4 = (((_Tp)1) / ((_Tp)14)); 
# 97
_Tp __xn = __x; 
# 98
_Tp __yn = __y; 
# 99
_Tp __zn = __z; 
# 101
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 102
const _Tp __errtol = std::pow(__eps, ((_Tp)1) / ((_Tp)6)); 
# 103
_Tp __mu; 
# 104
_Tp __xndev, __yndev, __zndev; 
# 106
const unsigned __max_iter = (100); 
# 107
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 108
{ 
# 109
__mu = (((__xn + __yn) + __zn) / ((_Tp)3)); 
# 110
__xndev = (2 - ((__mu + __xn) / __mu)); 
# 111
__yndev = (2 - ((__mu + __yn) / __mu)); 
# 112
__zndev = (2 - ((__mu + __zn) / __mu)); 
# 113
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 114
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 115
if (__epsilon < __errtol) { 
# 116
break; }  
# 117
const _Tp __xnroot = std::sqrt(__xn); 
# 118
const _Tp __ynroot = std::sqrt(__yn); 
# 119
const _Tp __znroot = std::sqrt(__zn); 
# 120
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 122
__xn = (__c0 * (__xn + __lambda)); 
# 123
__yn = (__c0 * (__yn + __lambda)); 
# 124
__zn = (__c0 * (__zn + __lambda)); 
# 125
}  
# 127
const _Tp __e2 = (__xndev * __yndev) - (__zndev * __zndev); 
# 128
const _Tp __e3 = (__xndev * __yndev) * __zndev; 
# 129
const _Tp __s = (((_Tp)1) + ((((__c1 * __e2) - __c2) - (__c3 * __e3)) * __e2)) + (__c4 * __e3); 
# 132
return __s / std::sqrt(__mu); 
# 133
}  }  
# 134
} 
# 153
template< class _Tp> _Tp 
# 155
__comp_ellint_1_series(_Tp __k) 
# 156
{ 
# 158
const _Tp __kk = __k * __k; 
# 160
_Tp __term = __kk / ((_Tp)4); 
# 161
_Tp __sum = ((_Tp)1) + __term; 
# 163
const unsigned __max_iter = (1000); 
# 164
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 165
{ 
# 166
__term *= (((((2) * __i) - (1)) * __kk) / ((2) * __i)); 
# 167
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 168
break; }  
# 169
__sum += __term; 
# 170
}  
# 172
return __numeric_constants< _Tp> ::__pi_2() * __sum; 
# 173
} 
# 191
template< class _Tp> _Tp 
# 193
__comp_ellint_1(_Tp __k) 
# 194
{ 
# 196
if (__isnan(__k)) { 
# 197
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 198
if (std::abs(__k) >= ((_Tp)1)) { 
# 199
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 201
return __ellint_rf((_Tp)0, ((_Tp)1) - (__k * __k), (_Tp)1); }  }  
# 202
} 
# 219
template< class _Tp> _Tp 
# 221
__ellint_1(_Tp __k, _Tp __phi) 
# 222
{ 
# 224
if (__isnan(__k) || __isnan(__phi)) { 
# 225
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 226
if (std::abs(__k) > ((_Tp)1)) { 
# 227
std::__throw_domain_error("Bad argument in __ellint_1."); } else 
# 229
{ 
# 231
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 233
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 236
const _Tp __s = std::sin(__phi_red); 
# 237
const _Tp __c = std::cos(__phi_red); 
# 239
const _Tp __F = __s * __ellint_rf(__c * __c, ((_Tp)1) - (((__k * __k) * __s) * __s), (_Tp)1); 
# 243
if (__n == 0) { 
# 244
return __F; } else { 
# 246
return __F + ((((_Tp)2) * __n) * __comp_ellint_1(__k)); }  
# 247
}  }  
# 248
} 
# 266
template< class _Tp> _Tp 
# 268
__comp_ellint_2_series(_Tp __k) 
# 269
{ 
# 271
const _Tp __kk = __k * __k; 
# 273
_Tp __term = __kk; 
# 274
_Tp __sum = __term; 
# 276
const unsigned __max_iter = (1000); 
# 277
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 278
{ 
# 279
const _Tp __i2m = ((2) * __i) - (1); 
# 280
const _Tp __i2 = (2) * __i; 
# 281
__term *= (((__i2m * __i2m) * __kk) / (__i2 * __i2)); 
# 282
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 283
break; }  
# 284
__sum += (__term / __i2m); 
# 285
}  
# 287
return __numeric_constants< _Tp> ::__pi_2() * (((_Tp)1) - __sum); 
# 288
} 
# 314
template< class _Tp> _Tp 
# 316
__ellint_rd(_Tp __x, _Tp __y, _Tp __z) 
# 317
{ 
# 318
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 319
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 320
const _Tp __max = std::template numeric_limits< _Tp> ::max(); 
# 321
const _Tp __lolim = (((_Tp)2) / std::pow(__max, ((_Tp)2) / ((_Tp)3))); 
# 323
if ((__x < ((_Tp)0)) || (__y < ((_Tp)0))) { 
# 324
std::__throw_domain_error("Argument less than zero in __ellint_rd."); } else { 
# 326
if (((__x + __y) < __lolim) || (__z < __lolim)) { 
# 327
std::__throw_domain_error("Argument too small in __ellint_rd."); } else 
# 330
{ 
# 331
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 332
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 333
const _Tp __c2 = (((_Tp)1) / ((_Tp)6)); 
# 334
const _Tp __c3 = (((_Tp)9) / ((_Tp)22)); 
# 335
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 337
_Tp __xn = __x; 
# 338
_Tp __yn = __y; 
# 339
_Tp __zn = __z; 
# 340
_Tp __sigma = ((_Tp)0); 
# 341
_Tp __power4 = ((_Tp)1); 
# 343
_Tp __mu; 
# 344
_Tp __xndev, __yndev, __zndev; 
# 346
const unsigned __max_iter = (100); 
# 347
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 348
{ 
# 349
__mu = (((__xn + __yn) + (((_Tp)3) * __zn)) / ((_Tp)5)); 
# 350
__xndev = ((__mu - __xn) / __mu); 
# 351
__yndev = ((__mu - __yn) / __mu); 
# 352
__zndev = ((__mu - __zn) / __mu); 
# 353
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 354
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 355
if (__epsilon < __errtol) { 
# 356
break; }  
# 357
_Tp __xnroot = std::sqrt(__xn); 
# 358
_Tp __ynroot = std::sqrt(__yn); 
# 359
_Tp __znroot = std::sqrt(__zn); 
# 360
_Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 362
__sigma += (__power4 / (__znroot * (__zn + __lambda))); 
# 363
__power4 *= __c0; 
# 364
__xn = (__c0 * (__xn + __lambda)); 
# 365
__yn = (__c0 * (__yn + __lambda)); 
# 366
__zn = (__c0 * (__zn + __lambda)); 
# 367
}  
# 369
_Tp __ea = __xndev * __yndev; 
# 370
_Tp __eb = __zndev * __zndev; 
# 371
_Tp __ec = __ea - __eb; 
# 372
_Tp __ed = __ea - (((_Tp)6) * __eb); 
# 373
_Tp __ef = (__ed + __ec) + __ec; 
# 374
_Tp __s1 = __ed * (((-__c1) + ((__c3 * __ed) / ((_Tp)3))) - ((((((_Tp)3) * __c4) * __zndev) * __ef) / ((_Tp)2))); 
# 377
_Tp __s2 = __zndev * ((__c2 * __ef) + (__zndev * ((((-__c3) * __ec) - (__zndev * __c4)) - __ea))); 
# 381
return (((_Tp)3) * __sigma) + ((__power4 * ((((_Tp)1) + __s1) + __s2)) / (__mu * std::sqrt(__mu))); 
# 383
}  }  
# 384
} 
# 399
template< class _Tp> _Tp 
# 401
__comp_ellint_2(_Tp __k) 
# 402
{ 
# 404
if (__isnan(__k)) { 
# 405
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 406
if (std::abs(__k) == 1) { 
# 407
return (_Tp)1; } else { 
# 408
if (std::abs(__k) > ((_Tp)1)) { 
# 409
std::__throw_domain_error("Bad argument in __comp_ellint_2."); } else 
# 411
{ 
# 412
const _Tp __kk = __k * __k; 
# 414
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) - ((__kk * __ellint_rd((_Tp)0, ((_Tp)1) - __kk, (_Tp)1)) / ((_Tp)3)); 
# 416
}  }  }  
# 417
} 
# 433
template< class _Tp> _Tp 
# 435
__ellint_2(_Tp __k, _Tp __phi) 
# 436
{ 
# 438
if (__isnan(__k) || __isnan(__phi)) { 
# 439
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 440
if (std::abs(__k) > ((_Tp)1)) { 
# 441
std::__throw_domain_error("Bad argument in __ellint_2."); } else 
# 443
{ 
# 445
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 447
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 450
const _Tp __kk = __k * __k; 
# 451
const _Tp __s = std::sin(__phi_red); 
# 452
const _Tp __ss = __s * __s; 
# 453
const _Tp __sss = __ss * __s; 
# 454
const _Tp __c = std::cos(__phi_red); 
# 455
const _Tp __cc = __c * __c; 
# 457
const _Tp __E = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) - (((__kk * __sss) * __ellint_rd(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) / ((_Tp)3)); 
# 463
if (__n == 0) { 
# 464
return __E; } else { 
# 466
return __E + ((((_Tp)2) * __n) * __comp_ellint_2(__k)); }  
# 467
}  }  
# 468
} 
# 492
template< class _Tp> _Tp 
# 494
__ellint_rc(_Tp __x, _Tp __y) 
# 495
{ 
# 496
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 497
const _Tp __lolim = ((_Tp)5) * __min; 
# 499
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || ((__x + __y) < __lolim)) { 
# 500
std::__throw_domain_error("Argument less than zero in __ellint_rc."); } else 
# 503
{ 
# 504
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 505
const _Tp __c1 = (((_Tp)1) / ((_Tp)7)); 
# 506
const _Tp __c2 = (((_Tp)9) / ((_Tp)22)); 
# 507
const _Tp __c3 = (((_Tp)3) / ((_Tp)10)); 
# 508
const _Tp __c4 = (((_Tp)3) / ((_Tp)8)); 
# 510
_Tp __xn = __x; 
# 511
_Tp __yn = __y; 
# 513
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 514
const _Tp __errtol = std::pow(__eps / ((_Tp)30), ((_Tp)1) / ((_Tp)6)); 
# 515
_Tp __mu; 
# 516
_Tp __sn; 
# 518
const unsigned __max_iter = (100); 
# 519
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 520
{ 
# 521
__mu = ((__xn + (((_Tp)2) * __yn)) / ((_Tp)3)); 
# 522
__sn = (((__yn + __mu) / __mu) - ((_Tp)2)); 
# 523
if (std::abs(__sn) < __errtol) { 
# 524
break; }  
# 525
const _Tp __lambda = ((((_Tp)2) * std::sqrt(__xn)) * std::sqrt(__yn)) + __yn; 
# 527
__xn = (__c0 * (__xn + __lambda)); 
# 528
__yn = (__c0 * (__yn + __lambda)); 
# 529
}  
# 531
_Tp __s = (__sn * __sn) * (__c3 + (__sn * (__c1 + (__sn * (__c4 + (__sn * __c2)))))); 
# 534
return (((_Tp)1) + __s) / std::sqrt(__mu); 
# 535
}  
# 536
} 
# 561
template< class _Tp> _Tp 
# 563
__ellint_rj(_Tp __x, _Tp __y, _Tp __z, _Tp __p) 
# 564
{ 
# 565
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 566
const _Tp __lolim = std::pow(((_Tp)5) * __min, ((_Tp)1) / ((_Tp)3)); 
# 568
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 569
std::__throw_domain_error("Argument less than zero in __ellint_rj."); } else { 
# 571
if (((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) || (__p < __lolim)) { 
# 573
std::__throw_domain_error("Argument too small in __ellint_rj"); } else 
# 576
{ 
# 577
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 578
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 579
const _Tp __c2 = (((_Tp)1) / ((_Tp)3)); 
# 580
const _Tp __c3 = (((_Tp)3) / ((_Tp)22)); 
# 581
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 583
_Tp __xn = __x; 
# 584
_Tp __yn = __y; 
# 585
_Tp __zn = __z; 
# 586
_Tp __pn = __p; 
# 587
_Tp __sigma = ((_Tp)0); 
# 588
_Tp __power4 = ((_Tp)1); 
# 590
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 591
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 593
_Tp __mu; 
# 594
_Tp __xndev, __yndev, __zndev, __pndev; 
# 596
const unsigned __max_iter = (100); 
# 597
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 598
{ 
# 599
__mu = ((((__xn + __yn) + __zn) + (((_Tp)2) * __pn)) / ((_Tp)5)); 
# 600
__xndev = ((__mu - __xn) / __mu); 
# 601
__yndev = ((__mu - __yn) / __mu); 
# 602
__zndev = ((__mu - __zn) / __mu); 
# 603
__pndev = ((__mu - __pn) / __mu); 
# 604
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 605
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 606
__epsilon = std::max(__epsilon, std::abs(__pndev)); 
# 607
if (__epsilon < __errtol) { 
# 608
break; }  
# 609
const _Tp __xnroot = std::sqrt(__xn); 
# 610
const _Tp __ynroot = std::sqrt(__yn); 
# 611
const _Tp __znroot = std::sqrt(__zn); 
# 612
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 614
const _Tp __alpha1 = (__pn * ((__xnroot + __ynroot) + __znroot)) + ((__xnroot * __ynroot) * __znroot); 
# 616
const _Tp __alpha2 = __alpha1 * __alpha1; 
# 617
const _Tp __beta = (__pn * (__pn + __lambda)) * (__pn + __lambda); 
# 619
__sigma += (__power4 * __ellint_rc(__alpha2, __beta)); 
# 620
__power4 *= __c0; 
# 621
__xn = (__c0 * (__xn + __lambda)); 
# 622
__yn = (__c0 * (__yn + __lambda)); 
# 623
__zn = (__c0 * (__zn + __lambda)); 
# 624
__pn = (__c0 * (__pn + __lambda)); 
# 625
}  
# 627
_Tp __ea = (__xndev * (__yndev + __zndev)) + (__yndev * __zndev); 
# 628
_Tp __eb = (__xndev * __yndev) * __zndev; 
# 629
_Tp __ec = __pndev * __pndev; 
# 630
_Tp __e2 = __ea - (((_Tp)3) * __ec); 
# 631
_Tp __e3 = __eb + ((((_Tp)2) * __pndev) * (__ea - __ec)); 
# 632
_Tp __s1 = ((_Tp)1) + (__e2 * (((-__c1) + (((((_Tp)3) * __c3) * __e2) / ((_Tp)4))) - (((((_Tp)3) * __c4) * __e3) / ((_Tp)2)))); 
# 634
_Tp __s2 = __eb * ((__c2 / ((_Tp)2)) + (__pndev * (((-__c3) - __c3) + (__pndev * __c4)))); 
# 636
_Tp __s3 = ((__pndev * __ea) * (__c2 - (__pndev * __c3))) - ((__c2 * __pndev) * __ec); 
# 639
return (((_Tp)3) * __sigma) + ((__power4 * ((__s1 + __s2) + __s3)) / (__mu * std::sqrt(__mu))); 
# 641
}  }  
# 642
} 
# 661
template< class _Tp> _Tp 
# 663
__comp_ellint_3(_Tp __k, _Tp __nu) 
# 664
{ 
# 666
if (__isnan(__k) || __isnan(__nu)) { 
# 667
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 668
if (__nu == ((_Tp)1)) { 
# 669
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 670
if (std::abs(__k) > ((_Tp)1)) { 
# 671
std::__throw_domain_error("Bad argument in __comp_ellint_3."); } else 
# 673
{ 
# 674
const _Tp __kk = __k * __k; 
# 676
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) + ((__nu * __ellint_rj((_Tp)0, ((_Tp)1) - __kk, (_Tp)1, ((_Tp)1) - __nu)) / ((_Tp)3)); 
# 680
}  }  }  
# 681
} 
# 701
template< class _Tp> _Tp 
# 703
__ellint_3(_Tp __k, _Tp __nu, _Tp __phi) 
# 704
{ 
# 706
if ((__isnan(__k) || __isnan(__nu)) || __isnan(__phi)) { 
# 707
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 708
if (std::abs(__k) > ((_Tp)1)) { 
# 709
std::__throw_domain_error("Bad argument in __ellint_3."); } else 
# 711
{ 
# 713
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 715
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 718
const _Tp __kk = __k * __k; 
# 719
const _Tp __s = std::sin(__phi_red); 
# 720
const _Tp __ss = __s * __s; 
# 721
const _Tp __sss = __ss * __s; 
# 722
const _Tp __c = std::cos(__phi_red); 
# 723
const _Tp __cc = __c * __c; 
# 725
const _Tp __Pi = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) + (((__nu * __sss) * __ellint_rj(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1, ((_Tp)1) - (__nu * __ss))) / ((_Tp)3)); 
# 731
if (__n == 0) { 
# 732
return __Pi; } else { 
# 734
return __Pi + ((((_Tp)2) * __n) * __comp_ellint_3(__k, __nu)); }  
# 735
}  }  
# 736
} 
# 737
}
# 743
}
# 50 "/usr/include/c++/13/tr1/exp_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 64 "/usr/include/c++/13/tr1/exp_integral.tcc" 3
namespace __detail { 
# 66
template< class _Tp> _Tp __expint_E1(_Tp); 
# 81
template< class _Tp> _Tp 
# 83
__expint_E1_series(_Tp __x) 
# 84
{ 
# 85
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 86
_Tp __term = ((_Tp)1); 
# 87
_Tp __esum = ((_Tp)0); 
# 88
_Tp __osum = ((_Tp)0); 
# 89
const unsigned __max_iter = (1000); 
# 90
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 91
{ 
# 92
__term *= ((-__x) / __i); 
# 93
if (std::abs(__term) < __eps) { 
# 94
break; }  
# 95
if (__term >= ((_Tp)0)) { 
# 96
__esum += (__term / __i); } else { 
# 98
__osum += (__term / __i); }  
# 99
}  
# 101
return (((-__esum) - __osum) - __numeric_constants< _Tp> ::__gamma_e()) - std::log(__x); 
# 103
} 
# 118
template< class _Tp> _Tp 
# 120
__expint_E1_asymp(_Tp __x) 
# 121
{ 
# 122
_Tp __term = ((_Tp)1); 
# 123
_Tp __esum = ((_Tp)1); 
# 124
_Tp __osum = ((_Tp)0); 
# 125
const unsigned __max_iter = (1000); 
# 126
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 127
{ 
# 128
_Tp __prev = __term; 
# 129
__term *= ((-__i) / __x); 
# 130
if (std::abs(__term) > std::abs(__prev)) { 
# 131
break; }  
# 132
if (__term >= ((_Tp)0)) { 
# 133
__esum += __term; } else { 
# 135
__osum += __term; }  
# 136
}  
# 138
return (std::exp(-__x) * (__esum + __osum)) / __x; 
# 139
} 
# 155
template< class _Tp> _Tp 
# 157
__expint_En_series(unsigned __n, _Tp __x) 
# 158
{ 
# 159
const unsigned __max_iter = (1000); 
# 160
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 161
const int __nm1 = __n - (1); 
# 162
_Tp __ans = (__nm1 != 0) ? ((_Tp)1) / __nm1 : ((-std::log(__x)) - __numeric_constants< _Tp> ::__gamma_e()); 
# 165
_Tp __fact = ((_Tp)1); 
# 166
for (int __i = 1; __i <= __max_iter; ++__i) 
# 167
{ 
# 168
__fact *= ((-__x) / ((_Tp)__i)); 
# 169
_Tp __del; 
# 170
if (__i != __nm1) { 
# 171
__del = ((-__fact) / ((_Tp)(__i - __nm1))); } else 
# 173
{ 
# 174
_Tp __psi = (-__numeric_constants< _Tp> ::gamma_e()); 
# 175
for (int __ii = 1; __ii <= __nm1; ++__ii) { 
# 176
__psi += (((_Tp)1) / ((_Tp)__ii)); }  
# 177
__del = (__fact * (__psi - std::log(__x))); 
# 178
}  
# 179
__ans += __del; 
# 180
if (std::abs(__del) < (__eps * std::abs(__ans))) { 
# 181
return __ans; }  
# 182
}  
# 183
std::__throw_runtime_error("Series summation failed in __expint_En_series."); 
# 185
} 
# 201
template< class _Tp> _Tp 
# 203
__expint_En_cont_frac(unsigned __n, _Tp __x) 
# 204
{ 
# 205
const unsigned __max_iter = (1000); 
# 206
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 207
const _Tp __fp_min = std::template numeric_limits< _Tp> ::min(); 
# 208
const int __nm1 = __n - (1); 
# 209
_Tp __b = __x + ((_Tp)__n); 
# 210
_Tp __c = ((_Tp)1) / __fp_min; 
# 211
_Tp __d = ((_Tp)1) / __b; 
# 212
_Tp __h = __d; 
# 213
for (unsigned __i = (1); __i <= __max_iter; ++__i) 
# 214
{ 
# 215
_Tp __a = (-((_Tp)(__i * (__nm1 + __i)))); 
# 216
__b += ((_Tp)2); 
# 217
__d = (((_Tp)1) / ((__a * __d) + __b)); 
# 218
__c = (__b + (__a / __c)); 
# 219
const _Tp __del = __c * __d; 
# 220
__h *= __del; 
# 221
if (std::abs(__del - ((_Tp)1)) < __eps) 
# 222
{ 
# 223
const _Tp __ans = __h * std::exp(-__x); 
# 224
return __ans; 
# 225
}  
# 226
}  
# 227
std::__throw_runtime_error("Continued fraction failed in __expint_En_cont_frac."); 
# 229
} 
# 246
template< class _Tp> _Tp 
# 248
__expint_En_recursion(unsigned __n, _Tp __x) 
# 249
{ 
# 250
_Tp __En; 
# 251
_Tp __E1 = __expint_E1(__x); 
# 252
if (__x < ((_Tp)__n)) 
# 253
{ 
# 255
__En = __E1; 
# 256
for (unsigned __j = (2); __j < __n; ++__j) { 
# 257
__En = ((std::exp(-__x) - (__x * __En)) / ((_Tp)(__j - (1)))); }  
# 258
} else 
# 260
{ 
# 262
__En = ((_Tp)1); 
# 263
const int __N = __n + (20); 
# 264
_Tp __save = ((_Tp)0); 
# 265
for (int __j = __N; __j > 0; --__j) 
# 266
{ 
# 267
__En = ((std::exp(-__x) - (__j * __En)) / __x); 
# 268
if (__j == __n) { 
# 269
__save = __En; }  
# 270
}  
# 271
_Tp __norm = __En / __E1; 
# 272
__En /= __norm; 
# 273
}  
# 275
return __En; 
# 276
} 
# 290
template< class _Tp> _Tp 
# 292
__expint_Ei_series(_Tp __x) 
# 293
{ 
# 294
_Tp __term = ((_Tp)1); 
# 295
_Tp __sum = ((_Tp)0); 
# 296
const unsigned __max_iter = (1000); 
# 297
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 298
{ 
# 299
__term *= (__x / __i); 
# 300
__sum += (__term / __i); 
# 301
if (__term < (std::template numeric_limits< _Tp> ::epsilon() * __sum)) { 
# 302
break; }  
# 303
}  
# 305
return (__numeric_constants< _Tp> ::__gamma_e() + __sum) + std::log(__x); 
# 306
} 
# 321
template< class _Tp> _Tp 
# 323
__expint_Ei_asymp(_Tp __x) 
# 324
{ 
# 325
_Tp __term = ((_Tp)1); 
# 326
_Tp __sum = ((_Tp)1); 
# 327
const unsigned __max_iter = (1000); 
# 328
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 329
{ 
# 330
_Tp __prev = __term; 
# 331
__term *= (__i / __x); 
# 332
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 333
break; }  
# 334
if (__term >= __prev) { 
# 335
break; }  
# 336
__sum += __term; 
# 337
}  
# 339
return (std::exp(__x) * __sum) / __x; 
# 340
} 
# 354
template< class _Tp> _Tp 
# 356
__expint_Ei(_Tp __x) 
# 357
{ 
# 358
if (__x < ((_Tp)0)) { 
# 359
return -__expint_E1(-__x); } else { 
# 360
if (__x < (-std::log(std::template numeric_limits< _Tp> ::epsilon()))) { 
# 361
return __expint_Ei_series(__x); } else { 
# 363
return __expint_Ei_asymp(__x); }  }  
# 364
} 
# 378
template< class _Tp> _Tp 
# 380
__expint_E1(_Tp __x) 
# 381
{ 
# 382
if (__x < ((_Tp)0)) { 
# 383
return -__expint_Ei(-__x); } else { 
# 384
if (__x < ((_Tp)1)) { 
# 385
return __expint_E1_series(__x); } else { 
# 386
if (__x < ((_Tp)100)) { 
# 387
return __expint_En_cont_frac(1, __x); } else { 
# 389
return __expint_E1_asymp(__x); }  }  }  
# 390
} 
# 408
template< class _Tp> _Tp 
# 410
__expint_asymp(unsigned __n, _Tp __x) 
# 411
{ 
# 412
_Tp __term = ((_Tp)1); 
# 413
_Tp __sum = ((_Tp)1); 
# 414
for (unsigned __i = (1); __i <= __n; ++__i) 
# 415
{ 
# 416
_Tp __prev = __term; 
# 417
__term *= ((-((__n - __i) + (1))) / __x); 
# 418
if (std::abs(__term) > std::abs(__prev)) { 
# 419
break; }  
# 420
__sum += __term; 
# 421
}  
# 423
return (std::exp(-__x) * __sum) / __x; 
# 424
} 
# 442
template< class _Tp> _Tp 
# 444
__expint_large_n(unsigned __n, _Tp __x) 
# 445
{ 
# 446
const _Tp __xpn = __x + __n; 
# 447
const _Tp __xpn2 = __xpn * __xpn; 
# 448
_Tp __term = ((_Tp)1); 
# 449
_Tp __sum = ((_Tp)1); 
# 450
for (unsigned __i = (1); __i <= __n; ++__i) 
# 451
{ 
# 452
_Tp __prev = __term; 
# 453
__term *= ((__n - (((2) * (__i - (1))) * __x)) / __xpn2); 
# 454
if (std::abs(__term) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 455
break; }  
# 456
__sum += __term; 
# 457
}  
# 459
return (std::exp(-__x) * __sum) / __xpn; 
# 460
} 
# 476
template< class _Tp> _Tp 
# 478
__expint(unsigned __n, _Tp __x) 
# 479
{ 
# 481
if (__isnan(__x)) { 
# 482
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 483
if ((__n <= (1)) && (__x == ((_Tp)0))) { 
# 484
return std::template numeric_limits< _Tp> ::infinity(); } else 
# 486
{ 
# 487
_Tp __E0 = std::exp(__x) / __x; 
# 488
if (__n == (0)) { 
# 489
return __E0; }  
# 491
_Tp __E1 = __expint_E1(__x); 
# 492
if (__n == (1)) { 
# 493
return __E1; }  
# 495
if (__x == ((_Tp)0)) { 
# 496
return ((_Tp)1) / (static_cast< _Tp>(__n - (1))); }  
# 498
_Tp __En = __expint_En_recursion(__n, __x); 
# 500
return __En; 
# 501
}  }  
# 502
} 
# 516
template< class _Tp> inline _Tp 
# 518
__expint(_Tp __x) 
# 519
{ 
# 520
if (__isnan(__x)) { 
# 521
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 523
return __expint_Ei(__x); }  
# 524
} 
# 525
}
# 531
}
# 44 "/usr/include/c++/13/tr1/hypergeometric.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/usr/include/c++/13/tr1/hypergeometric.tcc" 3
namespace __detail { 
# 83
template< class _Tp> _Tp 
# 85
__conf_hyperg_series(_Tp __a, _Tp __c, _Tp __x) 
# 86
{ 
# 87
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 89
_Tp __term = ((_Tp)1); 
# 90
_Tp __Fac = ((_Tp)1); 
# 91
const unsigned __max_iter = (100000); 
# 92
unsigned __i; 
# 93
for (__i = (0); __i < __max_iter; ++__i) 
# 94
{ 
# 95
__term *= (((__a + ((_Tp)__i)) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 97
if (std::abs(__term) < __eps) 
# 98
{ 
# 99
break; 
# 100
}  
# 101
__Fac += __term; 
# 102
}  
# 103
if (__i == __max_iter) { 
# 104
std::__throw_runtime_error("Series failed to converge in __conf_hyperg_series."); }  
# 107
return __Fac; 
# 108
} 
# 120
template< class _Tp> _Tp 
# 122
__conf_hyperg_luke(_Tp __a, _Tp __c, _Tp __xin) 
# 123
{ 
# 124
const _Tp __big = std::pow(std::template numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 125
const int __nmax = 20000; 
# 126
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 127
const _Tp __x = (-__xin); 
# 128
const _Tp __x3 = (__x * __x) * __x; 
# 129
const _Tp __t0 = __a / __c; 
# 130
const _Tp __t1 = (__a + ((_Tp)1)) / (((_Tp)2) * __c); 
# 131
const _Tp __t2 = (__a + ((_Tp)2)) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 132
_Tp __F = ((_Tp)1); 
# 133
_Tp __prec; 
# 135
_Tp __Bnm3 = ((_Tp)1); 
# 136
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 137
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 139
_Tp __Anm3 = ((_Tp)1); 
# 140
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 141
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 144
int __n = 3; 
# 145
while (1) 
# 146
{ 
# 147
_Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 148
_Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 149
_Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 150
_Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 151
_Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 152
_Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 153
_Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 154
_Tp __F1 = (((_Tp)(__n - 2)) - __a) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 155
_Tp __F2 = ((((_Tp)__n) + __a) * __npam1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 157
_Tp __F3 = (((-__npam2) * __npam1) * (((_Tp)(__n - 2)) - __a)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 160
_Tp __E = ((-__npam1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 163
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 165
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 167
_Tp __r = __An / __Bn; 
# 169
__prec = std::abs((__F - __r) / __F); 
# 170
__F = __r; 
# 172
if ((__prec < __eps) || (__n > __nmax)) { 
# 173
break; }  
# 175
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 176
{ 
# 177
__An /= __big; 
# 178
__Bn /= __big; 
# 179
__Anm1 /= __big; 
# 180
__Bnm1 /= __big; 
# 181
__Anm2 /= __big; 
# 182
__Bnm2 /= __big; 
# 183
__Anm3 /= __big; 
# 184
__Bnm3 /= __big; 
# 185
} else { 
# 186
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 188
{ 
# 189
__An *= __big; 
# 190
__Bn *= __big; 
# 191
__Anm1 *= __big; 
# 192
__Bnm1 *= __big; 
# 193
__Anm2 *= __big; 
# 194
__Bnm2 *= __big; 
# 195
__Anm3 *= __big; 
# 196
__Bnm3 *= __big; 
# 197
}  }  
# 199
++__n; 
# 200
__Bnm3 = __Bnm2; 
# 201
__Bnm2 = __Bnm1; 
# 202
__Bnm1 = __Bn; 
# 203
__Anm3 = __Anm2; 
# 204
__Anm2 = __Anm1; 
# 205
__Anm1 = __An; 
# 206
}  
# 208
if (__n >= __nmax) { 
# 209
std::__throw_runtime_error("Iteration failed to converge in __conf_hyperg_luke."); }  
# 212
return __F; 
# 213
} 
# 227
template< class _Tp> _Tp 
# 229
__conf_hyperg(_Tp __a, _Tp __c, _Tp __x) 
# 230
{ 
# 232
const _Tp __c_nint = std::nearbyint(__c); 
# 236
if ((__isnan(__a) || __isnan(__c)) || __isnan(__x)) { 
# 237
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 238
if ((__c_nint == __c) && (__c_nint <= 0)) { 
# 239
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 240
if (__a == ((_Tp)0)) { 
# 241
return (_Tp)1; } else { 
# 242
if (__c == __a) { 
# 243
return std::exp(__x); } else { 
# 244
if (__x < ((_Tp)0)) { 
# 245
return __conf_hyperg_luke(__a, __c, __x); } else { 
# 247
return __conf_hyperg_series(__a, __c, __x); }  }  }  }  }  
# 248
} 
# 271
template< class _Tp> _Tp 
# 273
__hyperg_series(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 274
{ 
# 275
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 277
_Tp __term = ((_Tp)1); 
# 278
_Tp __Fabc = ((_Tp)1); 
# 279
const unsigned __max_iter = (100000); 
# 280
unsigned __i; 
# 281
for (__i = (0); __i < __max_iter; ++__i) 
# 282
{ 
# 283
__term *= ((((__a + ((_Tp)__i)) * (__b + ((_Tp)__i))) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 285
if (std::abs(__term) < __eps) 
# 286
{ 
# 287
break; 
# 288
}  
# 289
__Fabc += __term; 
# 290
}  
# 291
if (__i == __max_iter) { 
# 292
std::__throw_runtime_error("Series failed to converge in __hyperg_series."); }  
# 295
return __Fabc; 
# 296
} 
# 304
template< class _Tp> _Tp 
# 306
__hyperg_luke(_Tp __a, _Tp __b, _Tp __c, _Tp __xin) 
# 307
{ 
# 308
const _Tp __big = std::pow(std::template numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 309
const int __nmax = 20000; 
# 310
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 311
const _Tp __x = (-__xin); 
# 312
const _Tp __x3 = (__x * __x) * __x; 
# 313
const _Tp __t0 = (__a * __b) / __c; 
# 314
const _Tp __t1 = ((__a + ((_Tp)1)) * (__b + ((_Tp)1))) / (((_Tp)2) * __c); 
# 315
const _Tp __t2 = ((__a + ((_Tp)2)) * (__b + ((_Tp)2))) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 318
_Tp __F = ((_Tp)1); 
# 320
_Tp __Bnm3 = ((_Tp)1); 
# 321
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 322
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 324
_Tp __Anm3 = ((_Tp)1); 
# 325
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 326
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 329
int __n = 3; 
# 330
while (1) 
# 331
{ 
# 332
const _Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 333
const _Tp __npbm1 = ((_Tp)(__n - 1)) + __b; 
# 334
const _Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 335
const _Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 336
const _Tp __npbm2 = ((_Tp)(__n - 2)) + __b; 
# 337
const _Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 338
const _Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 339
const _Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 340
const _Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 341
const _Tp __n2 = __n * __n; 
# 342
const _Tp __F1 = (((((((_Tp)3) * __n2) + (((__a + __b) - ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b)) - (((_Tp)2) * (__a + __b))) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 345
const _Tp __F2 = (((-((((((_Tp)3) * __n2) - (((__a + __b) + ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b))) * __npam1) * __npbm1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 348
const _Tp __F3 = (((((__npam2 * __npam1) * __npbm2) * __npbm1) * (((_Tp)(__n - 2)) - __a)) * (((_Tp)(__n - 2)) - __b)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 352
const _Tp __E = (((-__npam1) * __npbm1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 355
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 357
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 359
const _Tp __r = __An / __Bn; 
# 361
const _Tp __prec = std::abs((__F - __r) / __F); 
# 362
__F = __r; 
# 364
if ((__prec < __eps) || (__n > __nmax)) { 
# 365
break; }  
# 367
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 368
{ 
# 369
__An /= __big; 
# 370
__Bn /= __big; 
# 371
__Anm1 /= __big; 
# 372
__Bnm1 /= __big; 
# 373
__Anm2 /= __big; 
# 374
__Bnm2 /= __big; 
# 375
__Anm3 /= __big; 
# 376
__Bnm3 /= __big; 
# 377
} else { 
# 378
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 380
{ 
# 381
__An *= __big; 
# 382
__Bn *= __big; 
# 383
__Anm1 *= __big; 
# 384
__Bnm1 *= __big; 
# 385
__Anm2 *= __big; 
# 386
__Bnm2 *= __big; 
# 387
__Anm3 *= __big; 
# 388
__Bnm3 *= __big; 
# 389
}  }  
# 391
++__n; 
# 392
__Bnm3 = __Bnm2; 
# 393
__Bnm2 = __Bnm1; 
# 394
__Bnm1 = __Bn; 
# 395
__Anm3 = __Anm2; 
# 396
__Anm2 = __Anm1; 
# 397
__Anm1 = __An; 
# 398
}  
# 400
if (__n >= __nmax) { 
# 401
std::__throw_runtime_error("Iteration failed to converge in __hyperg_luke."); }  
# 404
return __F; 
# 405
} 
# 438
template< class _Tp> _Tp 
# 440
__hyperg_reflect(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 441
{ 
# 442
const _Tp __d = (__c - __a) - __b; 
# 443
const int __intd = std::floor(__d + ((_Tp)(0.5L))); 
# 444
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 445
const _Tp __toler = ((_Tp)1000) * __eps; 
# 446
const _Tp __log_max = std::log(std::template numeric_limits< _Tp> ::max()); 
# 447
const bool __d_integer = std::abs(__d - __intd) < __toler; 
# 449
if (__d_integer) 
# 450
{ 
# 451
const _Tp __ln_omx = std::log(((_Tp)1) - __x); 
# 452
const _Tp __ad = std::abs(__d); 
# 453
_Tp __F1, __F2; 
# 455
_Tp __d1, __d2; 
# 456
if (__d >= ((_Tp)0)) 
# 457
{ 
# 458
__d1 = __d; 
# 459
__d2 = ((_Tp)0); 
# 460
} else 
# 462
{ 
# 463
__d1 = ((_Tp)0); 
# 464
__d2 = __d; 
# 465
}  
# 467
const _Tp __lng_c = __log_gamma(__c); 
# 470
if (__ad < __eps) 
# 471
{ 
# 473
__F1 = ((_Tp)0); 
# 474
} else 
# 476
{ 
# 478
bool __ok_d1 = true; 
# 479
_Tp __lng_ad, __lng_ad1, __lng_bd1; 
# 480
try 
# 481
{ 
# 482
__lng_ad = __log_gamma(__ad); 
# 483
__lng_ad1 = __log_gamma(__a + __d1); 
# 484
__lng_bd1 = __log_gamma(__b + __d1); 
# 485
} 
# 486
catch (...) 
# 487
{ 
# 488
__ok_d1 = false; 
# 489
}  
# 491
if (__ok_d1) 
# 492
{ 
# 496
_Tp __sum1 = ((_Tp)1); 
# 497
_Tp __term = ((_Tp)1); 
# 498
_Tp __ln_pre1 = (((__lng_ad + __lng_c) + (__d2 * __ln_omx)) - __lng_ad1) - __lng_bd1; 
# 503
for (int __i = 1; __i < __ad; ++__i) 
# 504
{ 
# 505
const int __j = __i - 1; 
# 506
__term *= ((((((__a + __d2) + __j) * ((__b + __d2) + __j)) / ((((_Tp)1) + __d2) + __j)) / __i) * (((_Tp)1) - __x)); 
# 508
__sum1 += __term; 
# 509
}  
# 511
if (__ln_pre1 > __log_max) { 
# 512
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_luke."); } else { 
# 515
__F1 = (std::exp(__ln_pre1) * __sum1); }  
# 516
} else 
# 518
{ 
# 521
__F1 = ((_Tp)0); 
# 522
}  
# 523
}  
# 526
bool __ok_d2 = true; 
# 527
_Tp __lng_ad2, __lng_bd2; 
# 528
try 
# 529
{ 
# 530
__lng_ad2 = __log_gamma(__a + __d2); 
# 531
__lng_bd2 = __log_gamma(__b + __d2); 
# 532
} 
# 533
catch (...) 
# 534
{ 
# 535
__ok_d2 = false; 
# 536
}  
# 538
if (__ok_d2) 
# 539
{ 
# 542
const int __maxiter = 2000; 
# 543
const _Tp __psi_1 = (-__numeric_constants< _Tp> ::__gamma_e()); 
# 544
const _Tp __psi_1pd = __psi(((_Tp)1) + __ad); 
# 545
const _Tp __psi_apd1 = __psi(__a + __d1); 
# 546
const _Tp __psi_bpd1 = __psi(__b + __d1); 
# 548
_Tp __psi_term = (((__psi_1 + __psi_1pd) - __psi_apd1) - __psi_bpd1) - __ln_omx; 
# 550
_Tp __fact = ((_Tp)1); 
# 551
_Tp __sum2 = __psi_term; 
# 552
_Tp __ln_pre2 = ((__lng_c + (__d1 * __ln_omx)) - __lng_ad2) - __lng_bd2; 
# 556
int __j; 
# 557
for (__j = 1; __j < __maxiter; ++__j) 
# 558
{ 
# 561
const _Tp __term1 = (((_Tp)1) / ((_Tp)__j)) + (((_Tp)1) / (__ad + __j)); 
# 563
const _Tp __term2 = (((_Tp)1) / ((__a + __d1) + ((_Tp)(__j - 1)))) + (((_Tp)1) / ((__b + __d1) + ((_Tp)(__j - 1)))); 
# 565
__psi_term += (__term1 - __term2); 
# 566
__fact *= (((((__a + __d1) + ((_Tp)(__j - 1))) * ((__b + __d1) + ((_Tp)(__j - 1)))) / ((__ad + __j) * __j)) * (((_Tp)1) - __x)); 
# 569
const _Tp __delta = __fact * __psi_term; 
# 570
__sum2 += __delta; 
# 571
if (std::abs(__delta) < (__eps * std::abs(__sum2))) { 
# 572
break; }  
# 573
}  
# 574
if (__j == __maxiter) { 
# 575
std::__throw_runtime_error("Sum F2 failed to converge in __hyperg_reflect"); }  
# 578
if (__sum2 == ((_Tp)0)) { 
# 579
__F2 = ((_Tp)0); } else { 
# 581
__F2 = (std::exp(__ln_pre2) * __sum2); }  
# 582
} else 
# 584
{ 
# 587
__F2 = ((_Tp)0); 
# 588
}  
# 590
const _Tp __sgn_2 = (((__intd % 2) == 1) ? -((_Tp)1) : ((_Tp)1)); 
# 591
const _Tp __F = __F1 + (__sgn_2 * __F2); 
# 593
return __F; 
# 594
} else 
# 596
{ 
# 601
bool __ok1 = true; 
# 602
_Tp __sgn_g1ca = ((_Tp)0), __ln_g1ca = ((_Tp)0); 
# 603
_Tp __sgn_g1cb = ((_Tp)0), __ln_g1cb = ((_Tp)0); 
# 604
try 
# 605
{ 
# 606
__sgn_g1ca = __log_gamma_sign(__c - __a); 
# 607
__ln_g1ca = __log_gamma(__c - __a); 
# 608
__sgn_g1cb = __log_gamma_sign(__c - __b); 
# 609
__ln_g1cb = __log_gamma(__c - __b); 
# 610
} 
# 611
catch (...) 
# 612
{ 
# 613
__ok1 = false; 
# 614
}  
# 616
bool __ok2 = true; 
# 617
_Tp __sgn_g2a = ((_Tp)0), __ln_g2a = ((_Tp)0); 
# 618
_Tp __sgn_g2b = ((_Tp)0), __ln_g2b = ((_Tp)0); 
# 619
try 
# 620
{ 
# 621
__sgn_g2a = __log_gamma_sign(__a); 
# 622
__ln_g2a = __log_gamma(__a); 
# 623
__sgn_g2b = __log_gamma_sign(__b); 
# 624
__ln_g2b = __log_gamma(__b); 
# 625
} 
# 626
catch (...) 
# 627
{ 
# 628
__ok2 = false; 
# 629
}  
# 631
const _Tp __sgn_gc = __log_gamma_sign(__c); 
# 632
const _Tp __ln_gc = __log_gamma(__c); 
# 633
const _Tp __sgn_gd = __log_gamma_sign(__d); 
# 634
const _Tp __ln_gd = __log_gamma(__d); 
# 635
const _Tp __sgn_gmd = __log_gamma_sign(-__d); 
# 636
const _Tp __ln_gmd = __log_gamma(-__d); 
# 638
const _Tp __sgn1 = ((__sgn_gc * __sgn_gd) * __sgn_g1ca) * __sgn_g1cb; 
# 639
const _Tp __sgn2 = ((__sgn_gc * __sgn_gmd) * __sgn_g2a) * __sgn_g2b; 
# 641
_Tp __pre1, __pre2; 
# 642
if (__ok1 && __ok2) 
# 643
{ 
# 644
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 645
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 647
if ((__ln_pre1 < __log_max) && (__ln_pre2 < __log_max)) 
# 648
{ 
# 649
__pre1 = std::exp(__ln_pre1); 
# 650
__pre2 = std::exp(__ln_pre2); 
# 651
__pre1 *= __sgn1; 
# 652
__pre2 *= __sgn2; 
# 653
} else 
# 655
{ 
# 656
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 658
}  
# 659
} else { 
# 660
if (__ok1 && (!__ok2)) 
# 661
{ 
# 662
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 663
if (__ln_pre1 < __log_max) 
# 664
{ 
# 665
__pre1 = std::exp(__ln_pre1); 
# 666
__pre1 *= __sgn1; 
# 667
__pre2 = ((_Tp)0); 
# 668
} else 
# 670
{ 
# 671
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 673
}  
# 674
} else { 
# 675
if ((!__ok1) && __ok2) 
# 676
{ 
# 677
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 679
if (__ln_pre2 < __log_max) 
# 680
{ 
# 681
__pre1 = ((_Tp)0); 
# 682
__pre2 = std::exp(__ln_pre2); 
# 683
__pre2 *= __sgn2; 
# 684
} else 
# 686
{ 
# 687
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 689
}  
# 690
} else 
# 692
{ 
# 693
__pre1 = ((_Tp)0); 
# 694
__pre2 = ((_Tp)0); 
# 695
std::__throw_runtime_error("Underflow of gamma functions in __hyperg_reflect"); 
# 697
}  }  }  
# 699
const _Tp __F1 = __hyperg_series(__a, __b, ((_Tp)1) - __d, ((_Tp)1) - __x); 
# 701
const _Tp __F2 = __hyperg_series(__c - __a, __c - __b, ((_Tp)1) + __d, ((_Tp)1) - __x); 
# 704
const _Tp __F = (__pre1 * __F1) + (__pre2 * __F2); 
# 706
return __F; 
# 707
}  
# 708
} 
# 728
template< class _Tp> _Tp 
# 730
__hyperg(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 731
{ 
# 733
const _Tp __a_nint = std::nearbyint(__a); 
# 734
const _Tp __b_nint = std::nearbyint(__b); 
# 735
const _Tp __c_nint = std::nearbyint(__c); 
# 741
const _Tp __toler = ((_Tp)1000) * std::template numeric_limits< _Tp> ::epsilon(); 
# 742
if (std::abs(__x) >= ((_Tp)1)) { 
# 743
std::__throw_domain_error("Argument outside unit circle in __hyperg."); } else { 
# 745
if (((__isnan(__a) || __isnan(__b)) || __isnan(__c)) || __isnan(__x)) { 
# 747
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 748
if ((__c_nint == __c) && (__c_nint <= ((_Tp)0))) { 
# 749
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 750
if ((std::abs(__c - __b) < __toler) || (std::abs(__c - __a) < __toler)) { 
# 751
return std::pow(((_Tp)1) - __x, (__c - __a) - __b); } else { 
# 752
if ((__a >= ((_Tp)0)) && (__b >= ((_Tp)0)) && (__c >= ((_Tp)0)) && (__x >= ((_Tp)0)) && (__x < ((_Tp)(0.995L)))) { 
# 754
return __hyperg_series(__a, __b, __c, __x); } else { 
# 755
if ((std::abs(__a) < ((_Tp)10)) && (std::abs(__b) < ((_Tp)10))) 
# 756
{ 
# 759
if ((__a < ((_Tp)0)) && (std::abs(__a - __a_nint) < __toler)) { 
# 760
return __hyperg_series(__a_nint, __b, __c, __x); } else { 
# 761
if ((__b < ((_Tp)0)) && (std::abs(__b - __b_nint) < __toler)) { 
# 762
return __hyperg_series(__a, __b_nint, __c, __x); } else { 
# 763
if (__x < (-((_Tp)(0.25L)))) { 
# 764
return __hyperg_luke(__a, __b, __c, __x); } else { 
# 765
if (__x < ((_Tp)(0.5L))) { 
# 766
return __hyperg_series(__a, __b, __c, __x); } else { 
# 768
if (std::abs(__c) > ((_Tp)10)) { 
# 769
return __hyperg_series(__a, __b, __c, __x); } else { 
# 771
return __hyperg_reflect(__a, __b, __c, __x); }  }  }  }  }  
# 772
} else { 
# 774
return __hyperg_luke(__a, __b, __c, __x); }  }  }  }  }  }  
# 775
} 
# 776
}
# 783
}
# 49 "/usr/include/c++/13/tr1/legendre_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/13/tr1/legendre_function.tcc" 3
namespace __detail { 
# 80
template< class _Tp> _Tp 
# 82
__poly_legendre_p(unsigned __l, _Tp __x) 
# 83
{ 
# 85
if (__isnan(__x)) { 
# 86
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 87
if (__x == (+((_Tp)1))) { 
# 88
return +((_Tp)1); } else { 
# 89
if (__x == (-((_Tp)1))) { 
# 90
return (((__l % (2)) == (1)) ? -((_Tp)1) : (+((_Tp)1))); } else 
# 92
{ 
# 93
_Tp __p_lm2 = ((_Tp)1); 
# 94
if (__l == (0)) { 
# 95
return __p_lm2; }  
# 97
_Tp __p_lm1 = __x; 
# 98
if (__l == (1)) { 
# 99
return __p_lm1; }  
# 101
_Tp __p_l = (0); 
# 102
for (unsigned __ll = (2); __ll <= __l; ++__ll) 
# 103
{ 
# 106
__p_l = ((((((_Tp)2) * __x) * __p_lm1) - __p_lm2) - (((__x * __p_lm1) - __p_lm2) / ((_Tp)__ll))); 
# 108
__p_lm2 = __p_lm1; 
# 109
__p_lm1 = __p_l; 
# 110
}  
# 112
return __p_l; 
# 113
}  }  }  
# 114
} 
# 136
template< class _Tp> _Tp 
# 138
__assoc_legendre_p(unsigned __l, unsigned __m, _Tp __x, _Tp 
# 139
__phase = (_Tp)(+1)) 
# 140
{ 
# 142
if (__m > __l) { 
# 143
return (_Tp)0; } else { 
# 144
if (__isnan(__x)) { 
# 145
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 146
if (__m == (0)) { 
# 147
return __poly_legendre_p(__l, __x); } else 
# 149
{ 
# 150
_Tp __p_mm = ((_Tp)1); 
# 151
if (__m > (0)) 
# 152
{ 
# 155
_Tp __root = std::sqrt(((_Tp)1) - __x) * std::sqrt(((_Tp)1) + __x); 
# 156
_Tp __fact = ((_Tp)1); 
# 157
for (unsigned __i = (1); __i <= __m; ++__i) 
# 158
{ 
# 159
__p_mm *= ((__phase * __fact) * __root); 
# 160
__fact += ((_Tp)2); 
# 161
}  
# 162
}  
# 163
if (__l == __m) { 
# 164
return __p_mm; }  
# 166
_Tp __p_mp1m = (((_Tp)(((2) * __m) + (1))) * __x) * __p_mm; 
# 167
if (__l == (__m + (1))) { 
# 168
return __p_mp1m; }  
# 170
_Tp __p_lm2m = __p_mm; 
# 171
_Tp __P_lm1m = __p_mp1m; 
# 172
_Tp __p_lm = ((_Tp)0); 
# 173
for (unsigned __j = __m + (2); __j <= __l; ++__j) 
# 174
{ 
# 175
__p_lm = ((((((_Tp)(((2) * __j) - (1))) * __x) * __P_lm1m) - (((_Tp)((__j + __m) - (1))) * __p_lm2m)) / ((_Tp)(__j - __m))); 
# 177
__p_lm2m = __P_lm1m; 
# 178
__P_lm1m = __p_lm; 
# 179
}  
# 181
return __p_lm; 
# 182
}  }  }  
# 183
} 
# 214
template< class _Tp> _Tp 
# 216
__sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 217
{ 
# 218
if (__isnan(__theta)) { 
# 219
return std::template numeric_limits< _Tp> ::quiet_NaN(); }  
# 221
const _Tp __x = std::cos(__theta); 
# 223
if (__m > __l) { 
# 224
return (_Tp)0; } else { 
# 225
if (__m == (0)) 
# 226
{ 
# 227
_Tp __P = __poly_legendre_p(__l, __x); 
# 228
_Tp __fact = std::sqrt(((_Tp)(((2) * __l) + (1))) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 230
__P *= __fact; 
# 231
return __P; 
# 232
} else { 
# 233
if ((__x == ((_Tp)1)) || (__x == (-((_Tp)1)))) 
# 234
{ 
# 236
return (_Tp)0; 
# 237
} else 
# 239
{ 
# 245
const _Tp __sgn = ((__m % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1); 
# 246
const _Tp __y_mp1m_factor = __x * std::sqrt((_Tp)(((2) * __m) + (3))); 
# 248
const _Tp __lncirc = std::log1p((-__x) * __x); 
# 254
const _Tp __lnpoch = std::lgamma((_Tp)(__m + ((_Tp)(0.5L)))) - std::lgamma((_Tp)__m); 
# 260
const _Tp __lnpre_val = ((-((_Tp)(0.25L))) * __numeric_constants< _Tp> ::__lnpi()) + (((_Tp)(0.5L)) * (__lnpoch + (__m * __lncirc))); 
# 263
const _Tp __sr = std::sqrt((((_Tp)2) + (((_Tp)1) / __m)) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 265
_Tp __y_mm = (__sgn * __sr) * std::exp(__lnpre_val); 
# 266
_Tp __y_mp1m = __y_mp1m_factor * __y_mm; 
# 268
if (__l == __m) { 
# 269
return __y_mm; } else { 
# 270
if (__l == (__m + (1))) { 
# 271
return __y_mp1m; } else 
# 273
{ 
# 274
_Tp __y_lm = ((_Tp)0); 
# 277
for (unsigned __ll = __m + (2); __ll <= __l; ++__ll) 
# 278
{ 
# 279
const _Tp __rat1 = ((_Tp)(__ll - __m)) / ((_Tp)(__ll + __m)); 
# 280
const _Tp __rat2 = ((_Tp)((__ll - __m) - (1))) / ((_Tp)((__ll + __m) - (1))); 
# 281
const _Tp __fact1 = std::sqrt((__rat1 * ((_Tp)(((2) * __ll) + (1)))) * ((_Tp)(((2) * __ll) - (1)))); 
# 283
const _Tp __fact2 = std::sqrt(((__rat1 * __rat2) * ((_Tp)(((2) * __ll) + (1)))) / ((_Tp)(((2) * __ll) - (3)))); 
# 285
__y_lm = ((((__x * __y_mp1m) * __fact1) - ((((__ll + __m) - (1)) * __y_mm) * __fact2)) / ((_Tp)(__ll - __m))); 
# 287
__y_mm = __y_mp1m; 
# 288
__y_mp1m = __y_lm; 
# 289
}  
# 291
return __y_lm; 
# 292
}  }  
# 293
}  }  }  
# 294
} 
# 295
}
# 302
}
# 51 "/usr/include/c++/13/tr1/modified_bessel_func.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/13/tr1/modified_bessel_func.tcc" 3
namespace __detail { 
# 83
template< class _Tp> void 
# 85
__bessel_ik(_Tp __nu, _Tp __x, _Tp &
# 86
__Inu, _Tp &__Knu, _Tp &__Ipnu, _Tp &__Kpnu) 
# 87
{ 
# 88
if (__x == ((_Tp)0)) 
# 89
{ 
# 90
if (__nu == ((_Tp)0)) 
# 91
{ 
# 92
__Inu = ((_Tp)1); 
# 93
__Ipnu = ((_Tp)0); 
# 94
} else { 
# 95
if (__nu == ((_Tp)1)) 
# 96
{ 
# 97
__Inu = ((_Tp)0); 
# 98
__Ipnu = ((_Tp)(0.5L)); 
# 99
} else 
# 101
{ 
# 102
__Inu = ((_Tp)0); 
# 103
__Ipnu = ((_Tp)0); 
# 104
}  }  
# 105
__Knu = std::template numeric_limits< _Tp> ::infinity(); 
# 106
__Kpnu = (-std::template numeric_limits< _Tp> ::infinity()); 
# 107
return; 
# 108
}  
# 110
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 111
const _Tp __fp_min = ((_Tp)10) * std::template numeric_limits< _Tp> ::epsilon(); 
# 112
const int __max_iter = 15000; 
# 113
const _Tp __x_min = ((_Tp)2); 
# 115
const int __nl = static_cast< int>(__nu + ((_Tp)(0.5L))); 
# 117
const _Tp __mu = __nu - __nl; 
# 118
const _Tp __mu2 = __mu * __mu; 
# 119
const _Tp __xi = ((_Tp)1) / __x; 
# 120
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 121
_Tp __h = __nu * __xi; 
# 122
if (__h < __fp_min) { 
# 123
__h = __fp_min; }  
# 124
_Tp __b = __xi2 * __nu; 
# 125
_Tp __d = ((_Tp)0); 
# 126
_Tp __c = __h; 
# 127
int __i; 
# 128
for (__i = 1; __i <= __max_iter; ++__i) 
# 129
{ 
# 130
__b += __xi2; 
# 131
__d = (((_Tp)1) / (__b + __d)); 
# 132
__c = (__b + (((_Tp)1) / __c)); 
# 133
const _Tp __del = __c * __d; 
# 134
__h *= __del; 
# 135
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 136
break; }  
# 137
}  
# 138
if (__i > __max_iter) { 
# 139
std::__throw_runtime_error("Argument x too large in __bessel_ik; try asymptotic expansion."); }  
# 142
_Tp __Inul = __fp_min; 
# 143
_Tp __Ipnul = __h * __Inul; 
# 144
_Tp __Inul1 = __Inul; 
# 145
_Tp __Ipnu1 = __Ipnul; 
# 146
_Tp __fact = __nu * __xi; 
# 147
for (int __l = __nl; __l >= 1; --__l) 
# 148
{ 
# 149
const _Tp __Inutemp = (__fact * __Inul) + __Ipnul; 
# 150
__fact -= __xi; 
# 151
__Ipnul = ((__fact * __Inutemp) + __Inul); 
# 152
__Inul = __Inutemp; 
# 153
}  
# 154
_Tp __f = __Ipnul / __Inul; 
# 155
_Tp __Kmu, __Knu1; 
# 156
if (__x < __x_min) 
# 157
{ 
# 158
const _Tp __x2 = __x / ((_Tp)2); 
# 159
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 160
const _Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 162
_Tp __d = (-std::log(__x2)); 
# 163
_Tp __e = __mu * __d; 
# 164
const _Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 166
_Tp __gam1, __gam2, __gampl, __gammi; 
# 167
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 168
_Tp __ff = __fact * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 170
_Tp __sum = __ff; 
# 171
__e = std::exp(__e); 
# 172
_Tp __p = __e / (((_Tp)2) * __gampl); 
# 173
_Tp __q = ((_Tp)1) / ((((_Tp)2) * __e) * __gammi); 
# 174
_Tp __c = ((_Tp)1); 
# 175
__d = (__x2 * __x2); 
# 176
_Tp __sum1 = __p; 
# 177
int __i; 
# 178
for (__i = 1; __i <= __max_iter; ++__i) 
# 179
{ 
# 180
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 181
__c *= (__d / __i); 
# 182
__p /= (__i - __mu); 
# 183
__q /= (__i + __mu); 
# 184
const _Tp __del = __c * __ff; 
# 185
__sum += __del; 
# 186
const _Tp __del1 = __c * (__p - (__i * __ff)); 
# 187
__sum1 += __del1; 
# 188
if (std::abs(__del) < (__eps * std::abs(__sum))) { 
# 189
break; }  
# 190
}  
# 191
if (__i > __max_iter) { 
# 192
std::__throw_runtime_error("Bessel k series failed to converge in __bessel_ik."); }  
# 194
__Kmu = __sum; 
# 195
__Knu1 = (__sum1 * __xi2); 
# 196
} else 
# 198
{ 
# 199
_Tp __b = ((_Tp)2) * (((_Tp)1) + __x); 
# 200
_Tp __d = ((_Tp)1) / __b; 
# 201
_Tp __delh = __d; 
# 202
_Tp __h = __delh; 
# 203
_Tp __q1 = ((_Tp)0); 
# 204
_Tp __q2 = ((_Tp)1); 
# 205
_Tp __a1 = ((_Tp)(0.25L)) - __mu2; 
# 206
_Tp __q = __c = __a1; 
# 207
_Tp __a = (-__a1); 
# 208
_Tp __s = ((_Tp)1) + (__q * __delh); 
# 209
int __i; 
# 210
for (__i = 2; __i <= __max_iter; ++__i) 
# 211
{ 
# 212
__a -= (2 * (__i - 1)); 
# 213
__c = (((-__a) * __c) / __i); 
# 214
const _Tp __qnew = (__q1 - (__b * __q2)) / __a; 
# 215
__q1 = __q2; 
# 216
__q2 = __qnew; 
# 217
__q += (__c * __qnew); 
# 218
__b += ((_Tp)2); 
# 219
__d = (((_Tp)1) / (__b + (__a * __d))); 
# 220
__delh = (((__b * __d) - ((_Tp)1)) * __delh); 
# 221
__h += __delh; 
# 222
const _Tp __dels = __q * __delh; 
# 223
__s += __dels; 
# 224
if (std::abs(__dels / __s) < __eps) { 
# 225
break; }  
# 226
}  
# 227
if (__i > __max_iter) { 
# 228
std::__throw_runtime_error("Steed\'s method failed in __bessel_ik."); }  
# 230
__h = (__a1 * __h); 
# 231
__Kmu = ((std::sqrt(__numeric_constants< _Tp> ::__pi() / (((_Tp)2) * __x)) * std::exp(-__x)) / __s); 
# 233
__Knu1 = ((__Kmu * (((__mu + __x) + ((_Tp)(0.5L))) - __h)) * __xi); 
# 234
}  
# 236
_Tp __Kpmu = ((__mu * __xi) * __Kmu) - __Knu1; 
# 237
_Tp __Inumu = __xi / ((__f * __Kmu) - __Kpmu); 
# 238
__Inu = ((__Inumu * __Inul1) / __Inul); 
# 239
__Ipnu = ((__Inumu * __Ipnu1) / __Inul); 
# 240
for (__i = 1; __i <= __nl; ++__i) 
# 241
{ 
# 242
const _Tp __Knutemp = (((__mu + __i) * __xi2) * __Knu1) + __Kmu; 
# 243
__Kmu = __Knu1; 
# 244
__Knu1 = __Knutemp; 
# 245
}  
# 246
__Knu = __Kmu; 
# 247
__Kpnu = (((__nu * __xi) * __Kmu) - __Knu1); 
# 250
} 
# 267
template< class _Tp> _Tp 
# 269
__cyl_bessel_i(_Tp __nu, _Tp __x) 
# 270
{ 
# 271
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 272
std::__throw_domain_error("Bad argument in __cyl_bessel_i."); } else { 
# 274
if (__isnan(__nu) || __isnan(__x)) { 
# 275
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 276
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 277
return __cyl_bessel_ij_series(__nu, __x, +((_Tp)1), 200); } else 
# 279
{ 
# 280
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 281
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 282
return __I_nu; 
# 283
}  }  }  
# 284
} 
# 303
template< class _Tp> _Tp 
# 305
__cyl_bessel_k(_Tp __nu, _Tp __x) 
# 306
{ 
# 307
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 308
std::__throw_domain_error("Bad argument in __cyl_bessel_k."); } else { 
# 310
if (__isnan(__nu) || __isnan(__x)) { 
# 311
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else 
# 313
{ 
# 314
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 315
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 316
return __K_nu; 
# 317
}  }  
# 318
} 
# 337
template< class _Tp> void 
# 339
__sph_bessel_ik(unsigned __n, _Tp __x, _Tp &
# 340
__i_n, _Tp &__k_n, _Tp &__ip_n, _Tp &__kp_n) 
# 341
{ 
# 342
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 344
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 345
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 347
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 350
__i_n = (__factor * __I_nu); 
# 351
__k_n = (__factor * __K_nu); 
# 352
__ip_n = ((__factor * __Ip_nu) - (__i_n / (((_Tp)2) * __x))); 
# 353
__kp_n = ((__factor * __Kp_nu) - (__k_n / (((_Tp)2) * __x))); 
# 356
} 
# 373
template< class _Tp> void 
# 375
__airy(_Tp __x, _Tp &__Ai, _Tp &__Bi, _Tp &__Aip, _Tp &__Bip) 
# 376
{ 
# 377
const _Tp __absx = std::abs(__x); 
# 378
const _Tp __rootx = std::sqrt(__absx); 
# 379
const _Tp __z = ((((_Tp)2) * __absx) * __rootx) / ((_Tp)3); 
# 380
const _Tp _S_inf = std::template numeric_limits< _Tp> ::infinity(); 
# 382
if (__isnan(__x)) { 
# 383
__Bip = (__Aip = (__Bi = (__Ai = std::template numeric_limits< _Tp> ::quiet_NaN()))); } else { 
# 384
if (__z == _S_inf) 
# 385
{ 
# 386
__Aip = (__Ai = ((_Tp)0)); 
# 387
__Bip = (__Bi = _S_inf); 
# 388
} else { 
# 389
if (__z == (-_S_inf)) { 
# 390
__Bip = (__Aip = (__Bi = (__Ai = ((_Tp)0)))); } else { 
# 391
if (__x > ((_Tp)0)) 
# 392
{ 
# 393
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 395
__bessel_ik(((_Tp)1) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 396
__Ai = ((__rootx * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 399
__Bi = (__rootx * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 402
__bessel_ik(((_Tp)2) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 403
__Aip = (((-__x) * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 406
__Bip = (__x * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 409
} else { 
# 410
if (__x < ((_Tp)0)) 
# 411
{ 
# 412
_Tp __J_nu, __Jp_nu, __N_nu, __Np_nu; 
# 414
__bessel_jn(((_Tp)1) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 415
__Ai = ((__rootx * (__J_nu - (__N_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 417
__Bi = (((-__rootx) * (__N_nu + (__J_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 420
__bessel_jn(((_Tp)2) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 421
__Aip = ((__absx * ((__N_nu / __numeric_constants< _Tp> ::__sqrt3()) + __J_nu)) / ((_Tp)2)); 
# 423
__Bip = ((__absx * ((__J_nu / __numeric_constants< _Tp> ::__sqrt3()) - __N_nu)) / ((_Tp)2)); 
# 425
} else 
# 427
{ 
# 431
__Ai = ((_Tp)(0.35502805388781723926L)); 
# 432
__Bi = (__Ai * __numeric_constants< _Tp> ::__sqrt3()); 
# 437
__Aip = (-((_Tp)(0.2588194037928067984L))); 
# 438
__Bip = ((-__Aip) * __numeric_constants< _Tp> ::__sqrt3()); 
# 439
}  }  }  }  }  
# 442
} 
# 443
}
# 449
}
# 42 "/usr/include/c++/13/tr1/poly_hermite.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 56 "/usr/include/c++/13/tr1/poly_hermite.tcc" 3
namespace __detail { 
# 72
template< class _Tp> _Tp 
# 74
__poly_hermite_recursion(unsigned __n, _Tp __x) 
# 75
{ 
# 77
_Tp __H_0 = (1); 
# 78
if (__n == (0)) { 
# 79
return __H_0; }  
# 82
_Tp __H_1 = 2 * __x; 
# 83
if (__n == (1)) { 
# 84
return __H_1; }  
# 87
_Tp __H_n, __H_nm1, __H_nm2; 
# 88
unsigned __i; 
# 89
for (((__H_nm2 = __H_0), (__H_nm1 = __H_1)), (__i = (2)); __i <= __n; ++__i) 
# 90
{ 
# 91
__H_n = (2 * ((__x * __H_nm1) - ((__i - (1)) * __H_nm2))); 
# 92
__H_nm2 = __H_nm1; 
# 93
__H_nm1 = __H_n; 
# 94
}  
# 96
return __H_n; 
# 97
} 
# 114
template< class _Tp> inline _Tp 
# 116
__poly_hermite(unsigned __n, _Tp __x) 
# 117
{ 
# 118
if (__isnan(__x)) { 
# 119
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 121
return __poly_hermite_recursion(__n, __x); }  
# 122
} 
# 123
}
# 129
}
# 44 "/usr/include/c++/13/tr1/poly_laguerre.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/usr/include/c++/13/tr1/poly_laguerre.tcc" 3
namespace __detail { 
# 75
template< class _Tpa, class _Tp> _Tp 
# 77
__poly_laguerre_large_n(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 78
{ 
# 79
const _Tp __a = (-((_Tp)__n)); 
# 80
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 81
const _Tp __eta = (((_Tp)2) * __b) - (((_Tp)4) * __a); 
# 82
const _Tp __cos2th = __x / __eta; 
# 83
const _Tp __sin2th = ((_Tp)1) - __cos2th; 
# 84
const _Tp __th = std::acos(std::sqrt(__cos2th)); 
# 85
const _Tp __pre_h = ((((__numeric_constants< _Tp> ::__pi_2() * __numeric_constants< _Tp> ::__pi_2()) * __eta) * __eta) * __cos2th) * __sin2th; 
# 90
const _Tp __lg_b = std::lgamma(((_Tp)__n) + __b); 
# 91
const _Tp __lnfact = std::lgamma((_Tp)(__n + (1))); 
# 97
_Tp __pre_term1 = (((_Tp)(0.5L)) * (((_Tp)1) - __b)) * std::log((((_Tp)(0.25L)) * __x) * __eta); 
# 99
_Tp __pre_term2 = ((_Tp)(0.25L)) * std::log(__pre_h); 
# 100
_Tp __lnpre = (((__lg_b - __lnfact) + (((_Tp)(0.5L)) * __x)) + __pre_term1) - __pre_term2; 
# 102
_Tp __ser_term1 = std::sin(__a * __numeric_constants< _Tp> ::__pi()); 
# 103
_Tp __ser_term2 = std::sin(((((_Tp)(0.25L)) * __eta) * ((((_Tp)2) * __th) - std::sin(((_Tp)2) * __th))) + __numeric_constants< _Tp> ::__pi_4()); 
# 107
_Tp __ser = __ser_term1 + __ser_term2; 
# 109
return std::exp(__lnpre) * __ser; 
# 110
} 
# 129
template< class _Tpa, class _Tp> _Tp 
# 131
__poly_laguerre_hyperg(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 132
{ 
# 133
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 134
const _Tp __mx = (-__x); 
# 135
const _Tp __tc_sgn = (__x < ((_Tp)0)) ? (_Tp)1 : (((__n % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1)); 
# 138
_Tp __tc = ((_Tp)1); 
# 139
const _Tp __ax = std::abs(__x); 
# 140
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 141
__tc *= (__ax / __k); }  
# 143
_Tp __term = __tc * __tc_sgn; 
# 144
_Tp __sum = __term; 
# 145
for (int __k = ((int)__n) - 1; __k >= 0; --__k) 
# 146
{ 
# 147
__term *= ((((__b + ((_Tp)__k)) / ((_Tp)(((int)__n) - __k))) * ((_Tp)(__k + 1))) / __mx); 
# 149
__sum += __term; 
# 150
}  
# 152
return __sum; 
# 153
} 
# 185
template< class _Tpa, class _Tp> _Tp 
# 187
__poly_laguerre_recursion(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 188
{ 
# 190
_Tp __l_0 = ((_Tp)1); 
# 191
if (__n == (0)) { 
# 192
return __l_0; }  
# 195
_Tp __l_1 = (((-__x) + ((_Tp)1)) + ((_Tp)__alpha1)); 
# 196
if (__n == (1)) { 
# 197
return __l_1; }  
# 200
_Tp __l_n2 = __l_0; 
# 201
_Tp __l_n1 = __l_1; 
# 202
_Tp __l_n = ((_Tp)0); 
# 203
for (unsigned __nn = (2); __nn <= __n; ++__nn) 
# 204
{ 
# 205
__l_n = (((((((_Tp)(((2) * __nn) - (1))) + ((_Tp)__alpha1)) - __x) * __l_n1) / ((_Tp)__nn)) - (((((_Tp)(__nn - (1))) + ((_Tp)__alpha1)) * __l_n2) / ((_Tp)__nn))); 
# 208
__l_n2 = __l_n1; 
# 209
__l_n1 = __l_n; 
# 210
}  
# 212
return __l_n; 
# 213
} 
# 244
template< class _Tpa, class _Tp> _Tp 
# 246
__poly_laguerre(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 247
{ 
# 248
if (__x < ((_Tp)0)) { 
# 249
std::__throw_domain_error("Negative argument in __poly_laguerre."); } else { 
# 252
if (__isnan(__x)) { 
# 253
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 254
if (__n == (0)) { 
# 255
return (_Tp)1; } else { 
# 256
if (__n == (1)) { 
# 257
return (((_Tp)1) + ((_Tp)__alpha1)) - __x; } else { 
# 258
if (__x == ((_Tp)0)) 
# 259
{ 
# 260
_Tp __prod = ((_Tp)__alpha1) + ((_Tp)1); 
# 261
for (unsigned __k = (2); __k <= __n; ++__k) { 
# 262
__prod *= ((((_Tp)__alpha1) + ((_Tp)__k)) / ((_Tp)__k)); }  
# 263
return __prod; 
# 264
} else { 
# 265
if ((__n > (10000000)) && (((_Tp)__alpha1) > (-((_Tp)1))) && (__x < ((((_Tp)2) * (((_Tp)__alpha1) + ((_Tp)1))) + ((_Tp)((4) * __n))))) { 
# 267
return __poly_laguerre_large_n(__n, __alpha1, __x); } else { 
# 268
if ((((_Tp)__alpha1) >= ((_Tp)0)) || ((__x > ((_Tp)0)) && (((_Tp)__alpha1) < (-((_Tp)(__n + (1))))))) { 
# 270
return __poly_laguerre_recursion(__n, __alpha1, __x); } else { 
# 272
return __poly_laguerre_hyperg(__n, __alpha1, __x); }  }  }  }  }  }  }  
# 273
} 
# 296
template< class _Tp> inline _Tp 
# 298
__assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 299
{ return __poly_laguerre< unsigned, _Tp> (__n, __m, __x); } 
# 316
template< class _Tp> inline _Tp 
# 318
__laguerre(unsigned __n, _Tp __x) 
# 319
{ return __poly_laguerre< unsigned, _Tp> (__n, 0, __x); } 
# 320
}
# 327
}
# 47 "/usr/include/c++/13/tr1/riemann_zeta.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 63 "/usr/include/c++/13/tr1/riemann_zeta.tcc" 3
namespace __detail { 
# 78
template< class _Tp> _Tp 
# 80
__riemann_zeta_sum(_Tp __s) 
# 81
{ 
# 83
if (__s < ((_Tp)1)) { 
# 84
std::__throw_domain_error("Bad argument in zeta sum."); }  
# 86
const unsigned max_iter = (10000); 
# 87
_Tp __zeta = ((_Tp)0); 
# 88
for (unsigned __k = (1); __k < max_iter; ++__k) 
# 89
{ 
# 90
_Tp __term = std::pow(static_cast< _Tp>(__k), -__s); 
# 91
if (__term < std::template numeric_limits< _Tp> ::epsilon()) 
# 92
{ 
# 93
break; 
# 94
}  
# 95
__zeta += __term; 
# 96
}  
# 98
return __zeta; 
# 99
} 
# 115
template< class _Tp> _Tp 
# 117
__riemann_zeta_alt(_Tp __s) 
# 118
{ 
# 119
_Tp __sgn = ((_Tp)1); 
# 120
_Tp __zeta = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (10000000); ++__i) 
# 122
{ 
# 123
_Tp __term = __sgn / std::pow(__i, __s); 
# 124
if (std::abs(__term) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__zeta += __term; 
# 127
__sgn *= ((_Tp)(-1)); 
# 128
}  
# 129
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 131
return __zeta; 
# 132
} 
# 157
template< class _Tp> _Tp 
# 159
__riemann_zeta_glob(_Tp __s) 
# 160
{ 
# 161
_Tp __zeta = ((_Tp)0); 
# 163
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 165
const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 170
if (__s < ((_Tp)0)) 
# 171
{ 
# 173
if (std::fmod(__s, (_Tp)2) == ((_Tp)0)) { 
# 174
return (_Tp)0; } else 
# 177
{ 
# 178
_Tp __zeta = __riemann_zeta_glob(((_Tp)1) - __s); 
# 179
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 188
return __zeta; 
# 189
}  
# 190
}  
# 192
_Tp __num = ((_Tp)(0.5L)); 
# 193
const unsigned __maxit = (10000); 
# 194
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 195
{ 
# 196
bool __punt = false; 
# 197
_Tp __sgn = ((_Tp)1); 
# 198
_Tp __term = ((_Tp)0); 
# 199
for (unsigned __j = (0); __j <= __i; ++__j) 
# 200
{ 
# 202
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 210
if (__bincoeff > __max_bincoeff) 
# 211
{ 
# 213
__punt = true; 
# 214
break; 
# 215
}  
# 216
__bincoeff = std::exp(__bincoeff); 
# 217
__term += ((__sgn * __bincoeff) * std::pow((_Tp)((1) + __j), -__s)); 
# 218
__sgn *= ((_Tp)(-1)); 
# 219
}  
# 220
if (__punt) { 
# 221
break; }  
# 222
__term *= __num; 
# 223
__zeta += __term; 
# 224
if (std::abs(__term / __zeta) < __eps) { 
# 225
break; }  
# 226
__num *= ((_Tp)(0.5L)); 
# 227
}  
# 229
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 231
return __zeta; 
# 232
} 
# 252
template< class _Tp> _Tp 
# 254
__riemann_zeta_product(_Tp __s) 
# 255
{ 
# 256
static const _Tp __prime[] = {((_Tp)2), ((_Tp)3), ((_Tp)5), ((_Tp)7), ((_Tp)11), ((_Tp)13), ((_Tp)17), ((_Tp)19), ((_Tp)23), ((_Tp)29), ((_Tp)31), ((_Tp)37), ((_Tp)41), ((_Tp)43), ((_Tp)47), ((_Tp)53), ((_Tp)59), ((_Tp)61), ((_Tp)67), ((_Tp)71), ((_Tp)73), ((_Tp)79), ((_Tp)83), ((_Tp)89), ((_Tp)97), ((_Tp)101), ((_Tp)103), ((_Tp)107), ((_Tp)109)}; 
# 262
static const unsigned __num_primes = (sizeof(__prime) / sizeof(_Tp)); 
# 264
_Tp __zeta = ((_Tp)1); 
# 265
for (unsigned __i = (0); __i < __num_primes; ++__i) 
# 266
{ 
# 267
const _Tp __fact = ((_Tp)1) - std::pow(__prime[__i], -__s); 
# 268
__zeta *= __fact; 
# 269
if ((((_Tp)1) - __fact) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 270
break; }  
# 271
}  
# 273
__zeta = (((_Tp)1) / __zeta); 
# 275
return __zeta; 
# 276
} 
# 293
template< class _Tp> _Tp 
# 295
__riemann_zeta(_Tp __s) 
# 296
{ 
# 297
if (__isnan(__s)) { 
# 298
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 299
if (__s == ((_Tp)1)) { 
# 300
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 301
if (__s < (-((_Tp)19))) 
# 302
{ 
# 303
_Tp __zeta = __riemann_zeta_product(((_Tp)1) - __s); 
# 304
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 312
return __zeta; 
# 313
} else { 
# 314
if (__s < ((_Tp)20)) 
# 315
{ 
# 317
bool __glob = true; 
# 318
if (__glob) { 
# 319
return __riemann_zeta_glob(__s); } else 
# 321
{ 
# 322
if (__s > ((_Tp)1)) { 
# 323
return __riemann_zeta_sum(__s); } else 
# 325
{ 
# 326
_Tp __zeta = ((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::tgamma(((_Tp)1) - __s)) * __riemann_zeta_sum(((_Tp)1) - __s); 
# 335
return __zeta; 
# 336
}  
# 337
}  
# 338
} else { 
# 340
return __riemann_zeta_product(__s); }  }  }  }  
# 341
} 
# 365
template< class _Tp> _Tp 
# 367
__hurwitz_zeta_glob(_Tp __a, _Tp __s) 
# 368
{ 
# 369
_Tp __zeta = ((_Tp)0); 
# 371
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 373
const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 376
const unsigned __maxit = (10000); 
# 377
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 378
{ 
# 379
bool __punt = false; 
# 380
_Tp __sgn = ((_Tp)1); 
# 381
_Tp __term = ((_Tp)0); 
# 382
for (unsigned __j = (0); __j <= __i; ++__j) 
# 383
{ 
# 385
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 393
if (__bincoeff > __max_bincoeff) 
# 394
{ 
# 396
__punt = true; 
# 397
break; 
# 398
}  
# 399
__bincoeff = std::exp(__bincoeff); 
# 400
__term += ((__sgn * __bincoeff) * std::pow((_Tp)(__a + __j), -__s)); 
# 401
__sgn *= ((_Tp)(-1)); 
# 402
}  
# 403
if (__punt) { 
# 404
break; }  
# 405
__term /= ((_Tp)(__i + (1))); 
# 406
if (std::abs(__term / __zeta) < __eps) { 
# 407
break; }  
# 408
__zeta += __term; 
# 409
}  
# 411
__zeta /= (__s - ((_Tp)1)); 
# 413
return __zeta; 
# 414
} 
# 430
template< class _Tp> inline _Tp 
# 432
__hurwitz_zeta(_Tp __a, _Tp __s) 
# 433
{ return __hurwitz_zeta_glob(__a, __s); } 
# 434
}
# 441
}
# 59 "/usr/include/c++/13/bits/specfun.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 204
inline float assoc_laguerref(unsigned __n, unsigned __m, float __x) 
# 205
{ return __detail::__assoc_laguerre< float> (__n, __m, __x); } 
# 214
inline long double assoc_laguerrel(unsigned __n, unsigned __m, long double __x) 
# 215
{ return __detail::__assoc_laguerre< long double> (__n, __m, __x); } 
# 248
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 250
assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 251
{ 
# 252
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 253
return __detail::__assoc_laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __m, __x); 
# 254
} 
# 265
inline float assoc_legendref(unsigned __l, unsigned __m, float __x) 
# 266
{ return __detail::__assoc_legendre_p< float> (__l, __m, __x); } 
# 274
inline long double assoc_legendrel(unsigned __l, unsigned __m, long double __x) 
# 275
{ return __detail::__assoc_legendre_p< long double> (__l, __m, __x); } 
# 294
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 296
assoc_legendre(unsigned __l, unsigned __m, _Tp __x) 
# 297
{ 
# 298
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 299
return __detail::__assoc_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __x); 
# 300
} 
# 310
inline float betaf(float __a, float __b) 
# 311
{ return __detail::__beta< float> (__a, __b); } 
# 320
inline long double betal(long double __a, long double __b) 
# 321
{ return __detail::__beta< long double> (__a, __b); } 
# 339
template< class _Tpa, class _Tpb> inline typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type 
# 341
beta(_Tpa __a, _Tpb __b) 
# 342
{ 
# 343
typedef typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type __type; 
# 344
return __detail::__beta< typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type> (__a, __b); 
# 345
} 
# 356
inline float comp_ellint_1f(float __k) 
# 357
{ return __detail::__comp_ellint_1< float> (__k); } 
# 366
inline long double comp_ellint_1l(long double __k) 
# 367
{ return __detail::__comp_ellint_1< long double> (__k); } 
# 387
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 389
comp_ellint_1(_Tp __k) 
# 390
{ 
# 391
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 392
return __detail::__comp_ellint_1< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 393
} 
# 404
inline float comp_ellint_2f(float __k) 
# 405
{ return __detail::__comp_ellint_2< float> (__k); } 
# 414
inline long double comp_ellint_2l(long double __k) 
# 415
{ return __detail::__comp_ellint_2< long double> (__k); } 
# 434
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 436
comp_ellint_2(_Tp __k) 
# 437
{ 
# 438
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 439
return __detail::__comp_ellint_2< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 440
} 
# 451
inline float comp_ellint_3f(float __k, float __nu) 
# 452
{ return __detail::__comp_ellint_3< float> (__k, __nu); } 
# 461
inline long double comp_ellint_3l(long double __k, long double __nu) 
# 462
{ return __detail::__comp_ellint_3< long double> (__k, __nu); } 
# 485
template< class _Tp, class _Tpn> inline typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type 
# 487
comp_ellint_3(_Tp __k, _Tpn __nu) 
# 488
{ 
# 489
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type __type; 
# 490
return __detail::__comp_ellint_3< typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type> (__k, __nu); 
# 491
} 
# 502
inline float cyl_bessel_if(float __nu, float __x) 
# 503
{ return __detail::__cyl_bessel_i< float> (__nu, __x); } 
# 512
inline long double cyl_bessel_il(long double __nu, long double __x) 
# 513
{ return __detail::__cyl_bessel_i< long double> (__nu, __x); } 
# 531
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 533
cyl_bessel_i(_Tpnu __nu, _Tp __x) 
# 534
{ 
# 535
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 536
return __detail::__cyl_bessel_i< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 537
} 
# 548
inline float cyl_bessel_jf(float __nu, float __x) 
# 549
{ return __detail::__cyl_bessel_j< float> (__nu, __x); } 
# 558
inline long double cyl_bessel_jl(long double __nu, long double __x) 
# 559
{ return __detail::__cyl_bessel_j< long double> (__nu, __x); } 
# 577
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 579
cyl_bessel_j(_Tpnu __nu, _Tp __x) 
# 580
{ 
# 581
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 582
return __detail::__cyl_bessel_j< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 583
} 
# 594
inline float cyl_bessel_kf(float __nu, float __x) 
# 595
{ return __detail::__cyl_bessel_k< float> (__nu, __x); } 
# 604
inline long double cyl_bessel_kl(long double __nu, long double __x) 
# 605
{ return __detail::__cyl_bessel_k< long double> (__nu, __x); } 
# 629
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 631
cyl_bessel_k(_Tpnu __nu, _Tp __x) 
# 632
{ 
# 633
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 634
return __detail::__cyl_bessel_k< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 635
} 
# 646
inline float cyl_neumannf(float __nu, float __x) 
# 647
{ return __detail::__cyl_neumann_n< float> (__nu, __x); } 
# 656
inline long double cyl_neumannl(long double __nu, long double __x) 
# 657
{ return __detail::__cyl_neumann_n< long double> (__nu, __x); } 
# 677
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 679
cyl_neumann(_Tpnu __nu, _Tp __x) 
# 680
{ 
# 681
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 682
return __detail::__cyl_neumann_n< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 683
} 
# 694
inline float ellint_1f(float __k, float __phi) 
# 695
{ return __detail::__ellint_1< float> (__k, __phi); } 
# 704
inline long double ellint_1l(long double __k, long double __phi) 
# 705
{ return __detail::__ellint_1< long double> (__k, __phi); } 
# 725
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 727
ellint_1(_Tp __k, _Tpp __phi) 
# 728
{ 
# 729
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 730
return __detail::__ellint_1< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 731
} 
# 742
inline float ellint_2f(float __k, float __phi) 
# 743
{ return __detail::__ellint_2< float> (__k, __phi); } 
# 752
inline long double ellint_2l(long double __k, long double __phi) 
# 753
{ return __detail::__ellint_2< long double> (__k, __phi); } 
# 773
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 775
ellint_2(_Tp __k, _Tpp __phi) 
# 776
{ 
# 777
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 778
return __detail::__ellint_2< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 779
} 
# 790
inline float ellint_3f(float __k, float __nu, float __phi) 
# 791
{ return __detail::__ellint_3< float> (__k, __nu, __phi); } 
# 800
inline long double ellint_3l(long double __k, long double __nu, long double __phi) 
# 801
{ return __detail::__ellint_3< long double> (__k, __nu, __phi); } 
# 826
template< class _Tp, class _Tpn, class _Tpp> inline typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type 
# 828
ellint_3(_Tp __k, _Tpn __nu, _Tpp __phi) 
# 829
{ 
# 830
typedef typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type __type; 
# 831
return __detail::__ellint_3< typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type> (__k, __nu, __phi); 
# 832
} 
# 842
inline float expintf(float __x) 
# 843
{ return __detail::__expint< float> (__x); } 
# 852
inline long double expintl(long double __x) 
# 853
{ return __detail::__expint< long double> (__x); } 
# 866
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 868
expint(_Tp __x) 
# 869
{ 
# 870
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 871
return __detail::__expint< typename __gnu_cxx::__promote< _Tp> ::__type> (__x); 
# 872
} 
# 883
inline float hermitef(unsigned __n, float __x) 
# 884
{ return __detail::__poly_hermite< float> (__n, __x); } 
# 893
inline long double hermitel(unsigned __n, long double __x) 
# 894
{ return __detail::__poly_hermite< long double> (__n, __x); } 
# 914
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 916
hermite(unsigned __n, _Tp __x) 
# 917
{ 
# 918
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 919
return __detail::__poly_hermite< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 920
} 
# 931
inline float laguerref(unsigned __n, float __x) 
# 932
{ return __detail::__laguerre< float> (__n, __x); } 
# 941
inline long double laguerrel(unsigned __n, long double __x) 
# 942
{ return __detail::__laguerre< long double> (__n, __x); } 
# 958
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 960
laguerre(unsigned __n, _Tp __x) 
# 961
{ 
# 962
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 963
return __detail::__laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 964
} 
# 975
inline float legendref(unsigned __l, float __x) 
# 976
{ return __detail::__poly_legendre_p< float> (__l, __x); } 
# 985
inline long double legendrel(unsigned __l, long double __x) 
# 986
{ return __detail::__poly_legendre_p< long double> (__l, __x); } 
# 1003
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1005
legendre(unsigned __l, _Tp __x) 
# 1006
{ 
# 1007
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1008
return __detail::__poly_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __x); 
# 1009
} 
# 1020
inline float riemann_zetaf(float __s) 
# 1021
{ return __detail::__riemann_zeta< float> (__s); } 
# 1030
inline long double riemann_zetal(long double __s) 
# 1031
{ return __detail::__riemann_zeta< long double> (__s); } 
# 1054
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1056
riemann_zeta(_Tp __s) 
# 1057
{ 
# 1058
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1059
return __detail::__riemann_zeta< typename __gnu_cxx::__promote< _Tp> ::__type> (__s); 
# 1060
} 
# 1071
inline float sph_besself(unsigned __n, float __x) 
# 1072
{ return __detail::__sph_bessel< float> (__n, __x); } 
# 1081
inline long double sph_bessell(unsigned __n, long double __x) 
# 1082
{ return __detail::__sph_bessel< long double> (__n, __x); } 
# 1098
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1100
sph_bessel(unsigned __n, _Tp __x) 
# 1101
{ 
# 1102
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1103
return __detail::__sph_bessel< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1104
} 
# 1115
inline float sph_legendref(unsigned __l, unsigned __m, float __theta) 
# 1116
{ return __detail::__sph_legendre< float> (__l, __m, __theta); } 
# 1126
inline long double sph_legendrel(unsigned __l, unsigned __m, long double __theta) 
# 1127
{ return __detail::__sph_legendre< long double> (__l, __m, __theta); } 
# 1145
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1147
sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 1148
{ 
# 1149
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1150
return __detail::__sph_legendre< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __theta); 
# 1151
} 
# 1162
inline float sph_neumannf(unsigned __n, float __x) 
# 1163
{ return __detail::__sph_neumann< float> (__n, __x); } 
# 1172
inline long double sph_neumannl(unsigned __n, long double __x) 
# 1173
{ return __detail::__sph_neumann< long double> (__n, __x); } 
# 1189
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1191
sph_neumann(unsigned __n, _Tp __x) 
# 1192
{ 
# 1193
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1194
return __detail::__sph_neumann< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1195
} 
# 1200
}
# 1203
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 1217
inline float airy_aif(float __x) 
# 1218
{ 
# 1219
float __Ai, __Bi, __Aip, __Bip; 
# 1220
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1221
return __Ai; 
# 1222
} 
# 1228
inline long double airy_ail(long double __x) 
# 1229
{ 
# 1230
long double __Ai, __Bi, __Aip, __Bip; 
# 1231
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1232
return __Ai; 
# 1233
} 
# 1238
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1240
airy_ai(_Tp __x) 
# 1241
{ 
# 1242
typedef typename __promote< _Tp> ::__type __type; 
# 1243
__type __Ai, __Bi, __Aip, __Bip; 
# 1244
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1245
return __Ai; 
# 1246
} 
# 1252
inline float airy_bif(float __x) 
# 1253
{ 
# 1254
float __Ai, __Bi, __Aip, __Bip; 
# 1255
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1256
return __Bi; 
# 1257
} 
# 1263
inline long double airy_bil(long double __x) 
# 1264
{ 
# 1265
long double __Ai, __Bi, __Aip, __Bip; 
# 1266
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1267
return __Bi; 
# 1268
} 
# 1273
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1275
airy_bi(_Tp __x) 
# 1276
{ 
# 1277
typedef typename __promote< _Tp> ::__type __type; 
# 1278
__type __Ai, __Bi, __Aip, __Bip; 
# 1279
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1280
return __Bi; 
# 1281
} 
# 1293
inline float conf_hypergf(float __a, float __c, float __x) 
# 1294
{ return std::__detail::__conf_hyperg< float> (__a, __c, __x); } 
# 1304
inline long double conf_hypergl(long double __a, long double __c, long double __x) 
# 1305
{ return std::__detail::__conf_hyperg< long double> (__a, __c, __x); } 
# 1323
template< class _Tpa, class _Tpc, class _Tp> inline typename __promote_3< _Tpa, _Tpc, _Tp> ::__type 
# 1325
conf_hyperg(_Tpa __a, _Tpc __c, _Tp __x) 
# 1326
{ 
# 1327
typedef typename __promote_3< _Tpa, _Tpc, _Tp> ::__type __type; 
# 1328
return std::__detail::__conf_hyperg< typename __promote_3< _Tpa, _Tpc, _Tp> ::__type> (__a, __c, __x); 
# 1329
} 
# 1341
inline float hypergf(float __a, float __b, float __c, float __x) 
# 1342
{ return std::__detail::__hyperg< float> (__a, __b, __c, __x); } 
# 1352
inline long double hypergl(long double __a, long double __b, long double __c, long double __x) 
# 1353
{ return std::__detail::__hyperg< long double> (__a, __b, __c, __x); } 
# 1372
template< class _Tpa, class _Tpb, class _Tpc, class _Tp> inline typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type 
# 1374
hyperg(_Tpa __a, _Tpb __b, _Tpc __c, _Tp __x) 
# 1375
{ 
# 1377
typedef typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type __type; 
# 1378
return std::__detail::__hyperg< typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type> (__a, __b, __c, __x); 
# 1379
} 
# 1383
}
# 3702 "/usr/include/c++/13/cmath" 3
}
# 24 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/include/cmath" 3
#pragma libm (acosf, acoshf, asinf, asinhf, atanhf, atan2f)
#pragma libm (cbrtf, ceilf, copysignf, cosf, coshf)
#pragma libm (erff, erfcf, expf, exp2f, exp10f, expm1f)
#pragma libm (fabsf, floorf, fmaf, fminf, fmaxf)
#pragma libm (ilogbf)
#pragma libm (ldexpf, lgammaf, llrintf, llroundf, logbf, log1pf, logf, log2f, log10f, lrintf, lroundf)
#pragma libm (modff)
#pragma libm (nanf, nearbyintf, nextafterf)
#pragma libm (powf)
#pragma libm (remainderf, remquof, rintf, roundf, rsqrtf)
#pragma libm (scalblnf, scalbnf, sinf, sinhf, sqrtf)
#pragma libm (tanf, tanhf, tgammaf, truncf)
#pragma libm (abs, acos, acosh, asin, asinh, atanh, atan2)
#pragma libm (cbrt, ceil, copysign, cos, cosh)
#pragma libm (erf, erfc, exp, exp2, exp10, expm1)
#pragma libm (fabs, floor, fma, fmin, fmax)
#pragma libm (ilogb, isinf, isfinite, isnan)
#pragma libm (ldexp, lgamma, llrint, llround, logb, log1p, log, log2, log10, lrint, lround)
#pragma libm (modf)
#pragma libm (nan, nearbyint, nextafter)
#pragma libm (pow)
#pragma libm (remainder, remquo, rint, round, rsqrt)
#pragma libm (scalbln, scalbn, sin, sinh, sqrt)
#pragma libm (tan, tanh, tgamma, trunc)
#pragma libm (llabs)
#pragma libm (cyl_bessel_i0f,cyl_bessel_i1f)
#pragma libm (erfcinvf,erfcxf,erfinvf)
#pragma libm (erfcinv,erfcx,erfinv)
#pragma libm (fdim,fdimf)
#pragma libm (normf,norm3df,norm4df,normcdff,normcdfinvf)
#pragma libm (norm,norm3d,norm4d,normcdf,normcdfinv)
#pragma libm (rnormf,rnorm3df,rnorm4df,rhypotf,rcbrtf)
#pragma libm (rnorm,rnorm3d,rnorm4d,rhypot,rcbrt)
#pragma libm (ynf,y1f,y0f)
#pragma libm (yn,y1,y0)
#pragma libm (jnf,j1f,j0f)
#pragma libm (jn,j1,j0)
# 38 "/usr/include/c++/13/math.h" 3
using std::abs;
# 39
using std::acos;
# 40
using std::asin;
# 41
using std::atan;
# 42
using std::atan2;
# 43
using std::cos;
# 44
using std::sin;
# 45
using std::tan;
# 46
using std::cosh;
# 47
using std::sinh;
# 48
using std::tanh;
# 49
using std::exp;
# 50
using std::frexp;
# 51
using std::ldexp;
# 52
using std::log;
# 53
using std::log10;
# 54
using std::modf;
# 55
using std::pow;
# 56
using std::sqrt;
# 57
using std::ceil;
# 58
using std::fabs;
# 59
using std::floor;
# 60
using std::fmod;
# 63
using std::fpclassify;
# 64
using std::isfinite;
# 65
using std::isinf;
# 66
using std::isnan;
# 67
using std::isnormal;
# 68
using std::signbit;
# 69
using std::isgreater;
# 70
using std::isgreaterequal;
# 71
using std::isless;
# 72
using std::islessequal;
# 73
using std::islessgreater;
# 74
using std::isunordered;
# 78
using std::acosh;
# 79
using std::asinh;
# 80
using std::atanh;
# 81
using std::cbrt;
# 82
using std::copysign;
# 83
using std::erf;
# 84
using std::erfc;
# 85
using std::exp2;
# 86
using std::expm1;
# 87
using std::fdim;
# 88
using std::fma;
# 89
using std::fmax;
# 90
using std::fmin;
# 91
using std::hypot;
# 92
using std::ilogb;
# 93
using std::lgamma;
# 94
using std::llrint;
# 95
using std::llround;
# 96
using std::log1p;
# 97
using std::log2;
# 98
using std::logb;
# 99
using std::lrint;
# 100
using std::lround;
# 101
using std::nearbyint;
# 102
using std::nextafter;
# 103
using std::nexttoward;
# 104
using std::remainder;
# 105
using std::remquo;
# 106
using std::rint;
# 107
using std::round;
# 108
using std::scalbln;
# 109
using std::scalbn;
# 110
using std::tgamma;
# 111
using std::trunc;
# 10626 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 10627
constexpr bool signbit(float x); 
# 10628
constexpr bool signbit(double x); 
# 10629
constexpr bool signbit(long double x); 
# 10630
constexpr bool isfinite(float x); 
# 10631
constexpr bool isfinite(double x); 
# 10632
constexpr bool isfinite(long double x); 
# 10633
constexpr bool isnan(float x); 
# 10638
constexpr bool isnan(double x); 
# 10640
constexpr bool isnan(long double x); 
# 10641
constexpr bool isinf(float x); 
# 10646
constexpr bool isinf(double x); 
# 10648
constexpr bool isinf(long double x); 
# 10649
}
# 10805 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 10807
template< class T> extern T __pow_helper(T, int); 
# 10808
template< class T> extern T __cmath_power(T, unsigned); 
# 10809
}
# 10811
using std::abs;
# 10812
using std::fabs;
# 10813
using std::ceil;
# 10814
using std::floor;
# 10815
using std::sqrt;
# 10817
using std::pow;
# 10819
using std::log;
# 10820
using std::log10;
# 10821
using std::fmod;
# 10822
using std::modf;
# 10823
using std::exp;
# 10824
using std::frexp;
# 10825
using std::ldexp;
# 10826
using std::asin;
# 10827
using std::sin;
# 10828
using std::sinh;
# 10829
using std::acos;
# 10830
using std::cos;
# 10831
using std::cosh;
# 10832
using std::atan;
# 10833
using std::atan2;
# 10834
using std::tan;
# 10835
using std::tanh;
# 11206 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 11215
extern inline long long abs(long long); 
# 11225
extern inline long abs(long); 
# 11226
extern constexpr float abs(float); 
# 11227
extern constexpr double abs(double); 
# 11228
extern constexpr float fabs(float); 
# 11229
extern constexpr float ceil(float); 
# 11230
extern constexpr float floor(float); 
# 11231
extern constexpr float sqrt(float); 
# 11232
extern constexpr float pow(float, float); 
# 11237
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 11247
extern constexpr float log(float); 
# 11248
extern constexpr float log10(float); 
# 11249
extern constexpr float fmod(float, float); 
# 11250
extern inline float modf(float, float *); 
# 11251
extern constexpr float exp(float); 
# 11252
extern inline float frexp(float, int *); 
# 11253
extern constexpr float ldexp(float, int); 
# 11254
extern constexpr float asin(float); 
# 11255
extern constexpr float sin(float); 
# 11256
extern constexpr float sinh(float); 
# 11257
extern constexpr float acos(float); 
# 11258
extern constexpr float cos(float); 
# 11259
extern constexpr float cosh(float); 
# 11260
extern constexpr float atan(float); 
# 11261
extern constexpr float atan2(float, float); 
# 11262
extern constexpr float tan(float); 
# 11263
extern constexpr float tanh(float); 
# 11350 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 11456 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 11457
constexpr float logb(float a); 
# 11458
constexpr int ilogb(float a); 
# 11459
constexpr float scalbn(float a, int b); 
# 11460
constexpr float scalbln(float a, long b); 
# 11461
constexpr float exp2(float a); 
# 11462
constexpr float expm1(float a); 
# 11463
constexpr float log2(float a); 
# 11464
constexpr float log1p(float a); 
# 11465
constexpr float acosh(float a); 
# 11466
constexpr float asinh(float a); 
# 11467
constexpr float atanh(float a); 
# 11468
constexpr float hypot(float a, float b); 
# 11469
constexpr float cbrt(float a); 
# 11470
constexpr float erf(float a); 
# 11471
constexpr float erfc(float a); 
# 11472
constexpr float lgamma(float a); 
# 11473
constexpr float tgamma(float a); 
# 11474
constexpr float copysign(float a, float b); 
# 11475
constexpr float nextafter(float a, float b); 
# 11476
constexpr float remainder(float a, float b); 
# 11477
inline float remquo(float a, float b, int * quo); 
# 11478
constexpr float round(float a); 
# 11479
constexpr long lround(float a); 
# 11480
constexpr long long llround(float a); 
# 11481
constexpr float trunc(float a); 
# 11482
constexpr float rint(float a); 
# 11483
constexpr long lrint(float a); 
# 11484
constexpr long long llrint(float a); 
# 11485
constexpr float nearbyint(float a); 
# 11486
constexpr float fdim(float a, float b); 
# 11487
constexpr float fma(float a, float b, float c); 
# 11488
constexpr float fmax(float a, float b); 
# 11489
constexpr float fmin(float a, float b); 
# 11490
}
# 11595 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float exp10(const float a); 
# 11597
static inline float rsqrt(const float a); 
# 11599
static inline float rcbrt(const float a); 
# 11601
static inline float sinpi(const float a); 
# 11603
static inline float cospi(const float a); 
# 11605
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
# 11607
static inline void sincos(const float a, float *const sptr, float *const cptr); 
# 11609
static inline float j0(const float a); 
# 11611
static inline float j1(const float a); 
# 11613
static inline float jn(const int n, const float a); 
# 11615
static inline float y0(const float a); 
# 11617
static inline float y1(const float a); 
# 11619
static inline float yn(const int n, const float a); 
# 11621
__attribute__((unused)) static inline float cyl_bessel_i0(const float a); 
# 11623
__attribute__((unused)) static inline float cyl_bessel_i1(const float a); 
# 11625
static inline float erfinv(const float a); 
# 11627
static inline float erfcinv(const float a); 
# 11629
static inline float normcdfinv(const float a); 
# 11631
static inline float normcdf(const float a); 
# 11633
static inline float erfcx(const float a); 
# 11635
static inline double copysign(const double a, const float b); 
# 11637
static inline double copysign(const float a, const double b); 
# 11645
static inline unsigned min(const unsigned a, const unsigned b); 
# 11653
static inline unsigned min(const int a, const unsigned b); 
# 11661
static inline unsigned min(const unsigned a, const int b); 
# 11669
static inline long min(const long a, const long b); 
# 11677
static inline unsigned long min(const unsigned long a, const unsigned long b); 
# 11685
static inline unsigned long min(const long a, const unsigned long b); 
# 11693
static inline unsigned long min(const unsigned long a, const long b); 
# 11701
static inline long long min(const long long a, const long long b); 
# 11709
static inline unsigned long long min(const unsigned long long a, const unsigned long long b); 
# 11717
static inline unsigned long long min(const long long a, const unsigned long long b); 
# 11725
static inline unsigned long long min(const unsigned long long a, const long long b); 
# 11736
static inline float min(const float a, const float b); 
# 11747
static inline double min(const double a, const double b); 
# 11757
static inline double min(const float a, const double b); 
# 11767
static inline double min(const double a, const float b); 
# 11775
static inline unsigned max(const unsigned a, const unsigned b); 
# 11783
static inline unsigned max(const int a, const unsigned b); 
# 11791
static inline unsigned max(const unsigned a, const int b); 
# 11799
static inline long max(const long a, const long b); 
# 11807
static inline unsigned long max(const unsigned long a, const unsigned long b); 
# 11815
static inline unsigned long max(const long a, const unsigned long b); 
# 11823
static inline unsigned long max(const unsigned long a, const long b); 
# 11831
static inline long long max(const long long a, const long long b); 
# 11839
static inline unsigned long long max(const unsigned long long a, const unsigned long long b); 
# 11847
static inline unsigned long long max(const long long a, const unsigned long long b); 
# 11855
static inline unsigned long long max(const unsigned long long a, const long long b); 
# 11866
static inline float max(const float a, const float b); 
# 11877
static inline double max(const double a, const double b); 
# 11887
static inline double max(const float a, const double b); 
# 11897
static inline double max(const double a, const float b); 
# 11909
extern "C" {
# 11910
__attribute__((unused)) inline void *__nv_aligned_device_malloc(size_t size, size_t align) 
# 11911
{int volatile ___ = 1;(void)size;(void)align;
# 11914
::exit(___);}
#if 0
# 11911
{ 
# 11912
__attribute__((unused)) void *__nv_aligned_device_malloc_impl(size_t, size_t); 
# 11913
return __nv_aligned_device_malloc_impl(size, align); 
# 11914
} 
#endif
# 11915 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 758 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float exp10(const float a) 
# 759
{ 
# 760
return exp10f(a); 
# 761
} 
# 763
static inline float rsqrt(const float a) 
# 764
{ 
# 765
return rsqrtf(a); 
# 766
} 
# 768
static inline float rcbrt(const float a) 
# 769
{ 
# 770
return rcbrtf(a); 
# 771
} 
# 773
static inline float sinpi(const float a) 
# 774
{ 
# 775
return sinpif(a); 
# 776
} 
# 778
static inline float cospi(const float a) 
# 779
{ 
# 780
return cospif(a); 
# 781
} 
# 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
# 784
{ 
# 785
sincospif(a, sptr, cptr); 
# 786
} 
# 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
# 789
{ 
# 790
sincosf(a, sptr, cptr); 
# 791
} 
# 793
static inline float j0(const float a) 
# 794
{ 
# 795
return j0f(a); 
# 796
} 
# 798
static inline float j1(const float a) 
# 799
{ 
# 800
return j1f(a); 
# 801
} 
# 803
static inline float jn(const int n, const float a) 
# 804
{ 
# 805
return jnf(n, a); 
# 806
} 
# 808
static inline float y0(const float a) 
# 809
{ 
# 810
return y0f(a); 
# 811
} 
# 813
static inline float y1(const float a) 
# 814
{ 
# 815
return y1f(a); 
# 816
} 
# 818
static inline float yn(const int n, const float a) 
# 819
{ 
# 820
return ynf(n, a); 
# 821
} 
# 823
__attribute__((unused)) static inline float cyl_bessel_i0(const float a) 
# 824
{int volatile ___ = 1;(void)a;
# 826
::exit(___);}
#if 0
# 824
{ 
# 825
return cyl_bessel_i0f(a); 
# 826
} 
#endif
# 828 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute__((unused)) static inline float cyl_bessel_i1(const float a) 
# 829
{int volatile ___ = 1;(void)a;
# 831
::exit(___);}
#if 0
# 829
{ 
# 830
return cyl_bessel_i1f(a); 
# 831
} 
#endif
# 833 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float erfinv(const float a) 
# 834
{ 
# 835
return erfinvf(a); 
# 836
} 
# 838
static inline float erfcinv(const float a) 
# 839
{ 
# 840
return erfcinvf(a); 
# 841
} 
# 843
static inline float normcdfinv(const float a) 
# 844
{ 
# 845
return normcdfinvf(a); 
# 846
} 
# 848
static inline float normcdf(const float a) 
# 849
{ 
# 850
return normcdff(a); 
# 851
} 
# 853
static inline float erfcx(const float a) 
# 854
{ 
# 855
return erfcxf(a); 
# 856
} 
# 858
static inline double copysign(const double a, const float b) 
# 859
{ 
# 860
return copysign(a, static_cast< double>(b)); 
# 861
} 
# 863
static inline double copysign(const float a, const double b) 
# 864
{ 
# 865
return copysign(static_cast< double>(a), b); 
# 866
} 
# 868
static inline unsigned min(const unsigned a, const unsigned b) 
# 869
{ 
# 870
return umin(a, b); 
# 871
} 
# 873
static inline unsigned min(const int a, const unsigned b) 
# 874
{ 
# 875
return umin(static_cast< unsigned>(a), b); 
# 876
} 
# 878
static inline unsigned min(const unsigned a, const int b) 
# 879
{ 
# 880
return umin(a, static_cast< unsigned>(b)); 
# 881
} 
# 883
static inline long min(const long a, const long b) 
# 884
{ 
# 885
long retval; 
# 892
if (sizeof(long) == sizeof(int)) { 
# 896
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
# 897
} else { 
# 898
retval = (static_cast< long>(llmin(static_cast< long long>(a), static_cast< long long>(b)))); 
# 899
}  
# 900
return retval; 
# 901
} 
# 903
static inline unsigned long min(const unsigned long a, const unsigned long b) 
# 904
{ 
# 905
unsigned long retval; 
# 910
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 914
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 915
} else { 
# 916
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 917
}  
# 918
return retval; 
# 919
} 
# 921
static inline unsigned long min(const long a, const unsigned long b) 
# 922
{ 
# 923
unsigned long retval; 
# 928
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 932
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 933
} else { 
# 934
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 935
}  
# 936
return retval; 
# 937
} 
# 939
static inline unsigned long min(const unsigned long a, const long b) 
# 940
{ 
# 941
unsigned long retval; 
# 946
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 950
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 951
} else { 
# 952
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 953
}  
# 954
return retval; 
# 955
} 
# 957
static inline long long min(const long long a, const long long b) 
# 958
{ 
# 959
return llmin(a, b); 
# 960
} 
# 962
static inline unsigned long long min(const unsigned long long a, const unsigned long long b) 
# 963
{ 
# 964
return ullmin(a, b); 
# 965
} 
# 967
static inline unsigned long long min(const long long a, const unsigned long long b) 
# 968
{ 
# 969
return ullmin(static_cast< unsigned long long>(a), b); 
# 970
} 
# 972
static inline unsigned long long min(const unsigned long long a, const long long b) 
# 973
{ 
# 974
return ullmin(a, static_cast< unsigned long long>(b)); 
# 975
} 
# 977
static inline float min(const float a, const float b) 
# 978
{ 
# 979
return fminf(a, b); 
# 980
} 
# 982
static inline double min(const double a, const double b) 
# 983
{ 
# 984
return fmin(a, b); 
# 985
} 
# 987
static inline double min(const float a, const double b) 
# 988
{ 
# 989
return fmin(static_cast< double>(a), b); 
# 990
} 
# 992
static inline double min(const double a, const float b) 
# 993
{ 
# 994
return fmin(a, static_cast< double>(b)); 
# 995
} 
# 997
static inline unsigned max(const unsigned a, const unsigned b) 
# 998
{ 
# 999
return umax(a, b); 
# 1000
} 
# 1002
static inline unsigned max(const int a, const unsigned b) 
# 1003
{ 
# 1004
return umax(static_cast< unsigned>(a), b); 
# 1005
} 
# 1007
static inline unsigned max(const unsigned a, const int b) 
# 1008
{ 
# 1009
return umax(a, static_cast< unsigned>(b)); 
# 1010
} 
# 1012
static inline long max(const long a, const long b) 
# 1013
{ 
# 1014
long retval; 
# 1020
if (sizeof(long) == sizeof(int)) { 
# 1024
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
# 1025
} else { 
# 1026
retval = (static_cast< long>(llmax(static_cast< long long>(a), static_cast< long long>(b)))); 
# 1027
}  
# 1028
return retval; 
# 1029
} 
# 1031
static inline unsigned long max(const unsigned long a, const unsigned long b) 
# 1032
{ 
# 1033
unsigned long retval; 
# 1038
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1042
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1043
} else { 
# 1044
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1045
}  
# 1046
return retval; 
# 1047
} 
# 1049
static inline unsigned long max(const long a, const unsigned long b) 
# 1050
{ 
# 1051
unsigned long retval; 
# 1056
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1060
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1061
} else { 
# 1062
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1063
}  
# 1064
return retval; 
# 1065
} 
# 1067
static inline unsigned long max(const unsigned long a, const long b) 
# 1068
{ 
# 1069
unsigned long retval; 
# 1074
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1078
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1079
} else { 
# 1080
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1081
}  
# 1082
return retval; 
# 1083
} 
# 1085
static inline long long max(const long long a, const long long b) 
# 1086
{ 
# 1087
return llmax(a, b); 
# 1088
} 
# 1090
static inline unsigned long long max(const unsigned long long a, const unsigned long long b) 
# 1091
{ 
# 1092
return ullmax(a, b); 
# 1093
} 
# 1095
static inline unsigned long long max(const long long a, const unsigned long long b) 
# 1096
{ 
# 1097
return ullmax(static_cast< unsigned long long>(a), b); 
# 1098
} 
# 1100
static inline unsigned long long max(const unsigned long long a, const long long b) 
# 1101
{ 
# 1102
return ullmax(a, static_cast< unsigned long long>(b)); 
# 1103
} 
# 1105
static inline float max(const float a, const float b) 
# 1106
{ 
# 1107
return fmaxf(a, b); 
# 1108
} 
# 1110
static inline double max(const double a, const double b) 
# 1111
{ 
# 1112
return fmax(a, b); 
# 1113
} 
# 1115
static inline double max(const float a, const double b) 
# 1116
{ 
# 1117
return fmax(static_cast< double>(a), b); 
# 1118
} 
# 1120
static inline double max(const double a, const float b) 
# 1121
{ 
# 1122
return fmax(a, static_cast< double>(b)); 
# 1123
} 
# 1135 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
inline int min(const int a, const int b) 
# 1136
{ 
# 1137
return (a < b) ? a : b; 
# 1138
} 
# 1140
inline unsigned umin(const unsigned a, const unsigned b) 
# 1141
{ 
# 1142
return (a < b) ? a : b; 
# 1143
} 
# 1145
inline long long llmin(const long long a, const long long b) 
# 1146
{ 
# 1147
return (a < b) ? a : b; 
# 1148
} 
# 1150
inline unsigned long long ullmin(const unsigned long long a, const unsigned long long 
# 1151
b) 
# 1152
{ 
# 1153
return (a < b) ? a : b; 
# 1154
} 
# 1156
inline int max(const int a, const int b) 
# 1157
{ 
# 1158
return (a > b) ? a : b; 
# 1159
} 
# 1161
inline unsigned umax(const unsigned a, const unsigned b) 
# 1162
{ 
# 1163
return (a > b) ? a : b; 
# 1164
} 
# 1166
inline long long llmax(const long long a, const long long b) 
# 1167
{ 
# 1168
return (a > b) ? a : b; 
# 1169
} 
# 1171
inline unsigned long long ullmax(const unsigned long long a, const unsigned long long 
# 1172
b) 
# 1173
{ 
# 1174
return (a > b) ? a : b; 
# 1175
} 
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" {
# 3215
static inline int __vimax_s32_relu(const int a, const int b); 
# 3227
static inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b); 
# 3236
static inline int __vimin_s32_relu(const int a, const int b); 
# 3248
static inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b); 
# 3257
static inline int __vimax3_s32(const int a, const int b, const int c); 
# 3269
static inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3278
static inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3290
static inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3299
static inline int __vimin3_s32(const int a, const int b, const int c); 
# 3311
static inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3320
static inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3332
static inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3341
static inline int __vimax3_s32_relu(const int a, const int b, const int c); 
# 3353
static inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3362
static inline int __vimin3_s32_relu(const int a, const int b, const int c); 
# 3374
static inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3383
static inline int __viaddmax_s32(const int a, const int b, const int c); 
# 3395
static inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3404
static inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3416
static inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3425
static inline int __viaddmin_s32(const int a, const int b, const int c); 
# 3437
static inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3446
static inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3458
static inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3468
static inline int __viaddmax_s32_relu(const int a, const int b, const int c); 
# 3480
static inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3490
static inline int __viaddmin_s32_relu(const int a, const int b, const int c); 
# 3502
static inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3511
static inline int __vibmax_s32(const int a, const int b, bool *const pred); 
# 3520
static inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred); 
# 3529
static inline int __vibmin_s32(const int a, const int b, bool *const pred); 
# 3538
static inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred); 
# 3552
static inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3566
static inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3580
static inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3594
static inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3601
}
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
static inline int __vimax_s32_relu(const int a, const int b) { 
# 115
int ans = max(a, b); 
# 117
return (ans > 0) ? ans : 0; 
# 119
} 
# 121
static inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b) { 
# 122
unsigned res; 
# 130
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 131
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 133
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 134
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 137
short aS_lo = *((short *)(&aU_lo)); 
# 138
short aS_hi = *((short *)(&aU_hi)); 
# 140
short bS_lo = *((short *)(&bU_lo)); 
# 141
short bS_hi = *((short *)(&bU_hi)); 
# 144
short ansS_lo = (short)max(aS_lo, bS_lo); 
# 145
short ansS_hi = (short)max(aS_hi, bS_hi); 
# 148
if (ansS_lo < 0) { ansS_lo = (0); }  
# 149
if (ansS_hi < 0) { ansS_hi = (0); }  
# 152
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 153
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 156
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 159
return res; 
# 160
} 
# 162
static inline int __vimin_s32_relu(const int a, const int b) { 
# 169
int ans = min(a, b); 
# 171
return (ans > 0) ? ans : 0; 
# 173
} 
# 175
static inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b) { 
# 176
unsigned res; 
# 184
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 185
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 187
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 188
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 191
short aS_lo = *((short *)(&aU_lo)); 
# 192
short aS_hi = *((short *)(&aU_hi)); 
# 194
short bS_lo = *((short *)(&bU_lo)); 
# 195
short bS_hi = *((short *)(&bU_hi)); 
# 198
short ansS_lo = (short)min(aS_lo, bS_lo); 
# 199
short ansS_hi = (short)min(aS_hi, bS_hi); 
# 202
if (ansS_lo < 0) { ansS_lo = (0); }  
# 203
if (ansS_hi < 0) { ansS_hi = (0); }  
# 206
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 207
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 210
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 213
return res; 
# 214
} 
# 216
static inline int __vimax3_s32(const int a, const int b, const int c) { 
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return max(max(a, b), c); 
# 228
} 
# 230
static inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 231
unsigned res; 
# 243 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 244
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 246
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 247
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 249
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 250
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 253
short aS_lo = *((short *)(&aU_lo)); 
# 254
short aS_hi = *((short *)(&aU_hi)); 
# 256
short bS_lo = *((short *)(&bU_lo)); 
# 257
short bS_hi = *((short *)(&bU_hi)); 
# 259
short cS_lo = *((short *)(&cU_lo)); 
# 260
short cS_hi = *((short *)(&cU_hi)); 
# 263
short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo); 
# 264
short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi); 
# 267
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 268
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 271
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 273
return res; 
# 274
} 
# 276
static inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 286 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return max(max(a, b), c); 
# 288
} 
# 290
static inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 291
unsigned res; 
# 302 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 303
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 305
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 306
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 308
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 309
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 312
unsigned short ansU_lo = (unsigned short)max(max(aU_lo, bU_lo), cU_lo); 
# 313
unsigned short ansU_hi = (unsigned short)max(max(aU_hi, bU_hi), cU_hi); 
# 316
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 319
return res; 
# 320
} 
# 322
static inline int __vimin3_s32(const int a, const int b, const int c) { 
# 332 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return min(min(a, b), c); 
# 334
} 
# 336
static inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 337
unsigned res; 
# 348 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 349
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 351
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 352
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 354
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 355
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 358
short aS_lo = *((short *)(&aU_lo)); 
# 359
short aS_hi = *((short *)(&aU_hi)); 
# 361
short bS_lo = *((short *)(&bU_lo)); 
# 362
short bS_hi = *((short *)(&bU_hi)); 
# 364
short cS_lo = *((short *)(&cU_lo)); 
# 365
short cS_hi = *((short *)(&cU_hi)); 
# 368
short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo); 
# 369
short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi); 
# 372
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 373
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 376
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 379
return res; 
# 380
} 
# 382
static inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return min(min(a, b), c); 
# 394
} 
# 396
static inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 397
unsigned res; 
# 408 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 409
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 411
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 412
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 414
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 415
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 418
unsigned short ansU_lo = (unsigned short)min(min(aU_lo, bU_lo), cU_lo); 
# 419
unsigned short ansU_hi = (unsigned short)min(min(aU_hi, bU_hi), cU_hi); 
# 422
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 425
return res; 
# 426
} 
# 428
static inline int __vimax3_s32_relu(const int a, const int b, const int c) { 
# 438 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = max(max(a, b), c); 
# 440
return (ans > 0) ? ans : 0; 
# 442
} 
# 444
static inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 445
unsigned res; 
# 456 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 457
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 459
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 460
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 462
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 463
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 466
short aS_lo = *((short *)(&aU_lo)); 
# 467
short aS_hi = *((short *)(&aU_hi)); 
# 469
short bS_lo = *((short *)(&bU_lo)); 
# 470
short bS_hi = *((short *)(&bU_hi)); 
# 472
short cS_lo = *((short *)(&cU_lo)); 
# 473
short cS_hi = *((short *)(&cU_hi)); 
# 476
short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo); 
# 477
short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi); 
# 480
if (ansS_lo < 0) { ansS_lo = (0); }  
# 481
if (ansS_hi < 0) { ansS_hi = (0); }  
# 484
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 485
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 488
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 491
return res; 
# 492
} 
# 494
static inline int __vimin3_s32_relu(const int a, const int b, const int c) { 
# 504 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = min(min(a, b), c); 
# 506
return (ans > 0) ? ans : 0; 
# 508
} 
# 510
static inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 511
unsigned res; 
# 522 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 523
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 525
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 526
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 528
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 529
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 532
short aS_lo = *((short *)(&aU_lo)); 
# 533
short aS_hi = *((short *)(&aU_hi)); 
# 535
short bS_lo = *((short *)(&bU_lo)); 
# 536
short bS_hi = *((short *)(&bU_hi)); 
# 538
short cS_lo = *((short *)(&cU_lo)); 
# 539
short cS_hi = *((short *)(&cU_hi)); 
# 542
short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo); 
# 543
short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi); 
# 546
if (ansS_lo < 0) { ansS_lo = (0); }  
# 547
if (ansS_hi < 0) { ansS_hi = (0); }  
# 550
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 551
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 554
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 557
return res; 
# 558
} 
# 560
static inline int __viaddmax_s32(const int a, const int b, const int c) { 
# 570 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return max(a + b, c); 
# 572
} 
# 574
static inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 575
unsigned res; 
# 586 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 587
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 589
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 590
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 592
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 593
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 596
short aS_lo = *((short *)(&aU_lo)); 
# 597
short aS_hi = *((short *)(&aU_hi)); 
# 599
short bS_lo = *((short *)(&bU_lo)); 
# 600
short bS_hi = *((short *)(&bU_hi)); 
# 602
short cS_lo = *((short *)(&cU_lo)); 
# 603
short cS_hi = *((short *)(&cU_hi)); 
# 606
short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo); 
# 607
short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi); 
# 610
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 611
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 614
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 617
return res; 
# 618
} 
# 620
static inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 630 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return max(a + b, c); 
# 632
} 
# 634
static inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 635
unsigned res; 
# 646 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 647
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 649
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 650
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 652
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 653
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 656
unsigned short ansU_lo = (unsigned short)max((unsigned short)(aU_lo + bU_lo), cU_lo); 
# 657
unsigned short ansU_hi = (unsigned short)max((unsigned short)(aU_hi + bU_hi), cU_hi); 
# 660
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 663
return res; 
# 664
} 
# 666
static inline int __viaddmin_s32(const int a, const int b, const int c) { 
# 676 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return min(a + b, c); 
# 678
} 
# 680
static inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 681
unsigned res; 
# 692 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 693
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 695
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 696
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 698
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 699
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 702
short aS_lo = *((short *)(&aU_lo)); 
# 703
short aS_hi = *((short *)(&aU_hi)); 
# 705
short bS_lo = *((short *)(&bU_lo)); 
# 706
short bS_hi = *((short *)(&bU_hi)); 
# 708
short cS_lo = *((short *)(&cU_lo)); 
# 709
short cS_hi = *((short *)(&cU_hi)); 
# 712
short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo); 
# 713
short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi); 
# 716
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 717
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 720
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 723
return res; 
# 724
} 
# 726
static inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 736 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
return min(a + b, c); 
# 738
} 
# 740
static inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 741
unsigned res; 
# 752 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 753
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 755
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 756
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 758
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 759
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 762
unsigned short ansU_lo = (unsigned short)min((unsigned short)(aU_lo + bU_lo), cU_lo); 
# 763
unsigned short ansU_hi = (unsigned short)min((unsigned short)(aU_hi + bU_hi), cU_hi); 
# 766
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 769
return res; 
# 770
} 
# 772
static inline int __viaddmax_s32_relu(const int a, const int b, const int c) { 
# 782 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = max(a + b, c); 
# 784
return (ans > 0) ? ans : 0; 
# 786
} 
# 788
static inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 789
unsigned res; 
# 800 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 801
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 803
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 804
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 806
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 807
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 810
short aS_lo = *((short *)(&aU_lo)); 
# 811
short aS_hi = *((short *)(&aU_hi)); 
# 813
short bS_lo = *((short *)(&bU_lo)); 
# 814
short bS_hi = *((short *)(&bU_hi)); 
# 816
short cS_lo = *((short *)(&cU_lo)); 
# 817
short cS_hi = *((short *)(&cU_hi)); 
# 820
short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo); 
# 821
short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi); 
# 823
if (ansS_lo < 0) { ansS_lo = (0); }  
# 824
if (ansS_hi < 0) { ansS_hi = (0); }  
# 827
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 828
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 831
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 834
return res; 
# 835
} 
# 837
static inline int __viaddmin_s32_relu(const int a, const int b, const int c) { 
# 847 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = min(a + b, c); 
# 849
return (ans > 0) ? ans : 0; 
# 851
} 
# 853
static inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 854
unsigned res; 
# 865 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 866
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 868
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 869
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 871
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 872
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 875
short aS_lo = *((short *)(&aU_lo)); 
# 876
short aS_hi = *((short *)(&aU_hi)); 
# 878
short bS_lo = *((short *)(&bU_lo)); 
# 879
short bS_hi = *((short *)(&bU_hi)); 
# 881
short cS_lo = *((short *)(&cU_lo)); 
# 882
short cS_hi = *((short *)(&cU_hi)); 
# 885
short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo); 
# 886
short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi); 
# 888
if (ansS_lo < 0) { ansS_lo = (0); }  
# 889
if (ansS_hi < 0) { ansS_hi = (0); }  
# 892
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 893
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 896
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
# 899
return res; 
# 900
} 
# 904
static inline int __vibmax_s32(const int a, const int b, bool *const pred) { 
# 918 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = max(a, b); 
# 920
(*pred) = (a >= b); 
# 921
return ans; 
# 923
} 
# 925
static inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred) { 
# 939 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned ans = max(a, b); 
# 941
(*pred) = (a >= b); 
# 942
return ans; 
# 944
} 
# 947
static inline int __vibmin_s32(const int a, const int b, bool *const pred) { 
# 961 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
int ans = min(a, b); 
# 963
(*pred) = (a <= b); 
# 964
return ans; 
# 966
} 
# 969
static inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred) { 
# 983 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned ans = min(a, b); 
# 985
(*pred) = (a <= b); 
# 986
return ans; 
# 988
} 
# 990
static inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1012 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1013
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1015
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1016
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1019
short aS_lo = *((short *)(&aU_lo)); 
# 1020
short aS_hi = *((short *)(&aU_hi)); 
# 1022
short bS_lo = *((short *)(&bU_lo)); 
# 1023
short bS_hi = *((short *)(&bU_hi)); 
# 1026
short ansS_lo = (short)max(aS_lo, bS_lo); 
# 1027
short ansS_hi = (short)max(aS_hi, bS_hi); 
# 1029
(*pred_hi) = (aS_hi >= bS_hi); 
# 1030
(*pred_lo) = (aS_lo >= bS_lo); 
# 1033
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 1034
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 1037
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1039
return ans; 
# 1041
} 
# 1043
static inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1065 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1066
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1068
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1069
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1072
unsigned short ansU_lo = (unsigned short)max(aU_lo, bU_lo); 
# 1073
unsigned short ansU_hi = (unsigned short)max(aU_hi, bU_hi); 
# 1075
(*pred_hi) = (aU_hi >= bU_hi); 
# 1076
(*pred_lo) = (aU_lo >= bU_lo); 
# 1079
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1081
return ans; 
# 1083
} 
# 1085
static inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1108
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1110
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1111
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1114
short aS_lo = *((short *)(&aU_lo)); 
# 1115
short aS_hi = *((short *)(&aU_hi)); 
# 1117
short bS_lo = *((short *)(&bU_lo)); 
# 1118
short bS_hi = *((short *)(&bU_hi)); 
# 1121
short ansS_lo = (short)min(aS_lo, bS_lo); 
# 1122
short ansS_hi = (short)min(aS_hi, bS_hi); 
# 1124
(*pred_hi) = (aS_hi <= bS_hi); 
# 1125
(*pred_lo) = (aS_lo <= bS_lo); 
# 1128
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 1129
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 1132
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1134
return ans; 
# 1136
} 
# 1138
static inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1160 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1161
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1163
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1164
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1167
unsigned short ansU_lo = (unsigned short)min(aU_lo, bU_lo); 
# 1168
unsigned short ansU_hi = (unsigned short)min(aU_hi, bU_hi); 
# 1170
(*pred_hi) = (aU_hi <= bU_hi); 
# 1171
(*pred_lo) = (aU_lo <= bU_lo); 
# 1174
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1176
return ans; 
# 1178
} 
# 89 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 91 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 91
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 97
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 119
{ } 
#endif
# 121 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 121
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 123
{ } 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 125
{ } 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 127
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 129
{ } 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C" {
# 160
}
# 169
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 169
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 171
{ } 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 175
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 177
{ } 
#endif
# 90 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern "C" {
# 1142
}
# 1150
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1154
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1156
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1158
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1160
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1162
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1164
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1166
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1168
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1170
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1172
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1174
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1176
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 88 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 88
{ } 
#endif
# 89 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 91 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 91
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 97
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 107
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 93
{ } 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 96
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 99
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 105
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 123
{ } 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 129
{ } 
#endif
# 132 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 135
{ } 
#endif
# 138 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 141
{ } 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 147
{ } 
#endif
# 150 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 150
{ } 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 153
{ } 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 156
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 159
{ } 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 162
{ } 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 165
{ } 
#endif
# 168 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 168
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 171
{ } 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 174
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 177
{ } 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 180
{ } 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 183
{ } 
#endif
# 186 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 186
{ } 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 192 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 192
{ } 
#endif
# 195 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 195
{ } 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 198
{ } 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 201
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 204
{ } 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 207
{ } 
#endif
# 210 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 210
{ } 
#endif
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 213
{ } 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 216
{ } 
#endif
# 219 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 219
{ } 
#endif
# 222 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 222
{ } 
#endif
# 225 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 225
{ } 
#endif
# 228 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 229
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 229
{ } 
#endif
# 232 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 233
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 233
{ } 
#endif
# 236 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 237
compare, unsigned long long 
# 238
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 238
{ } 
#endif
# 241 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 242
compare, unsigned long long 
# 243
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 243
{ } 
#endif
# 246 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 246
{ } 
#endif
# 249 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 249
{ } 
#endif
# 252 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 252
{ } 
#endif
# 255 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 255
{ } 
#endif
# 258 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 258
{ } 
#endif
# 261 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 261
{ } 
#endif
# 264 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 264
{ } 
#endif
# 267 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 267
{ } 
#endif
# 270 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 270
{ } 
#endif
# 273 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 273
{ } 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 276
{ } 
#endif
# 279 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 279
{ } 
#endif
# 282 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 282
{ } 
#endif
# 285 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 285
{ } 
#endif
# 288 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 288
{ } 
#endif
# 291 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 291
{ } 
#endif
# 294 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 294
{ } 
#endif
# 297 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 297
{ } 
#endif
# 300 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 300
{ } 
#endif
# 303 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 303
{ } 
#endif
# 306 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 306
{ } 
#endif
# 309 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 309
{ } 
#endif
# 312 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 312
{ } 
#endif
# 315 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 315
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern "C" {
# 1508
}
# 1515
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1515
{ } 
#endif
# 1517 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1517
{ } 
#endif
# 1519 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1519
{ } 
#endif
# 1521 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1521
{ } 
#endif
# 1526 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1526
{ } 
#endif
# 1527 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1527
{ } 
#endif
# 1528 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1528
{ } 
#endif
# 1529 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1529
{ } 
#endif
# 1531 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGridConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1531
{ } 
#endif
# 1533 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1533
{ } 
#endif
# 1534 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1534
{ } 
#endif
# 1535 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1535
{ } 
#endif
# 1536 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1536
{ } 
#endif
# 1538 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_grid_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1538
{ } 
#endif
# 1541 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1541
{ } 
#endif
# 1542 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1542
{ } 
#endif
# 1543 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1543
{ } 
#endif
# 1544 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1544
{ } 
#endif
# 1546 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_grid_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1546
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 124
{ } 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 131
{ } 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 154 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 160
{ } 
#endif
# 161 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 169 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 205
{ } 
#endif
# 208 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 209
{ } 
#endif
# 210 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 210
{ } 
#endif
# 211 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 215
{ } 
#endif
# 91 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 94 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 121
{ } 
#endif
# 122 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 122
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 128 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 167 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 192 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 194
{ } 
#endif
# 195 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 200 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 203 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 210 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 210
{ } 
#endif
# 211 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 222 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 222
{ } 
#endif
# 223 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 228 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 228
{ } 
#endif
# 229 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 229
{ } 
#endif
# 230 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 230
{ } 
#endif
# 231 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 231
{ } 
#endif
# 232 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 232
{ } 
#endif
# 236 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 237
{ } 
#endif
# 239 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 239
{ } 
#endif
# 240 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 241
{ } 
#endif
# 242 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 242
{ } 
#endif
# 243 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldlu(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 243
{ } 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 244
{ } 
#endif
# 245 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 245
{ } 
#endif
# 246 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 246
{ } 
#endif
# 247 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 247
{ } 
#endif
# 248 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 248
{ } 
#endif
# 249 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 249
{ } 
#endif
# 250 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 250
{ } 
#endif
# 252 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 252
{ } 
#endif
# 253 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 253
{ } 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 254
{ } 
#endif
# 255 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldlu(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 255
{ } 
#endif
# 256 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 256
{ } 
#endif
# 257 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 257
{ } 
#endif
# 258 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 258
{ } 
#endif
# 259 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 259
{ } 
#endif
# 260 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 260
{ } 
#endif
# 261 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 261
{ } 
#endif
# 262 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 262
{ } 
#endif
# 264 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 264
{ } 
#endif
# 265 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 265
{ } 
#endif
# 266 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 266
{ } 
#endif
# 267 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 267
{ } 
#endif
# 268 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 268
{ } 
#endif
# 272 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 272
{ } 
#endif
# 273 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 273
{ } 
#endif
# 275 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 275
{ } 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 276
{ } 
#endif
# 277 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 277
{ } 
#endif
# 278 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 278
{ } 
#endif
# 279 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcv(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 279
{ } 
#endif
# 280 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 280
{ } 
#endif
# 281 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 281
{ } 
#endif
# 282 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 282
{ } 
#endif
# 283 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 283
{ } 
#endif
# 284 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 284
{ } 
#endif
# 285 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 285
{ } 
#endif
# 286 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 286
{ } 
#endif
# 288 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 288
{ } 
#endif
# 289 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 289
{ } 
#endif
# 290 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 290
{ } 
#endif
# 291 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcv(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 291
{ } 
#endif
# 292 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 292
{ } 
#endif
# 293 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 293
{ } 
#endif
# 294 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 294
{ } 
#endif
# 295 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 295
{ } 
#endif
# 296 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 296
{ } 
#endif
# 297 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 297
{ } 
#endif
# 298 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 298
{ } 
#endif
# 300 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 300
{ } 
#endif
# 301 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 301
{ } 
#endif
# 302 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 302
{ } 
#endif
# 303 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 303
{ } 
#endif
# 304 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 304
{ } 
#endif
# 308 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 308
{ } 
#endif
# 309 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 309
{ } 
#endif
# 311 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 311
{ } 
#endif
# 312 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 312
{ } 
#endif
# 313 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 313
{ } 
#endif
# 314 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 314
{ } 
#endif
# 315 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 315
{ } 
#endif
# 316 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 316
{ } 
#endif
# 317 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 317
{ } 
#endif
# 318 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 318
{ } 
#endif
# 319 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 319
{ } 
#endif
# 320 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 320
{ } 
#endif
# 321 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 321
{ } 
#endif
# 322 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 322
{ } 
#endif
# 324 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 324
{ } 
#endif
# 325 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 325
{ } 
#endif
# 326 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 326
{ } 
#endif
# 327 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 327
{ } 
#endif
# 328 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 328
{ } 
#endif
# 329 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 329
{ } 
#endif
# 330 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 330
{ } 
#endif
# 331 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 331
{ } 
#endif
# 332 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 332
{ } 
#endif
# 333 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 333
{ } 
#endif
# 334 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 334
{ } 
#endif
# 336 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 336
{ } 
#endif
# 337 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 337
{ } 
#endif
# 338 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 338
{ } 
#endif
# 339 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 339
{ } 
#endif
# 340 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 340
{ } 
#endif
# 344 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 344
{ } 
#endif
# 345 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 345
{ } 
#endif
# 347 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 347
{ } 
#endif
# 348 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 348
{ } 
#endif
# 349 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 349
{ } 
#endif
# 350 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 350
{ } 
#endif
# 351 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 351
{ } 
#endif
# 352 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 352
{ } 
#endif
# 353 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 353
{ } 
#endif
# 354 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 354
{ } 
#endif
# 355 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 355
{ } 
#endif
# 356 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 356
{ } 
#endif
# 357 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 357
{ } 
#endif
# 358 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 358
{ } 
#endif
# 360 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 360
{ } 
#endif
# 361 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 361
{ } 
#endif
# 362 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 362
{ } 
#endif
# 363 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 363
{ } 
#endif
# 364 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 364
{ } 
#endif
# 365 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 365
{ } 
#endif
# 366 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 366
{ } 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 367
{ } 
#endif
# 368 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 368
{ } 
#endif
# 369 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 369
{ } 
#endif
# 370 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 370
{ } 
#endif
# 372 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 372
{ } 
#endif
# 373 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 373
{ } 
#endif
# 374 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 374
{ } 
#endif
# 375 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 375
{ } 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 376
{ } 
#endif
# 380 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 380
{ } 
#endif
# 381 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 381
{ } 
#endif
# 383 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 383
{ } 
#endif
# 384 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 384
{ } 
#endif
# 385 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 385
{ } 
#endif
# 386 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 386
{ } 
#endif
# 387 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 387
{ } 
#endif
# 388 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 388
{ } 
#endif
# 389 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 389
{ } 
#endif
# 390 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 390
{ } 
#endif
# 391 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 391
{ } 
#endif
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 392
{ } 
#endif
# 393 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 393
{ } 
#endif
# 394 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 394
{ } 
#endif
# 396 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 396
{ } 
#endif
# 397 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 397
{ } 
#endif
# 398 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 398
{ } 
#endif
# 399 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 399
{ } 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 400
{ } 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 401
{ } 
#endif
# 402 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 402
{ } 
#endif
# 403 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 403
{ } 
#endif
# 404 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 404
{ } 
#endif
# 405 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 405
{ } 
#endif
# 406 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 406
{ } 
#endif
# 408 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 408
{ } 
#endif
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 409
{ } 
#endif
# 410 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 410
{ } 
#endif
# 411 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 411
{ } 
#endif
# 412 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 412
{ } 
#endif
# 416 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 416
{ } 
#endif
# 417 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 417
{ } 
#endif
# 419 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 419
{ } 
#endif
# 420 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 420
{ } 
#endif
# 421 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 421
{ } 
#endif
# 422 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 422
{ } 
#endif
# 423 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 423
{ } 
#endif
# 424 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 424
{ } 
#endif
# 425 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 425
{ } 
#endif
# 426 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 426
{ } 
#endif
# 427 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 427
{ } 
#endif
# 428 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 428
{ } 
#endif
# 429 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 429
{ } 
#endif
# 430 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 430
{ } 
#endif
# 432 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 432
{ } 
#endif
# 433 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 433
{ } 
#endif
# 434 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 434
{ } 
#endif
# 435 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 435
{ } 
#endif
# 436 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 436
{ } 
#endif
# 437 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 437
{ } 
#endif
# 438 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 438
{ } 
#endif
# 439 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 439
{ } 
#endif
# 440 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 440
{ } 
#endif
# 441 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 441
{ } 
#endif
# 442 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 442
{ } 
#endif
# 444 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 444
{ } 
#endif
# 445 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 445
{ } 
#endif
# 446 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 446
{ } 
#endif
# 447 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 447
{ } 
#endif
# 448 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 448
{ } 
#endif
# 465 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 465
{ } 
#endif
# 477 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 477
{ } 
#endif
# 490 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 490
{ } 
#endif
# 502 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 502
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 102
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 113
{ } 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 125
{ } 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 136
{ } 
#endif
# 148 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 148
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 159
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 171
{ } 
#endif
# 182 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 182
{ } 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 197
{ } 
#endif
# 206 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 206
{ } 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 216
{ } 
#endif
# 225 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 225
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 107
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
extern "C" {
# 113
__attribute__((unused)) inline void *__nv_associate_access_property(const void *ptr, unsigned long long 
# 114
property) {int volatile ___ = 1;(void)ptr;(void)property;
# 118
::exit(___);}
#if 0
# 114
{ 
# 115
__attribute__((unused)) extern void *__nv_associate_access_property_impl(const void *, unsigned long long); 
# 117
return __nv_associate_access_property_impl(ptr, property); 
# 118
} 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_4(void *dst, const void *
# 121
src, unsigned 
# 122
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 127
::exit(___);}
#if 0
# 122
{ 
# 123
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_4_impl(void *, const void *, unsigned); 
# 126
__nv_memcpy_async_shared_global_4_impl(dst, src, src_size); 
# 127
} 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_8(void *dst, const void *
# 130
src, unsigned 
# 131
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 136
::exit(___);}
#if 0
# 131
{ 
# 132
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_8_impl(void *, const void *, unsigned); 
# 135
__nv_memcpy_async_shared_global_8_impl(dst, src, src_size); 
# 136
} 
#endif
# 138 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_16(void *dst, const void *
# 139
src, unsigned 
# 140
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 145
::exit(___);}
#if 0
# 140
{ 
# 141
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_16_impl(void *, const void *, unsigned); 
# 144
__nv_memcpy_async_shared_global_16_impl(dst, src, src_size); 
# 145
} 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
}
# 92 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __isCtaShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __isClusterShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline void *__cluster_map_shared_rank(const void *ptr, unsigned target_block_rank) {int volatile ___ = 1;(void)ptr;(void)target_block_rank;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __cluster_query_shared_rank(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline uint2 __cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask) {int volatile ___ = 1;(void)ptr;(void)cluster_cta_mask;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __clusterDimIsSpecified() {int volatile ___ = 1;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline dim3 __clusterDim() {int volatile ___ = 1;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline dim3 __clusterRelativeBlockIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline dim3 __clusterGridDimInClusters() {int volatile ___ = 1;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline dim3 __clusterIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __clusterRelativeBlockRank() {int volatile ___ = 1;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline unsigned __clusterSizeInBlocks() {int volatile ___ = 1;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline void __cluster_barrier_arrive() {int volatile ___ = 1;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline void __cluster_barrier_arrive_relaxed() {int volatile ___ = 1;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline void __cluster_barrier_wait() {int volatile ___ = 1;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline void __threadfence_cluster() {int volatile ___ = 1;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float2 atomicAdd(float2 *__address, float2 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float2 atomicAdd_block(float2 *__address, float2 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float2 atomicAdd_system(float2 *__address, float2 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float4 atomicAdd(float4 *__address, float4 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float4 atomicAdd_block(float4 *__address, float4 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
__attribute__((unused)) static inline float4 atomicAdd_system(float4 *__address, float4 val) {int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
extern "C" {
# 132
}
# 139
template< bool __b, class _T> 
# 140
struct __nv_atomic_enable_if { }; 
# 142
template< class _T> 
# 143
struct __nv_atomic_enable_if< true, _T>  { typedef _T __type; }; 
# 153
template< class _T> 
# 154
struct __nv_atomic_triv_cp_helper { 
# 161 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
static const bool __val = __is_trivially_copyable(_T); 
# 166
}; 
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 203
atomicCAS(_T *__address, _T __compare, _T __val) {int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;
# 210
::exit(___);}
#if 0
# 203
{ 
# 204
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 204
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 205
__u128AtomicCAS((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 209
return __u.__ret; 
# 210
} 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 214
atomicCAS_block(_T *__address, _T __compare, _T __val) {int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;
# 221
::exit(___);}
#if 0
# 214
{ 
# 215
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 215
{ } 
#endif
# 215 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 216
__u128AtomicCAS_block((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 220
return __u.__ret; 
# 221
} 
#endif
# 223 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 225
atomicCAS_system(_T *__address, _T __compare, _T __val) {int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;
# 232
::exit(___);}
#if 0
# 225
{ 
# 226
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 226
{ } 
#endif
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 227
__u128AtomicCAS_system((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 231
return __u.__ret; 
# 232
} 
#endif
# 234 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 236
atomicExch(_T *__address, _T __val) {int volatile ___ = 1;(void)__address;(void)__val;
# 242
::exit(___);}
#if 0
# 236
{ 
# 237
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 237
{ } 
#endif
# 237 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 238
__u128AtomicExch((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 241
return __u.__ret; 
# 242
} 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 246
atomicExch_block(_T *__address, _T __val) {int volatile ___ = 1;(void)__address;(void)__val;
# 252
::exit(___);}
#if 0
# 246
{ 
# 247
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 247
{ } 
#endif
# 247 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 248
__u128AtomicExch_block((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 251
return __u.__ret; 
# 252
} 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
template< class _T> __attribute__((unused)) static inline typename __nv_atomic_enable_if< (sizeof(_T) == (16)) && (__alignof__(_T) >= (16)) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
# 256
atomicExch_system(_T *__address, _T __val) {int volatile ___ = 1;(void)__address;(void)__val;
# 262
::exit(___);}
#if 0
# 256
{ 
# 257
union _U { _T __ret; _U() {int *volatile ___ = 0;::free(___);}
#if 0
# 257
{ } 
#endif
# 257 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h"
}; _U __u; 
# 258
__u128AtomicExch_system((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
# 261
return __u.__ret; 
# 262
} 
#endif
# 65 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 66
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 87
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 88
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 89
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 90
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 102
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 103
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 104
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 108
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 109
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 110
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 112
::exit(___);}
#if 0
# 110
{ 
# 111
__nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x); 
# 112
} 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 115
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 116
{int volatile ___ = 1;(void)texObject;(void)x;
# 120
::exit(___);}
#if 0
# 116
{ 
# 117
T ret; 
# 118
tex1Dfetch(&ret, texObject, x); 
# 119
return ret; 
# 120
} 
#endif
# 122 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 123
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 124
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 126
::exit(___);}
#if 0
# 124
{ 
# 125
__nv_tex_surf_handler("__itex1D", ptr, obj, x); 
# 126
} 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 130
tex1D(cudaTextureObject_t texObject, float x) 
# 131
{int volatile ___ = 1;(void)texObject;(void)x;
# 135
::exit(___);}
#if 0
# 131
{ 
# 132
T ret; 
# 133
tex1D(&ret, texObject, x); 
# 134
return ret; 
# 135
} 
#endif
# 138 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 139
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 140
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 142
::exit(___);}
#if 0
# 140
{ 
# 141
__nv_tex_surf_handler("__itex2D", ptr, obj, x, y); 
# 142
} 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 145
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 146
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 150
::exit(___);}
#if 0
# 146
{ 
# 147
T ret; 
# 148
tex2D(&ret, texObject, x, y); 
# 149
return ret; 
# 150
} 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 154
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, bool *
# 155
isResident) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;
# 160
::exit(___);}
#if 0
# 156
{ 
# 157
unsigned char res; 
# 158
__nv_tex_surf_handler("__itex2D_sparse", ptr, obj, x, y, &res); 
# 159
(*isResident) = (res != 0); 
# 160
} 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 163
tex2D(cudaTextureObject_t texObject, float x, float y, bool *isResident) 
# 164
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;
# 168
::exit(___);}
#if 0
# 164
{ 
# 165
T ret; 
# 166
tex2D(&ret, texObject, x, y, isResident); 
# 167
return ret; 
# 168
} 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 174
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 175
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 177
::exit(___);}
#if 0
# 175
{ 
# 176
__nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z); 
# 177
} 
#endif
# 179 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 180
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 181
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 185
::exit(___);}
#if 0
# 181
{ 
# 182
T ret; 
# 183
tex3D(&ret, texObject, x, y, z); 
# 184
return ret; 
# 185
} 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 189
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, bool *
# 190
isResident) 
# 191
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;
# 195
::exit(___);}
#if 0
# 191
{ 
# 192
unsigned char res; 
# 193
__nv_tex_surf_handler("__itex3D_sparse", ptr, obj, x, y, z, &res); 
# 194
(*isResident) = (res != 0); 
# 195
} 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 198
tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
# 199
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;
# 203
::exit(___);}
#if 0
# 199
{ 
# 200
T ret; 
# 201
tex3D(&ret, texObject, x, y, z, isResident); 
# 202
return ret; 
# 203
} 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 208
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 209
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 211
::exit(___);}
#if 0
# 209
{ 
# 210
__nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer); 
# 211
} 
#endif
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 214
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 215
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 219
::exit(___);}
#if 0
# 215
{ 
# 216
T ret; 
# 217
tex1DLayered(&ret, texObject, x, layer); 
# 218
return ret; 
# 219
} 
#endif
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 222
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 223
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 225
::exit(___);}
#if 0
# 223
{ 
# 224
__nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer); 
# 225
} 
#endif
# 227 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 228
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 229
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 233
::exit(___);}
#if 0
# 229
{ 
# 230
T ret; 
# 231
tex2DLayered(&ret, texObject, x, y, layer); 
# 232
return ret; 
# 233
} 
#endif
# 236 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 237
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
# 238
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;
# 242
::exit(___);}
#if 0
# 238
{ 
# 239
unsigned char res; 
# 240
__nv_tex_surf_handler("__itex2DLayered_sparse", ptr, obj, x, y, layer, &res); 
# 241
(*isResident) = (res != 0); 
# 242
} 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 245
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
# 246
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;
# 250
::exit(___);}
#if 0
# 246
{ 
# 247
T ret; 
# 248
tex2DLayered(&ret, texObject, x, y, layer, isResident); 
# 249
return ret; 
# 250
} 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 255
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 256
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 258
::exit(___);}
#if 0
# 256
{ 
# 257
__nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z); 
# 258
} 
#endif
# 261 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 262
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 263
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 267
::exit(___);}
#if 0
# 263
{ 
# 264
T ret; 
# 265
texCubemap(&ret, texObject, x, y, z); 
# 266
return ret; 
# 267
} 
#endif
# 270 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 271
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 272
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 274
::exit(___);}
#if 0
# 272
{ 
# 273
__nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer); 
# 274
} 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 277
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 278
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 282
::exit(___);}
#if 0
# 278
{ 
# 279
T ret; 
# 280
texCubemapLayered(&ret, texObject, x, y, z, layer); 
# 281
return ret; 
# 282
} 
#endif
# 284 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 285
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 286
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 288
::exit(___);}
#if 0
# 286
{ 
# 287
__nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp); 
# 288
} 
#endif
# 290 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 291
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 292
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 296
::exit(___);}
#if 0
# 292
{ 
# 293
T ret; 
# 294
tex2Dgather(&ret, to, x, y, comp); 
# 295
return ret; 
# 296
} 
#endif
# 299 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 300
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
# 301
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;
# 305
::exit(___);}
#if 0
# 301
{ 
# 302
unsigned char res; 
# 303
__nv_tex_surf_handler("__itex2Dgather_sparse", ptr, obj, x, y, comp, &res); 
# 304
(*isResident) = (res != 0); 
# 305
} 
#endif
# 307 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 308
tex2Dgather(cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
# 309
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;
# 313
::exit(___);}
#if 0
# 309
{ 
# 310
T ret; 
# 311
tex2Dgather(&ret, to, x, y, isResident, comp); 
# 312
return ret; 
# 313
} 
#endif
# 317 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 318
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 319
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 321
::exit(___);}
#if 0
# 319
{ 
# 320
__nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level); 
# 321
} 
#endif
# 323 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 324
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 325
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 329
::exit(___);}
#if 0
# 325
{ 
# 326
T ret; 
# 327
tex1DLod(&ret, texObject, x, level); 
# 328
return ret; 
# 329
} 
#endif
# 332 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 333
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 334
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 336
::exit(___);}
#if 0
# 334
{ 
# 335
__nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level); 
# 336
} 
#endif
# 338 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 339
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 340
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 344
::exit(___);}
#if 0
# 340
{ 
# 341
T ret; 
# 342
tex2DLod(&ret, texObject, x, y, level); 
# 343
return ret; 
# 344
} 
#endif
# 348 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 349
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
# 350
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;
# 354
::exit(___);}
#if 0
# 350
{ 
# 351
unsigned char res; 
# 352
__nv_tex_surf_handler("__itex2DLod_sparse", ptr, obj, x, y, level, &res); 
# 353
(*isResident) = (res != 0); 
# 354
} 
#endif
# 356 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 357
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
# 358
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;
# 362
::exit(___);}
#if 0
# 358
{ 
# 359
T ret; 
# 360
tex2DLod(&ret, texObject, x, y, level, isResident); 
# 361
return ret; 
# 362
} 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 368
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 369
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 371
::exit(___);}
#if 0
# 369
{ 
# 370
__nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level); 
# 371
} 
#endif
# 373 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 374
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 375
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 379
::exit(___);}
#if 0
# 375
{ 
# 376
T ret; 
# 377
tex3DLod(&ret, texObject, x, y, z, level); 
# 378
return ret; 
# 379
} 
#endif
# 382 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 383
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
# 384
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 388
::exit(___);}
#if 0
# 384
{ 
# 385
unsigned char res; 
# 386
__nv_tex_surf_handler("__itex3DLod_sparse", ptr, obj, x, y, z, level, &res); 
# 387
(*isResident) = (res != 0); 
# 388
} 
#endif
# 390 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 391
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
# 392
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 396
::exit(___);}
#if 0
# 392
{ 
# 393
T ret; 
# 394
tex3DLod(&ret, texObject, x, y, z, level, isResident); 
# 395
return ret; 
# 396
} 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 402
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 403
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 405
::exit(___);}
#if 0
# 403
{ 
# 404
__nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level); 
# 405
} 
#endif
# 407 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 408
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 409
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 413
::exit(___);}
#if 0
# 409
{ 
# 410
T ret; 
# 411
tex1DLayeredLod(&ret, texObject, x, layer, level); 
# 412
return ret; 
# 413
} 
#endif
# 416 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 417
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 418
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 420
::exit(___);}
#if 0
# 418
{ 
# 419
__nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level); 
# 420
} 
#endif
# 422 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 423
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 424
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 428
::exit(___);}
#if 0
# 424
{ 
# 425
T ret; 
# 426
tex2DLayeredLod(&ret, texObject, x, y, layer, level); 
# 427
return ret; 
# 428
} 
#endif
# 431 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 432
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
# 433
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 437
::exit(___);}
#if 0
# 433
{ 
# 434
unsigned char res; 
# 435
__nv_tex_surf_handler("__itex2DLayeredLod_sparse", ptr, obj, x, y, layer, level, &res); 
# 436
(*isResident) = (res != 0); 
# 437
} 
#endif
# 439 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 440
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
# 441
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 445
::exit(___);}
#if 0
# 441
{ 
# 442
T ret; 
# 443
tex2DLayeredLod(&ret, texObject, x, y, layer, level, isResident); 
# 444
return ret; 
# 445
} 
#endif
# 448 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 449
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 450
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 452
::exit(___);}
#if 0
# 450
{ 
# 451
__nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level); 
# 452
} 
#endif
# 454 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 455
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 456
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 460
::exit(___);}
#if 0
# 456
{ 
# 457
T ret; 
# 458
texCubemapLod(&ret, texObject, x, y, z, level); 
# 459
return ret; 
# 460
} 
#endif
# 463 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 464
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 465
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 467
::exit(___);}
#if 0
# 465
{ 
# 466
__nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
# 467
} 
#endif
# 469 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 470
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 471
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 475
::exit(___);}
#if 0
# 471
{ 
# 472
T ret; 
# 473
texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
# 474
return ret; 
# 475
} 
#endif
# 477 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 478
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 479
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 481
::exit(___);}
#if 0
# 479
{ 
# 480
__nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level); 
# 481
} 
#endif
# 483 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 484
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 485
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 489
::exit(___);}
#if 0
# 485
{ 
# 486
T ret; 
# 487
texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level); 
# 488
return ret; 
# 489
} 
#endif
# 491 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 492
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 493
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 495
::exit(___);}
#if 0
# 493
{ 
# 494
__nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy); 
# 495
} 
#endif
# 497 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 498
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 499
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 503
::exit(___);}
#if 0
# 499
{ 
# 500
T ret; 
# 501
tex1DGrad(&ret, texObject, x, dPdx, dPdy); 
# 502
return ret; 
# 503
} 
#endif
# 506 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 507
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 508
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 510
::exit(___);}
#if 0
# 508
{ 
# 509
__nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy); 
# 510
} 
#endif
# 512 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 513
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 514
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 518
::exit(___);}
#if 0
# 514
{ 
# 515
T ret; 
# 516
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy); 
# 517
return ret; 
# 518
} 
#endif
# 521 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 522
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 523
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 527
::exit(___);}
#if 0
# 523
{ 
# 524
unsigned char res; 
# 525
__nv_tex_surf_handler("__itex2DGrad_sparse", ptr, obj, x, y, &dPdx, &dPdy, &res); 
# 526
(*isResident) = (res != 0); 
# 527
} 
#endif
# 529 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 530
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 531
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 535
::exit(___);}
#if 0
# 531
{ 
# 532
T ret; 
# 533
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy, isResident); 
# 534
return ret; 
# 535
} 
#endif
# 539 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 540
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 541
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 543
::exit(___);}
#if 0
# 541
{ 
# 542
__nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
# 543
} 
#endif
# 545 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 546
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 547
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 551
::exit(___);}
#if 0
# 547
{ 
# 548
T ret; 
# 549
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
# 550
return ret; 
# 551
} 
#endif
# 554 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 555
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 556
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 560
::exit(___);}
#if 0
# 556
{ 
# 557
unsigned char res; 
# 558
__nv_tex_surf_handler("__itex3DGrad_sparse", ptr, obj, x, y, z, &dPdx, &dPdy, &res); 
# 559
(*isResident) = (res != 0); 
# 560
} 
#endif
# 562 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 563
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 564
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 568
::exit(___);}
#if 0
# 564
{ 
# 565
T ret; 
# 566
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy, isResident); 
# 567
return ret; 
# 568
} 
#endif
# 573 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 574
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 575
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 577
::exit(___);}
#if 0
# 575
{ 
# 576
__nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy); 
# 577
} 
#endif
# 579 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 580
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 581
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 585
::exit(___);}
#if 0
# 581
{ 
# 582
T ret; 
# 583
tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy); 
# 584
return ret; 
# 585
} 
#endif
# 588 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 589
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 590
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 590
{ 
# 591
__nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy); 
# 592
} 
#endif
# 594 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 595
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 596
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 600
::exit(___);}
#if 0
# 596
{ 
# 597
T ret; 
# 598
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy); 
# 599
return ret; 
# 600
} 
#endif
# 603 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 604
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 605
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 609
::exit(___);}
#if 0
# 605
{ 
# 606
unsigned char res; 
# 607
__nv_tex_surf_handler("__itex2DLayeredGrad_sparse", ptr, obj, x, y, layer, &dPdx, &dPdy, &res); 
# 608
(*isResident) = (res != 0); 
# 609
} 
#endif
# 611 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 612
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 613
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 617
::exit(___);}
#if 0
# 613
{ 
# 614
T ret; 
# 615
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy, isResident); 
# 616
return ret; 
# 617
} 
#endif
# 621 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 622
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 623
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 625
::exit(___);}
#if 0
# 623
{ 
# 624
__nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy); 
# 625
} 
#endif
# 627 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 628
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 629
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 633
::exit(___);}
#if 0
# 629
{ 
# 630
T ret; 
# 631
texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy); 
# 632
return ret; 
# 633
} 
#endif
# 58 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 59
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 60
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 79
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 89
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 98
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 99
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 100
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 102
::exit(___);}
#if 0
# 100
{ 
# 101
__nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode); 
# 102
} 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 105
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 106
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 110
::exit(___);}
#if 0
# 106
{ 
# 107
T ret; 
# 108
surf1Dread(&ret, surfObject, x, boundaryMode); 
# 109
return ret; 
# 110
} 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 113
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 114
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 116
::exit(___);}
#if 0
# 114
{ 
# 115
__nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode); 
# 116
} 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 119
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 120
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 124
::exit(___);}
#if 0
# 120
{ 
# 121
T ret; 
# 122
surf2Dread(&ret, surfObject, x, y, boundaryMode); 
# 123
return ret; 
# 124
} 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 128
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 129
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 131
::exit(___);}
#if 0
# 129
{ 
# 130
__nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode); 
# 131
} 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 134
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 135
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 139
::exit(___);}
#if 0
# 135
{ 
# 136
T ret; 
# 137
surf3Dread(&ret, surfObject, x, y, z, boundaryMode); 
# 138
return ret; 
# 139
} 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 142
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 143
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 145
::exit(___);}
#if 0
# 143
{ 
# 144
__nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode); 
# 145
} 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 148
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 149
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 153
::exit(___);}
#if 0
# 149
{ 
# 150
T ret; 
# 151
surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode); 
# 152
return ret; 
# 153
} 
#endif
# 155 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 156
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 157
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 159
::exit(___);}
#if 0
# 157
{ 
# 158
__nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode); 
# 159
} 
#endif
# 161 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 162
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 163
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 167
::exit(___);}
#if 0
# 163
{ 
# 164
T ret; 
# 165
surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode); 
# 166
return ret; 
# 167
} 
#endif
# 169 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 170
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 171
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 173
::exit(___);}
#if 0
# 171
{ 
# 172
__nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode); 
# 173
} 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 176
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 177
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 181
::exit(___);}
#if 0
# 177
{ 
# 178
T ret; 
# 179
surfCubemapread(&ret, surfObject, x, y, face, boundaryMode); 
# 180
return ret; 
# 181
} 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 184
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 185
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 187
::exit(___);}
#if 0
# 185
{ 
# 186
__nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode); 
# 187
} 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 190
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 191
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 195
::exit(___);}
#if 0
# 191
{ 
# 192
T ret; 
# 193
surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode); 
# 194
return ret; 
# 195
} 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 198
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 199
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 201
::exit(___);}
#if 0
# 199
{ 
# 200
__nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode); 
# 201
} 
#endif
# 203 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 204
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 205
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 207
::exit(___);}
#if 0
# 205
{ 
# 206
__nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode); 
# 207
} 
#endif
# 209 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 210
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 211
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 213
::exit(___);}
#if 0
# 211
{ 
# 212
__nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode); 
# 213
} 
#endif
# 215 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 216
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 217
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 219
::exit(___);}
#if 0
# 217
{ 
# 218
__nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode); 
# 219
} 
#endif
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 222
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 223
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 225
::exit(___);}
#if 0
# 223
{ 
# 224
__nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode); 
# 225
} 
#endif
# 227 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 228
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 229
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 231
::exit(___);}
#if 0
# 229
{ 
# 230
__nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode); 
# 231
} 
#endif
# 233 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 234
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 235
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 237
::exit(___);}
#if 0
# 235
{ 
# 236
__nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode); 
# 237
} 
#endif
# 3641 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 62 "/usr/include/c++/13/bits/stl_relops.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 66
namespace rel_ops { 
# 86
template< class _Tp> inline bool 
# 88
operator!=(const _Tp &__x, const _Tp &__y) 
# 89
{ return !(__x == __y); } 
# 99
template< class _Tp> inline bool 
# 101
operator>(const _Tp &__x, const _Tp &__y) 
# 102
{ return __y < __x; } 
# 112
template< class _Tp> inline bool 
# 114
operator<=(const _Tp &__x, const _Tp &__y) 
# 115
{ return !(__y < __x); } 
# 125
template< class _Tp> inline bool 
# 127
operator>=(const _Tp &__x, const _Tp &__y) 
# 128
{ return !(__x < __y); } 
# 129
}
# 132
}
# 41 "/usr/include/c++/13/initializer_list" 3
namespace std __attribute((__visibility__("default"))) { 
# 44
template< class _E> 
# 45
class initializer_list { 
# 48
public: typedef _E value_type; 
# 49
typedef const _E &reference; 
# 50
typedef const _E &const_reference; 
# 51
typedef size_t size_type; 
# 52
typedef const _E *iterator; 
# 53
typedef const _E *const_iterator; 
# 56
private: iterator _M_array; 
# 57
size_type _M_len; 
# 60
constexpr initializer_list(const_iterator __a, size_type __l) : _M_array(__a), _M_len(__l) 
# 61
{ } 
# 64
public: constexpr initializer_list() noexcept : _M_array((0)), _M_len((0)) 
# 65
{ } 
# 69
constexpr size_type size() const noexcept { return _M_len; } 
# 73
constexpr const_iterator begin() const noexcept { return _M_array; } 
# 77
constexpr const_iterator end() const noexcept { return begin() + size(); } 
# 78
}; 
# 86
template< class _Tp> constexpr const _Tp *
# 88
begin(initializer_list< _Tp>  __ils) noexcept 
# 89
{ return __ils.begin(); } 
# 97
template< class _Tp> constexpr const _Tp *
# 99
end(initializer_list< _Tp>  __ils) noexcept 
# 100
{ return __ils.end(); } 
# 101
}
# 82 "/usr/include/c++/13/utility" 3
namespace std __attribute((__visibility__("default"))) { 
# 94
template< class _Tp, class _Up = _Tp> inline _Tp 
# 97
exchange(_Tp &__obj, _Up &&__new_val) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_assignable< _Tp &, _Up> > ::value) 
# 100
{ return std::__exchange(__obj, std::forward< _Up> (__new_val)); } 
# 105
template< class _Tp> 
# 106
[[nodiscard]] constexpr add_const_t< _Tp>  &
# 108
as_const(_Tp &__t) noexcept 
# 109
{ return __t; } 
# 111
template < typename _Tp >
    void as_const ( const _Tp && ) = delete;
# 225 "/usr/include/c++/13/utility" 3
}
# 206 "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 207
cudaLaunchKernel(const T *
# 208
func, dim3 
# 209
gridDim, dim3 
# 210
blockDim, void **
# 211
args, size_t 
# 212
sharedMem = 0, cudaStream_t 
# 213
stream = 0) 
# 215
{ 
# 216
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 217
} 
# 277
template< class ...ExpTypes, class ...ActTypes> static inline cudaError_t 
# 278
cudaLaunchKernelEx(const cudaLaunchConfig_t *
# 279
config, void (*
# 280
kernel)(ExpTypes ...), ActTypes &&...
# 281
args) 
# 283
{ 
# 284
return [&](ExpTypes ...coercedArgs) { 
# 285
void *pArgs[] = {(&coercedArgs)...}; 
# 286
return ::cudaLaunchKernelExC(config, (const void *)(kernel), pArgs); 
# 287
} (std::forward< ActTypes> (args)...); 
# 288
} 
# 340
template< class T> static inline cudaError_t 
# 341
cudaLaunchCooperativeKernel(const T *
# 342
func, dim3 
# 343
gridDim, dim3 
# 344
blockDim, void **
# 345
args, size_t 
# 346
sharedMem = 0, cudaStream_t 
# 347
stream = 0) 
# 349
{ 
# 350
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 351
} 
# 384
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 385
event, unsigned 
# 386
flags) 
# 388
{ 
# 389
return ::cudaEventCreateWithFlags(event, flags); 
# 390
} 
# 428
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *
# 429
pGraphExec, cudaGraph_t 
# 430
graph, cudaGraphNode_t *
# 431
pErrorNode, char *
# 432
pLogBuffer, size_t 
# 433
bufferSize) 
# 435
{ 
# 436
(void)pErrorNode; 
# 437
(void)pLogBuffer; 
# 438
(void)bufferSize; 
# 439
return ::cudaGraphInstantiate(pGraphExec, graph, 0); 
# 440
} 
# 499
static inline cudaError_t cudaMallocHost(void **
# 500
ptr, size_t 
# 501
size, unsigned 
# 502
flags) 
# 504
{ 
# 505
return ::cudaHostAlloc(ptr, size, flags); 
# 506
} 
# 508
template< class T> static inline cudaError_t 
# 509
cudaHostAlloc(T **
# 510
ptr, size_t 
# 511
size, unsigned 
# 512
flags) 
# 514
{ 
# 515
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 516
} 
# 518
template< class T> static inline cudaError_t 
# 519
cudaHostGetDevicePointer(T **
# 520
pDevice, void *
# 521
pHost, unsigned 
# 522
flags) 
# 524
{ 
# 525
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 526
} 
# 628
template< class T> static inline cudaError_t 
# 629
cudaMallocManaged(T **
# 630
devPtr, size_t 
# 631
size, unsigned 
# 632
flags = 1) 
# 634
{ 
# 635
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 636
} 
# 646
template< class T> cudaError_t 
# 647
cudaMemAdvise(T *
# 648
devPtr, size_t 
# 649
count, cudaMemoryAdvise 
# 650
advice, cudaMemLocation 
# 651
location) 
# 653
{ 
# 654
return ::cudaMemAdvise_v2((const void *)devPtr, count, advice, location); 
# 655
} 
# 657
template< class T> static inline cudaError_t 
# 658
cudaMemPrefetchAsync(T *
# 659
devPtr, size_t 
# 660
count, cudaMemLocation 
# 661
location, unsigned 
# 662
flags, cudaStream_t 
# 663
stream = 0) 
# 665
{ 
# 666
return ::cudaMemPrefetchAsync_v2((const void *)devPtr, count, location, flags, stream); 
# 667
} 
# 749
template< class T> static inline cudaError_t 
# 750
cudaStreamAttachMemAsync(cudaStream_t 
# 751
stream, T *
# 752
devPtr, size_t 
# 753
length = 0, unsigned 
# 754
flags = 4) 
# 756
{ 
# 757
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 758
} 
# 760
template< class T> inline cudaError_t 
# 761
cudaMalloc(T **
# 762
devPtr, size_t 
# 763
size) 
# 765
{ 
# 766
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 767
} 
# 769
template< class T> static inline cudaError_t 
# 770
cudaMallocHost(T **
# 771
ptr, size_t 
# 772
size, unsigned 
# 773
flags = 0) 
# 775
{ 
# 776
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 777
} 
# 779
template< class T> static inline cudaError_t 
# 780
cudaMallocPitch(T **
# 781
devPtr, size_t *
# 782
pitch, size_t 
# 783
width, size_t 
# 784
height) 
# 786
{ 
# 787
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 788
} 
# 799
static inline cudaError_t cudaMallocAsync(void **
# 800
ptr, size_t 
# 801
size, cudaMemPool_t 
# 802
memPool, cudaStream_t 
# 803
stream) 
# 805
{ 
# 806
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
# 807
} 
# 809
template< class T> static inline cudaError_t 
# 810
cudaMallocAsync(T **
# 811
ptr, size_t 
# 812
size, cudaMemPool_t 
# 813
memPool, cudaStream_t 
# 814
stream) 
# 816
{ 
# 817
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 818
} 
# 820
template< class T> static inline cudaError_t 
# 821
cudaMallocAsync(T **
# 822
ptr, size_t 
# 823
size, cudaStream_t 
# 824
stream) 
# 826
{ 
# 827
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
# 828
} 
# 830
template< class T> static inline cudaError_t 
# 831
cudaMallocFromPoolAsync(T **
# 832
ptr, size_t 
# 833
size, cudaMemPool_t 
# 834
memPool, cudaStream_t 
# 835
stream) 
# 837
{ 
# 838
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 839
} 
# 878
template< class T> static inline cudaError_t 
# 879
cudaMemcpyToSymbol(const T &
# 880
symbol, const void *
# 881
src, size_t 
# 882
count, size_t 
# 883
offset = 0, cudaMemcpyKind 
# 884
kind = cudaMemcpyHostToDevice) 
# 886
{ 
# 887
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 888
} 
# 932
template< class T> static inline cudaError_t 
# 933
cudaMemcpyToSymbolAsync(const T &
# 934
symbol, const void *
# 935
src, size_t 
# 936
count, size_t 
# 937
offset = 0, cudaMemcpyKind 
# 938
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 939
stream = 0) 
# 941
{ 
# 942
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 943
} 
# 980
template< class T> static inline cudaError_t 
# 981
cudaMemcpyFromSymbol(void *
# 982
dst, const T &
# 983
symbol, size_t 
# 984
count, size_t 
# 985
offset = 0, cudaMemcpyKind 
# 986
kind = cudaMemcpyDeviceToHost) 
# 988
{ 
# 989
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 990
} 
# 1034
template< class T> static inline cudaError_t 
# 1035
cudaMemcpyFromSymbolAsync(void *
# 1036
dst, const T &
# 1037
symbol, size_t 
# 1038
count, size_t 
# 1039
offset = 0, cudaMemcpyKind 
# 1040
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 1041
stream = 0) 
# 1043
{ 
# 1044
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 1045
} 
# 1103
template< class T> static inline cudaError_t 
# 1104
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *
# 1105
pGraphNode, cudaGraph_t 
# 1106
graph, const cudaGraphNode_t *
# 1107
pDependencies, size_t 
# 1108
numDependencies, const T &
# 1109
symbol, const void *
# 1110
src, size_t 
# 1111
count, size_t 
# 1112
offset, cudaMemcpyKind 
# 1113
kind) 
# 1114
{ 
# 1115
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
# 1116
} 
# 1174
template< class T> static inline cudaError_t 
# 1175
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *
# 1176
pGraphNode, cudaGraph_t 
# 1177
graph, const cudaGraphNode_t *
# 1178
pDependencies, size_t 
# 1179
numDependencies, void *
# 1180
dst, const T &
# 1181
symbol, size_t 
# 1182
count, size_t 
# 1183
offset, cudaMemcpyKind 
# 1184
kind) 
# 1185
{ 
# 1186
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
# 1187
} 
# 1225
template< class T> static inline cudaError_t 
# 1226
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t 
# 1227
node, const T &
# 1228
symbol, const void *
# 1229
src, size_t 
# 1230
count, size_t 
# 1231
offset, cudaMemcpyKind 
# 1232
kind) 
# 1233
{ 
# 1234
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
# 1235
} 
# 1273
template< class T> static inline cudaError_t 
# 1274
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t 
# 1275
node, void *
# 1276
dst, const T &
# 1277
symbol, size_t 
# 1278
count, size_t 
# 1279
offset, cudaMemcpyKind 
# 1280
kind) 
# 1281
{ 
# 1282
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
# 1283
} 
# 1331
template< class T> static inline cudaError_t 
# 1332
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t 
# 1333
hGraphExec, cudaGraphNode_t 
# 1334
node, const T &
# 1335
symbol, const void *
# 1336
src, size_t 
# 1337
count, size_t 
# 1338
offset, cudaMemcpyKind 
# 1339
kind) 
# 1340
{ 
# 1341
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
# 1342
} 
# 1390
template< class T> static inline cudaError_t 
# 1391
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t 
# 1392
hGraphExec, cudaGraphNode_t 
# 1393
node, void *
# 1394
dst, const T &
# 1395
symbol, size_t 
# 1396
count, size_t 
# 1397
offset, cudaMemcpyKind 
# 1398
kind) 
# 1399
{ 
# 1400
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
# 1401
} 
# 1404
static inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) 
# 1405
{ 
# 1406
cudaGraphExecUpdateResultInfo resultInfo; 
# 1407
cudaError_t status = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo); 
# 1408
if (hErrorNode_out) { 
# 1409
(*hErrorNode_out) = (resultInfo.errorNode); 
# 1410
}  
# 1411
if (updateResult_out) { 
# 1412
(*updateResult_out) = (resultInfo.result); 
# 1413
}  
# 1414
return status; 
# 1415
} 
# 1443
template< class T> static inline cudaError_t 
# 1444
cudaUserObjectCreate(cudaUserObject_t *
# 1445
object_out, T *
# 1446
objectToWrap, unsigned 
# 1447
initialRefcount, unsigned 
# 1448
flags) 
# 1449
{ 
# 1450
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
# 1453
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
# 1456
} 
# 1458
template< class T> static inline cudaError_t 
# 1459
cudaUserObjectCreate(cudaUserObject_t *
# 1460
object_out, T *
# 1461
objectToWrap, unsigned 
# 1462
initialRefcount, cudaUserObjectFlags 
# 1463
flags) 
# 1464
{ 
# 1465
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
# 1466
} 
# 1493
template< class T> static inline cudaError_t 
# 1494
cudaGetSymbolAddress(void **
# 1495
devPtr, const T &
# 1496
symbol) 
# 1498
{ 
# 1499
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 1500
} 
# 1525
template< class T> static inline cudaError_t 
# 1526
cudaGetSymbolSize(size_t *
# 1527
size, const T &
# 1528
symbol) 
# 1530
{ 
# 1531
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 1532
} 
# 1577
template< class T> static inline cudaError_t 
# 1578
cudaFuncSetCacheConfig(T *
# 1579
func, cudaFuncCache 
# 1580
cacheConfig) 
# 1582
{ 
# 1583
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1584
} 
# 1586
template< class T> 
# 1588
__attribute((deprecated)) static inline cudaError_t 
# 1589
cudaFuncSetSharedMemConfig(T *
# 1590
func, cudaSharedMemConfig 
# 1591
config) 
# 1593
{ 
# 1595
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
# 1600
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1602
#pragma GCC diagnostic pop
# 1604
} 
# 1636
template< class T> inline cudaError_t 
# 1637
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1638
numBlocks, T 
# 1639
func, int 
# 1640
blockSize, size_t 
# 1641
dynamicSMemSize) 
# 1642
{ 
# 1643
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1644
} 
# 1688
template< class T> inline cudaError_t 
# 1689
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1690
numBlocks, T 
# 1691
func, int 
# 1692
blockSize, size_t 
# 1693
dynamicSMemSize, unsigned 
# 1694
flags) 
# 1695
{ 
# 1696
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1697
} 
# 1702
class __cudaOccupancyB2DHelper { 
# 1703
size_t n; 
# 1705
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1706
size_t operator()(int) 
# 1707
{ 
# 1708
return n; 
# 1709
} 
# 1710
}; 
# 1758
template< class UnaryFunction, class T> static inline cudaError_t 
# 1759
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1760
minGridSize, int *
# 1761
blockSize, T 
# 1762
func, UnaryFunction 
# 1763
blockSizeToDynamicSMemSize, int 
# 1764
blockSizeLimit = 0, unsigned 
# 1765
flags = 0) 
# 1766
{ 
# 1767
cudaError_t status; 
# 1770
int device; 
# 1771
cudaFuncAttributes attr; 
# 1774
int maxThreadsPerMultiProcessor; 
# 1775
int warpSize; 
# 1776
int devMaxThreadsPerBlock; 
# 1777
int multiProcessorCount; 
# 1778
int funcMaxThreadsPerBlock; 
# 1779
int occupancyLimit; 
# 1780
int granularity; 
# 1783
int maxBlockSize = 0; 
# 1784
int numBlocks = 0; 
# 1785
int maxOccupancy = 0; 
# 1788
int blockSizeToTryAligned; 
# 1789
int blockSizeToTry; 
# 1790
int blockSizeLimitAligned; 
# 1791
int occupancyInBlocks; 
# 1792
int occupancyInThreads; 
# 1793
size_t dynamicSMemSize; 
# 1799
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1800
return cudaErrorInvalidValue; 
# 1801
}  
# 1807
status = ::cudaGetDevice(&device); 
# 1808
if (status != (cudaSuccess)) { 
# 1809
return status; 
# 1810
}  
# 1812
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1816
if (status != (cudaSuccess)) { 
# 1817
return status; 
# 1818
}  
# 1820
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1824
if (status != (cudaSuccess)) { 
# 1825
return status; 
# 1826
}  
# 1828
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1832
if (status != (cudaSuccess)) { 
# 1833
return status; 
# 1834
}  
# 1836
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1840
if (status != (cudaSuccess)) { 
# 1841
return status; 
# 1842
}  
# 1844
status = cudaFuncGetAttributes(&attr, func); 
# 1845
if (status != (cudaSuccess)) { 
# 1846
return status; 
# 1847
}  
# 1849
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1855
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1856
granularity = warpSize; 
# 1858
if (blockSizeLimit == 0) { 
# 1859
blockSizeLimit = devMaxThreadsPerBlock; 
# 1860
}  
# 1862
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1863
blockSizeLimit = devMaxThreadsPerBlock; 
# 1864
}  
# 1866
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1867
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1868
}  
# 1870
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1872
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1876
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1877
blockSizeToTry = blockSizeLimit; 
# 1878
} else { 
# 1879
blockSizeToTry = blockSizeToTryAligned; 
# 1880
}  
# 1882
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1884
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1891
if (status != (cudaSuccess)) { 
# 1892
return status; 
# 1893
}  
# 1895
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1897
if (occupancyInThreads > maxOccupancy) { 
# 1898
maxBlockSize = blockSizeToTry; 
# 1899
numBlocks = occupancyInBlocks; 
# 1900
maxOccupancy = occupancyInThreads; 
# 1901
}  
# 1905
if (occupancyLimit == maxOccupancy) { 
# 1906
break; 
# 1907
}  
# 1908
}  
# 1916
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1917
(*blockSize) = maxBlockSize; 
# 1919
return status; 
# 1920
} 
# 1954
template< class UnaryFunction, class T> static inline cudaError_t 
# 1955
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1956
minGridSize, int *
# 1957
blockSize, T 
# 1958
func, UnaryFunction 
# 1959
blockSizeToDynamicSMemSize, int 
# 1960
blockSizeLimit = 0) 
# 1961
{ 
# 1962
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1963
} 
# 2000
template< class T> static inline cudaError_t 
# 2001
cudaOccupancyMaxPotentialBlockSize(int *
# 2002
minGridSize, int *
# 2003
blockSize, T 
# 2004
func, size_t 
# 2005
dynamicSMemSize = 0, int 
# 2006
blockSizeLimit = 0) 
# 2007
{ 
# 2008
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 2009
} 
# 2038
template< class T> static inline cudaError_t 
# 2039
cudaOccupancyAvailableDynamicSMemPerBlock(size_t *
# 2040
dynamicSmemSize, T 
# 2041
func, int 
# 2042
numBlocks, int 
# 2043
blockSize) 
# 2044
{ 
# 2045
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
# 2046
} 
# 2097
template< class T> static inline cudaError_t 
# 2098
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 2099
minGridSize, int *
# 2100
blockSize, T 
# 2101
func, size_t 
# 2102
dynamicSMemSize = 0, int 
# 2103
blockSizeLimit = 0, unsigned 
# 2104
flags = 0) 
# 2105
{ 
# 2106
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 2107
} 
# 2141
template< class T> static inline cudaError_t 
# 2142
cudaOccupancyMaxPotentialClusterSize(int *
# 2143
clusterSize, T *
# 2144
func, const cudaLaunchConfig_t *
# 2145
config) 
# 2146
{ 
# 2147
return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void *)func, config); 
# 2148
} 
# 2184
template< class T> static inline cudaError_t 
# 2185
cudaOccupancyMaxActiveClusters(int *
# 2186
numClusters, T *
# 2187
func, const cudaLaunchConfig_t *
# 2188
config) 
# 2189
{ 
# 2190
return ::cudaOccupancyMaxActiveClusters(numClusters, (const void *)func, config); 
# 2191
} 
# 2224
template< class T> inline cudaError_t 
# 2225
cudaFuncGetAttributes(cudaFuncAttributes *
# 2226
attr, T *
# 2227
entry) 
# 2229
{ 
# 2230
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 2231
} 
# 2286
template< class T> static inline cudaError_t 
# 2287
cudaFuncSetAttribute(T *
# 2288
entry, cudaFuncAttribute 
# 2289
attr, int 
# 2290
value) 
# 2292
{ 
# 2293
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2294
} 
# 2318
template< class T> static inline cudaError_t 
# 2319
cudaFuncGetName(const char **
# 2320
name, const T *
# 2321
func) 
# 2323
{ 
# 2324
return ::cudaFuncGetName(name, (const void *)func); 
# 2325
} 
# 2341
template< class T> static inline cudaError_t 
# 2342
cudaGetKernel(cudaKernel_t *
# 2343
kernelPtr, const T *
# 2344
entryFuncAddr) 
# 2346
{ 
# 2347
return ::cudaGetKernel(kernelPtr, (const void *)entryFuncAddr); 
# 2348
} 
# 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 66
const char *info_simulate = ("INFO:simulate[GNU]"); 
# 377 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((12 / 10000000) % 10)), (('0') + ((12 / 1000000) % 10)), (('0') + ((12 / 100000) % 10)), (('0') + ((12 / 10000) % 10)), (('0') + ((12 / 1000) % 10)), (('0') + ((12 / 100) % 10)), (('0') + ((12 / 10) % 10)), (('0') + (12 % 10)), '.', (('0') + ((4 / 10000000) % 10)), (('0') + ((4 / 1000000) % 10)), (('0') + ((4 / 100000) % 10)), (('0') + ((4 / 10000) % 10)), (('0') + ((4 / 1000) % 10)), (('0') + ((4 / 100) % 10)), (('0') + ((4 / 10) % 10)), (('0') + (4 % 10)), '.', (('0') + ((131 / 10000000) % 10)), (('0') + ((131 / 1000000) % 10)), (('0') + ((131 / 100000) % 10)), (('0') + ((131 / 10000) % 10)), (('0') + ((131 / 1000) % 10)), (('0') + ((131 / 100) % 10)), (('0') + ((131 / 10) % 10)), (('0') + (131 % 10)), ']', '\000'}; 
# 406 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((13 / 10000000) % 10)), (('0') + ((13 / 1000000) % 10)), (('0') + ((13 / 100000) % 10)), (('0') + ((13 / 10000) % 10)), (('0') + ((13 / 1000) % 10)), (('0') + ((13 / 100) % 10)), (('0') + ((13 / 10) % 10)), (('0') + (13 % 10)), '.', (('0') + ((2 / 10000000) % 10)), (('0') + ((2 / 1000000) % 10)), (('0') + ((2 / 100000) % 10)), (('0') + ((2 / 10000) % 10)), (('0') + ((2 / 1000) % 10)), (('0') + ((2 / 100) % 10)), (('0') + ((2 / 10) % 10)), (('0') + (2 % 10)), ']', '\000'}; 
# 426 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Linux]"); 
# 427
const char *info_arch = ("INFO:arch[]"); 
# 447 "CMakeCUDACompilerId.cu"
const char *info_language_standard_default = ("INFO:standard_default[17]"); 
# 465 "CMakeCUDACompilerId.cu"
const char *info_language_extensions_default = ("INFO:extensions_default[ON]"); 
# 476
int main(int argc, char *argv[]) 
# 477
{ 
# 478
int require = 0; 
# 479
require += (info_compiler[argc]); 
# 480
require += (info_platform[argc]); 
# 482
require += (info_version[argc]); 
# 485
require += (info_simulate[argc]); 
# 488
require += (info_simulate_version[argc]); 
# 490
require += (info_language_standard_default[argc]); 
# 491
require += (info_language_extensions_default[argc]); 
# 492
(void)argv; 
# 493
return require; 
# 494
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__740f3253_22_CMakeCUDACompilerId_cu_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
