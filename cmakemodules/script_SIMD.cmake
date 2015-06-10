# SSE{2,3,4} extensions? 
# This file is an adaptation of cmakemodules/script_SIMD.cmake from "MRPT"
# ===================================================
SET(AUTODETECT_SSE ON CACHE BOOL "Check /proc/cpuinfo to determine if SSE{2,3,4} and AVX optimizations are available")
MARK_AS_ADVANCED(AUTODETECT_SSE)

# Read info about CPUs:
SET(DO_SSE_AUTODETECT 0)
IF(AUTODETECT_SSE AND EXISTS "/proc/cpuinfo")
	SET(DO_SSE_AUTODETECT 1)
ENDIF(AUTODETECT_SSE AND EXISTS "/proc/cpuinfo")

IF (DO_SSE_AUTODETECT)
	FILE(READ "/proc/cpuinfo" CPU_INFO)
ENDIF (DO_SSE_AUTODETECT)

# Macro for each SSE* var: Invoke with name in uppercase:
macro(DEFINE_SSE_VAR  _setname)
	string(TOLOWER ${_setname} _set)

	IF (DO_SSE_AUTODETECT)
		# Automatic detection:
		SET(CMAKE_HAS_${_setname} 0)
		IF (${CPU_INFO} MATCHES ".*${_set}.*")
			SET(CMAKE_HAS_${_setname} 1)
		ENDIF()
	ELSE (DO_SSE_AUTODETECT)
		# Manual:
		SET("DISABLE_${_setname}" OFF CACHE BOOL "Forces compilation WITHOUT ${_setname} extensions")
		MARK_AS_ADVANCED("DISABLE_${_setname}")
		SET(CMAKE_HAS_${_setname} 0)
		IF (NOT DISABLE_${_setname})
			SET(CMAKE_HAS_${_setname} 1)
		ENDIF (NOT DISABLE_${_setname})	
	ENDIF (DO_SSE_AUTODETECT)
endmacro(DEFINE_SSE_VAR)

# SSE optimizations:
DEFINE_SSE_VAR(SSE2)
DEFINE_SSE_VAR(SSE3)
DEFINE_SSE_VAR(SSE4_1)
DEFINE_SSE_VAR(SSE4_2)
#DEFINE_SSE_VAR(SSE4_A)
DEFINE_SSE_VAR(AVX)
#DEFINE_SSE_VAR(AVX2)
#DEFINE_SSE_VAR(-march=, -mfpmath=sse)

add_definitions(-D_SSE2=${CMAKE_HAS_SSE2})
add_definitions(-D_SSE3=${CMAKE_HAS_SSE3})
add_definitions(-D_SSE4_1=${CMAKE_HAS_SSE4_1})
add_definitions(-D_SSE4_2=${CMAKE_HAS_SSE4_2})
#add_definitions(-D_SSE4_A=${CMAKE_HAS_SSE4_A})
add_definitions(-D_AVX=${CMAKE_HAS_AVX})

