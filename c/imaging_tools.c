#include <inttypes.h>
#include <memory.h>

#define EXPORT __declspec(dllexport)

EXPORT int unpack_raw_data(uint8_t *raw, uint16_t *out, uint32_t N, uint8_t bits, uint8_t packed)
{
	uint32_t n;
	uint8_t *rp = raw;
	uint16_t *dp = out;
	
	if ( packed )
	{
		if ( bits!= 12 ) return -1; // only implemented for Mono12Packed data
		for ( n=N; n>0; --n, ++dp,++rp )
		{
			*dp = ( uint16_t )*rp << 4;
			*dp |= *++rp >> 4;
			if ( --n == 0 ) break; ++dp;
			*dp = *rp & 15u;
			*dp |= ( uint16_t )*++rp << 4;
		}
	}
	else
	{
		if ( bits <=8 ) // insert spacing byte(s)
			for ( n=N; n>0; --n, ++dp, ++rp )
				*dp = *rp;
		else if ( bits <= 16 ) // assume hi bits are properly zero'd
			memcpy(dp,rp,N*2);
		/*
		else if ( bits <= 16 ) // don't trust the hi bits
		{
			uint16_t *rp2 = ( uint16_t* )raw;
			uint16_t mask = ~0 >> ( 16 - bits );
			for ( n=N; n>0; --n, ++dp, ++rp2 )
				*dp = *rp2 & mask;
		}
		*/
		else
			return -1;
	}
	
	return 0;
}
