// Source-based slice around line 38
// Method: com.google.common.hash.SipHashFunction.SIP_HASH_24

/**
 * {@link HashFunction} implementation of SipHash-c-d.
 *
 * @author Kurt Alfred Kluever
 * @author Jean-Philippe Aumasson
 * @author Daniel J. Bernstein
 */
@Immutable
final class SipHashFunction extends AbstractHashFunction implements Serializable {
  static final HashFunction SIP_HASH_24 =
      new SipHashFunction(2, 4, 0x0706050403020100L, 0x0f0e0d0c0b0a0908L);

  // The number of compression rounds.
  private final int c;
  // The number of finalization rounds.
  private final int d;
  // Two 64-bit keys (represent a single 128-bit key).
  private final long k0;
  private final long k1;

