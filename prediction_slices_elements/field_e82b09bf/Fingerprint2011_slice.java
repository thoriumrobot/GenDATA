// Source-based slice around line 32
// Method: com.google.common.hash.Fingerprint2011.FINGERPRINT_2011

 *   <li>comparisons (signed values can be negative)
 *   <li>division (avoided here)
 *   <li>shifting (right shift must be unsigned)
 * </ul>
 *
 * @author kylemaddison@google.com (Kyle Maddison)
 * @author gpike@google.com (Geoff Pike)
 */
final class Fingerprint2011 extends AbstractNonStreamingHashFunction {
  static final HashFunction FINGERPRINT_2011 = new Fingerprint2011();

  // Some primes between 2^63 and 2^64 for various uses.
  private static final long K0 = 0xa5b85c5e198ed849L;
  private static final long K1 = 0x8d58ac26afe12e47L;
  private static final long K2 = 0xc47b6e9e3a970ed3L;
  private static final long K3 = 0xc6a4a7935bd1e995L;

  @Override
  public HashCode hashBytes(byte[] input, int off, int len) {
    checkPositionIndexes(off, off + len, input.length);
