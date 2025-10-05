// Source-based slice around line 57
// Method: com.google.common.util.concurrent.AtomicDoubleArray.serialVersionUID

 * }
 *
 * @author Doug Lea
 * @author Martin Buchholz
 * @since 11.0
 */
@GwtIncompatible
@J2ktIncompatible
public class AtomicDoubleArray implements Serializable {
  private static final long serialVersionUID = 0L;

  // Making this non-final is the lesser evil according to Effective
  // Java 2nd Edition Item 76: Write readObject methods defensively.
  private transient AtomicLongArray longs;

  /**
   * Creates a new {@code AtomicDoubleArray} of the given length, with all elements initially zero.
   *
   * @param length the length of the array
   */
