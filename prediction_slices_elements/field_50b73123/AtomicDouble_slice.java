// Source-based slice around line 65
// Method: com.google.common.util.concurrent.AtomicDouble.serialVersionUID

 *
 * @author Doug Lea
 * @author Martin Buchholz
 * @since 11.0
 */
@GwtIncompatible
@J2ktIncompatible
@ReflectionSupport(value = ReflectionSupport.Level.FULL)
public class AtomicDouble extends Number {
  private static final long serialVersionUID = 0L;

  private transient volatile long value;

  private static final AtomicLongFieldUpdater<AtomicDouble> updater =
      AtomicLongFieldUpdater.newUpdater(AtomicDouble.class, "value");

  /**
   * Creates a new {@code AtomicDouble} with the given initial value.
   *
   * @param initialValue the initial value
