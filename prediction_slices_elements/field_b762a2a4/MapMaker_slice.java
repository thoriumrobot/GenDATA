// Source-based slice around line 93
// Method: com.google.common.collect.MapMaker.DEFAULT_CONCURRENCY_LEVEL

 * @author Bob Lee
 * @author Charles Fry
 * @author Kevin Bourrillion
 * @since 2.0
 */
@J2ktIncompatible
@GwtCompatible
public final class MapMaker {
  private static final int DEFAULT_INITIAL_CAPACITY = 16;
  private static final int DEFAULT_CONCURRENCY_LEVEL = 4;

  static final int UNSET_INT = -1;

  // TODO(kevinb): dispense with this after benchmarking
  boolean useCustomMap;

  int initialCapacity = UNSET_INT;
  int concurrencyLevel = UNSET_INT;

  @Nullable Strength keyStrength;
