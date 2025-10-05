// Source-based slice around line 97
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray of(long)>

public final class ImmutableLongArray implements Serializable {
  private static final ImmutableLongArray EMPTY = new ImmutableLongArray(new long[0]);

  /** Returns the empty array. */
  public static ImmutableLongArray of() {
    return EMPTY;
  }

  /** Returns an immutable array containing a single value. */
  public static ImmutableLongArray of(long e0) {
    return new ImmutableLongArray(new long[] {e0});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray of(long e0, long e1) {
    return new ImmutableLongArray(new long[] {e0, e1});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray of(long e0, long e1, long e2) {
