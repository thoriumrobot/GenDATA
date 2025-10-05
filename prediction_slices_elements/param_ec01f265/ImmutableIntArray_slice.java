// Source-based slice around line 97
// Method: <com.google.common.primitives.ImmutableIntArray: ImmutableIntArray of(int)>

public final class ImmutableIntArray implements Serializable {
  private static final ImmutableIntArray EMPTY = new ImmutableIntArray(new int[0]);

  /** Returns the empty array. */
  public static ImmutableIntArray of() {
    return EMPTY;
  }

  /** Returns an immutable array containing a single value. */
  public static ImmutableIntArray of(int e0) {
    return new ImmutableIntArray(new int[] {e0});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableIntArray of(int e0, int e1) {
    return new ImmutableIntArray(new int[] {e0, e1});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableIntArray of(int e0, int e1, int e2) {
