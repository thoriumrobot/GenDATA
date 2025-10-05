// Source-based slice around line 97
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray of(double)>

public final class ImmutableDoubleArray implements Serializable {
  private static final ImmutableDoubleArray EMPTY = new ImmutableDoubleArray(new double[0]);

  /** Returns the empty array. */
  public static ImmutableDoubleArray of() {
    return EMPTY;
  }

  /** Returns an immutable array containing a single value. */
  public static ImmutableDoubleArray of(double e0) {
    return new ImmutableDoubleArray(new double[] {e0});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray of(double e0, double e1) {
    return new ImmutableDoubleArray(new double[] {e0, e1});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray of(double e0, double e1, double e2) {
