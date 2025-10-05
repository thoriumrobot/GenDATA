// Source-based slice around line 89
// Method: com.google.common.primitives.ImmutableDoubleArray.EMPTY

 *       {@code List} (though the most common utilities do have replacements here, and there is a
 *       lazy {@link #asList} view).
 * </ul>
 *
 * @since 22.0
 */
@GwtCompatible
@Immutable
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
