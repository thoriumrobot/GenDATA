// Source-based slice around line 89
// Method: com.google.common.primitives.ImmutableLongArray.EMPTY

 *       {@code List} (though the most common utilities do have replacements here, and there is a
 *       lazy {@link #asList} view).
 * </ul>
 *
 * @since 22.0
 */
@GwtCompatible
@Immutable
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
