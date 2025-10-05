// Source-based slice around line 89
// Method: com.google.common.primitives.ImmutableIntArray.EMPTY

 *       {@code List} (though the most common utilities do have replacements here, and there is a
 *       lazy {@link #asList} view).
 * </ul>
 *
 * @since 22.0
 */
@GwtCompatible
@Immutable
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
