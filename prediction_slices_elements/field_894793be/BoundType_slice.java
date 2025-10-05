// Source-based slice around line 32
// Method: com.google.common.collect.BoundType.inclusive

 *
 * @since 10.0
 */
@GwtCompatible
public enum BoundType {
  /** The endpoint value <i>is not</i> considered part of the set ("exclusive"). */
  OPEN(false),
  CLOSED(true);

  final boolean inclusive;

  BoundType(boolean inclusive) {
    this.inclusive = inclusive;
  }

  /** Returns the bound type corresponding to a boolean value for inclusivity. */
  static BoundType forBoolean(boolean inclusive) {
    return inclusive ? CLOSED : OPEN;
  }
}
