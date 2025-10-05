// Source-based slice around line 39
// Method: <com.google.common.collect.BoundType: BoundType forBoolean(boolean)>

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
