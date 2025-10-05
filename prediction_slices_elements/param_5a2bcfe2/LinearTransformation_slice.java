// Source-based slice around line 121
// Method: <com.google.common.math.LinearTransformation: LinearTransformation horizontal(double)>

  public static LinearTransformation vertical(double x) {
    checkArgument(isFinite(x));
    return new VerticalLinearTransformation(x);
  }

  /**
   * Builds an instance representing a horizontal transformation with a constant value of {@code y}.
   * (The inverse of this will be a vertical transformation.)
   */
  public static LinearTransformation horizontal(double y) {
    checkArgument(isFinite(y));
    double slope = 0.0;
    return new RegularLinearTransformation(slope, y);
  }

  /**
   * Builds an instance for datasets which contains {@link Double#NaN}. The {@link #isHorizontal}
   * and {@link #isVertical} methods return {@code false} and the {@link #slope}, and {@link
   * #transform} methods all return {@link Double#NaN}. The {@link #inverse} method returns the same
   * instance.
