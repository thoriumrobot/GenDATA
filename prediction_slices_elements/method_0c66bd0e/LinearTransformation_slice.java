// Source-based slice around line 138
// Method: <com.google.common.math.LinearTransformation: boolean isVertical()>

   * and {@link #isVertical} methods return {@code false} and the {@link #slope}, and {@link
   * #transform} methods all return {@link Double#NaN}. The {@link #inverse} method returns the same
   * instance.
   */
  public static LinearTransformation forNaN() {
    return NaNLinearTransformation.INSTANCE;
  }

  /** Returns whether this is a vertical transformation. */
  public abstract boolean isVertical();

  /** Returns whether this is a horizontal transformation. */
  public abstract boolean isHorizontal();

  /**
   * Returns the slope of the transformation, i.e. the rate of change of {@code y} with respect to
   * {@code x}. This must not be called on a vertical transformation (i.e. when {@link
   * #isVertical()} is true).
   */
  public abstract double slope();
