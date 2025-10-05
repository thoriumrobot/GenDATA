// Source-based slice around line 148
// Method: <com.google.common.math.LinearTransformation: double slope()>


  /** Returns whether this is a horizontal transformation. */
  public abstract boolean isHorizontal();

  /**
   * Returns the slope of the transformation, i.e. the rate of change of {@code y} with respect to
   * {@code x}. This must not be called on a vertical transformation (i.e. when {@link
   * #isVertical()} is true).
   */
  public abstract double slope();

  /**
   * Returns the {@code y} corresponding to the given {@code x}. This must not be called on a
   * vertical transformation (i.e. when {@link #isVertical()} is true).
   */
  public abstract double transform(double x);

  /**
   * Returns the inverse linear transformation. The inverse of a horizontal transformation is a
   * vertical transformation, and vice versa. The inverse of the {@link #forNaN} transformation is
