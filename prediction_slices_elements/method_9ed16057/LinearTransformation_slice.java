// Source-based slice around line 133
// Method: <com.google.common.math.LinearTransformation: LinearTransformation forNaN()>

    return new RegularLinearTransformation(slope, y);
  }

  /**
   * Builds an instance for datasets which contains {@link Double#NaN}. The {@link #isHorizontal}
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
