// Source-based slice around line 112
// Method: <com.google.common.math.LinearTransformation: LinearTransformation vertical(double)>

        return new VerticalLinearTransformation(x1);
      }
    }
  }

  /**
   * Builds an instance representing a vertical transformation with a constant value of {@code x}.
   * (The inverse of this will be a horizontal transformation.)
   */
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
