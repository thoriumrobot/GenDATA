// Source-based slice around line 162
// Method: <com.google.common.math.Quantiles: Scale scale(int)>

    return scale(100);
  }

  /**
   * Specifies the computation of q-quantiles.
   *
   * @param scale the scale for the quantiles to be calculated, i.e. the q of the q-quantiles, which
   *     must be positive
   */
  public static Scale scale(int scale) {
    return new Scale(scale);
  }

  /**
   * Describes the point in a fluent API chain where only the scale (i.e. the q in q-quantiles) has
   * been specified.
   *
   * @since 20.0
   */
  public static final class Scale {
