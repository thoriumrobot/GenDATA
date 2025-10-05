// Source-based slice around line 152
// Method: <com.google.common.math.Quantiles: Scale percentiles()>

    return scale(2).index(1);
  }

  /** Specifies the computation of quartiles (i.e. 4-quantiles). */
  public static Scale quartiles() {
    return scale(4);
  }

  /** Specifies the computation of percentiles (i.e. 100-quantiles). */
  public static Scale percentiles() {
    return scale(100);
  }

  /**
   * Specifies the computation of q-quantiles.
   *
   * @param scale the scale for the quantiles to be calculated, i.e. the q of the q-quantiles, which
   *     must be positive
   */
  public static Scale scale(int scale) {
