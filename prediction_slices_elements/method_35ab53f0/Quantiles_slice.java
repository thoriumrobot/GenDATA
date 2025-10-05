// Source-based slice around line 147
// Method: <com.google.common.math.Quantiles: Scale quartiles()>

  @Deprecated
  public Quantiles() {}

  /** Specifies the computation of a median (i.e. the 1st 2-quantile). */
  public static ScaleAndIndex median() {
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
