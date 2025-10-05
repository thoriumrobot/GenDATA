// Source-based slice around line 142
// Method: <com.google.common.math.Quantiles: ScaleAndIndex median()>

   * Constructor for a type that is not meant to be instantiated.
   *
   * @deprecated Use the static factory methods of the class. There is no reason to create an
   *     instance of {@link Quantiles}.
   */
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
