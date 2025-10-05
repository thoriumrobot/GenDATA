// Source-based slice around line 53
// Method: <com.google.common.math.LinearTransformation: LinearTransformationBuilder mapping(double,double)>

   */
  @Deprecated
  public LinearTransformation() {}

  /**
   * Start building an instance which maps {@code x = x1} to {@code y = y1}. Both arguments must be
   * finite. Call either {@link LinearTransformationBuilder#and} or {@link
   * LinearTransformationBuilder#withSlope} on the returned object to finish building the instance.
   */
  public static LinearTransformationBuilder mapping(double x1, double y1) {
    checkArgument(isFinite(x1) && isFinite(y1));
    return new LinearTransformationBuilder(x1, y1);
  }

  /**
   * This is an intermediate stage in the construction process. It is returned by {@link
   * LinearTransformation#mapping}. You almost certainly don't want to keep instances around, but
   * instead use method chaining. This represents a single point mapping, i.e. a mapping between one
   * {@code x} and {@code y} value pair.
   *
