// Source-based slice around line 154
// Method: <com.google.common.math.LinearTransformation: double transform(double)>

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
   * itself. In all other cases, the inverse is a transformation such that applying both the
   * original transformation and its inverse to a value gives you the original value give-or-take
   * numerical errors. Calling this method multiple times on the same instance will always return
   * the same instance. Calling this method on the result of calling this method on an instance will
   * always return that original instance.
   */
