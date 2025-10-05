// Source-based slice around line 37
// Method: <com.google.common.base.FinalizableReference: void finalizeReferent()>

@DoNotMock("Use an instance of one of the Finalizable*Reference classes")
@J2ktIncompatible
@GwtIncompatible
public interface FinalizableReference {
  /**
   * Invoked on a background thread after the referent has been garbage collected unless security
   * restrictions prevented starting a background thread, in which case this method is invoked when
   * new references are created.
   */
  void finalizeReferent();
}
