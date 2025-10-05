// Source-based slice around line 34
// Method: <com.google.common.collect.testing.features.FeatureEnumTest: void assertGoodTesterAnnotation(Class)>

import junit.framework.TestCase;

/**
 * Since annotations have some reusability issues that force copy and paste all over the place, it's
 * worth having a test to ensure that all our Feature enums have their annotations correctly set up.
 *
 * @author George van den Driessche
 */
public class FeatureEnumTest extends TestCase {
  private static void assertGoodTesterAnnotation(Class<? extends Annotation> annotationClass) {
    assertNotNull(
        rootLocaleFormat("%s must be annotated with @TesterAnnotation.", annotationClass),
        annotationClass.getAnnotation(TesterAnnotation.class));
    Retention retentionPolicy = annotationClass.getAnnotation(Retention.class);
    assertNotNull(
        rootLocaleFormat("%s must have a @Retention annotation.", annotationClass),
        retentionPolicy);
    assertEquals(
        rootLocaleFormat("%s must have RUNTIME RetentionPolicy.", annotationClass),
        RetentionPolicy.RUNTIME,
