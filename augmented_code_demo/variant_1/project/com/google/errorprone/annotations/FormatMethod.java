    @Positive
    @Positive
    @Positive
package com.google.errorprone.annotations;

    @Positive
    @Positive
    @Positive
import java.lang.annotation.Documented;
    @Positive
    @Positive
    @Positive
import java.lang.annotation.ElementType;
    @Positive
    @Positive
    @Positive
import java.lang.annotation.Retention;
    @Positive
    @Positive
    @Positive
import java.lang.annotation.RetentionPolicy;
    @Positive
    @Positive
    @Positive
import java.lang.annotation.Target;

/**
    @Positive
    @Positive
    @Positive
 * Annotation for a method that takes a printf-style format string as an argument followed by
    @Positive
    @Positive
    @Positive
 * arguments for that format string.
    @Positive
    @NonNegative
    @Positive
 */
    @Positive
    @Positive
    @Positive
@Documented
    @Positive
    @Positive
    @Positive
@Retention(RetentionPolicy.RUNTIME)
    @Positive
    @Positive
    @Positive
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR})
    @Positive
    @Positive
    @Positive
public @interface FormatMethod {}

// CFWR semantic augmentation - variant 1
