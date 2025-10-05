/*
    @Positive
    @Positive
    @Positive
TODO: Implement the functionality for @PolyGuardedBy and uncomment this.

    @Positive
    @Positive
    @Positive
package org.checkerframework.checker.lock.qual;

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

    @Positive
    @Positive
    @Positive
import org.checkerframework.framework.qual.PolymorphicQualifier;

/**
    @Positive
    @Positive
    @Positive
 * A polymorphic qualifier for the GuardedBy type system.
    @Positive
    @NonNegative
    @Positive
 * Indicates that it is unknown what the guards are or whether they are held.
    @Positive
    @Positive
    @Positive
 * An expression whose type is {@code @PolyGuardedBy} cannot be dereferenced.
    @Positive
    @Positive
    @Positive
 * Hence, unlike for {@code @GuardSatisfied}, when an expression of type {@code @PolyGuardedBy}
    @Positive
    @Positive
    @Positive
 * is the LHS of an assignment, the locks guarding the RHS do not need to be held.
    @Positive
    @NonNegative
    @Positive
 *
    @Positive
    @NonNegative
    @Positive
 * <p>Any method written using {@code @PolyGuardedBy} conceptually has an
    @Positive
    @NonNegative
    @Positive
 * arbitrary number of versions:  one in which every instance of
    @Positive
    @Positive
    @Positive
 * {@code @PolyGuardedBy} has been replaced by {@code @}{@link GuardedByUnknown},
    @Positive
    @Positive
    @Positive
 * one in which every instance of {@code @PolyGuardedBy} has been
    @Positive
    @Positive
    @Positive
 * replaced by {@code @}{@link GuardedByBottom}, and ones in which every
    @Positive
    @Positive
    @Positive
 * instance of {@code @PolyGuardedBy} has been replaced by {@code @}{@link GuardedBy},
    @Positive
    @Positive
    @Positive
 * for every possible combination of map arguments.
    @Positive
    @NonNegative
    @Positive
 *
    @Positive
    @Positive
    @Positive
 * @see GuardedBy
    @Positive
    @Positive
    @Positive
 * @checker_framework.manual #lock-checker Lock Checker
    @Positive
    @Positive
    @Positive
 * @checker_framework.manual #qualifier-polymorphism Qualifier polymorphism
    @Positive
    @NonNegative
    @Positive
 */
// @PolymorphicQualifier(GuardedByUnknown.class)
// @Documented
// @Retention(RetentionPolicy.RUNTIME)
// @Target({ElementType.TYPE_USE, ElementType.TYPE_PARAMETER})
// public @interface PolyGuardedBy {}

// CFWR semantic augmentation - variant 1
