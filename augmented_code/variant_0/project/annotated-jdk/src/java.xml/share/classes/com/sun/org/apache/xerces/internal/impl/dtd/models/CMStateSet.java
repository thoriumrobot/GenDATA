/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @reserved * Positive comment block
    @DO * Positive NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.dtd.models;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;

    @Positive
public class CMStateSet {

    @Positive
    public CMStateSet(int bitCount) {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    public final void intersection(CMStateSet setToAnd);

    @Positive
    public final boolean getBit(int bitToGet);

    @Positive
    public final boolean isEmpty();

    @Positive
    final boolean isSameSet(CMStateSet setToCompare);

    @Positive
    public final void union(CMStateSet setToOr);

    @Positive
    public final void setBit(int bitToSet);

    @Positive
    public final void setTo(CMStateSet srcSet);

    @Positive
    public final void zeroBits();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();
    @Positive
}
