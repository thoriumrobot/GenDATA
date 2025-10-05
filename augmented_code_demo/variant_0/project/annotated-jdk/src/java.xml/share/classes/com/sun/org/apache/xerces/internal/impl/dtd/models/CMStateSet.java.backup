/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.impl.dtd.models;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;

public class CMStateSet {

    public CMStateSet(int bitCount) {
    }

    public String toString();

    public final void intersection(CMStateSet setToAnd);

    public final boolean getBit(int bitToGet);

    public final boolean isEmpty();

    final boolean isSameSet(CMStateSet setToCompare);

    public final void union(CMStateSet setToOr);

    public final void setBit(int bitToSet);

    public final void setTo(CMStateSet srcSet);

    public final void zeroBits();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();
}
